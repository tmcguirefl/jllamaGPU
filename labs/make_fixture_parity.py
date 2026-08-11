#!/usr/bin/env python3
"""Write a tiny Llama F16 GGUF with weights + GPT-2 BPE vocab for M6 parity.

llama.cpp requires fuller metadata than jllama M4 (e.g. rms eps).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32

VT_UINT32 = 4
VT_INT32 = 5
VT_FLOAT32 = 6
VT_BOOL = 7
VT_STRING = 8
VT_ARRAY = 9

# tensor types (ggml_type)
TTYPE_F32 = 0
TTYPE_F16 = 1
# metadata array element types = value types
ARR_F32 = VT_FLOAT32
ARR_I32 = VT_INT32
ARR_STRING = VT_STRING


def align_up(n: int, a: int = ALIGN) -> int:
    return n + (a - n % a) % a


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Writer:
    def __init__(self) -> None:
        self.buf = bytearray()

    def raw(self, b: bytes | bytearray) -> None:
        self.buf.extend(b)

    def u32(self, v: int) -> None:
        self.raw(struct.pack("<I", int(v)))

    def u64(self, v: int) -> None:
        self.raw(struct.pack("<Q", int(v)))

    def f32(self, v: float) -> None:
        self.raw(struct.pack("<f", float(v)))

    def i32(self, v: int) -> None:
        self.raw(struct.pack("<i", int(v)))

    def string(self, s: str) -> None:
        b = s.encode("utf-8")
        self.u64(len(b))
        self.raw(b)

    def kv_string(self, k: str, v: str) -> None:
        self.string(k)
        self.u32(VT_STRING)
        self.string(v)

    def kv_u32(self, k: str, v: int) -> None:
        self.string(k)
        self.u32(VT_UINT32)
        self.u32(v)

    def kv_f32(self, k: str, v: float) -> None:
        self.string(k)
        self.u32(VT_FLOAT32)
        self.f32(v)

    def kv_bool(self, k: str, v: bool) -> None:
        self.string(k)
        self.u32(VT_BOOL)
        self.raw(bytes([1 if v else 0]))

    def kv_arr_str(self, k: str, arr: list[str]) -> None:
        self.string(k)
        self.u32(VT_ARRAY)
        self.u32(ARR_STRING)
        self.u64(len(arr))
        for s in arr:
            self.string(s)

    def kv_arr_f32(self, k: str, arr: list[float]) -> None:
        self.string(k)
        self.u32(VT_ARRAY)
        self.u32(ARR_F32)
        self.u64(len(arr))
        for x in arr:
            self.f32(x)

    def kv_arr_i32(self, k: str, arr: list[int]) -> None:
        self.string(k)
        self.u32(VT_ARRAY)
        self.u32(ARR_I32)
        self.u64(len(arr))
        for x in arr:
            self.i32(x)


def pack_tensor_info(w: Writer, name: str, dims: list[int], typ: int, offset: int) -> None:
    w.string(name)
    w.u32(len(dims))
    for d in dims:
        w.u64(d)
    w.u32(typ)
    w.u64(offset)


def gpt2_bpe_encode(text: str, encoder: dict[str, int], ranks: dict[tuple[str, str], int]) -> list[int]:
    b2u = bytes_to_unicode()
    chars = [b2u[b] for b in text.encode("utf-8")]
    if not chars:
        return []

    def get_pairs(word):
        pairs = set()
        prev = word[0]
        for c in word[1:]:
            pairs.add((prev, c))
            prev = c
        return pairs

    word = chars[:]
    while True:
        pairs = get_pairs(word)
        if not pairs:
            break
        bigram = min(pairs, key=lambda p: ranks.get(p, 10**9))
        if bigram not in ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
    return [encoder[t] for t in word]


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/tiny_parity_f16.gguf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Larger than M4 fixture so generation is less degenerate, still tiny.
    n_embd = 64
    n_head = 4
    n_layer = 2
    n_ff = 128
    theta = 10000.0
    n_ctx = 128
    rms_eps = 1e-5
    d_head = n_embd // n_head

    b2u = bytes_to_unicode()
    specials = ["<unk>", "<s>", "</s>"]
    byte_tokens = [b2u[b] for b in range(256)]
    a, bch = b2u[ord("a")], b2u[ord("b")]
    sp = b2u[32]
    c = b2u[ord("c")]
    merge_pairs = [
        (a, bch),  # ab
        (sp, a),  # Ġa
        (sp + a, bch),  # Ġab
        (a, c),  # ac
        (b2u[ord("h")], b2u[ord("e")]),  # he
        (b2u[ord("l")], b2u[ord("l")]),  # ll
        ("he", "ll"),  # hell
        ("hell", b2u[ord("o")]),  # hello
    ]
    tokens = specials + byte_tokens
    merges: list[str] = []
    for left, right in merge_pairs:
        merged = left + right
        merges.append(f"{left} {right}")
        if merged not in tokens:
            tokens.append(merged)

    n_vocab = len(tokens)
    encoder = {t: i for i, t in enumerate(tokens)}
    ranks = {}
    for i, m in enumerate(merges):
        left, right = m.split(" ", 1)
        ranks[(left, right)] = i

    scores = [0.0] * n_vocab
    for i, m in enumerate(merges):
        left, right = m.split(" ", 1)
        scores[encoder[left + right]] = float(-i)
    token_types = [1] * n_vocab
    token_types[0] = 2
    token_types[1] = 3
    token_types[2] = 3

    rng = np.random.default_rng(42)

    def mat(shape, scale=0.02):
        return (scale * rng.standard_normal(shape)).astype(np.float32)

    wte_j = mat((n_vocab, n_embd), 0.05)
    out_j = mat((n_embd, n_vocab), 0.03)
    output_norm = np.ones(n_embd, dtype=np.float32)

    specs: list[tuple[str, list[int], int, np.ndarray, str]] = [
        ("token_embd.weight", [n_embd, n_vocab], TTYPE_F16, wte_j, "embd"),
    ]
    for li in range(n_layer):
        p = f"blk.{li}."
        specs.extend(
            [
                (p + "attn_norm.weight", [n_embd], TTYPE_F32, np.linspace(0.5, 1.5, n_embd, dtype=np.float32), "vec"),
                (p + "attn_q.weight", [n_embd, n_embd], TTYPE_F16, mat((n_embd, n_embd)), "w"),
                (p + "attn_k.weight", [n_embd, n_embd], TTYPE_F16, mat((n_embd, n_embd)), "w"),
                (p + "attn_v.weight", [n_embd, n_embd], TTYPE_F16, mat((n_embd, n_embd)), "w"),
                (p + "attn_output.weight", [n_embd, n_embd], TTYPE_F16, mat((n_embd, n_embd)), "w"),
                (p + "ffn_norm.weight", [n_embd], TTYPE_F32, np.linspace(1.0, 0.25, n_embd, dtype=np.float32), "vec"),
                (p + "ffn_gate.weight", [n_embd, n_ff], TTYPE_F16, mat((n_embd, n_ff)), "w"),
                (p + "ffn_up.weight", [n_embd, n_ff], TTYPE_F16, mat((n_embd, n_ff)), "w"),
                (p + "ffn_down.weight", [n_ff, n_embd], TTYPE_F16, mat((n_ff, n_embd)), "w"),
            ]
        )
    specs.extend(
        [
            ("output_norm.weight", [n_embd], TTYPE_F32, output_norm, "vec"),
            ("output.weight", [n_embd, n_vocab], TTYPE_F16, out_j, "w"),
        ]
    )

    file_arrays = []
    for name, dims, typ, arr, kind in specs:
        dt = np.float16 if typ == TTYPE_F16 else np.float32
        if kind == "vec":
            data = np.ascontiguousarray(arr.astype(dt).reshape(-1))
        elif kind == "embd":
            data = np.ascontiguousarray(arr.astype(dt))
            assert data.shape == (dims[1], dims[0])
        else:
            a = arr.astype(dt)
            assert a.shape == (dims[0], dims[1]), (name, a.shape, dims)
            data = np.ascontiguousarray(a.T)
        assert data.size == int(np.prod(dims))
        file_arrays.append(data)

    data_blob = bytearray()
    offsets = []
    for data in file_arrays:
        offsets.append(len(data_blob))
        data_blob.extend(data.tobytes(order="C"))
        pad = align_up(len(data_blob)) - len(data_blob)
        data_blob.extend(b"\x00" * pad)

    w = Writer()
    # KV writers built after we know counts
    kv_ops = []

    def add(fn):
        kv_ops.append(fn)

    add(lambda: w.kv_string("general.architecture", "llama"))
    add(lambda: w.kv_string("general.name", "jllama-tiny-parity"))
    add(lambda: w.kv_u32("general.alignment", ALIGN))
    add(lambda: w.kv_u32("llama.context_length", n_ctx))
    add(lambda: w.kv_u32("llama.embedding_length", n_embd))
    add(lambda: w.kv_u32("llama.block_count", n_layer))
    add(lambda: w.kv_u32("llama.feed_forward_length", n_ff))
    add(lambda: w.kv_u32("llama.attention.head_count", n_head))
    add(lambda: w.kv_u32("llama.attention.head_count_kv", n_head))
    add(lambda: w.kv_f32("llama.rope.freq_base", theta))
    add(lambda: w.kv_u32("llama.rope.dimension_count", d_head))
    add(lambda: w.kv_f32("llama.attention.layer_norm_rms_epsilon", rms_eps))
    add(lambda: w.kv_u32("llama.vocab_size", n_vocab))
    add(lambda: w.kv_string("tokenizer.ggml.model", "gpt2"))
    add(lambda: w.kv_string("tokenizer.ggml.pre", "default"))
    add(lambda: w.kv_arr_str("tokenizer.ggml.tokens", tokens))
    add(lambda: w.kv_arr_f32("tokenizer.ggml.scores", scores))
    add(lambda: w.kv_arr_i32("tokenizer.ggml.token_type", token_types))
    add(lambda: w.kv_arr_str("tokenizer.ggml.merges", merges))
    add(lambda: w.kv_u32("tokenizer.ggml.bos_token_id", 1))
    add(lambda: w.kv_u32("tokenizer.ggml.eos_token_id", 2))
    add(lambda: w.kv_u32("tokenizer.ggml.unknown_token_id", 0))
    add(lambda: w.kv_bool("tokenizer.ggml.add_bos_token", False))
    add(lambda: w.kv_bool("tokenizer.ggml.add_eos_token", False))

    w.u32(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(len(specs))
    w.u64(len(kv_ops))
    for fn in kv_ops:
        fn()
    for (name, dims, typ, _a, _k), off in zip(specs, offsets):
        pack_tensor_info(w, name, dims, typ, off)
    pad = align_up(len(w.buf)) - len(w.buf)
    w.raw(b"\x00" * pad)
    data_off = len(w.buf)
    w.raw(data_blob)

    out.write_bytes(w.buf)
    print(
        f"wrote {out} ({len(w.buf)} bytes) data_off={data_off} "
        f"n_vocab={n_vocab} n_embd={n_embd} n_head={n_head} n_layer={n_layer} n_ff={n_ff}"
    )

    prompts = ["ab", "hello", "a b", " cab"]
    side = out.with_suffix(".meta.txt")
    lines = [
        f"n_vocab {n_vocab}",
        f"n_embd {n_embd}",
        f"n_head {n_head}",
        f"n_layer {n_layer}",
        f"n_ff {n_ff}",
        f"theta {theta}",
        f"rms_eps {rms_eps}",
        f"data_off {data_off}",
        f"n_tensors {len(specs)}",
        f"bos_id 1",
        f"eos_id 2",
        f"unk_id 0",
        f"add_bos 0",
    ]
    for p in prompts:
        ids = gpt2_bpe_encode(p, encoder, ranks)
        lines.append(f"tok_{p.replace(' ', '_').replace(' ', '_')} " + " ".join(map(str, ids)))
        # safer keys without spaces
    # explicit prompt keys
    lines = [ln for ln in lines if not ln.startswith("tok_")]
    lines.extend(
        [
            "prompt_ab " + " ".join(map(str, gpt2_bpe_encode("ab", encoder, ranks))),
            "prompt_hello " + " ".join(map(str, gpt2_bpe_encode("hello", encoder, ranks))),
            "prompt_a_b " + " ".join(map(str, gpt2_bpe_encode("a b", encoder, ranks))),
            "prompt_cab " + " ".join(map(str, gpt2_bpe_encode(" cab", encoder, ranks))),
        ]
    )
    side.write_text("\n".join(lines) + "\n")
    print(f"wrote {side}")
    for p in prompts:
        print(f"  encode {p!r} -> {gpt2_bpe_encode(p, encoder, ranks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
