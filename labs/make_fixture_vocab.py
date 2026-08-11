#!/usr/bin/env python3
"""Write a tiny GPT-2-style BPE vocab as a GGUF (metadata only, 0 tensors)."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32

VT_UINT32 = 4
VT_FLOAT32 = 6
VT_STRING = 8
VT_ARRAY = 9
VT_BOOL = 7

GGML_STRING = 8  # array elem type for string arrays
GGML_F32 = 6
GGML_I32 = 5


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

    def kv_bool(self, k: str, v: bool) -> None:
        self.string(k)
        self.u32(VT_BOOL)
        self.raw(bytes([1 if v else 0]))

    def kv_arr_str(self, k: str, arr: list[str]) -> None:
        self.string(k)
        self.u32(VT_ARRAY)
        self.u32(GGML_STRING)  # elem type STRING
        self.u64(len(arr))
        for s in arr:
            self.string(s)

    def kv_arr_f32(self, k: str, arr: list[float]) -> None:
        self.string(k)
        self.u32(VT_ARRAY)
        self.u32(GGML_F32)
        self.u64(len(arr))
        for x in arr:
            self.f32(x)

    def kv_arr_i32(self, k: str, arr: list[int]) -> None:
        self.string(k)
        self.u32(VT_ARRAY)
        self.u32(GGML_I32)
        self.u64(len(arr))
        for x in arr:
            self.i32(x)


def gpt2_bpe_encode(text: str, encoder: dict[str, int], ranks: dict[tuple[str, str], int]) -> list[int]:
    """Reference encode for expect file (whole-string byte BPE)."""
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
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/tiny_bpe_vocab.gguf")
    out.parent.mkdir(parents=True, exist_ok=True)

    b2u = bytes_to_unicode()
    # Specials then all 256 byte tokens (stable order)
    specials = ["<unk>", "<s>", "</s>"]
    byte_tokens = [b2u[b] for b in range(256)]
    # Extra merged tokens we will create via merges
    # merges applied in order; each merge introduces a new vocab entry in GPT-2 style?
    # In GPT-2, merges produce strings that are already the concatenation; vocab contains
    # all merge results as tokens. Build merges and add results to vocab.
    base = specials + byte_tokens
    # Planned merges (as symbol pairs using byte-unicode chars)
    a, b = b2u[ord("a")], b2u[ord("b")]
    sp = b2u[32]  # Ġ
    c = b2u[ord("c")]
    merge_pairs = [
        (a, b),  # ab
        (sp, a),  # Ġa
        (sp + a, b),  # Ġab
        (a, c),  # ac
    ]
    tokens = base[:]
    merges = []
    for left, right in merge_pairs:
        merged = left + right
        merges.append(f"{left} {right}")
        if merged not in tokens:
            tokens.append(merged)

    encoder = {t: i for i, t in enumerate(tokens)}
    ranks = {}
    for i, m in enumerate(merges):
        left, right = m.split(" ", 1)
        ranks[(left, right)] = i

    scores = [0.0] * len(tokens)
    # lower score = higher priority in SPM; for BPE unused mostly
    for i, m in enumerate(merges):
        left, right = m.split(" ", 1)
        mid = left + right
        scores[encoder[mid]] = float(-i)

    token_types = [1] * len(tokens)  # normal
    token_types[0] = 2  # unk
    token_types[1] = 3  # control bos
    token_types[2] = 3  # control eos

    # Gold strings
    samples = {
        "ab": gpt2_bpe_encode("ab", encoder, ranks),
        "a b": gpt2_bpe_encode("a b", encoder, ranks),
        " cab": gpt2_bpe_encode(" cab", encoder, ranks),
        "hello": gpt2_bpe_encode("hello", encoder, ranks),
        "": gpt2_bpe_encode("", encoder, ranks),
    }

    kvs = []
    # Use a list of callables to write kvs
    w = Writer()
    # count kvs
    kv_writers = [
        lambda: w.kv_string("general.architecture", "llama"),
        lambda: w.kv_string("general.name", "jllama-tiny-bpe-vocab"),
        lambda: w.kv_u32("general.alignment", ALIGN),
        lambda: w.kv_string("tokenizer.ggml.model", "gpt2"),
        lambda: w.kv_string("tokenizer.ggml.pre", "byte"),
        lambda: w.kv_arr_str("tokenizer.ggml.tokens", tokens),
        lambda: w.kv_arr_f32("tokenizer.ggml.scores", scores),
        lambda: w.kv_arr_i32("tokenizer.ggml.token_type", token_types),
        lambda: w.kv_arr_str("tokenizer.ggml.merges", merges),
        lambda: w.kv_u32("tokenizer.ggml.bos_token_id", 1),
        lambda: w.kv_u32("tokenizer.ggml.eos_token_id", 2),
        lambda: w.kv_u32("tokenizer.ggml.unknown_token_id", 0),
        lambda: w.kv_bool("tokenizer.ggml.add_bos_token", False),
        lambda: w.kv_bool("tokenizer.ggml.add_eos_token", False),
    ]

    w.u32(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(0)  # no tensors
    w.u64(len(kv_writers))
    for fn in kv_writers:
        fn()
    pad = align_up(len(w.buf)) - len(w.buf)
    w.raw(b"\x00" * pad)

    out.write_bytes(w.buf)
    print(f"wrote {out} ({len(w.buf)} bytes) n_vocab={len(tokens)} n_merges={len(merges)}")

    def jids(ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    side = out.with_suffix(".expect.txt")
    lines = [
        f"n_vocab {len(tokens)}",
        f"n_merges {len(merges)}",
        f"bos_id 1",
        f"eos_id 2",
        f"unk_id 0",
        f"id_ab {encoder['ab']}",
        f"id_Gab {encoder[sp + 'ab']}",
        f"tok_ab {encoder['a']} {encoder['b']}",  # before merge check separate
        f"enc_ab {jids(samples['ab'])}",
        f"enc_a_b {jids(samples['a b'])}",
        f"enc_cab {jids(samples[' cab'])}",
        f"enc_hello {jids(samples['hello'])}",
        f"dec_ab ab",
        f"dec_a_b a b",
    ]
    side.write_text("\n".join(lines) + "\n")
    print(f"wrote {side}")
    for k, v in samples.items():
        print(f"  sample {k!r} -> {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
