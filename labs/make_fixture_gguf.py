#!/usr/bin/env python3
"""Write a tiny Llama-arch GGUF (F16+F32 mix) for jllama M4 tests."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747  # LE u32 for bytes GGUF
GGUF_VERSION = 3
ALIGN = 32

VT_UINT32 = 4
VT_FLOAT32 = 6
VT_STRING = 8
VT_UINT64 = 10

GGML_F32 = 0
GGML_F16 = 1


def align_up(n: int, a: int = ALIGN) -> int:
    return n + (a - n % a) % a


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


def pack_tensor_info(w: Writer, name: str, dims: list[int], typ: int, offset: int) -> None:
    w.string(name)
    w.u32(len(dims))
    for d in dims:
        w.u64(d)
    w.u32(typ)
    w.u64(offset)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/tiny_llama_f16.gguf")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_vocab = 8
    n_embd = 4
    n_head = 2
    n_layer = 1
    n_ff = 8
    theta = 10000.0
    n_ctx = 32

    rng = np.random.default_rng(0)

    def mat(shape, scale=0.02):
        return (scale * rng.standard_normal(shape)).astype(np.float32)

    # Keep jllama-layout arrays for expect dump; convert to ggml file order on write.
    # jllama 2d weight: (n_in, n_out) = (ne0, ne1)
    # token_embd jllama: (n_vocab, n_embd); ggml dims [ne0=n_embd, ne1=n_vocab]
    wte_j = mat((n_vocab, n_embd), 0.05)  # rows = tokens
    q_j = mat((n_embd, n_embd))
    k_j = mat((n_embd, n_embd))
    v_j = mat((n_embd, n_embd))
    o_j = mat((n_embd, n_embd))
    gate_j = mat((n_embd, n_ff))
    up_j = mat((n_embd, n_ff))
    down_j = mat((n_ff, n_embd))
    out_j = mat((n_embd, n_vocab), 0.03)
    attn_norm = np.linspace(0.5, 1.5, n_embd, dtype=np.float32)
    ffn_norm = np.linspace(1.0, 0.25, n_embd, dtype=np.float32)
    output_norm = np.ones(n_embd, dtype=np.float32)

    # list of (name, ggml_dims, typ, jllama_array)
    # jllama_array: 1d vector OR 2d (ne0,ne1) for weights; embd special (n_vocab,n_embd)
    specs: list[tuple[str, list[int], int, np.ndarray, str]] = [
        ("token_embd.weight", [n_embd, n_vocab], GGML_F16, wte_j, "embd"),
        ("blk.0.attn_norm.weight", [n_embd], GGML_F32, attn_norm, "vec"),
        ("blk.0.attn_q.weight", [n_embd, n_embd], GGML_F16, q_j, "w"),
        ("blk.0.attn_k.weight", [n_embd, n_embd], GGML_F16, k_j, "w"),
        ("blk.0.attn_v.weight", [n_embd, n_embd], GGML_F16, v_j, "w"),
        ("blk.0.attn_output.weight", [n_embd, n_embd], GGML_F16, o_j, "w"),
        ("blk.0.ffn_norm.weight", [n_embd], GGML_F32, ffn_norm, "vec"),
        ("blk.0.ffn_gate.weight", [n_embd, n_ff], GGML_F16, gate_j, "w"),
        ("blk.0.ffn_up.weight", [n_embd, n_ff], GGML_F16, up_j, "w"),
        ("blk.0.ffn_down.weight", [n_ff, n_embd], GGML_F16, down_j, "w"),
        ("output_norm.weight", [n_embd], GGML_F32, output_norm, "vec"),
        ("output.weight", [n_embd, n_vocab], GGML_F16, out_j, "w"),
    ]

    file_arrays = []
    for name, dims, typ, arr, kind in specs:
        dt = np.float16 if typ == GGML_F16 else np.float32
        if kind == "vec":
            data = np.ascontiguousarray(arr.astype(dt).reshape(-1))
        elif kind == "embd":
            # jllama (n_vocab,n_embd); ggml ne0=n_embd, ne1=n_vocab
            # file: (ne1, ne0) = (n_vocab, n_embd) ravel - same as jllama embd
            data = np.ascontiguousarray(arr.astype(dt))
            assert data.shape == (dims[1], dims[0])
        else:
            # jllama (ne0,ne1); file ravel of (ne1,ne0) = arr.T
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

    kvs = [
        ("general.architecture", "string", "llama"),
        ("general.name", "string", "jllama-tiny-fixture"),
        ("general.alignment", "u32", ALIGN),
        ("llama.context_length", "u32", n_ctx),
        ("llama.embedding_length", "u32", n_embd),
        ("llama.block_count", "u32", n_layer),
        ("llama.feed_forward_length", "u32", n_ff),
        ("llama.attention.head_count", "u32", n_head),
        ("llama.attention.head_count_kv", "u32", n_head),
        ("llama.rope.freq_base", "f32", theta),
        ("llama.vocab_size", "u32", n_vocab),
    ]

    w = Writer()
    w.u32(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(len(specs))
    w.u64(len(kvs))
    for k, ty, v in kvs:
        if ty == "string":
            w.kv_string(k, v)
        elif ty == "u32":
            w.kv_u32(k, v)
        elif ty == "f32":
            w.kv_f32(k, v)
        else:
            raise RuntimeError(ty)

    for (name, dims, typ, _a, _k), off in zip(specs, offsets):
        pack_tensor_info(w, name, dims, typ, off)

    pad = align_up(len(w.buf)) - len(w.buf)
    w.raw(b"\x00" * pad)
    data_off = len(w.buf)
    w.raw(data_blob)

    out.write_bytes(w.buf)
    print(f"wrote {out} ({len(w.buf)} bytes) data_off={data_off} tensors={len(specs)}")

    def jfloat(x: float) -> str:
        """Format a float for J ". parse (use _ for negatives)."""
        s = f"{float(x):.8f}"
        return s.replace("-", "_")

    side = out.with_suffix(".expect.txt")
    lines = [
        f"n_vocab {n_vocab}",
        f"n_embd {n_embd}",
        f"n_head {n_head}",
        f"n_layer {n_layer}",
        f"n_ff {n_ff}",
        f"theta {theta}",
        f"data_off {data_off}",
        f"n_tensors {len(specs)}",
        "wte0 " + " ".join(jfloat(x) for x in wte_j[0].tolist()),
        "attn_norm " + " ".join(jfloat(x) for x in attn_norm.tolist()),
        f"wq00 {jfloat(float(q_j[0, 0]))}",
        f"wq_shape {q_j.shape[0]} {q_j.shape[1]}",
        f"lm_shape {out_j.shape[0]} {out_j.shape[1]}",
        f"wte_shape {wte_j.shape[0]} {wte_j.shape[1]}",
    ]
    side.write_text("\n".join(lines) + "\n")
    print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
