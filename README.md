# jllama

Llama-style dense decoder inference in the **J** programming language (Jsoftware J9.8).

Not a port of the llama.cpp C++ codebase. jllama reimplements a **thin vertical slice** of the algorithm in J arrays: GGUF load, tokenize, prefill, KV-cache decode, sample - validated against llama.cpp as an oracle.

## Requirements

- J 9.8 (`jconsole` at `/Applications/j9.8/bin/jconsole` on this machine)
- Always use the **full path** to jconsole (avoids the Java tool of the same name)
- Optional addons: `general/misc` (trace), `math/lapack2`, `convert/pjson`

## Quick start

```sh
/Applications/j9.8/bin/jconsole /Users/tomdevel/jdev/jllama/jllama.ijs
```

Inside the session:

```j
jllama_help ''
jllama_smoke ''
jllama_test ''     NB. M1-M6 unit tests (M6 needs tools/oracle_greedy)

NB. synthetic or fixture GGUF generate + tokenize
loadcore_jllama_ ''
m =. make_synthetic_jllamamodel_ 32;8;2;2;16
m generate_jllamamodel_ (1 2 3) ; 5
m =. model_from_gguf_jllamagguf_ jllama_root '' , 'test/fixtures/tiny_parity_f16.gguf'
v =. vocab_from_gguf_jllamavocab_ jllama_root '' , 'test/fixtures/tiny_parity_f16.gguf'
ids =. v encode_jllamavocab_ 'ab'
m generate_jllamamodel_ ids ; 3
```

### M6 oracle (llama.cpp)

```sh
# Homebrew llama.cpp required
make -C labs
labs/run_oracle.sh test/fixtures/tiny_parity_f16.gguf ab 3 --ids 259
```


## Project goal (v1)

| In scope | Out of scope |
|----------|----------------|
| One Llama-ish dense arch | Model zoo / MoE / VLM |
| GGUF F16 (F32) weights | ggml backends (Metal/CUDA) |
| RMSNorm, RoPE, MHA/GQA, SwiGLU | Flash-attention kernels |
| KV-cache prefill + decode | Quant matmul kernels (maybe dequant later) |
| Greedy + basic sampling | Server / GBNF / chat Jinja |
| Greedy parity vs llama.cpp | Training |

## Test model (this machine: MacBook Pro M2, 32 GB)

See [docs/hardware.md](docs/hardware.md). Short version:

- **Primary target:** ~0.5B-1B param **F16** GGUF (Llama-arch or close), ctx 512-2048 for dev
- **Comfortable lab model:** ~135M-360M class if F16 footprint or J overhead bites
- **Avoid for v1 in J F64:** 7B+ F16 (weights alone ~14 GB F16 -> ~28 GB as J floats before KV/activations)

J has no float32 arrays in 64-bit J; loaded F16/F32 weights become **float64** (~2x weight RAM vs F32 engines).

## Layout

```text
jllama.ijs          NB. entry: load path, help, smoke
core/               NB. array ops, transformer (milestones M1+)
io/                 NB. GGUF, vocab (M4-M5)
test/               NB. unit + parity tests
docs/               NB. design notes
labs/               NB. scratch / experiments
```

## Milestones

See [docs/milestones.md](docs/milestones.md).

## References

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) - oracle and GGUF ecosystem
- [NPN/picoGPT-in-j](https://github.com/NPN/picoGPT-in-j) - J GPT-2 style reference
- [jonghough/jlearn](https://github.com/jonghough/jlearn) - optional J ML idioms (not a dependency yet)
- GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## License

TBD.
