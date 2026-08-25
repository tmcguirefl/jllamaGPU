# jllama

Llama-style dense decoder inference in the **J** programming language (Jsoftware J9.8).

Not a port of the llama.cpp C++ codebase. jllama reimplements a **thin vertical slice** of the algorithm in J arrays: GGUF load, tokenize, prefill, KV-cache decode, sample — validated against llama.cpp as an oracle.

## Requirements

- J 9.8 (`jconsole` at `/Applications/j9.8/bin/jconsole` on this machine)
- Always use the **full path** to jconsole (avoids the Java tool of the same name)
- Optional addons: `general/misc` (trace), `math/lapack2`, `convert/pjson`

## Quick start

Clone this tree and run from the repo root. Scripts load from `ROOT` (the directory that contains `jllama_cli.ijs` / `jllama_dev.ijs`). A `manifest.ijs` will later install the same files as a J addon.

### Run the CLI

From the repo root, either invoke the J script directly (shebang → jconsole) or use the thin wrapper.

#### TinyStories lab model (~15M, ~47 MB F16) — fast English smoke

```sh
# direct J script
./jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32

# equivalent wrapper
bin/jllama_cli -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
```

Download once if needed:

```sh
export PATH="$HOME/.local/bin:$PATH"   # after: uv tool install huggingface_hub
hf download shibatch/stories-converted stories15M.F16.gguf --local-dir models
```

#### Llama 3.2 1B Instruct (F16 GGUF) — larger lab model

```sh
# direct J script (preferred)
./jllama_cli.ijs -m models/Llama-3.2-1B-Instruct-f16.gguf -p "Once upon a time" -n 32

# equivalent wrapper
bin/jllama_cli -m models/Llama-3.2-1B-Instruct-f16.gguf -p "Once upon a time" -n 32
```

Notes for the 1B model:

- First run is slow in pure J f64 (load + prefill); keep `-n` modest while checking.
- Instruct checkpoints have **no chat template** in jllama yet — prompts are raw text (optionally with manual Llama-3 headers).
- BOS is prepended for Llama-3 BPE (`pre=llama-bpe`) to match libllama.
- Weights are F16 on disk → **float64** in J (~4× F16 RAM for the weight tensors).

Place the GGUF under `models/` (gitignored). Example name used in this repo:

```text
models/Llama-3.2-1B-Instruct-f16.gguf
```

#### Qwen3.5-2B (hybrid Gated DeltaNet + gated attention)

Filename `Qwen3.5` / `qwen35` → noun `Qwen35` (`". '0!:0 Qwen35'`). F16/F32 only. ~2B F16 is a large f64 RAM load.

```sh
./jllama_cli.ijs -m models/Qwen3.5-2B.f16.gguf -p "Once upon a time" -n 16
```

Text-only: gated attention (QK-norm, Q-gate, partial NeoX RoPE) interleaved with Gated DeltaNet. MTP/NextN and the vision encoder are skipped.

#### Phi-4-mini (phi3 graph, NeoX RoPE)

Filename `Phi-4-mini` / `phi4-mini` / `phi-4` / `phi3` → noun `Phi4Mini`. Dense GQA decoder; fused QKV / fused SwiGLU split at load. ~3.8B F16 → ~30 GB as J f64 (tight on 32 GB).

```sh
./jllama_cli.ijs -m models/Phi-4-mini-instruct-f16.gguf -p "What is the capital of France?" -n 16
```

#### Fixture plumbing + help

```sh
./jllama_cli.ijs -m test/fixtures/tiny_parity_f16.gguf -p ab -n 3 --tokens
./jllama_cli.ijs --help
./jllama_cli.ijs --version
```

#### Useful flags

| Flag | Meaning |
|------|---------|
| `-m, --model PATH` | GGUF model (required) |
| `-p, --prompt TEXT` | prompt string |
| `-f, --file PATH` | read prompt from file |
| `-n, --n-predict N` | new tokens (default 16) |
| `--temp F` | temperature; `<=0` greedy (default 0) |
| `--top-k K` | top-k; `<=0` off |
| `--top-p P` | nucleus; `>=1` off |
| `--seed S` | RNG seed |
| `--tokens` | also print token ids |
| `-h, --help` | help |
| `--version` | version and exit |

Without a shebang exec bit, you can always use jconsole explicitly:

```sh
/Applications/j9.8/bin/jconsole jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
```

### 3. REPL (dev / tests)

```sh
/Applications/j9.8/bin/jconsole jllama_dev.ijs
```

Inside the session:

```j
jllama_help ''
jllama_smoke ''
jllama_test ''     NB. M1-M13 unit tests (M6 needs tools/oracle_greedy)

NB. synthetic or fixture GGUF generate + tokenize
loadcore_jllama_ ''
m =. make_synthetic_jllamamodel_ 32;8;2;2;16
m generate_jllamamodel_ (1 2 3) ; 5
m =. model_from_gguf_jllamagguf_ jllama_root '' , 'test/fixtures/tiny_parity_f16.gguf'
v =. vocab_from_gguf_jllamavocab_ jllama_root '' , 'test/fixtures/tiny_parity_f16.gguf'
ids =. v encode_jllamavocab_ 'ab'
m generate_jllamamodel_ ids ; 3
```

`jllama_cli.ijs` is standalone (shebang → jconsole). It loads `sysutils.ijs`, then `core/`, `io/`, and `cli/` from the same directory.

### 4. M6 oracle (llama.cpp)

```sh
# Homebrew llama.cpp required
make -C labs
labs/run_oracle.sh test/fixtures/tiny_parity_f16.gguf ab 3 --ids 259
```

## Project goal (v1)

| In scope | Out of scope (for now) |
|----------|-------------------------|
| Llama dense + Qwen3.5 hybrid (`Llama3` / `Qwen35` nouns) | MoE / VLM / other GGUF arches |
| GGUF F16 (F32) weights | ggml backends (Metal/CUDA) |
| RMSNorm, RoPE, MHA/GQA, SwiGLU | Flash-attention kernels |
| KV-cache prefill + decode | Quant matmul kernels (dequant later) |
| Greedy + basic sampling | GBNF / chat Jinja |
| Greedy parity vs llama.cpp | Training |
| **CLI** (`jllama_cli.ijs`) | **Server** (optional later milestone) |

jllama executes the transformer **eagerly in J** (readable `core/*.ijs`). Architecture graphs live as **nouns** `Llama3` and `Qwen35` (`0 : 0` multiline scripts). The CLI detects the arch from the `.gguf` filename and brings the noun in with do: `". '0!:0 Qwen35'` (script-do on the character noun). llama.cpp remains the external oracle.

## Test model (this machine: MacBook Pro M2, 32 GB)

See [docs/hardware.md](docs/hardware.md). Short version:

| Model | Role | Approx |
|-------|------|--------|
| `models/stories15M.F16.gguf` | Fast English lab (MHA) | ~47 MB F16 file |
| `models/Llama-3.2-1B-Instruct-f16.gguf` | Primary 1B GQA lab | ~2.3 GB F16 file → large f64 RAM |
| `test/fixtures/*.gguf` | Unit / parity fixtures | tiny |

- **Primary target:** ~0.5B–1B param **F16** Llama-arch GGUF, ctx 512–2048 for dev
- **Comfortable lab model:** stories15M-class if you want quick iteration
- **Avoid for v1 in pure J f64:** 7B+ F16 (weights alone ~14 GB F16 → tens of GB as J floats before KV/activations)

J has no float32 arrays in 64-bit J; loaded F16/F32 weights become **float64**.

## Layout

```text
jllama_cli.ijs      NB. standalone CLI (shebang jconsole; loads this tree)
jllama_dev.ijs      NB. REPL: help, smoke, test, loadcore
sysutils.ijs        NB. ROOT, setroot, jload, jrequire, VERSION
bin/jllama_cli      NB. thin wrapper → ./jllama_cli.ijs
cli/                NB. jllamacli parse + run
core/               NB. tensor, rope, attn, block, sample, model
io/                 NB. GGUF, vocab
test/               NB. unit + parity tests
docs/               NB. milestones, hardware, models, conventions
labs/               NB. fixtures, oracle, experiments
models/             NB. large GGUFs (gitignored)
# later: manifest.ijs  NB. J addon install into ~addons
```

## Milestones

| Now | Next | Later |
|-----|------|--------|
| **M0–M8 + M10 + M13 done** (engine, CLI, stories15M, GQA 1B path) | quant / server / second arch | M9 perf deferred |

Full board: [docs/milestones.md](docs/milestones.md). Models detail: [docs/models.md](docs/models.md).

## References

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — oracle and GGUF ecosystem
- [NPN/picoGPT-in-j](https://github.com/NPN/picoGPT-in-j) — J GPT-2 style reference
- [jonghough/jlearn](https://github.com/jonghough/jlearn) — optional J ML idioms (not a dependency yet)
- GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## License

TBD.
