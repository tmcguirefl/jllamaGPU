# jllama

**GPU rewrite workspace.** Read **[GPU_ENGINE.md](GPU_ENGINE.md)** first (ggml-backed J 9.8, verb table, layout). Named wrappers: `jgpu.ijs`. Runtime: `C:\Users\tmcguire\j9.8\bin\jconsole.exe` (never the Java `jconsole`).

Llama-style dense decoder inference in the **J** programming language (Jsoftware J9.8).

Not a port of the llama.cpp C++ codebase. jllama reimplements a **thin vertical slice** of the algorithm in J arrays: GGUF load, tokenize, prefill, KV-cache decode, sample — validated against llama.cpp as an oracle.

## Requirements

- J 9.8 (`jconsole` at `C:\Users\tmcguire\j9.8\bin\jconsole.exe` on this machine; GPU `j.dll` is already installed there)
- Always use the **full path** to jconsole (avoids the Java tool of the same name)
- Optional addons: `general/misc` (trace), `math/lapack2`, `convert/pjson`

## Quick start

Clone this tree and run from the repo root. Scripts load from `ROOT` (the directory that contains `jllama_cli.ijs` / `jllama_dev.ijs`). A `manifest.ijs` will later install the same files as a J addon.

### Download models (Hugging Face `hf`)

Lab GGUFs live in `models/` (gitignored). Fixtures under `test/fixtures/` are already in the repo. Install the Hub CLI once, then pull **one file** per model (do not clone the whole repo):

```sh
uv tool install huggingface_hub
export PATH="$HOME/.local/bin:$PATH"
# equivalent: pip install -U "huggingface_hub[cli]"
```

From the **jllamaGPU repo root**:

```sh
hf download shibatch/stories-converted stories15M.F16.gguf --local-dir models
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-f16.gguf --local-dir models
hf download mradermacher/Qwen3.5-2B-GGUF Qwen3.5-2B.f16.gguf --local-dir models
hf download second-state/Phi-4-mini-instruct-GGUF Phi-4-mini-instruct-Q4_K_M.gguf --local-dir models
hf download bartowski/phi-4-GGUF phi-4-Q4_K_M.gguf --local-dir models
```

| Local path | Hub repo | File | Size (approx) |
|---|---|---|---|
| `models/stories15M.F16.gguf` | [shibatch/stories-converted](https://huggingface.co/shibatch/stories-converted) | `stories15M.F16.gguf` | 47 MB |
| `models/Llama-3.2-1B-Instruct-f16.gguf` | [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) | `Llama-3.2-1B-Instruct-f16.gguf` | 2.3 GB |
| `models/Qwen3.5-2B.f16.gguf` | [mradermacher/Qwen3.5-2B-GGUF](https://huggingface.co/mradermacher/Qwen3.5-2B-GGUF) | `Qwen3.5-2B.f16.gguf` | 3.8 GB |
| `models/Phi-4-mini-instruct-Q4_K_M.gguf` | [second-state/Phi-4-mini-instruct-GGUF](https://huggingface.co/second-state/Phi-4-mini-instruct-GGUF) | `Phi-4-mini-instruct-Q4_K_M.gguf` | 2.3 GB |
| `models/phi-4-Q4_K_M.gguf` | [bartowski/phi-4-GGUF](https://huggingface.co/bartowski/phi-4-GGUF) | `phi-4-Q4_K_M.gguf` | 8.4–9.1 GB |

`hf download <repo> <filename> --local-dir models` writes `models/<filename>`. If Hub nests an extra directory, move the `.gguf` up into `models/` so the CLI paths below match. Qwen F16 from Unsloth is the same weights under a different name (`unsloth/Qwen3.5-2B-GGUF` → `Qwen3.5-2B-BF16.gguf`); rename or pass that path to `-m`.

On 32 GB unified memory, prefer the Q4 Phi files. Qwen 2B **F16** is a large GPU load (activations stay F32). Phi-4 **14B F16** does not fit; use Q4_K_M.

### Run the CLI

From the repo root, either invoke the J script directly (shebang → jconsole on macOS/Linux) or use the thin wrapper. **Do not remove the `#!` line** — J skips it (`xs.c`); Windows cannot honor shebang, so run `jconsole.exe` or `bin\jllama_cli.cmd`.

Windows (PowerShell), using the GPU `j.dll` already in `C:\Users\tmcguire\j9.8\bin`:

```powershell
cd C:\Users\tmcguire\jdev\jllamaGPU
& 'C:\Users\tmcguire\j9.8\bin\jconsole.exe' .\jllama_cli.ijs --help
.\bin\jllama_cli.cmd -m test\fixtures\tiny_parity_f16.gguf -p ab -n 3
```

#### TinyStories lab model (~15M, ~47 MB F16) — fast English smoke

```sh
# direct J script
./jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32

# equivalent wrapper
bin/jllama_cli -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
```

```powershell
.\bin\jllama_cli.cmd -m models\stories15M.F16.gguf -p "Once upon a time" -n 32
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

#### Phi-4-mini and Phi-4 14B (same phi3 graph)

Filename `Phi-4-mini` / `phi4-mini` / `phi-4` / `phi4` / `phi3` → noun `Phi4Mini`. There is **no separate 14B architecture**. Both GGUFs are `general.architecture=phi3`: NeoX RoPE, GQA, fused QKV / fused SwiGLU split at load.

| | Phi-4-mini instruct | Phi-4 14B |
|---|---|---|
| Params | 3.8B | 14.7B |
| Typical quant on 32 GB | `Q4_K_M` (~2.3 GB) | `Q4_K_M` (~8.4–9.1 GB) |
| F16 on 32 GB | tight | too large (~29 GB weights) |
| `n_embd` / layers / GQA | 3072 / 32 / 24×8 | 5120 / 40 / 40×10 |
| `n_rot` | 96 of 128 (partial) | 128 (full) |
| RoPE θ | 10000 | 250000 |
| Tokenizer | gpt2-family | gpt2, `pre=dbrx` |
| Chat template | ChatML | ChatML (`<\|im_start\|>…`) |

Downloads: see [Download models](#download-models-hugging-face-hf) above.

**Why 14B “only repeats the prompt” or answers in one line.** The CLI always prints **prompt + new tokens**. Mini was usually run as **completion** (`-p "Paris is the capital of France"`): it keeps writing the next sentence of a document, so `-n 16` looks detailed. 14B is a **chat** checkpoint. A raw question is out of distribution: it may echo the user turn, emit `<|im_end|>` (id `100265`) immediately, or answer `Paris.` and stop. Default `-n 16` plus EOS-stop makes that look empty or terse. The graph is the same; the prompt format is not.

**ChatML (required for 14B, optional but better for mini).** Wrap the user turn and leave the assistant header open. Ask for length in the user or system text, and raise `-n` so EOS — not the cap — ends the turn:

```sh
./jllama_cli.ijs -m models/phi-4-Q4_K_M.gguf -n 128 --tokens \
  -p '<|im_start|>user<|im_sep|>What is the capital of France? Write a short paragraph.<|im_end|><|im_start|>assistant<|im_sep|>'
```

With a system turn:

```sh
./jllama_cli.ijs -m models/phi-4-Q4_K_M.gguf -n 128 \
  -p '<|im_start|>system<|im_sep|>Answer in a few sentences.<|im_end|><|im_start|>user<|im_sep|>What is the capital of France?<|im_end|><|im_start|>assistant<|im_sep|>'
```

`--tokens` shows whether new ids are `100265` (chat turn finished), a copy of the prompt, or real continuation. `--no-stop` ignores EOS and fills `-n` tokens (can ramble past `<|im_end|>`).

Mini still accepts a raw completion string; 14B will not behave like stories15M / Llama-3.2-1B until the string is a ChatML turn. jllama does not apply the GGUF Jinja template — wrap it in `-p` as above. BOS is omitted for `pre=dbrx` (unlike Llama-3 `pre=llama-bpe`).

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
"C:\Users\tmcguire\j9.8\bin\jconsole.exe" jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
```

### 3. REPL (dev / tests)

```sh
"C:\Users\tmcguire\j9.8\bin\jconsole.exe" jllama_dev.ijs
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
| GGUF F16/F32/quant weights on GPU | Flash-attention kernels |
| RMSNorm, RoPE, MHA/GQA, SwiGLU | Flash-attention kernels |
| KV-cache prefill + decode | Quant matmul kernels (dequant later) |
| Greedy + basic sampling | GBNF / chat Jinja |
| Greedy parity vs llama.cpp | Training |
| **CLI** (`jllama_cli.ijs`) | **Server** (optional later milestone) |

jllama executes the transformer in J on the **GPU J engine** (`G.` nouns, `+/ .*`, `'silu' g.` / `'softmax' g.` / `'rmsnorm' g.` / `'rope' g.`). Architecture graphs live as **nouns** `Llama3`, `Qwen35`, and `Phi4Mini`. The CLI detects the arch from the `.gguf` filename: `". '0!:0 Qwen35'`. llama.cpp remains the external oracle.

Qwen Gated DeltaNet still needs extra `libj` wraps (`ggml_sigmoid`, `ggml_ssm_conv`, `ggml_gated_delta_net`) — see GPU_ENGINE.md.

## Test model (this machine: Windows, AMD 8060S, Vulkan)

See [docs/hardware.md](docs/hardware.md). Short version:

| Model | Role | Approx |
|-------|------|--------|
| `models/stories15M.F16.gguf` | Fast English lab (MHA) | ~47 MB F16 file |
| `models/Llama-3.2-1B-Instruct-f16.gguf` | Primary 1B GQA lab | ~2.3 GB F16 file |
| `models/Qwen3.5-2B.f16.gguf` | 2B hybrid GDN | ~3.8 GB F16 |
| `models/Phi-4-mini-instruct-Q4_K_M.gguf` | 3.8B phi3, Q4 packed | ~2.3 GB; raw `-p` often enough |
| `models/phi-4-Q4_K_M.gguf` | 14B phi3, Q4 packed | ~8.4 GB; **ChatML `-p`**, `-n 128` |
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

**GNU GPL v3 only** (`LICENSE`, `COPYRIGHT`).

This program depends on a fork of the J language runtime (Jsoftware jsource,
GPU/ggml backend) used under GPL v3, so this tree is GPL-3 as well — not MIT.

- **jllamaGPU** (this repo): GPL-3.0-only — `COPYRIGHT`, `LICENSE`
- **J engine** (`libj`): Jsoftware dual license; this project uses the **GPL-3** option — `THIRD_PARTY.md`
- **ggml**: MIT (compatible with GPL-3) — `THIRD_PARTY.md`

A GitHub release that includes `j.dll` is a GPL-3 distribution of both
the application and the engine.
