# Milestones

jllama reimplements a thin Llama-style decode path in J, validated against llama.cpp where it matters. Product surface is **CLI first**, server later (optional).

## Status board

| ID | Name | Exit criteria | Status |
|----|------|----------------|--------|
| **M0** | Skeleton | `jllama.ijs` loads; smoke + help | **done** |
| **M1** | Tensor lab | rmsnorm, softmax, silu, linear, mask tested | **done** |
| **M2** | Attn + RoPE + KV | cache decode == full recompute (synthetic) | **done** |
| **M3** | Tiny stack | greedy generate on random Llama shapes | **done** |
| **M4** | GGUF F16/F32 | load real file; names/shapes match dump | **done** |
| **M5** | Tokenizer | GPT-2 byte BPE encode/decode from GGUF vocab | **done** |
| **M6** | Parity (fixture) | greedy tokens match llama.cpp on tiny F16 fixture | **done** |
| **M7** | Sampling | temp / top-k / top-p, EOS stop | **done** |
| **M8** | **CLI** | `jllama_cli` shell entry: model, prompt, n, sample flags, print text/ids | **next** |
| **M9** | Performance | profile CLI path; fewer copies; LAPACK/`mp` where useful | planned |
| **M10** | Real-model lab | pin ~0.5B–1B F16 GGUF under `models/`; smoke + optional greedy spot-check | planned |
| **M11** | Server (opt) | thin HTTP/SSE wrapper around same engine as CLI | optional |
| **M12** | Quant (opt) | dequant-to-f64 first (Q4/Q5/Q8), then optional faster paths | optional |
| **M13** | Second arch (opt) | shared core, second golden model (e.g. Qwen-dense) | optional |

**Completed critical path:** M0 → M7 (engine vertical slice).  
**Product path:** M8 → M9 → M10.  
**Optional:** M11–M13.

## What “done” means for open milestones

### M8 — CLI (next)

Thin front-end over existing APIs — not a llama.cpp clone.

```text
jllama_cli -m PATH.gguf -p "prompt" -n 64 \
  --temp 0.8 --top-k 40 --top-p 0.95 [--eos ID] [--tokens]
```

Exit criteria:

- Runnable from shell via jconsole (wrapper script or `jllama_cli.ijs` + `bin/jllama_cli`)
- Flags: model, prompt (or `-f`), `n_predict`, temp/top-k/top-p/seed, optional EOS, optional print token ids
- Uses `model_from_gguf`, `vocab_from_gguf`, `encode`, `generate` / `generate_sample`, `decode`
- Works on `test/fixtures/tiny_parity_f16.gguf` without a large download
- Documented in README; smoke/help mentions it
- **Out of scope for M8:** chat templates, REPL conversation, GBNF, HTTP, GPU

### M9 — Performance

- Profile prefill vs decode on CLI path (tiny fixture + optional real model)
- Cut obvious J copies / re-boxes on hot path
- Try `math/lapack2` (or equivalent) for large `mp` if win is real
- Record before/after notes in `docs/` (tok/s ballpark on this M2)

### M10 — Real-model lab

- Choose and document one ~0.5B–1B **Llama-arch F16** GGUF in `docs/models.md` (URL, sha256, hparams)
- File lives in `models/` (gitignored)
- CLI can load it and generate short completions
- Optional: greedy spot-check vs `oracle_greedy` / `llama-cli` on a short prompt (best-effort; not full CI if model is huge)

### M11 — Server (optional)

Only after CLI is pleasant:

- Same completion engine as CLI
- Minimal HTTP (e.g. one `/completion` or SSE stream)
- No requirement to match llama-server API 1:1 in v1

### M12 — Quant (optional)

- M12a: GGUF quant tensor → f64, reuse float graph
- Later: faster kernels only if needed

### M13 — Second arch (optional)

- Shared tensor/attn/sample; arch-specific load + naming

## Design rules (carry forward)

- **J only** in product code (Python OK for fixture writers / labs)
- Locale names: no `_` (`jllamamodel`, not `jllama_model`)
- Nested packs: `(<a),(<b),box` — not chained `;` across box lists
- jconsole: `/Applications/j9.8/bin/jconsole` full path
- Oracle: Homebrew llama.cpp + `tools/oracle_greedy` for greedy id parity
- Test model sizing: [hardware.md](hardware.md)

## Suggested “what next” order

1. **M8 CLI** — makes the stack usable and demos M4–M7  
2. **M9 perf** — profile the path people actually run  
3. **M10 real GGUF** — one documented ~1B F16 when you want a real LM feel  
4. **M11 server** only if you need a networked client  
5. **M12/M13** as interest/need dictates  

## Version mapping

| Version | Milestone |
|---------|-----------|
| 0.1.x | M1 |
| 0.2.x | M2 |
| 0.3.x | M3 |
| 0.4.x | M4 |
| 0.5.x | M5 |
| 0.6.x | M6 |
| 0.7.x | M7 |
| 0.8.x | M8 CLI (upcoming) |
| 0.9.x | M9 perf |
| 1.0? | CLI + real-model lab solid enough to call v1 |
