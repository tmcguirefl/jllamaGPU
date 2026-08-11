# Models

## Test fixture (M4)

| Field | Value |
|-------|--------|
| Path | `test/fixtures/tiny_llama_f16.gguf` |
| Arch | llama |
| Dtypes | F16 weights + F32 norms |
| Shape | n_vocab=8, n_embd=4, n_head=2, n_layer=1, n_ff=8 |
| Regen | `python3 labs/make_fixture_gguf.py` |
| Expect | `test/fixtures/tiny_llama_f16.expect.txt` |

Not a real LM - only for loader/shape/generate smoke.

## Parity fixture (M6)

| Field | Value |
|-------|--------|
| Path | `test/fixtures/tiny_parity_f16.gguf` |
| Arch | llama dense MHA |
| Dtypes | F16 weights + F32 norms |
| Shape | n_vocab=267, n_embd=64, n_head=4, n_layer=2, n_ff=128 |
| Tokenizer | GPT-2 byte BPE in same GGUF |
| Regen | `python3 labs/make_fixture_parity.py` |
| Meta | `test/fixtures/tiny_parity_f16.meta.txt` |
| Oracle | `tools/oracle_greedy` via `labs/run_oracle.sh` (libllama) |

## Planned primary (larger M6+)

| Field | Value |
|-------|--------|
| Class | ~1B dense Llama-arch |
| File dtype | F16 GGUF |
| Machine | M2 32 GB (see hardware.md) |
| Path | _TBD - download into `models/` (gitignored)_ |
| Oracle | same `oracle_greedy` on the real file |

When a file is chosen, add: URL, filename, sha256, `n_vocab`, `n_embd`, `n_layer`, `n_head`, `n_head_kv`, `n_ff`, `n_ctx_train`, rope settings.

## Lab / synthetic

| Name | Notes |
|------|--------|
| `make_synthetic` | Deterministic tiny weights - M2-M3 |
| `model_from_gguf` | Llama GGUF F16/F32 - M4 |
