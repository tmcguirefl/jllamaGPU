# Models

No production GGUF pinned yet (pre-M4).

## Planned primary (M6)

| Field | Value |
|-------|--------|
| Class | ~1B dense Llama-arch |
| File dtype | F16 GGUF |
| Machine | M2 32 GB (see hardware.md) |
| Path | _TBD - download into `models/` (gitignored)_ |
| Oracle | llama.cpp greedy on the same file |

## Lab / synthetic

| Name | Notes |
|------|--------|
| `synthetic` | Random weights, tiny hparams - M2-M3 |

When a file is chosen, add: URL, filename, sha256, `n_vocab`, `n_embd`, `n_layer`, `n_head`, `n_head_kv`, `n_ff`, `n_ctx_train`, rope settings.
