# io/

| File | Milestone | Status |
|------|-----------|--------|
| `gguf.ijs` | M4 | **done** - F16/F32 GGUF reader + Llama model pack |
| `vocab.ijs` | M5 | planned |

## jllamagguf (M4)

| Verb | Role |
|------|------|
| `gguf_load` | path -> load box (meta, tensor infos, align, data_off, bytes) |
| `gguf_meta` | load meta key |
| `gguf_tensor` | load tensor name -> J f64 (jllama layout) |
| `gguf_names` | list tensor names |
| `model_from_gguf` | path -> jllama model box |
| `gguf_summary` | dump header/tensors |

**Supported:** little-endian GGUF v3, `GGML_TYPE_F32` / `F16` only, Llama-arch dense MHA (`n_head = n_head_kv`).

**Layout:** ggml 2d `ne0` contiguous; non-embd weights become `n_in x n_out` for `x mp w`. `token_embd.weight` stays `n_vocab x n_embd`.

**Fixture:** `test/fixtures/tiny_llama_f16.gguf` (regen via `python3 labs/make_fixture_gguf.py`).

```j
loadcore_jllama_ ''
m =. model_from_gguf_jllamagguf_ jllama_root '' , 'test/fixtures/tiny_llama_f16.gguf'
m generate_jllamamodel_ (0 1) ; 3
```
