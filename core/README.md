# core/

Array ops and transformer math.

| File | Milestone | Status |
|------|-----------|--------|
| `tensor.ijs` | M0-M1 | done |
| `rope.ijs` | M2 | done |
| `attention.ijs` | M2 | done |
| `block.ijs` | M3 | **done** - SwiGLU + pre-norm block |
| `model.ijs` | M3 | **done** - stack + greedy generate |

## Locales

### jllamatensor (M1)

`mp` `silu` `softmax` `rmsnorm` `linear` `causal_mask` `allclose`

### jllamarope (M2)

`rope` `rotate_half`

### jllamaattn (M2)

`mha_full` `mha_step` `mha_prefill_cached` `kv_empty`

### jllamablock (M3)

| Verb | Role |
|------|------|
| `ffn_swiglu` | SiLU(gate)*up then down |
| `block_full` | full-seq pre-norm block |
| `block_step` | one token + layer KV |
| `block_prefill_cached` | prefill via steps |

Layer (one scalar box):
`<"_ (attn_norm ; wq ; wk ; wv ; wo ; ffn_norm ; w_gate ; w_up ; w_down)`

Call packs use `,` of scalar boxes, e.g.
`block_full (<x) , (<n_head) , layer [ , (<theta) ]`

### jllamamodel (M3)

| Verb | Role |
|------|------|
| `make_synthetic` | tiny deterministic model |
| `forward_full` | no-cache stack |
| `forward_prefill` / `forward_step` | KV path |
| `generate` | greedy with cache |
| `generate_fullrecompute` | slow oracle |

Model (one scalar box):
`<"_ (hparams ; wte ; layers ; ln_f ; lm_head)`  
hparams: `n_vocab; n_embd; n_head; n_layer; n_ff; theta`

See `docs/conventions.md` for why nested args use `(<a),(<b),box` not chained `;`.

Run tests: `jllama_test ''`
