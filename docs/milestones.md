# Milestones

Full narrative was developed in the jllama design session. Summary:

| ID | Name | Exit criteria | Status |
|----|------|----------------|--------|
| **M0** | Skeleton | `jllama.ijs` loads; smoke + help work | done |
| **M1** | Tensor lab | rmsnorm, softmax, silu, linear, mask tested | **done** |
| **M2** | Attn + RoPE + KV | cache decode == full recompute on synthetic weights | **done** |
| **M3** | Tiny stack | greedy generate on random Llama shapes | **done** |
| **M4** | GGUF F16/F32 | load real file; names/shapes match dump | **done** |
| **M5** | Tokenizer | GPT-2 byte BPE encode/decode from GGUF vocab | **done** |
| **M6** | Parity * | greedy tokens match llama.cpp on tiny F16 fixture | **done** |
| **M7** | Sampling UX | temp/top-k/top-p, EOS stop | |
| **M8** | Performance | profile; fewer copies; LAPACK where useful | |
| **M9** | Quant (opt) | 9a dequant-to-float first | |
| **M10** | Second arch (opt) | shared core, second golden model | |

**Critical path:** M4 -> M5 -> M6.

**Test model sizing:** [hardware.md](hardware.md).
