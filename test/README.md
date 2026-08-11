# test/

| File | Milestone | Status |
|------|-----------|--------|
| `test_tensor.ijs` | M1 | done - tensor primitives |
| `test_m2.ijs` | M2 | done - RoPE, MHA, KV parity |
| `test_m3.ijs` | M3 | done - block, stack, generate |
| `test_m4.ijs` | M4 | done - GGUF F16/F32 + model_from_gguf |
| `test_m5.ijs` | M5 | done - GPT-2 byte BPE encode/decode |
| `test_m6.ijs` | M6 | done - greedy parity vs libllama oracle |
| `test_m7.ijs` | M7 | done - temp/top-k/top-p + EOS stop |
| `test_m8.ijs` | M8 | done - jllama_cli parse + run |
| `test_m10.ijs` | M10 | done - stories15M + Llama SPM (skips if model missing) |
| `test_m13.ijs` | M13 | done - GQA synthetic parity |
| `fixtures/tiny_llama_f16.gguf` | M4 | tiny Llama-arch weight fixture |
| `fixtures/tiny_bpe_vocab.gguf` | M5 | tiny BPE vocab-only GGUF |
| `fixtures/tiny_parity_f16.gguf` | M6 | weights + tokenizer for oracle parity |

M6 requires Homebrew `llama.cpp` and a built oracle:

```sh
make -C labs
jllama_test ''   # from jconsole after loading jllama.ijs
```
