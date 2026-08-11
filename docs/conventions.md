# J conventions for jllama

## Runtime

```text
/Applications/j9.8/bin/jconsole
```

Never bare `jconsole` (Java clash). No Java code in this project.

## Language

- Expressions parse **right to left**
- Operators are **verbs** (monadic and/or dyadic)
- Dyadic verbs are **infix**: `left verb right`
- **Adverbs** often derive a new verb that is still dyadic - always find the **left** argument of that derived verb
- When an adverb phrase is unclear, load and use:

```j
load 'general/misc/trace'
```

## Array layout (v1)

Unless a verb documents otherwise:

| Tensor | J shape (leading axis first) |
|--------|------------------------------|
| Token embeddings batch | `n_tok , n_embd` |
| Weights `linear` W | `n_in , n_out` with `x mp w` and `mp=: +/ . *` |
| Heads (M2 locked) | `n_tok , n_head , d_head` |
| KV cache per layer (M2 locked) | `n_past , n_head , d_head` (K and V separate) |

**Dtype:** J float64 in core math. Loaders may read F16/F32 from disk then promote.

## Locales

- Entry and user verbs: `z` or documented exports from `jllama.ijs`
- Implementation: dedicated locales (`jllama`, `jllamatensor`, later `jllamagguf`, etc.)
- **Locale names must not contain `_`.** In J, `name_locale_` uses underscore as the locale separator, so a locale like `jllama_tensor` is ill-formed.
- Avoid silent globals for model weights; keep them in a model locale/noun

## Boxing and multi-arg packs (M3 lesson)

J link (`;`) and box (`<`) interact in ways that break nested weight packs if you treat them like Python tuples.

1. **Enclose a whole open list** with `<"_ y` when you need one scalar box around many items.
2. **Multiple assignment spreads open lists only:**
   - `'a b c' =. open_list` works
   - `'a b c' =. <open_list` does **not** - open first with `> y`
3. **Chained `a ; b ; c` re-boxes when the left side is already a box list.** Nested layer/model boxes get an extra box level, then multi-assign peels the wrong depth.
4. **Safe mixed packs** (nested boxes + numerics): catenate scalar boxes:
   ```j
   (<x) , (<n_head) , layerbox , (<theta)
   (<out) , (<kc) , (<vc)
   ```
5. **Pure-numeric packs** (`x ; wq ; wk ; ...`) may still use `;`.

Layer / model shapes:

```text
layer  = <"_ (attn_n ; wq ; wk ; wv ; wo ; ffn_n ; wg ; wu ; wd)
model  = <"_ (hparams ; wte ; layers ; ln_f ; lm_head)
hparams = n_vocab ; n_embd ; n_head ; n_layer ; n_ff ; theta
```

Generate API: `m generate ids ; n_new` (numeric `;` is fine).

## GGUF binary (M4)

- Magic LE u32 for bytes `GGUF` = `0x46554747`; version **3**
- J codecs: `_2 (3!:4)` u32, `_3 (3!:4)` u64, `_1 fc` f32; F16 via bit unpack (not `b.` shifts)
- Locale: `jllamagguf` (no `_` in locale name)
- Loader promotes F16/F32 disk tensors to J f64
- Quant types rejected until M9
- GGUF metadata strings are raw UTF-8 bytes; decode with `utf8_decode` before BPE

## Tokenizer (M5)

- Locale: `jllamavocab`
- GPT-2-style byte BPE: UTF-8 bytes -> `bytes_to_unicode` -> greedy merge by rank
- Vocab box packed with `(<a),(<b),...` then `<"_` (same nesting rules as model/layer)
- `encode` / `decode` are dyadic: `vocab encode text`, `vocab decode ids`
- BOS/EOS skipped on decode; optional add_bos/add_eos from GGUF flags

## Testing

- Prefer small explicit matrices with known results
- Parity tests (M6+) compare to llama.cpp on the same GGUF and prompt
- Trace before rewriting opaque tacit trains
