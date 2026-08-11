NB. jllama tokenizer (M5)
NB.
NB. GPT-2-style byte-level BPE (GGUF tokenizer.ggml.model = gpt2|bpe).
NB. Pre-tokenizer "byte": UTF-8 bytes -> GPT-2 unicode map -> BPE on whole string.
NB. (No full GPT-2 regex split; enough for fixtures and simple prompts.)
NB.
NB. Locale: jllamavocab
NB.
NB. Vocab box (scalar):
NB.   <"_ ((<model),(<tokens),(<ranks),(<bos),(<eos),(<unk),(<add_bos),(<add_eos),(<pre))
NB. tokens = boxed list of unicode token strings (id = index)
NB. ranks  = boxed list of <(left;right) in merge order (rank = index)
NB.
NB. Public:
NB.   vocab_from_gguf path
NB.   vocab_from_load loadbox
NB.   encode  vocab encode text  -> int ids
NB.   decode  vocab decode ids   -> text
NB.   vocab_token / vocab_bos / vocab_eos / vocab_unk / vocab_tokens

cocurrent 'jllamavocab'

NB. ---------------------------------------------------------------
NB. GPT-2 bytes_to_unicode
NB. ---------------------------------------------------------------

NB. Build maps:
NB.   BYTE_UCP  - 256 ints, byte -> unicode codepoint
NB.   BYTE_RMAP - table codepoint -> byte (_1 if unused); length 400
make_byte_maps =: 3 : 0
  bs =. (33 + i. 94) , (161 + i. 12) , (174 + i. 82)
  cs =. bs
  n =. 0
  for_b. i. 256 do.
    if. -. b e. bs do.
      bs =. bs , b
      cs =. cs , 256 + n
      n =. n + 1
    end.
  end.
  u =. 256 $ 0
  for_i. i. # bs do.
    u =. (i { cs) (i { bs) } u
  end.
  r =. 400 $ _1
  for_b. i. 256 do.
    r =. b (b { u) } r
  end.
  r ; u
)

'BYTE_RMAP BYTE_UCP' =: make_byte_maps ''

NB. ---------------------------------------------------------------
NB. UTF-8 (GGUF strings are raw UTF-8 bytes as 8-bit chars)
NB. ---------------------------------------------------------------

NB. raw byte-char string (from 1!:1 / rd_string) -> unicode codepoints
utf8_cps =: 3 : 0
  b =. a. i. , y
  if. 0 = # b do. 0 $ 0 return. end.
  out =. 0 $ 0
  i =. 0
  n =. # b
  while. i < n do.
    c =. i { b
    if. c < 128 do.
      out =. out , c
      i =. i + 1
    elseif. c < 224 do.
      'vocab: truncated utf8' assert (i + 1) < n
      c2 =. (i + 1) { b
      out =. out , (64 * 31 (17 b.) c) + 63 (17 b.) c2
      i =. i + 2
    elseif. c < 240 do.
      'vocab: truncated utf8' assert (i + 2) < n
      c2 =. (i + 1) { b
      c3 =. (i + 2) { b
      out =. out , (4096 * 15 (17 b.) c) + (64 * 63 (17 b.) c2) + 63 (17 b.) c3
      i =. i + 3
    elseif. do.
      'vocab: unsupported utf8' assert 0
    end.
  end.
  out
)

NB. raw GGUF string -> J unicode string
utf8_decode =: 3 : 0
  cp =. utf8_cps y
  if. 0 = # cp do. '' return. end.
  u: cp
)

NB. J unicode string -> UTF-8 byte values 0..255
utf8_bytes =: 3 : 0
  if. 0 = # y do. 0 $ 0 return. end.
  cp =. 3 u: y
  if. *./ cp < 128 do. cp return. end.
  out =. 0 $ 0
  for_c. cp do.
    if. c < 128 do.
      out =. out , c
    elseif. c < 2048 do.
      out =. out , (192 + <. c % 64) , (128 + 63 (17 b.) c)
    elseif. c < 65536 do.
      out =. out , (224 + <. c % 4096) , (128 + 63 (17 b.) <. c % 64) , (128 + 63 (17 b.) c)
    elseif. do.
      'vocab: codepoint too large' assert 0
    end.
  end.
  out
)

NB. bytes -> GPT-2 BPE unicode string
bytes_to_bpe_chars =: 3 : 0
  b =. , y
  if. 0 = # b do. '' return. end.
  u: b { BYTE_UCP
)

NB. GPT-2 BPE unicode string -> bytes
bpe_chars_to_bytes =: 3 : 0
  if. 0 = # y do. 0 $ 0 return. end.
  cp =. 3 u: y
  out =. 0 $ 0
  for_c. cp do.
    'vocab: bad bpe char' assert (c >: 0) *. c < # BYTE_RMAP
    b =. c { BYTE_RMAP
    'vocab: bad bpe char in decode' assert b >: 0
    out =. out , b
  end.
  out
)

NB. bytes -> J text (UTF-8 decode)
bytes_to_text =: 3 : 0
  b =. , y
  if. 0 = # b do. '' return. end.
  if. *./ b < 128 do. b { a. return. end.
  utf8_decode b { a.
)

NB. ---------------------------------------------------------------
NB. BPE core
NB. ---------------------------------------------------------------

NB. x = ranks (list of <(left;right) in rank order)
NB. y = word (boxed list of symbol strings)
bpe_merge =: 4 : 0
  ranks =. x
  word =. y
  if. 2 > # word do. word return. end.
  maxr =. # ranks
  whilst. 1 do.
    if. 2 > # word do. word return. end.
    best =. maxr
    besti =. _1
    for_i. i. <: # word do.
      L =. > i { word
      R =. > (i + 1) { word
      r =. maxr
      for_j. i. # ranks do.
        pair =. > j { ranks
        a =. > 0 { pair
        b =. > 1 { pair
        if. (L -: a) *. R -: b do. r =. j break. end.
      end.
      if. r < best do.
        best =. r
        besti =. i
      end.
    end.
    if. besti < 0 do. word return. end.
    if. best >: maxr do. word return. end.
    pair =. > best { ranks
    a =. > 0 { pair
    b =. > 1 { pair
    nw =. 0 $ a:
    i =. 0
    n =. # word
    while. i < n do.
      NB. J *. does not short-circuit; gate index before (i+1){
      if. i < n - 1 do.
        if. ((> i { word) -: a) *. (> (i + 1) { word) -: b do.
          nw =. nw , < a , b
          i =. i + 2
        else.
          nw =. nw , i { word
          i =. i + 1
        end.
      else.
        nw =. nw , i { word
        i =. i + 1
      end.
    end.
    word =. nw
  end.
  word
)

NB. ---------------------------------------------------------------
NB. Vocab load
NB. ---------------------------------------------------------------

NB. Compare token strings by codepoints (literal vs unicode same text).
tok_eq =: 4 : '(3 u: x) -: (3 u: y)'

token_to_id =: 4 : 0
  tokens =. x
  want =. y
  for_i. i. # tokens do.
    if. want tok_eq > i { tokens do. i return. end.
  end.
  _1
)

NB. y = gguf load box
vocab_from_load =: 3 : 0
  load =. y
  model =. load gguf_meta_jllamagguf_ 'tokenizer.ggml.model'
  'vocab: unsupported tokenizer model (want gpt2|bpe)' assert +./ model&-: &> 'gpt2' ; 'bpe'
  raw_tokens =. load gguf_meta_jllamagguf_ 'tokenizer.ggml.tokens'
  'vocab: tokens must be boxed list' assert 32 = 3!:0 raw_tokens
  tokens =. 0 $ a:
  for_t. raw_tokens do.
    tokens =. tokens , < utf8_decode > t
  end.
  merges =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.merges' ; (0 $ a:)
  ranks =. 0 $ a:
  for_m. merges do.
    s =. utf8_decode > m
    sp =. s i. ' '
    'vocab: bad merge (no space)' assert sp < # s
    left =. sp {. s
    right =. (sp + 1) }. s
    ranks =. ranks , < (<left) , (<right)
  end.
  bos =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.bos_token_id' ; 1
  eos =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.eos_token_id' ; 2
  unk =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.unknown_token_id' ; 0
  add_bos =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.add_bos_token' ; 0
  add_eos =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.add_eos_token' ; 0
  pre =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.pre' ; 'byte'
  NB. safe pack: never chain ; across box-lists
  pack =. (<model) , (<tokens) , (<ranks) , (<bos) , (<eos) , (<unk) , (<add_bos) , (<add_eos) , (<pre)
  <"_ pack
)

vocab_from_gguf =: 3 : 0
  vocab_from_load gguf_load_jllamagguf_ y
)

vocab_open =: 3 : '> y'

vocab_tokens =: 3 : 0
  'model tokens ranks bos eos unk add_bos add_eos pre' =. vocab_open y
  tokens
)

vocab_bos =: 3 : 0
  'model tokens ranks bos eos unk add_bos add_eos pre' =. vocab_open y
  bos
)

vocab_eos =: 3 : 0
  'model tokens ranks bos eos unk add_bos add_eos pre' =. vocab_open y
  eos
)

vocab_unk =: 3 : 0
  'model tokens ranks bos eos unk add_bos add_eos pre' =. vocab_open y
  unk
)

vocab_token =: 4 : 0
  tokens =. vocab_tokens x
  id =. y
  'vocab_token: bad id' assert (id >: 0) *. id < # tokens
  > id { tokens
)

NB. ---------------------------------------------------------------
NB. encode / decode
NB. ---------------------------------------------------------------

NB. vocab encode text
encode =: 4 : 0
  v =. x
  text =. y
  'model tokens ranks bos eos unk add_bos add_eos pre' =. vocab_open v
  ids =. 0 $ 0
  if. add_bos do. ids =. ids , bos end.
  if. # text do.
    raw =. utf8_bytes text
    chars =. bytes_to_bpe_chars raw
    word =. 0 $ a:
    for_c. chars do. word =. word , < , c end.
    word =. ranks bpe_merge word
    for_s. word do.
      t =. > s
      id =. tokens token_to_id t
      if. id < 0 do. id =. unk end.
      ids =. ids , id
    end.
  end.
  if. add_eos do. ids =. ids , eos end.
  ids
)

NB. vocab decode ids
decode =: 4 : 0
  v =. x
  ids =. , y
  'model tokens ranks bos eos unk add_bos add_eos pre' =. vocab_open v
  pieces =. ''
  for_id. ids do.
    if. (id = bos) +. id = eos do. continue. end.
    if. (id < 0) +. id >: # tokens do.
      t =. > unk { tokens
    else.
      t =. > id { tokens
    end.
    pieces =. pieces , t
  end.
  b =. bpe_chars_to_bytes pieces
  bytes_to_text b
)

cocurrent 'base'
