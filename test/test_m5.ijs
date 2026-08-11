NB. M5 tests: GPT-2-style byte BPE encode/decode from GGUF vocab

cocurrent 'jllamatestm5'

vocab_from_gguf =: vocab_from_gguf_jllamavocab_
encode =: encode_jllamavocab_
decode =: decode_jllamavocab_
vocab_bos =: vocab_bos_jllamavocab_
vocab_eos =: vocab_eos_jllamavocab_
vocab_unk =: vocab_unk_jllamavocab_
vocab_tokens =: vocab_tokens_jllamavocab_
vocab_token =: vocab_token_jllamavocab_
gguf_load =: gguf_load_jllamagguf_
gguf_meta =: gguf_meta_jllamagguf_

FIXTURE =: (root_jllama_ '') , 'test/fixtures/tiny_bpe_vocab.gguf'
EXPECT =: (root_jllama_ '') , 'test/fixtures/tiny_bpe_vocab.expect.txt'

read_expect =: 3 : 0
  t =. 1!:1 < EXPECT
  lines =. <;._2 t , LF
  d =. 0 $ a:
  for_L. lines do.
    s =. > L
    if. # s do. d =. d , < s end.
  end.
  d
)

expect_line =: 4 : 0
  key =. x
  lines =. y
  pref =. key , ' '
  for_L. lines do.
    s =. > L
    if. pref -: (# pref) {. s , pref do.
      (# pref) }. s return.
    end.
  end.
  'missing expect key ' , key assert 0
)

test_fixture_exists =: 3 : 0
  assert. fexist FIXTURE
  assert. fexist EXPECT
  1
)

NB. codepoint equality (unicode vs literal)
same_text =: 4 : '(3 u: , x) -: (3 u: , y)'

test_vocab_load =: 3 : 0
  v =. vocab_from_gguf FIXTURE
  exp =. read_expect ''
  n_vocab =. ". 'n_vocab' expect_line exp
  n_merges =. ". 'n_merges' expect_line exp
  toks =. vocab_tokens v
  assert. n_vocab = # toks
  assert. 1 = vocab_bos v
  assert. 2 = vocab_eos v
  assert. 0 = vocab_unk v
  assert. '<unk>' same_text v vocab_token 0
  assert. '<s>' same_text v vocab_token 1
  assert. '</s>' same_text v vocab_token 2
  load =. gguf_load FIXTURE
  merges =. load gguf_meta 'tokenizer.ggml.merges'
  assert. n_merges = # merges
  assert. 'gpt2' -: load gguf_meta 'tokenizer.ggml.model'
  1
)

test_encode_samples =: 3 : 0
  v =. vocab_from_gguf FIXTURE
  exp =. read_expect ''
  NB. ". of a single number is scalar; encode always returns a list
  assert. (, ". 'enc_ab' expect_line exp) -: , v encode 'ab'
  assert. (, ". 'enc_a_b' expect_line exp) -: , v encode 'a b'
  assert. (, ". 'enc_cab' expect_line exp) -: , v encode ' cab'
  assert. (, ". 'enc_hello' expect_line exp) -: , v encode 'hello'
  assert. 0 = # v encode ''
  id_ab =. ". 'id_ab' expect_line exp
  assert. id_ab = {. v encode 'ab'
  1
)

test_decode_roundtrip =: 3 : 0
  v =. vocab_from_gguf FIXTURE
  for_s. 'ab' ; 'a b' ; ' cab' ; 'hello' ; 'c' ; 'aaa' do.
    t =. > s
    ids =. v encode t
    out =. v decode ids
    assert. t same_text out
  end.
  ids =. v encode 'ab'
  ids2 =. (vocab_bos v) , ids , vocab_eos v
  assert. 'ab' same_text v decode ids2
  1
)

test_byte_coverage =: 3 : 0
  v =. vocab_from_gguf FIXTURE
  toks =. vocab_tokens v
  assert. 259 <: # toks  NB. 3 specials + 256 bytes + merges
  assert. 1 = # v encode 'z'
  assert. 'z' same_text v decode v encode 'z'
  assert. ' ' same_text v decode v encode ' '
  1
)

TESTS =: 'test_fixture_exists'; 'test_vocab_load'; 'test_encode_samples'; 'test_decode_roundtrip'; 'test_byte_coverage'

run =: 3 : 0
  failed =. 0
  for_t. TESTS do.
    name =. > t
    try.
      r =. (name ~ '')
      if. 1 -: r do.
        smoutput '  pass  ' , name
      else.
        smoutput '  FAIL  ' , name , ' (bad return)'
        failed =. failed + 1
      end.
    catch.
      smoutput '  FAIL  ' , name
      smoutput '        ' , (13!:12 '')
      failed =. failed + 1
    end.
  end.
  if. failed do.
    smoutput 'jllamatestm5: ' , (": failed) , ' failed'
    0
  else.
    smoutput 'jllamatestm5: ' , (": # TESTS) , ' passed'
    1
  end.
)

cocurrent 'base'
