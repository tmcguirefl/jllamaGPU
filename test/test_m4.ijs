NB. M4 tests: GGUF F16/F32 load + model_from_gguf

cocurrent 'jllamatestm4'

gguf_load =: gguf_load_jllamagguf_
gguf_meta =: gguf_meta_jllamagguf_
gguf_meta_default =: gguf_meta_default_jllamagguf_
gguf_tensor =: gguf_tensor_jllamagguf_
gguf_names =: gguf_names_jllamagguf_
model_from_gguf =: model_from_gguf_jllamagguf_
load ROOT_jllamasys_ , 'core/tensor.ijs'
forward_full =: forward_full_jllamamodel_
generate =: generate_jllamamodel_
logits_last =: logits_last_jllamamodel_

NB. fixture path relative to project root
FIXTURE =: (root_jllama_ '') , 'test/fixtures/tiny_llama_f16.gguf'
EXPECT =: (root_jllama_ '') , 'test/fixtures/tiny_llama_f16.expect.txt'

read_expect =: 3 : 0
  t =. 1!:1 < EXPECT
  lines =. <;._2 t , LF
  d =. 0 $ a:
  for_L. lines do.
    s =. > L
    if. # s do.
      d =. d , < s
    end.
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
      NB. rest after key+space
      (# pref) }. s return.
    end.
  end.
  'missing expect key ' , key assert 0
)

NB. parse floats from a char string of numbers
parse_floats =: 3 : 0
  ". y
)

test_fixture_exists =: 3 : 0
  assert. fexist FIXTURE
  assert. fexist EXPECT
  1
)

test_gguf_header_meta =: 3 : 0
  load =. gguf_load FIXTURE
  'meta tinfos align data_off path' =. > load
  assert. 32 = align
  exp =. read_expect ''
  want_off =. ". 'data_off' expect_line exp
  assert. want_off = data_off
  assert. 'llama' -: load gguf_meta 'general.architecture'
  assert. 4 = load gguf_meta 'llama.embedding_length'
  assert. 1 = load gguf_meta 'llama.block_count'
  assert. 8 = load gguf_meta 'llama.feed_forward_length'
  assert. 2 = load gguf_meta 'llama.attention.head_count'
  assert. 2 = load gguf_meta 'llama.attention.head_count_kv'
  assert. 8 = load gguf_meta 'llama.vocab_size'
  th =. load gguf_meta 'llama.rope.freq_base'
  assert. 1e_3 > | th - 10000
  n_tensors =. ". 'n_tensors' expect_line exp
  assert. n_tensors = # tinfos
  names =. gguf_names load
  assert. (< 'token_embd.weight') e. names
  assert. (< 'blk.0.attn_q.weight') e. names
  assert. (< 'output.weight') e. names
  1
)

test_tensor_shapes_values =: 3 : 0
  load =. gguf_load FIXTURE
  exp =. read_expect ''
  wte =. G.^:_1 load gguf_tensor 'token_embd.weight'
  assert. 8 4 -: $ wte
  wte0 =. parse_floats 'wte0' expect_line exp
  NB. F16 roundtrip - looser tol
  assert. *./ (1e_3 > | wte0 - 0 { wte)
  an =. G.^:_1 load gguf_tensor 'blk.0.attn_norm.weight'
  assert. 1 = #$ an
  assert. 4 = # an
  an_e =. parse_floats 'attn_norm' expect_line exp
  assert. an_e allclose an
  wq =. G.^:_1 load gguf_tensor 'blk.0.attn_q.weight'
  assert. 4 4 -: $ wq
  wq00 =. ". 'wq00' expect_line exp
  NB. GGUF layout is n_out x n_in (not the old n_in x n_out transpose)
  assert. 1e_3 > | wq00 - (<0 0) { wq
  lm =. G.^:_1 load gguf_tensor 'output.weight'
  assert. 8 4 -: $ lm
  wd =. G.^:_1 load gguf_tensor 'blk.0.ffn_down.weight'
  assert. 4 8 -: $ wd
  wg =. G.^:_1 load gguf_tensor 'blk.0.ffn_gate.weight'
  assert. 8 4 -: $ wg
  1
)

test_model_from_gguf =: 3 : 0
  m =. model_from_gguf FIXTURE
  'hp wte layers ln_f lm_head' =. > m
  'n_vocab n_embd n_head n_layer n_ff theta n_head_kv' =. hp
  assert. 8 4 2 1 8 -: n_vocab , n_embd , n_head , n_layer , n_ff
  assert. 2 = n_head_kv
  assert. 1e_3 > | theta - 10000
  assert. 8 4 -: $ wte
  assert. 4 = # ln_f
  assert. 8 4 -: $ lm_head
  assert. 1 = # layers
  L =. 0 { layers
  'attn_n wq wk wv wo ffn_n wg wu wd' =. > L
  assert. 4 4 -: $ wq
  assert. 8 4 -: $ wg
  assert. 4 8 -: $ wd
  1
)

test_forward_generate =: 3 : 0
  m =. model_from_gguf FIXTURE
  h =. m forward_full 1 2 3
  assert. 3 4 -: $ h
  z =. m logits_last h
  assert. 8 = # z
  ids =. m generate (0 1) ; 3
  assert. 5 = # ids
  assert. *./ (0 <: ids) *. ids < 8
  NB. deterministic
  assert. ids -: m generate (0 1) ; 3
  1
)

TESTS =: 'test_fixture_exists'; 'test_gguf_header_meta'; 'test_tensor_shapes_values'; 'test_model_from_gguf'; 'test_forward_generate'

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
    smoutput 'jllamatestm4: ' , (": failed) , ' failed'
    0
  else.
    smoutput 'jllamatestm4: ' , (": # TESTS) , ' passed'
    1
  end.
)

cocurrent 'base'
