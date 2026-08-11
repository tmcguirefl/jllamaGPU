NB. M6 tests: greedy token parity vs llama.cpp (libllama oracle)

cocurrent 'jllamatestm6'

model_from_gguf =: model_from_gguf_jllamagguf_
generate =: generate_jllamamodel_
vocab_from_gguf =: vocab_from_gguf_jllamavocab_
encode =: encode_jllamavocab_
gguf_meta =: gguf_meta_jllamagguf_
gguf_load =: gguf_load_jllamagguf_

FIXTURE =: (root_jllama_ '') , 'test/fixtures/tiny_parity_f16.gguf'
META =: (root_jllama_ '') , 'test/fixtures/tiny_parity_f16.meta.txt'
ORACLE_SH =: (root_jllama_ '') , 'labs/run_oracle.sh'
ORACLE_BIN =: (root_jllama_ '') , 'tools/oracle_greedy'

read_lines =: 3 : 0
  t =. 1!:1 < y
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
  'missing meta key ' , key assert 0
)

NB. strip leading LABEL from a section string
strip_lab =: 4 : 0
  t =. deb y
  lab =. x
  if. lab -: (# lab) {. t do.
    t =. (# lab) }. t
    if. # t do. if. ' ' = {. t do. t =. }. t end. end.
  end.
  deb t
)

NB. parse "PROMPT a b | GEN c d | FULL a b c d" -> three int lists as boxes
parse_oracle =: 3 : 0
  s =. y -. CR
  s =. LF taketo s , LF
  parts =. <;._1 '|', s
  'parse_oracle: need 3 parts' assert 3 <: # parts
  p =. 'PROMPT' strip_lab > 0 { parts
  g =. 'GEN' strip_lab > 1 { parts
  f =. 'FULL' strip_lab > 2 { parts
  (". p) ; (". g) ; ". f
)

NB. run oracle: path ; prompt ; n_new ; ids  -> full ids list
run_oracle_ids =: 3 : 0
  'path prompt n_new ids' =. y
  cmd =. ORACLE_SH , ' ' , path , ' ' , (quote prompt) , ' ' , (": n_new) , ' --ids'
  for_i. ids do. cmd =. cmd , ' ' , ": i end.
  NB. 2!:0 captures stdout
  out =. 2!:0 cmd
  'p g f' =. parse_oracle out
  'oracle prompt ids mismatch' assert (, ids) -: , p
  f
)

test_fixture_and_oracle =: 3 : 0
  assert. fexist FIXTURE
  assert. fexist META
  assert. fexist ORACLE_SH
  assert. fexist ORACLE_BIN
  1
)

test_load_shapes =: 3 : 0
  m =. model_from_gguf FIXTURE
  'hp wte layers ln_f lm_head' =. > m
  'n_vocab n_embd n_head n_layer n_ff theta n_head_kv' =. hp
  meta =. read_lines META
  assert. n_vocab = ". 'n_vocab' expect_line meta
  assert. n_embd = ". 'n_embd' expect_line meta
  assert. n_head = ". 'n_head' expect_line meta
  assert. n_layer = ". 'n_layer' expect_line meta
  assert. n_ff = ". 'n_ff' expect_line meta
  assert. n_vocab = # wte
  assert. n_embd = {: $ wte
  assert. n_layer = # layers
  load =. gguf_load FIXTURE
  assert. 'gpt2' -: load gguf_meta 'tokenizer.ggml.model'
  1
)

test_tokenizer_vs_meta =: 3 : 0
  v =. vocab_from_gguf FIXTURE
  meta =. read_lines META
  assert. (, ". 'prompt_ab' expect_line meta) -: , v encode 'ab'
  assert. (, ". 'prompt_hello' expect_line meta) -: , v encode 'hello'
  assert. (, ". 'prompt_a_b' expect_line meta) -: , v encode 'a b'
  assert. (, ". 'prompt_cab' expect_line meta) -: , v encode ' cab'
  1
)

test_greedy_parity_ids =: 3 : 0
  m =. model_from_gguf FIXTURE
  meta =. read_lines META
  cases =. 'prompt_ab' ; 'prompt_hello' ; 'prompt_a_b' ; 'prompt_cab'
  prompts =. 'ab' ; 'hello' ; 'a b' ; ' cab'
  n_new =. 3
  for_i. i. # cases do.
    key =. > i { cases
    prompt =. > i { prompts
    ids =. , ". key expect_line meta
    jl =. , m generate ids ; n_new
    or =. , run_oracle_ids FIXTURE ; prompt ; n_new ; ids
    if. -. jl -: or do.
      smoutput 'parity fail prompt=' , prompt
      smoutput '  jllama  ' , ": jl
      smoutput '  oracle  ' , ": or
      assert. 0
    end.
  end.
  1
)

test_greedy_parity_text_encode =: 3 : 0
  NB. encode in jllama, generate, compare to oracle on same ids
  m =. model_from_gguf FIXTURE
  v =. vocab_from_gguf FIXTURE
  n_new =. 4
  for_p. 'ab' ; 'hello' ; 'a b' do.
    prompt =. > p
    ids =. , v encode prompt
    jl =. , m generate ids ; n_new
    or =. , run_oracle_ids FIXTURE ; prompt ; n_new ; ids
    assert. jl -: or
  end.
  1
)

TESTS =: 'test_fixture_and_oracle'; 'test_load_shapes'; 'test_tokenizer_vs_meta'; 'test_greedy_parity_ids'; 'test_greedy_parity_text_encode'

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
    smoutput 'jllamatestm6: ' , (": failed) , ' failed'
    0
  else.
    smoutput 'jllamatestm6: ' , (": # TESTS) , ' passed'
    1
  end.
)

cocurrent 'base'
