NB. M10 tests: real-model lab (stories15M F16) + Llama SPM tokenizer
NB. Skips gracefully if models/stories15M.F16.gguf is absent.

cocurrent 'jllamatestm10'

vocab_from_gguf =: vocab_from_gguf_jllamavocab_
encode =: encode_jllamavocab_
decode =: decode_jllamavocab_
model_from_gguf =: model_from_gguf_jllamagguf_
generate =: generate_jllamamodel_
cli_run =: run_jllamacli_

MODEL =: (root_jllama_ '') , 'models/stories15M.F16.gguf'
ORACLE_TOK =: (root_jllama_ '') , 'tools/oracle_tokenize'
ORACLE_GEN =: (root_jllama_ '') , 'tools/oracle_greedy'

have_model =: 3 : 'fexist MODEL'
have_oracle_tok =: 3 : 'fexist ORACLE_TOK'

test_model_present_or_skip =: 3 : 0
  if. -. have_model '' do.
    smoutput '  SKIP (no models/stories15M.F16.gguf)'
    1 return.
  end.
  1
)

test_load_hparams =: 3 : 0
  if. -. have_model '' do. 1 return. end.
  m =. model_from_gguf MODEL
  'hp wte layers ln_f lm_head' =. > m
  'n_vocab n_embd n_head n_layer n_ff theta n_head_kv' =. hp
  assert. 32000 = n_vocab
  assert. 288 = n_embd
  assert. 6 = n_head
  assert. 6 = n_head_kv
  assert. 6 = n_layer
  assert. 768 = n_ff
  assert. 6 = # layers
  assert. 32000 288 -: $ wte
  assert. 32000 288 -: $ lm_head
  1
)

test_spm_encode_oracle =: 3 : 0
  if. -. have_model '' do. 1 return. end.
  v =. vocab_from_gguf MODEL
  a =. v encode 'Once upon a time'
  b =. v encode 'Hello'
  assert. 1 9038 2501 263 931 -: a
  assert. 1 15043 -: b
  if. have_oracle_tok '' do.
    NB. libllama prints load logs on stderr; keep only the id line
    cmd =. ORACLE_TOK , ' ' , MODEL , ' "Once upon a time" 2>/dev/null | grep -E ''^[0-9 ]+$'' | tail -1'
    o =. 2!:0 cmd
    o =. deb o -. CR , LF
    assert. '1 9038 2501 263 931' -: o
  end.
  1
)

test_spm_decode_space_prefix =: 3 : 0
  if. -. have_model '' do. 1 return. end.
  v =. vocab_from_gguf MODEL
  ids =. v encode 'Once upon a time'
  t =. v decode ids
  assert. ' Once upon a time' -: t
  1
)

test_greedy_prefix_vs_oracle =: 3 : 0
  if. -. have_model '' do. 1 return. end.
  if. -. fexist ORACLE_GEN do.
    smoutput '  SKIP greedy oracle (no tools/oracle_greedy)'
    1 return.
  end.
  v =. vocab_from_gguf MODEL
  m =. model_from_gguf MODEL
  ids =. v encode 'Once upon a time'
  out =. m generate ids ; 4
  NB. oracle FULL line: PROMPT ... | GEN ... | FULL ...
  cmd =. ORACLE_GEN , ' ' , MODEL , ' "Once upon a time" 4 2>/dev/null | grep FULL | tail -1'
  line =. deb ((2!:0 cmd) -. CR) -. LF
  NB. take FULL segment after last | and drop leading "FULL" label
  parts =. <;._1 '|', line
  full =. deb > {: parts
  if. 'FULL' -: 4 {. full do. full =. deb 4 }. full end.
  want =. ". full
  assert. out -: want
  1
)

test_cli_english =: 3 : 0
  if. -. have_model '' do. 1 return. end.
  args =. (<'-m') , (<MODEL) , (<'-p') , (<'Once upon a time') , (<'-n') , (<'8')
  rc =. cli_run args
  assert. 0 = rc
  1
)

test_cli_story_contains_words =: 3 : 0
  if. -. have_model '' do. 1 return. end.
  v =. vocab_from_gguf MODEL
  m =. model_from_gguf MODEL
  ids =. v encode 'Once upon a time'
  out =. m generate ids ; 24
  t =. v decode out
  NB. crude english-ish: should contain "time" and a comma or "there"
  assert. +./ 'time' E. t
  assert. (+./ 'there' E. t) +. +./ ',' E. t
  1
)

run =: 3 : 0
  smoutput 'M10 tests...'
  assert. test_model_present_or_skip ''
  assert. test_load_hparams ''
  smoutput '  load_hparams ok/skip'
  assert. test_spm_encode_oracle ''
  smoutput '  spm_encode_oracle ok/skip'
  assert. test_spm_decode_space_prefix ''
  smoutput '  spm_decode ok/skip'
  assert. test_greedy_prefix_vs_oracle ''
  smoutput '  greedy_prefix ok/skip'
  assert. test_cli_english ''
  smoutput '  cli_english ok/skip'
  assert. test_cli_story_contains_words ''
  smoutput '  story_words ok/skip'
  smoutput 'M10 tests passed (7)'
  1
)

cocurrent 'base'
