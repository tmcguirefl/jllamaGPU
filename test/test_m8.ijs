NB. M8 tests: jllama_cli parse + run on parity fixture
NB. Box strings with <'text' — not (<,'text') which is a hook.

cocurrent 'jllamatestm8'

parse_args =: parse_args_jllamacli_
cli_run =: run_jllamacli_
fmt_ids =: fmt_ids_jllamacli_
default_opts =: default_opts_jllamacli_
model_from_gguf =: model_from_gguf_jllamagguf_
vocab_from_gguf =: vocab_from_gguf_jllamavocab_
encode =: encode_jllamavocab_
generate =: generate_jllamamodel_

FIX =: (root_jllama_ '') , 'test/fixtures/tiny_parity_f16.gguf'

test_default_opts =: 3 : 0
  o =. default_opts ''
  assert. 12 = # o
  'm p n t k pp s e st tok h v' =. o
  assert. 0 = # m
  assert. 16 = n
  assert. 0 = t
  assert. _2 = e
  assert. 1 = st
  1
)

test_parse_help =: 3 : 0
  o =. parse_args <'--help'
  'm p n t k pp s e st tok h v' =. o
  assert. 1 = h
  o2 =. parse_args <'-h'
  'm p n t k pp s e st tok h v' =. o2
  assert. 1 = h
  1
)

test_parse_flags =: 3 : 0
  args =. <;._1 '|-m|mod.gguf|-p|hello|-n|8|--temp|0.5|--top-k|40|--top-p|0.9|--seed|7|--eos|3|--tokens|--no-stop'
  o =. parse_args args
  'm p n t k pp s e st tok h v' =. o
  assert. 'mod.gguf' -: m
  assert. 'hello' -: p
  assert. 8 = n
  assert. 0.5 = t
  assert. 40 = k
  assert. 0.9 = pp
  assert. 7 = s
  assert. 3 = e
  assert. 0 = st
  assert. 1 = tok
  assert. 0 = h
  assert. 0 = v
  1
)

test_parse_long_model =: 3 : 0
  args =. (<'--model') , (<'x.gguf') , (<'-p') , (<'z')
  o =. parse_args args
  'm p n t k pp s e st tok h v' =. o
  assert. 'x.gguf' -: m
  assert. 'z' -: p
  1
)

test_fmt_ids =: 3 : 0
  assert. '1 2 3' -: fmt_ids 1 2 3
  assert. '42' -: fmt_ids , 42
  assert. '' -: fmt_ids 0 $ 0
  1
)

test_run_missing_model =: 3 : 0
  rc =. cli_run <;._1 '|-p|hi'
  assert. 2 = rc
  1
)

test_run_help =: 3 : 0
  rc =. cli_run <'--help'
  assert. 0 = rc
  rc =. cli_run <'--version'
  assert. 0 = rc
  1
)

test_run_fixture_greedy =: 3 : 0
  'test_m8: missing fixture' assert fexist FIX
  args =. (<'-m') , (<FIX) , (<'-p') , (<'ab') , (<'-n') , (<'3') , (<'--tokens')
  rc =. cli_run args
  assert. 0 = rc
  m =. model_from_gguf FIX
  v =. vocab_from_gguf FIX
  ids =. v encode 'ab'
  out =. m generate ids ; 3
  assert. 4 = # out
  1
)

test_run_bad_model =: 3 : 0
  args =. (<'-m') , (<'/no/such/model.gguf') , (<'-p') , (<'x')
  rc =. cli_run args
  assert. 1 = rc
  1
)

test_parse_unknown =: 3 : 0
  rc =. cli_run <'--not-a-flag'
  assert. 1 = rc
  1
)

run =: 3 : 0
  smoutput 'M8 tests...'
  assert. test_default_opts ''
  smoutput '  default_opts ok'
  assert. test_parse_help ''
  smoutput '  parse_help ok'
  assert. test_parse_flags ''
  smoutput '  parse_flags ok'
  assert. test_parse_long_model ''
  smoutput '  parse_long_model ok'
  assert. test_fmt_ids ''
  smoutput '  fmt_ids ok'
  assert. test_run_missing_model ''
  smoutput '  run_missing_model ok'
  assert. test_run_help ''
  smoutput '  run_help ok'
  assert. test_run_fixture_greedy ''
  smoutput '  run_fixture_greedy ok'
  assert. test_run_bad_model ''
  smoutput '  run_bad_model ok'
  assert. test_parse_unknown ''
  smoutput '  parse_unknown ok'
  smoutput 'M8 tests passed (10)'
  1
)

cocurrent 'base'
