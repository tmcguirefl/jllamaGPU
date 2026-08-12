NB. jllama - Llama-style decoder inference in J
NB. Entry script. Prefer:
NB.   /Applications/j9.8/bin/jconsole /Users/tomdevel/jdev/jllama/jllama.ijs

cocurrent 'jllama'

VERSION =: '0.13.3'
MILESTONE =: 'M13'

NB. Directory containing this script (works when loaded by full path).
ROOT =: (jpath '~user') NB. placeholder overwritten below

NB. ---------------------------------------------------------------
NB. Bootstrap root from 4!:3 (list of loaded scripts)
NB. ---------------------------------------------------------------
setroot =: 3 : 0
  scripts =. 4!:3 ''
  hit =. scripts #~ +./@('jllama.ijs'&E.)@> scripts
  if. #hit do.
    p =. > {: hit
    NB. directory of script, always with trailing /
    ROOT =: ((p i: '/') {. p) , '/'
  else.
    ROOT =: (1!:43 '') , '/'
  end.
  ROOT
)

NB. Load a project script relative to ROOT
jrequire =: 3 : 0
  path =. ROOT , y
  if. -. fexist path do.
    echo 'jllama: missing ' , path
    'missing script' assert 0
  end.
  load path
)

NB. ---------------------------------------------------------------
NB. Public help / smoke / test (also exported to z below)
NB. Note: do not nest 0 : 0 inside 3 : 0 - ) terminates both.
NB. ---------------------------------------------------------------
HELP =: 0 : 0
jllama - Llama-style inference in J
  version    jllama_version ''
  smoke      jllama_smoke ''
  test       jllama_test ''
  root       jllama_root ''
  milestone  M13 GQA (n_head_kv) + M10 lab model

Locales / modules:
  core/tensor   silu softmax rmsnorm linear causal_mask allclose
                (no locale — load into caller; matmul is +/ . *)
  jllamarope    rope rotate_half
  jllamaattn    mha_full mha_step mha_prefill_cached
  jllamablock   ffn_swiglu block_full block_step
  jllamamodel   make_synthetic generate generate_sample
  jllamasample  sample_next top_k_filter top_p_filter
  jllamagguf    gguf_load gguf_tensor model_from_gguf
  jllamavocab   vocab_from_gguf encode decode (gpt2/llama-bpe + llama SPM)
  jllamacli     parse_args run main (after loadcli)

CLI:
  bin/jllama_cli -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
  bin/jllama_cli --help

Example:
  loadcore_jllama_ ''
  m =. make_synthetic_jllamamodel_ 32;8;2;2;16
  m generate_jllamamodel_ (1 2 3) ; 5
  cfg =. 0.8 ; 40 ; 0.95 ; 1 ; _1 ; 1
  m generate_sample_jllamamodel_ (<1 2 3) , (<8) , (<cfg)

Oracle (M6/M10):
  make -C labs oracle_greedy   NB. needs brew llama.cpp
  labs/run_oracle.sh test/fixtures/tiny_parity_f16.gguf ab 3 --ids 259

Next: optional quant/server; M9 perf deferred

jconsole: /Applications/j9.8/bin/jconsole
trace:    load 'general/misc/trace'
docs:     README.md  docs/hardware.md  docs/milestones.md  docs/models.md
)

help =: 3 : 0
  smoutput HELP
)

version =: 3 : 0
  VERSION , ' (' , MILESTONE , ')'
)

root =: 3 : 0
  ROOT
)

NB. Load M1-M7 modules in order.
NB. core/tensor.ijs is locale-free: each consumer loads it into its own locale
NB. (attention, block, sample, model, tests). Do not load it into jllama here.
loadcore =: 3 : 0
  jrequire 'core/rope.ijs'
  jrequire 'core/attention.ijs'
  jrequire 'core/block.ijs'
  jrequire 'core/sample.ijs'
  jrequire 'core/model.ijs'
  jrequire 'io/gguf.ijs'
  jrequire 'io/vocab.ijs'
)

NB. Load CLI (M8) after core
loadcli =: 3 : 0
  loadcore ''
  jrequire 'cli/cli.ijs'
)

NB. Smoke: prior checks + synthetic generate + optional fixture GGUF
smoke =: 3 : 0
  assert. *# ROOT
  assert. fexist ROOT , 'jllama.ijs'
  loadcore ''
  a =. 2 3 $ 1 2 3 4 5 6
  b =. 3 2 $ 7 8 9 10 11 12
  assert. (2 2 $ 58 64 139 154) -: a +/ . * b
  x =. 2 4 $ 0 1 2 3 4 5 6 7
  r =. rope_jllamarope_ x
  assert. (0 { x) allclose_jllamamodel_ 0 { r
  m =. make_synthetic_jllamamodel_ 16 ; 8 ; 2 ; 1 ; 16
  ids =. m generate_jllamamodel_ (1 2) ; 3
  assert. 5 = # ids
  assert. *./ (0 <: ids) *. ids < 16
  fix =. ROOT , 'test/fixtures/tiny_llama_f16.gguf'
  if. fexist fix do.
    g =. model_from_gguf_jllamagguf_ fix
    'hp wte layers ln_f lm_head' =. > g
    'n_vocab n_embd n_head n_layer n_ff theta n_head_kv' =. hp
    assert. 8 4 2 1 8 -: n_vocab , n_embd , n_head , n_layer , n_ff
    assert. 2 = n_head_kv
    gids =. g generate_jllamamodel_ (0 1) ; 2
    assert. 4 = # gids
  end.
  vfix =. ROOT , 'test/fixtures/tiny_bpe_vocab.gguf'
  if. fexist vfix do.
    v =. vocab_from_gguf_jllamavocab_ vfix
    ids =. v encode_jllamavocab_ 'ab'
    assert. 1 = # ids
    assert. 'ab' -: v decode_jllamavocab_ ids
  end.
  pfix =. ROOT , 'test/fixtures/tiny_parity_f16.gguf'
  if. fexist pfix do.
    pm =. model_from_gguf_jllamagguf_ pfix
    pv =. vocab_from_gguf_jllamavocab_ pfix
    pids =. pv encode_jllamavocab_ 'ab'
    assert. 1 = # pids
    g2 =. pm generate_jllamamodel_ pids ; 2
    assert. 3 = # g2
  end.
  cfg =. 0.8 ; 5 ; 0.9 ; 7 ; _1 ; 1
  sids =. m generate_sample_jllamamodel_ (<1 2) , (<3) , (<cfg)
  assert. 5 = # sids
  smoutput 'jllama smoke OK  ' , version ''
  smoutput 'ROOT=' , ROOT
  i. 0 0
)

NB. M1-M13 unit tests (M6 needs tools/oracle_greedy)
test =: 3 : 0
  loadcore ''
  jrequire 'test/test_tensor.ijs'
  r1 =. run_jllamatest_ ''
  jrequire 'test/test_m2.ijs'
  r2 =. run_jllamatestm2_ ''
  jrequire 'test/test_m3.ijs'
  r3 =. run_jllamatestm3_ ''
  jrequire 'test/test_m4.ijs'
  r4 =. run_jllamatestm4_ ''
  jrequire 'test/test_m5.ijs'
  r5 =. run_jllamatestm5_ ''
  jrequire 'test/test_m6.ijs'
  r6 =. run_jllamatestm6_ ''
  jrequire 'test/test_m7.ijs'
  r7 =. run_jllamatestm7_ ''
  jrequire 'cli/cli.ijs'
  jrequire 'test/test_m8.ijs'
  r8 =. run_jllamatestm8_ ''
  jrequire 'test/test_m10.ijs'
  r10 =. run_jllamatestm10_ ''
  jrequire 'test/test_m13.ijs'
  r13 =. run_jllamatestm13_ ''
  assert. r1
  assert. r2
  assert. r3
  assert. r4
  assert. r5
  assert. r6
  assert. r7
  assert. r8
  assert. r10
  assert. r13
  smoutput 'jllama test OK  ' , version ''
  i. 0 0
)

NB. ---------------------------------------------------------------
NB. Init
NB. QUIET_z_ = 1 before load suppresses the REPL banner (used by CLI).
NB. ---------------------------------------------------------------
setroot ''
3 : 0 ''
  quiet =. 0
  if. 0 = nc <'QUIET_z_' do. quiet =. QUIET_z_ end.
  if. -. quiet do.
    smoutput 'jllama ' , version ''
    smoutput 'ROOT ' , ROOT
    smoutput 'Type jllama_help ''''  jllama_smoke ''''  jllama_test '''''
  end.
  i. 0 0
)

cocurrent 'z'
jllama_help_z_ =: help_jllama_
jllama_smoke_z_ =: smoke_jllama_
jllama_test_z_ =: test_jllama_
jllama_version_z_ =: version_jllama_
jllama_root_z_ =: root_jllama_
