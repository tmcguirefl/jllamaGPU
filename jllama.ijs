NB. jllama - Llama-style decoder inference in J
NB. Entry script. Prefer:
NB.   /Applications/j9.8/bin/jconsole /Users/tomdevel/jdev/jllama/jllama.ijs

cocurrent 'jllama'

VERSION =: '0.6.0'
MILESTONE =: 'M6'

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
  milestone  M6 greedy parity vs llama.cpp

Locales:
  jllamatensor  mp silu softmax rmsnorm linear causal_mask allclose
  jllamarope    rope rotate_half
  jllamaattn    mha_full mha_step mha_prefill_cached
  jllamablock   ffn_swiglu block_full block_step
  jllamamodel   make_synthetic generate forward_full
  jllamagguf    gguf_load gguf_tensor model_from_gguf
  jllamavocab   vocab_from_gguf encode decode

Example:
  loadcore_jllama_ ''
  m =. make_synthetic_jllamamodel_ 32;8;2;2;16
  m generate_jllamamodel_ (1 2 3) ; 5
  m =. model_from_gguf_jllamagguf_ jllama_root '' , 'test/fixtures/tiny_parity_f16.gguf'
  v =. vocab_from_gguf_jllamavocab_ jllama_root '' , 'test/fixtures/tiny_parity_f16.gguf'
  ids =. v encode_jllamavocab_ 'ab'
  m generate_jllamamodel_ ids ; 3

Oracle (M6):
  make -C labs oracle_greedy   NB. needs brew llama.cpp
  labs/run_oracle.sh test/fixtures/tiny_parity_f16.gguf ab 3 --ids 259

Next: M7 sampling UX

jconsole: /Applications/j9.8/bin/jconsole
trace:    load 'general/misc/trace'
docs:     README.md  docs/hardware.md  docs/milestones.md
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

NB. Load M1-M6 modules in order
loadcore =: 3 : 0
  jrequire 'core/tensor.ijs'
  jrequire 'core/rope.ijs'
  jrequire 'core/attention.ijs'
  jrequire 'core/block.ijs'
  jrequire 'core/model.ijs'
  jrequire 'io/gguf.ijs'
  jrequire 'io/vocab.ijs'
)

NB. Smoke: prior checks + synthetic generate + optional fixture GGUF
smoke =: 3 : 0
  assert. *# ROOT
  assert. fexist ROOT , 'jllama.ijs'
  loadcore ''
  a =. 2 3 $ 1 2 3 4 5 6
  b =. 3 2 $ 7 8 9 10 11 12
  assert. (2 2 $ 58 64 139 154) -: a mp_jllamatensor_ b
  x =. 2 4 $ 0 1 2 3 4 5 6 7
  r =. rope_jllamarope_ x
  assert. (0 { x) allclose_jllamatensor_ 0 { r
  m =. make_synthetic_jllamamodel_ 16 ; 8 ; 2 ; 1 ; 16
  ids =. m generate_jllamamodel_ (1 2) ; 3
  assert. 5 = # ids
  assert. *./ (0 <: ids) *. ids < 16
  fix =. ROOT , 'test/fixtures/tiny_llama_f16.gguf'
  if. fexist fix do.
    g =. model_from_gguf_jllamagguf_ fix
    'hp wte layers ln_f lm_head' =. > g
    'n_vocab n_embd n_head n_layer n_ff theta' =. hp
    assert. 8 4 2 1 8 -: n_vocab , n_embd , n_head , n_layer , n_ff
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
  smoutput 'jllama smoke OK  ' , version ''
  smoutput 'ROOT=' , ROOT
  i. 0 0
)

NB. M1-M6 unit tests (M6 needs tools/oracle_greedy)
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
  assert. r1
  assert. r2
  assert. r3
  assert. r4
  assert. r5
  assert. r6
  smoutput 'jllama test OK  ' , version ''
  i. 0 0
)

NB. ---------------------------------------------------------------
NB. Init
NB. ---------------------------------------------------------------
setroot ''
smoutput 'jllama ' , version ''
smoutput 'ROOT ' , ROOT
smoutput 'Type jllama_help ''''  jllama_smoke ''''  jllama_test '''''

cocurrent 'z'
jllama_help_z_ =: help_jllama_
jllama_smoke_z_ =: smoke_jllama_
jllama_test_z_ =: test_jllama_
jllama_version_z_ =: version_jllama_
jllama_root_z_ =: root_jllama_
