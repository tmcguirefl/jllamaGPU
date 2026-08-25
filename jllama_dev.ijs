NB. jllama_dev - development / testing entry (not used by jllama_cli)
NB. Prefer:
NB.   /Applications/j9.8/bin/jconsole /path/to/jllama_dev.ijs
NB.
NB. Requires published engine under jpath '~temp/jllama' (bin/publish_jllama).
NB. Provides help, smoke, full test runner, and loadcore for REPL work.

cocurrent 'base'

NB. Bootstrap from published sysutils; ROOT still from this script (project tree).
load jpath '~temp/jllama/sysutils.ijs'
3 : 0 ''
  setroot_jllamasys_ 'jllama_dev.ijs'
  i. 0 0
)

cocurrent 'jllama'

NB. Mirror sysutils into jllama locale for existing root_jllama_ / VERSION_jllama_ callers.
VERSION =: VERSION_jllamasys_
MILESTONE =: MILESTONE_jllamasys_
jrequire =: jrequire_jllamasys_
jrequire_temp =: jrequire_temp_jllamasys_
setroot =: setroot_jllamasys_
ROOT =: ROOT_jllamasys_

NB. ---------------------------------------------------------------
NB. Public help / smoke / test (also exported to z below)
NB. Note: do not nest 0 : 0 inside 3 : 0 - ) terminates both.
NB. ---------------------------------------------------------------
HELP =: 0 : 0
jllama_dev - development entry for jllama
  version    jllama_version ''
  smoke      jllama_smoke ''
  test       jllama_test ''
  root       jllama_root ''
  milestone  M14 Qwen3.5 arch nouns (Llama3 / Qwen35) + GQA + M10 lab model

Publish engine (once / after edits):
  bin/publish_jllama
  -> jpath '~temp/jllama/...'

Locales / modules:
  jllamasys     ROOT setroot jrequire jrequire_temp VERSION
  core/tensor   silu softmax rmsnorm linear causal_mask allclose
                (no locale — load into caller; matmul is +/ . *)
  jllamarope    rope rotate_half
  jllamaattn    mha_full mha_step mha_prefill_cached
  jllamablock   ffn_swiglu block_full block_step
  jllamamodel   make_synthetic generate generate_sample
  jllamasample  sample_next top_k_filter top_p_filter
  jllamagguf    gguf_load gguf_tensor model_from_gguf
  jllamavocab   vocab_from_gguf encode decode (gpt2/llama-bpe + llama SPM)
  jllamacli     parse_args run main (CLI loads its own stack)

CLI (standalone; does not load this file):
  ./jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
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
  ROOT_jllamasys_
)

NB. Load M1-M7 engine modules from published ~temp/jllama.
NB. core/tensor.ijs is locale-free: each consumer loads it into its own locale.
loadcore =: 3 : 0
  jrequire_temp 'core/rope.ijs'
  jrequire_temp 'core/attention.ijs'
  jrequire_temp 'core/block.ijs'
  jrequire_temp 'core/sample.ijs'
  jrequire_temp 'core/model.ijs'
  jrequire_temp 'io/gguf.ijs'
  jrequire_temp 'io/vocab.ijs'
  jrequire_temp 'core/arch.ijs'
  ". '0!:0 Llama3'
)

NB. Load CLI module after core (for REPL experiments; production CLI is standalone)
loadcli =: 3 : 0
  loadcore ''
  jrequire_temp 'cli/cli.ijs'
)

NB. Smoke: prior checks + synthetic generate + optional fixture GGUF
smoke =: 3 : 0
  assert. *# ROOT_jllamasys_
  assert. fexist ROOT_jllamasys_ , 'jllama_dev.ijs'
  assert. fexist jpath '~temp/jllama/sysutils.ijs'
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
  fix =. ROOT_jllamasys_ , 'test/fixtures/tiny_llama_f16.gguf'
  if. fexist fix do.
    g =. model_from_gguf_jllamagguf_ fix
    'hp wte layers ln_f lm_head' =. > g
    'n_vocab n_embd n_head n_layer n_ff theta n_head_kv' =. hp
    assert. 8 4 2 1 8 -: n_vocab , n_embd , n_head , n_layer , n_ff
    assert. 2 = n_head_kv
    gids =. g generate_jllamamodel_ (0 1) ; 2
    assert. 4 = # gids
  end.
  vfix =. ROOT_jllamasys_ , 'test/fixtures/tiny_bpe_vocab.gguf'
  if. fexist vfix do.
    v =. vocab_from_gguf_jllamavocab_ vfix
    ids =. v encode_jllamavocab_ 'ab'
    assert. 1 = # ids
    assert. 'ab' -: v decode_jllamavocab_ ids
  end.
  pfix =. ROOT_jllamasys_ , 'test/fixtures/tiny_parity_f16.gguf'
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
  smoutput 'ROOT=' , ROOT_jllamasys_
  smoutput 'JTEMP=' , jpath '~temp/jllama'
  i. 0 0
)

NB. M1-M14 unit tests (M6 needs tools/oracle_greedy)
NB. Tests load from project ROOT; engine already published under ~temp.
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
  jrequire_temp 'cli/cli.ijs'
  jrequire 'test/test_m8.ijs'
  r8 =. run_jllamatestm8_ ''
  jrequire 'test/test_m10.ijs'
  r10 =. run_jllamatestm10_ ''
  jrequire 'test/test_m13.ijs'
  r13 =. run_jllamatestm13_ ''
  jrequire 'test/test_m14.ijs'
  r14 =. run_jllamatestm14_ ''
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
  assert. r14
  smoutput 'jllama test OK  ' , version ''
  i. 0 0
)

NB. ---------------------------------------------------------------
NB. Init banner (dev REPL only)
NB. ---------------------------------------------------------------
3 : 0 ''
  smoutput 'jllama_dev ' , version ''
  smoutput 'ROOT ' , ROOT_jllamasys_
  smoutput 'JTEMP ' , jpath '~temp/jllama'
  smoutput 'Type jllama_help ''''  jllama_smoke ''''  jllama_test '''''
  i. 0 0
)

cocurrent 'z'
jllama_help_z_ =: help_jllama_
jllama_smoke_z_ =: smoke_jllama_
jllama_test_z_ =: test_jllama_
jllama_version_z_ =: version_jllama_
jllama_root_z_ =: root_jllama_
