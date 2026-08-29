NB. M14 tests: architecture nouns, filename detect, Qwen3.5 primitives

cocurrent 'jllamatestm14'

load ROOT_jllamasys_ , 'core/tensor.ijs'
load ROOT_jllamasys_ , 'core/gdn.ijs'
load ROOT_jllamasys_ , 'core/arch.ijs'
rope_neox =: rope_neox_jllamarope_
gdn_seq =: gdn_seq_jllamagdn_
causal_conv1d =: causal_conv1d_jllamagdn_
sigmoid =: sigmoid_jllamagdn_
make_synthetic =: make_synthetic_jllamamodel_
generate =: generate_jllamamodel_
detect_arch =: detect_arch_jllamaarch_

test_detect_arch_filename =: 3 : 0
  assert. 'Qwen35' -: detect_arch 'models/Qwen3.5-2B-Instruct-f16.gguf'
  assert. 'Qwen35' -: detect_arch '/tmp/qwen35-2b.Q8_0.gguf'
  assert. 'Qwen35' -: detect_arch 'Qwen3_5-2B.gguf'
  assert. 'Qwen35' -: detect_arch 'qwen-3.5-2b.gguf'
  assert. 'Llama3' -: detect_arch 'models/Llama-3.2-1B-Instruct-f16.gguf'
  assert. 'Llama3' -: detect_arch 'models/stories15M.F16.gguf'
  assert. 'Llama3' -: detect_arch 'test/fixtures/tiny_parity_f16.gguf'
  assert. 'Phi4Mini' -: detect_arch 'models/Phi-4-mini-instruct-f16.gguf'
  assert. 'Phi4Mini' -: detect_arch 'phi4-mini-Q8_0.gguf'
  assert. 'Phi4Mini' -: detect_arch 'Phi-4-14B-f16.gguf'
  1
)

test_arch_nouns =: 3 : 0
  assert. 2 = 3!:0 Llama3
  assert. 2 = 3!:0 Qwen35
  assert. 2 = 3!:0 Phi4Mini
  assert. LF e. Llama3
  assert. LF e. Qwen35
  assert. LF e. Phi4Mini
  assert. 1 e. 'jllamaqwen' E. Qwen35
  assert. 1 e. 'jllamaphi' E. Phi4Mini
  assert. 1 e. 'model_from_gguf_llama' E. Llama3
  assert. 1 e. 'model_from_gguf_qwen' E. Qwen35
  assert. 1 e. 'model_from_gguf_phi' E. Phi4Mini
  1
)

test_do_llama3_generate =: 3 : 0
  ". '0!:0 Llama3'
  m =. make_synthetic 16 ; 8 ; 2 ; 1 ; 16
  ids =. m generate (1 2) ; 3
  assert. 5 = # ids
  assert. *./ (0 <: ids) *. ids < 16
  1
)

test_switch_qwen35_then_llama3 =: 3 : 0
  ". '0!:0 Qwen35'
  assert. 3 = 4!:0 <'block_full_jllamaqwen_'
  ". '0!:0 Llama3'
  m =. make_synthetic 16 ; 8 ; 2 ; 1 ; 16
  ids =. m generate (1 2) ; 2
  assert. 4 = # ids
  1
)

test_apply_arch_do =: 3 : 0
  apply_arch_jllamaarch_ 'Qwen35'
  assert. 3 = 4!:0 <'block_full_jllamablock_'
  apply_arch_jllamaarch_ 'Llama3'
  m =. make_synthetic 12 ; 8 ; 2 ; 1 ; 16
  assert. 4 = # m generate (0 1) ; 2
  1
)

test_switch_phi4mini_then_llama3 =: 3 : 0
  ". '0!:0 Phi4Mini'
  assert. 3 = 4!:0 <'block_full_jllamaphi_'
  m =. make_synthetic 16 ; 8 ; 2 ; 1 ; 16
  ids =. m generate (1 2) ; 2
  assert. 4 = # ids
  ". '0!:0 Llama3'
  ids2 =. m generate (1 2) ; 2
  assert. 4 = # ids2
  1
)

test_rope_neox_pos0 =: 3 : 0
  x =. 1 8 $ 0.1 * 1 + i. 8
  got =. 0 rope_neox x
  assert. x allclose got
  1
)

test_rope_neox_partial =: 3 : 0
  NB. n_rot=4 of d=8 at pos=1: last 4 dims copied, first 4 rotate
  x =. 1 8 $ 1 2 3 4 5 6 7 8
  got =. (1 ; 10000 ; 4) rope_neox x
  assert. (4 }. , x) allclose 4 }. , got
  assert. -. (4 {. , x) allclose 4 {. , got
  1
)

test_sigmoid =: 3 : 0
  assert. 0.5 allclose sigmoid 0
  1
)

test_gdn_one_token =: 3 : 0
  Q =. 1 1 2 $ 1 0
  K =. 1 1 2 $ 1 0
  V =. 1 1 2 $ 1 0
  g =. 1 1 $ 0
  beta =. 1 1 $ 1
  'out st' =. gdn_seq Q ; K ; V ; g ; beta
  out =. $.^:_1 out
  st =. $.^:_1 st
  assert. 1 1 2 -: $ out
  scale =. % %: 2
  assert. (scale , 0) allclose , out
  assert. 1 2 2 -: $ st
  1
)

test_causal_conv1d =: 3 : 0
  NB. C=1, d_conv=2, kernel [1 0] -> output is previous-padded input
  k =. 1 2 $ 0 1
  x =. 3 1 $ 2 3 4
  got =. k causal_conv1d x
  NB. windows: [0,2] [2,3] [3,4] dotted with [0,1] -> 2 3 4
  assert. (3 1 $ 2 3 4) allclose got
  1
)

run =: 3 : 0
  smoutput 'M14 tests...'
  assert. test_detect_arch_filename ''
  assert. test_arch_nouns ''
  assert. test_do_llama3_generate ''
  assert. test_switch_qwen35_then_llama3 ''
  assert. test_switch_phi4mini_then_llama3 ''
  assert. test_apply_arch_do ''
  assert. test_rope_neox_pos0 ''
  assert. test_rope_neox_partial ''
  assert. test_sigmoid ''
  assert. test_gdn_one_token ''
  assert. test_causal_conv1d ''
  smoutput 'M14 OK'
  1
)

cocurrent 'base'
