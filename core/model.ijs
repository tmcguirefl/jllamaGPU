NB. jllama tiny model stack + greedy generate (M3)
NB.
NB. Model is ONE scalar box:
NB.   <"_ (hparams ; wte ; layers ; ln_f ; lm_head)
NB. hparams = open list: n_vocab ; n_embd ; n_head ; n_layer ; n_ff ; theta
NB. layers  = list of layer boxes (each layer is one scalar box of 9 weights)
NB.
NB. Synthetic models use deterministic integer-derived weights (no RNG).
NB.
NB. Boxing / packing rules (critical in J):
NB.   1. Enclose a whole open list with  <"_ y   (not bare < if intent is enclose).
NB.   2. 'a b c' =. open_list     spreads items.
NB.   3. 'a b c' =. <open_list    does NOT spread - open first:  > y
NB.   4. Chained  a ; b ; c  re-boxes when left is already a box list.
NB.      Pack mixed nested args with:
NB.        (<a) , (<b) , already_boxed_c , (<d)
NB.   5. Pure-numeric packs may still use  ;

cocurrent 'jllamamodel'

mp =: mp_jllamatensor_
rmsnorm =: rmsnorm_jllamatensor_
block_full =: block_full_jllamablock_
block_step =: block_step_jllamablock_
block_prefill_cached =: block_prefill_cached_jllamablock_
DEFAULT_THETA =: DEFAULT_THETA_jllamarope_
sample_next =: sample_next_jllamasample_
sample_cfg_pack =: sample_cfg_pack_jllamasample_
default_cfg =: default_cfg_jllamasample_

NB. ---------------------------------------------------------------
NB. Weight builders
NB. ---------------------------------------------------------------

NB. shape packw seed -> small deterministic weights
packw =: 4 : 0
  shape =. x
  seed =. y
  n =. */ shape
  shape $ 0.02 * <: 23 | seed + 3 * i. n
)

NB. y = n_embd ; n_head ; n_ff ; seed
NB. returns ONE scalar box of 9 weights
make_layer =: 3 : 0
  'n_embd n_head n_ff seed' =. y
  'make_layer: bad head split' assert 0 = n_head | n_embd
  attn_n =. n_embd $ 1 + 0.01 * i. n_embd
  ffn_n =. n_embd $ 1 + 0.01 * |. i. n_embd
  wq =. (n_embd , n_embd) packw seed + 1
  wk =. (n_embd , n_embd) packw seed + 2
  wv =. (n_embd , n_embd) packw seed + 3
  wo =. (n_embd , n_embd) packw seed + 4
  wg =. (n_embd , n_ff) packw seed + 5
  wu =. (n_embd , n_ff) packw seed + 6
  wd =. (n_ff , n_embd) packw seed + 7
  <"_ (attn_n ; wq ; wk ; wv ; wo ; ffn_n ; wg ; wu ; wd)
)

NB. y = n_vocab ; n_embd ; n_head ; n_layer ; n_ff [; theta] [; seed]
NB. returns ONE model box
make_synthetic =: 3 : 0
  theta =. DEFAULT_THETA
  seed =. 0
  select. # y
  case. 5 do. 'n_vocab n_embd n_head n_layer n_ff' =. y
  case. 6 do. 'n_vocab n_embd n_head n_layer n_ff theta' =. y
  case. 7 do. 'n_vocab n_embd n_head n_layer n_ff theta seed' =. y
  case. do. 'make_synthetic: bad arg count' assert 0
  end.
  wte =. (n_vocab , n_embd) packw seed + 100
  layers =. 0 $ a:
  for_i. i. n_layer do.
    layers =. layers , make_layer n_embd ; n_head ; n_ff ; seed + 1000 + 50 * i
  end.
  ln_f =. n_embd $ 1 + 0.005 * i. n_embd
  lm_head =. (n_embd , n_vocab) packw seed + 200
  hparams =. n_vocab ; n_embd ; n_head ; n_layer ; n_ff ; theta
  <"_ (hparams ; wte ; layers ; ln_f ; lm_head)
)

NB. ---------------------------------------------------------------
NB. Forward
NB. ---------------------------------------------------------------

NB. model embed ids
embed =: 4 : 0
  'hp wte layers ln_f lm_head' =. > x
  y { wte
)

NB. model logits_last xhidden  (last token only)
logits_last =: 4 : 0
  'hp wte layers ln_f lm_head' =. > x
  h =. ln_f rmsnorm y
  ({: h) mp lm_head
)

NB. model logits_all xhidden
logits_all =: 4 : 0
  'hp wte layers ln_f lm_head' =. > x
  h =. ln_f rmsnorm y
  h mp lm_head
)

NB. model forward_full ids -> hidden
forward_full =: 4 : 0
  model =. x
  'hp wte layers ln_f lm_head' =. > model
  'n_vocab n_embd n_head n_layer n_ff theta' =. hp
  h =. model embed y
  for_L. layers do.
    h =. block_full (<h) , (<n_head) , L , (<theta)
  end.
  h
)

NB. model forward_prefill ids -> (<hidden) , (<caches)
NB. caches = list of per-layer (<kc;vc>) boxes
forward_prefill =: 4 : 0
  model =. x
  'hp wte layers ln_f lm_head' =. > model
  'n_vocab n_embd n_head n_layer n_ff theta' =. hp
  h =. model embed y
  caches =. 0 $ a:
  for_L. layers do.
    'h kc vc' =. block_prefill_cached (<h) , (<n_head) , L , (<theta)
    caches =. caches , <"_ (kc ; vc)
  end.
  (<h) , (<caches)
)

NB. model forward_step (<id) , (<caches) , (<pos)
NB. returns (<hidden1) , (<caches2)
forward_step =: 4 : 0
  model =. x
  'id caches pos' =. y
  'hp wte layers ln_f lm_head' =. > model
  'n_vocab n_embd n_head n_layer n_ff theta' =. hp
  h =. model embed , id
  caches2 =. 0 $ a:
  i =. 0
  for_L. layers do.
    'kc vc' =. > i { caches
    'h kc vc' =. block_step (<h) , (<n_head) , L , (<kc) , (<vc) , (<pos) , (<theta)
    caches2 =. caches2 , <"_ (kc ; vc)
    i =. i + 1
  end.
  (<h) , (<caches2)
)

NB. ---------------------------------------------------------------
NB. Generate
NB. model generate ids ; n_new
NB.   Greedy (temp=0). Backward-compatible with M3-M6.
NB.
NB. model generate_sample (<ids) , (<n_new) , (<cfg)
NB.   cfg = temp ; top_k ; top_p ; seed ; eos_id ; stop_on_eos
NB.   See core/sample.ijs. Pack with catenate of scalar boxes.
NB.   On EOS (if eos_id>=0 and stop_on_eos): append EOS and stop.
NB. ---------------------------------------------------------------
generate =: 4 : 0
  model =. x
  'ids n_new' =. y
  model generate_sample (<,ids) , (<n_new) , (<default_cfg '')
)

generate_sample =: 4 : 0
  model =. x
  'ids n_new cfg' =. y
  ids =. , ids
  n_new =. ,. n_new
  n_new =. {. n_new
  cfg =. sample_cfg_pack cfg
  'temp top_k top_p seed eos_id stop_on_eos' =. cfg
  if. 0 = # ids do.
    'generate_sample: empty prompt' assert 0
  end.
  'hp wte layers ln_f lm_head' =. > model
  'h caches' =. model forward_prefill ids
  for. i. n_new do.
    logits =. model logits_last h
    'nxt seed' =. sample_next (<cfg) , (<logits)
    NB. refresh seed inside cfg (open numeric list)
    cfg =. (temp , top_k , top_p , seed , eos_id , stop_on_eos)
    ids =. ids , nxt
    if. (eos_id >: 0) *. stop_on_eos *. nxt = eos_id do. break. end.
    pos =. <: # ids
    'h caches' =. model forward_step (<nxt) , (<caches) , (<pos)
  end.
  ids
)

NB. Slow oracle: full recompute every token (greedy)
generate_fullrecompute =: 4 : 0
  model =. x
  'ids n_new' =. y
  ids =. , ids
  for. i. n_new do.
    h =. model forward_full ids
    logits =. model logits_last h
    nxt =. (i. >./) logits
    ids =. ids , nxt
  end.
  ids
)

cocurrent 'base'
