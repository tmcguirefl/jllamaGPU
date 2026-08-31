#!/usr/bin/env jconsole
NB. jllamaGPU — GNU GPL v3 only. Copyright (C) 2026 Tom McGuire. See LICENSE.
NB. jllama_cli - standalone shell entry
NB.
NB. From a clone of this tree:
NB.   ./jllama_cli.ijs -m MODEL.gguf -p PROMPT ...          (macOS/Linux shebang)
NB.   bin\jllama_cli.cmd -m MODEL.gguf -p PROMPT ...       (Windows)
NB.   jconsole jllama_cli.ijs -m MODEL.gguf -p PROMPT ...  (any OS)
NB.
NB. The #! line is kept for Unix. J skips it (xs.c); Windows jconsole does too.
NB. Loads sibling scripts from this directory (ROOT). No ~temp publish step.
NB. Later: manifest.ijs will install the same tree as a J addon.

cocurrent 'base'

NB. This script is already in 4!:3; use it to find the clone root.
NB. Windows 4!:3 uses \; Unix uses /. Take the last of either.
3 : 0 ''
  p =. > {: 4!:3 ''
  i =. (p i: '/') >. p i: '\'
  if. i < # p do. d =. (i {. p) , '/' else. d =. (1!:43 '') , '/' end.
  load d , 'sysutils.ijs'
  setroot_jllamasys_ 'jllama_cli.ijs'
  i. 0 0
)

load ROOT_jllamasys_ , 'jgpu.ijs'
load ROOT_jllamasys_ , 'core/rope.ijs'
load ROOT_jllamasys_ , 'core/attention.ijs'
load ROOT_jllamasys_ , 'core/block.ijs'
load ROOT_jllamasys_ , 'core/sample.ijs'
load ROOT_jllamasys_ , 'core/model.ijs'
load ROOT_jllamasys_ , 'io/gguf.ijs'
load ROOT_jllamasys_ , 'io/vocab.ijs'
load ROOT_jllamasys_ , 'core/arch.ijs'
". '0!:0 Llama3'
load ROOT_jllamasys_ , 'cli/cli.ijs'

main_jllamacli_ ''
