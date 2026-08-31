NB. jllamaGPU — GNU GPL v3 only. Copyright (C) 2026 Tom McGuire. See LICENSE.
NB. jllama system utilities
NB. Locale: jllamasys
NB.
NB. Clone-and-run (and later J addon) bootstrap.
NB.   setroot marker  - ROOT from a loaded script path (4!:3)
NB.   jload relpath   - load ROOT , relpath
NB.   ROOT            - tree root with trailing /
NB.   VERSION MILESTONE
NB.
NB. A future manifest.ijs will place this tree under ~addons.

cocurrent 'jllamasys'

VERSION =: '0.15.0'
MILESTONE =: 'M15'

ROOT =: (1!:43 '') , '/'

NB. Directory of a file path, trailing /. Accepts / and \ (Windows 4!:3).
scriptdir =: 3 : 0
  p =. , y
  i =. (p i: '/') >. p i: '\'
  if. i < # p do. (i {. p) , '/' else. (1!:43 '') , '/' end.
)

NB. y = basename (or unique substring) of a script already in 4!:3,
NB. e.g. 'jllama_cli.ijs'. Sets ROOT to that script's directory.
setroot =: 3 : 0
  marker =. y
  'setroot: marker required' assert 0 < # marker
  scripts =. 4!:3 ''
  hit =. scripts #~ +./@(marker&E.)@> scripts
  if. # hit do.
    ROOT =: scriptdir > {: hit
  else.
    ROOT =: (1!:43 '') , '/'
  end.
  ROOT
)

NB. Load a script relative to ROOT.
jload =: 3 : 0
  path =. ROOT , y
  if. -. fexist path do.
    echo 'jllama: missing ' , path
    'missing script' assert 0
  end.
  load path
)

NB. Load a project script relative to ROOT (tests, fixtures, labs).
jrequire =: jload

root =: 3 : 'ROOT'

cocurrent 'base'
