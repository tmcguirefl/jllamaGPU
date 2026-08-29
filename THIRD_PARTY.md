# Third-party licenses

This repository is the J application (`jllama` GPU rewrite). It is **not**
jsource. Runtime and tensor backend licenses:

## J engine (libj)

The J interpreter this program loads is **J SOURCE** from Jsoftware Inc.

- Copyright 1990-2026, Jsoftware Inc. All rights reserved.
- Dual license: **commercial from Jsoftware** **or** **GNU GPL v3**.
- Full terms in the engine tree: `/Users/tomdevel/jdev/jsource/LICENSE`,
  `license.txt`, and `LICENSE-GPL3`.
- This machine's runtime is the GPL3 build:
  `/Users/tomdevel/j9.8/bin/libj.dylib`.

Distributing `libj.dylib` (or a combined binary that includes it) requires
complying with that dual license. The J **scripts** in this repo are separate
(see `LICENSE` here) and load the engine at runtime.

## ggml

The GPU engine links **ggml** (MIT).

- Copyright (c) 2023-2026 The ggml authors
- License: MIT
- Copy of the notice: `/Users/tomdevel/jdev/jsource/third_party/ggml.LICENSE`
  (same text as llama.cpp's `LICENSE`)

MIT is compatible with GPL-3 and with a Jsoftware commercial license. Keep the
ggml copyright notice with any distribution of ggml object code.
