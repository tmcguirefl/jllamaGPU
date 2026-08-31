# Third-party licenses

This repository is **jllamaGPU**: a J application that depends on a **fork of
the J runtime** (`libj` from Jsoftware jsource, GPU/ggml backend). Because of
that dependency, **this project is GPL-3.0-only** (`LICENSE`, `COPYRIGHT`).

## J engine (libj)

The interpreter this program loads is **J SOURCE** from Jsoftware Inc.

- Copyright 1990-2026, Jsoftware Inc. All rights reserved.
- Dual license: commercial from Jsoftware **or** GNU GPL v3.
- This project uses the engine **under GPL v3** (the GPL3 build of the
  gpu-resident fork).
- Engine tree: `C:\Users\tmcguire\jdev\jsource` (`LICENSE`, `license.txt`,
  `LICENSE-GPL3`).
- This machine: `C:\Users\tmcguire\j9.8\bin\j.dll`.

Distributing `j.dll` together with this program is a GPL-3 source
distribution of both the application and the engine (plus ggml, below).

## ggml

The GPU engine links **ggml**.

- Copyright (c) 2023-2026 The ggml authors
- License: MIT (compatible with GPL-3)
- Notice: `C:\Users\tmcguire\jdev\jsource\third_party\ggml.LICENSE`
  (same text as llama.cpp `LICENSE`)

Keep the MIT copyright notice with any distribution of ggml object code.
