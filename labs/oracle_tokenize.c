/* Print token ids for a prompt via libllama */
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf PROMPT\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load fail\n"); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const char *prompt = argv[2];
    int n_max = 512;
    llama_token *toks = calloc(n_max, sizeof(llama_token));
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks, n_max, true, true);
    if (n < 0) { fprintf(stderr, "tokenize fail %d\n", n); return 1; }
    for (int i = 0; i < n; ++i) {
        if (i) putchar(' ');
        printf("%d", (int)toks[i]);
    }
    putchar('\n');
    free(toks);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
