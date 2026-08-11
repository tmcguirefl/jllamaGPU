/* jllama M6 oracle: greedy generate token ids via libllama.
 *
 * Usage:
 *   oracle_greedy MODEL.gguf "prompt text" N_NEW [--ids ID ID ...]
 *
 * Prints one line:
 *   PROMPT_IDS ... | GEN_IDS ... | FULL_IDS ...
 *
 * If --ids is given, those token ids are used as the prompt (no tokenize).
 * Temperature is greedy argmax on logits (no sampling chain randomness).
 */
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void die(const char *msg) {
    fprintf(stderr, "oracle_greedy: %s\n", msg);
    exit(1);
}

static llama_token argmax_token(const float *logits, int n_vocab) {
    int best = 0;
    float bestv = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > bestv) {
            bestv = logits[i];
            best = i;
        }
    }
    return (llama_token)best;
}

static void print_ids(const llama_token *ids, int n) {
    for (int i = 0; i < n; ++i) {
        if (i) putchar(' ');
        printf("%d", (int)ids[i]);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf PROMPT N_NEW [--ids ID...]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *prompt = argv[2];
    int n_new = atoi(argv[3]);
    if (n_new < 0) die("N_NEW must be >= 0");

    int use_ids = 0;
    int id_start = -1;
    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--ids") == 0) {
            use_ids = 1;
            id_start = i + 1;
            break;
        }
    }

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; /* CPU for deterministic parity */
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) die("failed to load model");

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 256;
    cparams.n_ubatch = 256;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    /* force f32 KV if available via cache type fields - leave defaults */
    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) die("failed to create context");

    llama_token prompt_ids[512];
    int n_prompt = 0;

    if (use_ids) {
        for (int i = id_start; i < argc; ++i) {
            if (n_prompt >= 512) die("prompt too long");
            prompt_ids[n_prompt++] = (llama_token)atoi(argv[i]);
        }
        if (n_prompt == 0) die("empty --ids");
    } else {
        int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), prompt_ids, 512, true, true);
        if (n < 0) {
            /* retry with larger or without specials */
            n = llama_tokenize(vocab, prompt, (int)strlen(prompt), prompt_ids, 512, false, true);
        }
        if (n < 0) die("tokenize failed");
        n_prompt = n;
        if (n_prompt == 0) die("empty prompt tokens");
    }

    /* decode prompt */
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        batch.token[i] = prompt_ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_prompt - 1);
    }
    batch.n_tokens = n_prompt;
    if (llama_decode(ctx, batch) != 0) die("decode prompt failed");

    llama_token full[1024];
    int n_full = n_prompt;
    memcpy(full, prompt_ids, sizeof(llama_token) * n_prompt);

    llama_token gen[512];
    int n_gen = 0;

    for (int t = 0; t < n_new; ++t) {
        float *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) {
            /* some builds put last logits at index -1 / last */
            logits = llama_get_logits_ith(ctx, -1);
        }
        if (!logits) die("null logits");
        llama_token nxt = argmax_token(logits, n_vocab);
        gen[n_gen++] = nxt;
        if (n_full >= 1024) die("full too long");
        full[n_full++] = nxt;

        /* next step: single token */
        batch.n_tokens = 1;
        batch.token[0] = nxt;
        batch.pos[0] = n_prompt + t;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) die("decode step failed");
    }

    printf("PROMPT ");
    print_ids(prompt_ids, n_prompt);
    printf(" | GEN ");
    print_ids(gen, n_gen);
    printf(" | FULL ");
    print_ids(full, n_full);
    printf("\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
