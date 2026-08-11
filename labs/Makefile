# Build jllama oracles against Homebrew llama.cpp + ggml
ROOT := $(abspath ..)
TOOLS := $(ROOT)/tools

LLAMA_PREFIX ?= $(shell brew --prefix llama.cpp 2>/dev/null)
GGML_PREFIX  ?= $(shell brew --prefix ggml 2>/dev/null)

CFLAGS  ?= -O2
CPPFLAGS += -I$(LLAMA_PREFIX)/include -I$(GGML_PREFIX)/include
LDFLAGS  += -L$(LLAMA_PREFIX)/lib -L$(GGML_PREFIX)/lib
LDLIBS   += -lllama -lggml -lggml-base
RPATH    += -Wl,-rpath,$(LLAMA_PREFIX)/lib -Wl,-rpath,$(GGML_PREFIX)/lib

.PHONY: all clean oracle_greedy oracle_tokenize

all: oracle_greedy oracle_tokenize
oracle_greedy: $(TOOLS)/oracle_greedy
oracle_tokenize: $(TOOLS)/oracle_tokenize

$(TOOLS)/oracle_greedy: oracle_greedy.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) $(RPATH)

$(TOOLS)/oracle_tokenize: oracle_tokenize.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) $(RPATH)

clean:
	rm -f $(TOOLS)/oracle_greedy $(TOOLS)/oracle_tokenize
