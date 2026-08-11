# Build jllama M6 oracle against Homebrew llama.cpp + ggml
ROOT := $(abspath ..)
BIN  := $(ROOT)/tools/oracle_greedy
SRC  := oracle_greedy.c

LLAMA_PREFIX ?= $(shell brew --prefix llama.cpp 2>/dev/null)
GGML_PREFIX  ?= $(shell brew --prefix ggml 2>/dev/null)

CFLAGS  ?= -O2
CPPFLAGS += -I$(LLAMA_PREFIX)/include -I$(GGML_PREFIX)/include
LDFLAGS  += -L$(LLAMA_PREFIX)/lib -L$(GGML_PREFIX)/lib
LDLIBS   += -lllama -lggml -lggml-base
RPATH    += -Wl,-rpath,$(LLAMA_PREFIX)/lib -Wl,-rpath,$(GGML_PREFIX)/lib

.PHONY: all clean oracle_greedy

all: $(BIN)
oracle_greedy: $(BIN)

$(BIN): $(SRC)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) $(RPATH)

clean:
	rm -f $(BIN)
