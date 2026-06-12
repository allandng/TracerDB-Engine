# TracerDB — header-only library; this builds the tools and tests.
CXX      ?= c++
CXXFLAGS ?= -O3 -ffast-math -std=c++17 -Wall -Wextra
CPPFLAGS += -Iinclude -Itools

BUILD := build
TOOLS := tracer_gen tracer_build tracer_query tracer_bench
TESTS := test_pager test_cache test_policy test_index

TOOL_BINS := $(addprefix $(BUILD)/,$(TOOLS))
TEST_BINS := $(addprefix $(BUILD)/,$(TESTS))

HDRS := $(wildcard include/tracerdb/*.h) tools/cli.h

all: $(TOOL_BINS) $(TEST_BINS)

tools: $(TOOL_BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%: tools/%.cpp $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BUILD)/%: tests/%.cpp $(HDRS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; $$t || exit 1; done
	@echo "all tests passed"

clean:
	rm -rf $(BUILD)

.PHONY: all tools test clean
