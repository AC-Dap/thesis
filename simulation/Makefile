# Compilation flags
CXX := g++
CXXFLAGS := -I. -O3 -std=c++20 -MMD -MP

# Files to compile
common_files := common/io/dataset.cc common/io/results.cc common/utils/hashing.cc \
	common/bucket_sketch.cc common/count_sketch.cc common/ppswor.cc

threshold_files := threshold/sketches/bucket_sketch.cc threshold/sketches/fake_swa.cc \
	threshold/sketches/swa.cc threshold/simulations.cc

SRC := $(common_files) $(threshold_files) main.cc

# Output files
OUT_DIR := out
OBJ := $(patsubst %.cc, $(OUT_DIR)/%.o, $(SRC))  # Store object files in out/

TARGET := $(OUT_DIR)/sim

all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJ) | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile object files
$(OUT_DIR)/%.o: %.cc | $(OUT_DIR)
	mkdir -p $(dir $@)  # Ensure the directory exists
	$(CXX) $(CXXFLAGS) -c $< -o $@


# Include dependency files for header tracking
-include $(wildcard $(OBJ:.o=.d))

# Create the output directory
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Clean target
clean:
	rm -f $(OBJ) $(OUT_DIR)/$(TARGET)

.PHONY: run_aol
run_aol: $(TARGET)
	$(TARGET) 30 \
  		../data/AOL-user-ct-collection/user-ct-test-collection-01.txt \
  		../data/AOL-user-ct-collection/user-ct-test-collection-02.txt \
  		aol_results

.PHONY: run_fake
run_fake: $(TARGET) simulate_fake_dataset.py
	python3 simulate_fake_dataset.py $(shape)

.PHONY: run_all
run_all:
	$(MAKE) run_aol
	$(MAKE) run_fake shape=0.1
	$(MAKE) run_fake shape=0.3
	$(MAKE) run_fake shape=0.5
