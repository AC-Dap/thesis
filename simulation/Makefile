FLAGS := -O3 -std=c++20

OUT_DIR := out
INCLUDE_DIR := include
lib_files := io/dataset.cc io/results.cpp sketches/count_sketch.cc sketches/fake_swa.cc sketches/ppswor.cc sketches/swa.cc sketches/bucket_sketch.cc utils/hashing.cc
files := $(lib_files) main.cc

.PHONY: all
all: sim sim_win64

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

.PHONY: sim
sim: $(files) | $(OUT_DIR)
	g++ $(FLAGS) $(files) -I $(INCLUDE_DIR) -o $(OUT_DIR)/sim

.PHONY: run
run: sim
	$(OUT_DIR)/sim