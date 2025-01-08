FLAGS := -O3 -std=c++20

OUT_DIR := out
INCLUDE_DIR := include
lib_files := io/dataset.cc io/results.cpp sketches/count_sketch.cc sketches/fake_swa.cc sketches/ppswor.cc sketches/swa.cc sketches/bucket_sketch.cc utils/hashing.cc
files := $(lib_files) main.cc

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/sim: $(files) | $(OUT_DIR)
	g++ $(FLAGS) $(files) -I $(INCLUDE_DIR) -o $(OUT_DIR)/sim

.PHONY: run_aol
run_aol: $(OUT_DIR)/sim
	$(OUT_DIR)/sim 30 \
  		../data/AOL-user-ct-collection/user-ct-test-collection-01.txt \
  		../data/AOL-user-ct-collection/user-ct-test-collection-02.txt \
  		aol_results

.PHONY: run_fake
run_fake: $(OUT_DIR)/sim simulate_fake_dataset.py
	python3 simulate_fake_dataset.py $(shape)

.PHONY: run_all
run_all:
	make run_aol
	make run_fake shape=0.1
	make run_fake shape=0.3
	make run_fake shape=0.5
