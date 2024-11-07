FLAGS := -O3 -std=c++20
lib_files := count_sketch.cc dataset.cc fake_swa.cc hashing.cc ppswor.cc swa.cc
OUT_DIR := out

.PHONY: all
all: sim sim_win64

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

.PHONY: sim
sim: $(lib_files) | $(OUT_DIR)
	g++ $(FLAGS) $(lib_files) main.cc -o $(OUT_DIR)/sim

.PHONY: sim_win64
sim_win64: $(lib_files) | $(OUT_DIR)
	x86_64-w64-mingw32-g++ $(FLAGS) -static-libgcc -static-libstdc++ -static $(lib_files) main.cc -o $(OUT_DIR)/sim_win64.exe
