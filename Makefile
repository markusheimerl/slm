CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -lcurl -flto
CUDAFLAGS = --cuda-gpu-arch=sm_87 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

slm.out: slm.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $^ $(CUDALIBS) $(LDFLAGS) -o $@

run: slm.out
	@time ./slm.out

cont: slm.out
	@time ./slm.out $(shell ls -t *_model.bin 2>/dev/null | head -1)

clean:
	rm -f *.out *.csv *.bin
