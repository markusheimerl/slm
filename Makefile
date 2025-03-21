CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

data.out: data.c
	$(CC) $(CFLAGS) $< -lcurl $(LDFLAGS) -o $@

slm.out: slm.c
	$(CC) $(CFLAGS) $< -lopenblas -lcurl $(LDFLAGS) -o $@

data: data.out
	@time ./data.out

run: slm.out
	@time ./slm.out

clean:
	rm -f *.out *.bin
