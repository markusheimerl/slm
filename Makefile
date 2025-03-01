CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -lcurl -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 \
    -x cuda \
    -fcuda-flush-denormals-to-zero \
    -fcuda-approx-transcendentals \
    -Wno-unknown-cuda-version

CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

data.out: data.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

slm.out: slm.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

run: data.out
	@time ./data.out | tee -a data.txt

train: slm.out
	@time ./slm.out

clean:
	rm -f *.out *.txt *.bin
