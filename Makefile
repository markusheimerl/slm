CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -lcurl -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 -x cuda -Wno-unknown-cuda-version

CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

data.out: data.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

slm.out: slm.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

generate.out: generate.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

data: data.out
	@time ./data.out

run: slm.out
	@time ./slm.out

gen: generate.out
	@time ./generate.out \
		$(shell ls -t *_encoder.bin | head -1) \
		$(shell ls -t *_reasoning.bin | head -1) \
		$(shell ls -t *_output.bin | head -1) \
		$(shell ls -t *_embeddings.bin | head -1)

clean:
	rm -f *.out *.bin
