CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

data.out: data.c
	$(CC) $(CFLAGS) $< -lcurl $(LDFLAGS) -o $@

slm.out: slm.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

generate.out: generate.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

data: data.out
	@time ./data.out

run: slm.out
	@time ./slm.out

cont: slm.out
	@time ./slm.out \
		$(shell ls -t *_layer1.bin | head -1) \
		$(shell ls -t *_layer2.bin | head -1) \
		$(shell ls -t *_layer3.bin | head -1) \
		$(shell ls -t *_layer4.bin | head -1) \
		$(shell ls -t *_layer5.bin | head -1) \
		$(shell ls -t *_layer6.bin | head -1) \
		$(shell ls -t *_embeddings.bin | head -1)

gen: generate.out
	@time ./generate.out \
		$(shell ls -t *_embeddings.bin | head -1) \
		$(shell ls -t *_layer1.bin | head -1) \
		$(shell ls -t *_layer2.bin | head -1) \
		$(shell ls -t *_layer3.bin | head -1) \
		$(shell ls -t *_layer4.bin | head -1) \
		$(shell ls -t *_layer5.bin | head -1) \
		$(shell ls -t *_layer6.bin | head -1) \
		"Once upon a time, " 100 0.8

clean:
	rm -f *.out *.bin