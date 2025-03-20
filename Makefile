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
		$(shell ls -t *_layer7.bin | head -1) \
		$(shell ls -t *_layer8.bin | head -1) \
		$(shell ls -t *_layer9.bin | head -1) \
		$(shell ls -t *_layer10.bin | head -1) \
		$(shell ls -t *_layer11.bin | head -1) \
		$(shell ls -t *_layer12.bin | head -1) \
		$(shell ls -t *_layer13.bin | head -1) \
		$(shell ls -t *_layer14.bin | head -1) \
		$(shell ls -t *_layer15.bin | head -1) \
		$(shell ls -t *_layer16.bin | head -1) \
		$(shell ls -t *_layer17.bin | head -1) \
		$(shell ls -t *_layer18.bin | head -1) \
		$(shell ls -t *_layer19.bin | head -1) \
		$(shell ls -t *_layer20.bin | head -1) \
		$(shell ls -t *_layer21.bin | head -1) \
		$(shell ls -t *_layer22.bin | head -1) \
		$(shell ls -t *_layer23.bin | head -1) \
		$(shell ls -t *_layer24.bin | head -1) \
		$(shell ls -t *_embeddings.bin | head -1)

gen: generate.out
	@time ./generate.out \
		$(shell ls -t *_layer1.bin | head -1) \
		$(shell ls -t *_layer2.bin | head -1) \
		$(shell ls -t *_layer3.bin | head -1) \
		$(shell ls -t *_layer4.bin | head -1) \
		$(shell ls -t *_layer5.bin | head -1) \
		$(shell ls -t *_layer6.bin | head -1) \
		$(shell ls -t *_layer7.bin | head -1) \
		$(shell ls -t *_layer8.bin | head -1) \
		$(shell ls -t *_layer9.bin | head -1) \
		$(shell ls -t *_layer10.bin | head -1) \
		$(shell ls -t *_layer11.bin | head -1) \
		$(shell ls -t *_layer12.bin | head -1) \
		$(shell ls -t *_layer13.bin | head -1) \
		$(shell ls -t *_layer14.bin | head -1) \
		$(shell ls -t *_layer15.bin | head -1) \
		$(shell ls -t *_layer16.bin | head -1) \
		$(shell ls -t *_layer17.bin | head -1) \
		$(shell ls -t *_layer18.bin | head -1) \
		$(shell ls -t *_layer19.bin | head -1) \
		$(shell ls -t *_layer20.bin | head -1) \
		$(shell ls -t *_layer21.bin | head -1) \
		$(shell ls -t *_layer22.bin | head -1) \
		$(shell ls -t *_layer23.bin | head -1) \
		$(shell ls -t *_layer24.bin | head -1) \
		$(shell ls -t *_embeddings.bin | head -1)

clean:
	rm -f *.out *.bin