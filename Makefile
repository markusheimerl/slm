CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: slm.o data.o train.o transformer.o attention.o mlp.o
	$(CC) slm.o data.o train.o transformer.o attention.o mlp.o $(CUDALIBS) $(LDFLAGS) -o $@

slm.o: slm.c slm.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c slm.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c data.c -o $@

train.o: train.c slm.h data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

transformer.o: transformer/gpu/transformer.c transformer/gpu/transformer.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c transformer/gpu/transformer.c -o $@

attention.o: transformer/attention/gpu/attention.c transformer/attention/gpu/attention.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c transformer/attention/gpu/attention.c -o $@

mlp.o: transformer/mlp/gpu/mlp.c transformer/mlp/gpu/mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c transformer/mlp/gpu/mlp.c -o $@

run: train.out
	@time ./train.out

cont: train.out
	@time ./train.out $(shell ls -t *_model_embeddings.bin 2>/dev/null | head -1 | sed 's/_embeddings\.bin/.bin/')

clean:
	rm -f *.out *.o *.csv
	$(MAKE) -C transformer clean