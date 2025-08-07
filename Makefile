CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -lcurl -flto

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: slm.o data.o train.o ssm/gpu/ssm.o mlp/gpu/mlp.o
	$(CC) slm.o data.o train.o ssm/gpu/ssm.o mlp/gpu/mlp.o $(CUDALIBS) $(LDFLAGS) -o $@

slm.o: slm.c slm.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c slm.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c data.c -o $@

train.o: train.c slm.h data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

ssm/gpu/ssm.o:
	$(MAKE) -C ssm/gpu ssm.o ARCH=$(ARCH)

mlp/gpu/mlp.o:
	$(MAKE) -C mlp/gpu mlp.o ARCH=$(ARCH)

ssm/gpu/data.o:
	$(MAKE) -C ssm/gpu data.o ARCH=$(ARCH)

mlp/gpu/data.o:
	$(MAKE) -C mlp/gpu data.o ARCH=$(ARCH)

run: train.out
	@time ./train.out

cont: train.out
	@time ./train.out $(shell ls -t *_model_embeddings.bin 2>/dev/null | head -1 | sed 's/_embeddings\.bin/.bin/')

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C ssm clean
	$(MAKE) -C mlp clean