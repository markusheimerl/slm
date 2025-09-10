CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt

train.out: slm.o ../transformer/gpu/transformer.o ../transformer/attention/gpu/attention.o ../transformer/mlp/gpu/mlp.o ../data.o train.o
	$(CC) slm.o ../transformer/gpu/transformer.o ../transformer/attention/gpu/attention.o ../transformer/mlp/gpu/mlp.o ../data.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

slm.o: slm.c slm.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c slm.c -o $@

../transformer/gpu/transformer.o:
	$(MAKE) -C ../transformer/gpu transformer.o

../transformer/attention/gpu/attention.o:
	$(MAKE) -C ../transformer/attention/gpu attention.o

../transformer/mlp/gpu/mlp.o:
	$(MAKE) -C ../transformer/mlp/gpu mlp.o

../data.o: ../data.c ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../data.c -o $@

train.o: train.c slm.h ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv
	$(MAKE) -C ../transformer/gpu clean
	$(MAKE) -C ../transformer/attention/gpu clean
	$(MAKE) -C ../transformer/mlp/gpu clean