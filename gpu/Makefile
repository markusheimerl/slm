CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra $(shell pkg-config --cflags glib-2.0)
LDFLAGS = -lopenblas -lm -flto -lstdc++

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublasLt
BPELIBS = -lpcre2-8 -fopenmp $(shell pkg-config --libs glib-2.0)

train.out: gpt.o ../transformer/gpu/transformer.o ../transformer/attention/gpu/attention.o ../transformer/mlp/gpu/mlp.o ../data.o ../bpe/bpe.o train.o
	$(CC) gpt.o ../transformer/gpu/transformer.o ../transformer/attention/gpu/attention.o ../transformer/mlp/gpu/mlp.o ../data.o ../bpe/bpe.o train.o $(CUDALIBS) $(BPELIBS) $(LDFLAGS) -o $@

gpt.o: gpt.c gpt.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c gpt.c -o $@

../transformer/gpu/transformer.o:
	$(MAKE) -C ../transformer/gpu transformer.o

../transformer/attention/gpu/attention.o:
	$(MAKE) -C ../transformer/attention/gpu attention.o

../transformer/mlp/gpu/mlp.o:
	$(MAKE) -C ../transformer/mlp/gpu mlp.o

../data.o: ../data.c ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../data.c -o $@

../bpe/bpe.o: ../bpe/bpe.c ../bpe/bpe.h
	$(MAKE) -C ../bpe bpe.o

train.o: train.c gpt.h ../data.h ../bpe/bpe.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

cont: train.out
	@time ./train.out $$(ls -t *_gpt.bin 2>/dev/null | head -n1)

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C ../transformer/gpu clean
	$(MAKE) -C ../transformer/attention/gpu clean
	$(MAKE) -C ../transformer/mlp/gpu clean