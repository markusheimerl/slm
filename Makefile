CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
CUDAFLAGS = --cuda-gpu-arch=sm_87 -x cuda -Wno-unknown-cuda-version
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -lcurl -flto

train.out: slm.o data.o train.o
	$(CC) slm.o data.o train.o $(LDFLAGS) -o $@

slm.o: slm.c slm.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c slm.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c slm.h data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

cont: train.out
	@time ./train.out $(shell ls -t *_model.bin 2>/dev/null | head -1)

clean:
	rm -f *.out *.o *.csv *.bin