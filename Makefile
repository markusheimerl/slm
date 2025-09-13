CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

train.out: slm.o transformer/transformer.o transformer/attention/attention.o transformer/mlp/mlp.o data.o train.o
	$(CC) slm.o transformer/transformer.o transformer/attention/attention.o transformer/mlp/mlp.o data.o train.o $(LDFLAGS) -o $@

slm.o: slm.c slm.h
	$(CC) $(CFLAGS) -c slm.c -o $@

transformer/transformer.o:
	$(MAKE) -C transformer transformer.o

transformer/attention/attention.o:
	$(MAKE) -C transformer/attention attention.o

transformer/mlp/mlp.o:
	$(MAKE) -C transformer/mlp mlp.o

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c slm.h data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

cont: train.out
	@time ./train.out $$(ls -t *_slm.bin 2>/dev/null | head -n1)

clean:
	rm -f *.out *.o *.csv
	$(MAKE) -C gpu clean
	$(MAKE) -C transformer clean
	$(MAKE) -C transformer/attention clean
	$(MAKE) -C transformer/mlp clean