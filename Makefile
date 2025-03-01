CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -lcurl -flto

data.out: data.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

run: data.out
	@time ./data.out | tee -a data.txt

clean:
	rm -f *.out *.txt
