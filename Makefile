clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C transformer clean