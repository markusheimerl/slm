clean:
	rm -f *.out *.o *.csv
	$(MAKE) -C gpu clean