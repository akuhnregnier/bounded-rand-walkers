LDLIBS=-Wall -L/usr/local/lib -lcnpy -lz -lnlopt -lm
# DO NOT USE -pg -O0 for final code!!!
# CFLAGS=-g -O0 -I /home/alex/.pyenv/versions/miniconda2-latest/include -I /home/alex/Documents/roboptim-core/include -I /usr/include/eigen3
# Release version
CFLAGS=-O3 -I ~/.pyenv/versions/miniconda2-latest/include -I ~/Documents/roboptim-core/include -I /usr/include/eigen3

data_generation: data_generation.o
	g++ $(CFLAGS) -o data_generation data_generation.o $(LDLIBS)
	# ./data_generation

data_generation.o: data_generation.cpp
	g++ $(CFLAGS) -c data_generation.cpp

rejection_sampling: rejection_sampling.o
	g++ $(CFLAGS) -o rejection_sampling rejection_sampling.o $(LDLIBS)
	./rejection_sampling

rejection_sampling.o: rejection_sampling.cpp rejection_sampling.h
	g++ $(CFLAGS) -c rejection_sampling.cpp

vtest: vectortest.o
	g++ $(CFLAGS) -o vtest vectortest.o $(LDLIBS)
	./vtest

vectortest.o: vectortest.cpp
	g++ $(CFLAGS) -c vectortest.cpp

roboptim_example: roboptim_example.o
	g++ $(CFLAGS) -o roboptim_example roboptim_example.o $(LDLIBS)
	./roboptim_example

roboptim_example.o: roboptim_example.cpp
	g++ $(CFLAGS) -c roboptim_example.cpp

nlopt_example: nlopt_example.o
	g++ $(CFLAGS) -o nlopt_example nlopt_example.o $(LDLIBS)
	./nlopt_example

nlopt_example.o: nlopt_example.cpp
	g++ $(CFLAGS) -c nlopt_example.cpp

nlopt_tophat_example: nlopt_tophat_example.o
	g++ $(CFLAGS) -o nlopt_tophat_example nlopt_tophat_example.o $(LDLIBS)
	./nlopt_tophat_example

nlopt_tophat_example.o: nlopt_tophat_example.cpp
	g++ $(CFLAGS) -c nlopt_tophat_example.cpp

method_calling: method_calling.cpp
	g++ $(CFLAGS) method_calling.cpp -o method_calling $(LDLIBS)
	./method_calling

clean:
	rm -f vtest vectortest.o rejection_sampling rejection_sampling.o roboptim_example.o robotim_example nlopt_example.o nlopt_example nlopt_tophat_example.o nlopt_tophat_example method_calling data_generation.o data_generation
