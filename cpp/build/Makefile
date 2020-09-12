LDLIBS=-Wall -L/usr/local/lib -lcnpy -lz -lnlopt -lm
CFLAGS=-O3 -I ~/.pyenv/versions/miniconda3-latest/envs/random-walkers/include -I ~/Documents/roboptim-core/include -I /usr/include/eigen3 -I ../includes

data_generation: data_generation.o
	g++ $(CFLAGS) -o data_generation data_generation.o $(LDLIBS)
	./data_generation

data_generation.o: ../src/data_generation.cpp
	g++ $(CFLAGS) -c ../src/data_generation.cpp

rejection_sampling: rejection_sampling.o
	g++ $(CFLAGS) -o rejection_sampling rejection_sampling.o $(LDLIBS)
	./rejection_sampling

rejection_sampling.o: ../src/rejection_sampling.cpp ../includes/rejection_sampling.h
	g++ $(CFLAGS) -c rejection_sampling.cpp

vtest: vectortest.o
	g++ $(CFLAGS) -o vtest vectortest.o $(LDLIBS)
	./vtest

vectortest.o: ../src/vectortest.cpp
	g++ $(CFLAGS) -c ../src/vectortest.cpp

nlopt_example: nlopt_example.o
	g++ $(CFLAGS) -o nlopt_example nlopt_example.o $(LDLIBS)
	./nlopt_example

nlopt_example.o: ../src/nlopt_example.cpp
	g++ $(CFLAGS) -c ../src/nlopt_example.cpp

nlopt_tophat_example: nlopt_tophat_example.o
	g++ $(CFLAGS) -o nlopt_tophat_example nlopt_tophat_example.o $(LDLIBS)
	./nlopt_tophat_example

nlopt_tophat_example.o: ../src/nlopt_tophat_example.cpp
	g++ $(CFLAGS) -c ../src/nlopt_tophat_example.cpp

method_calling: ../src/method_calling.cpp
	g++ $(CFLAGS) ../src/method_calling.cpp -o method_calling $(LDLIBS)
	./method_calling

test_pdf: ../src/test_pdf.cpp
	g++ $(CFLAGS) ../src/test_pdf.cpp -o test_pdf $(LDLIBS)
	./test_pdf

clean:
	rm -f vtest vectortest.o rejection_sampling rejection_sampling.o roboptim_example.o robotim_example nlopt_example.o nlopt_example nlopt_tophat_example.o nlopt_tophat_example method_calling data_generation.o data_generation test_pdf
