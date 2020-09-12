INCPATH = ../includes/
LDFLAGS = -L/usr/local/lib
LDLIBS = -lcnpy -lz -lnlopt -lm
VPATH = ../src

CXX = g++
CC = $(CXX)

CXXFLAGS = -O3 --std=c++17 -I$(HOME)/.pyenv/versions/miniconda3-latest/envs/random-walkers/include -I$(INCPATH)

data_generation: data_generation.o

rejection_sampling: rejection_sampling.o

vectortest: vectortest.o

nlopt_example: nlopt_example.o

nlopt_tophat_example: nlopt_tophat_example.o

test_pdf: test_pdf.o

clean:
	rm -f *.o
