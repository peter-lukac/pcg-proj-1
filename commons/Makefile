# Paralelní programování na GPU (PCG 2020)
# Projekt c. 1 (cuda)
# Login: xlukac11

LIBS=-lhdf5

.PHONY: all clean

all: gen

gen: gen.cpp h5Helper.cpp
	g++ gen.cpp h5Helper.cpp ${LIBS} -o gen

clean:
	rm -f *.o nbody
