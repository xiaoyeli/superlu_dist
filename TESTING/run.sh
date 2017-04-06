#!/bin/bash

#mpiexec -n 2 pdtest -r 1 -c 2 -x 4 -m 10 -b 5 -s 1 -f ../EXAMPLE/g20.rua

# Use valgrind 
 mpiexec -n 1 valgrind --leak-check=full --track-origins=yes  \
    pdtest -r 1 -c 1 -x 4 -m 10 -b 5 -s 1 -f ../EXAMPLE/g4.rua

