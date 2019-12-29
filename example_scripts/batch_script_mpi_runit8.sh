#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue


EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2

export MPICH_MAX_THREAD_SAFETY=multiple

for i in `seq 1 200`;
do
	srun -n 2 ./EXAMPLE/pddrive -r 2 -c 1 ../EXAMPLE/big.rua | tee -a a.out
done


exit $EXIT_SUCCESS

