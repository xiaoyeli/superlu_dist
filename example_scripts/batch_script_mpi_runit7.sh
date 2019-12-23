#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue

EXIT_SUCCESS=0
EXIT_HOST=1
EXIT_PARAM=2


for i in `seq 1 200`;
do
	srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ./EXAMPLE/torso3.mtx | tee -a a.out
done


exit $EXIT_SUCCESS

