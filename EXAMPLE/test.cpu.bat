#PBS -q debug
#PBS -l nodes=6:ppn=1
#PBS -l walltime=00:09:00
#PBS -j oe
#PBS -o test-cpu.out
#PBS -m e

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

# module load cuda
module load mkl

## you can try changing the following parameters

## setenv NSUP 144
# setenv NSUP 10
## setenv NREL 32    ## 144
# setenv NREL 4    ## 144
setenv OMP_NUM_THREADS 8
setenv MAX_BUFFER_SIZE 64000000

mpirun -np 6 pddrive -r 3 -c 2 $SCRATCH/torso3.rua
#mpirun -np 4 pddrive -r 2 -c 2 big.rua
#mpirun -np 6 pzdrive -r 3 -c 2 cg20.cua
#mpirun -np 1 pzdrive -c 1 g4.cua


#-- For Intel Phi
# setenv THREAD_PER_PROCESS 4

#-- For CUDA
# setenv ACC GPU
# setenv CUBLAS_NB 128
setenv N_GEMM 0

