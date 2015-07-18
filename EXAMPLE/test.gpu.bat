#PBS -q dirac_reg
# #PBS -l nodes=6:ppn=1:mfermi
#PBS -l nodes=6:ppn=1
# #PBS -l nodes=2:ppn=1:mtesla
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o test-gpu.out
#PBS -m e

#module unload pgi
#module unload openmpi
#module unload cuda
#module load gcc-sl6
#module load openmpi-gcc-sl6
#module load cuda
 
limit vmemoryuse unlimited 
limit memorylocked unlimited

module load  cuda mkl

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

## you can try changing the following parameters

setenv ACC GPU
setenv N_GEMM 0
setenv CUBLAS_NB 128
setenv MAX_BUFFER_SIZE 64000000

setenv OMP_NUM_THREADS  8
# setenv NSUP 2   ## 144
# setenv NREL 2   ## 32

mpirun -np 6 -bysocket pddrive1_ABglobal -r 3 -c 2 g20.rua
#mpirun -np 6 -bysocket pddrive -r 3 -c 2 $SCRATCH/torso3.rua
# mpirun -np 2 -bysocket ./pddrive -r 2 -c 1 g20.rua
#mpirun -np 2 -bysocket pzdrive -r 2 cg20.cua

#-- For Intel Phi
# setenv THREAD_PER_PROCESS 4
