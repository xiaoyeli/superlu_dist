#!/bin/bash
#
# Run SuperLU_dist examples built with GNU compiler on NERSC Perlmutter
#
# Last update: July 22, 2022
# Perlmutter is not in production and the software environment changes rapidly.
# Expect this file to be frequently updated

module load PrgEnv-gnu
module load gcc/11.2.0
module load cmake/3.22.0
module load cudatoolkit/11.7
# avoid bug in cray-libsci/21.08.1.2
module load cray-libsci/22.06.1.3
# avoid bug in cudatoolkit
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.5\/compat:/}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.7\/compat:/}
# please make sure the above module load and export LD_LIBRARY_PATH commands match your build script



export MAX_BUFFER_SIZE=50000000
export OMP_NUM_THREADS=1
export NUM_GPU_STREAMS=1
# export SUPERLU_BIND_MPI_GPU=1
# srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/g20.rua

# export NSUP=5
# export NREL=5
for MAT in big.rua 
# for MAT in g4.rua 
# for MAT in s1_mat_0_126936.bin 
# for MAT in s1_mat_0_126936.bin s1_mat_0_253872.bin s1_mat_0_507744.bin 
# for MAT in matrix_ACTIVSg70k_AC_00.mtx matrix_ACTIVSg10k_AC_00.mtx
# for MAT in temp_13k.mtx temp_25k.mtx temp_75k.mtx
do
srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 $CFS/ntrain9/YangLiu/matrix/$MAT
done 

