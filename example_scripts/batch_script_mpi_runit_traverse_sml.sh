#!/bin/sh

#SBATCH --qos=test
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J superlu_test
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH --gpus=1


module purge
export ALLINEA_FORCE_CUDA_VERSION=20.0.1
module load cudatoolkit/11.2 pgi/20.4 openmpi/pgi-20.4/4.0.4/64
module load hdf5/pgi-20.4/openmpi-4.0.4/1.10.6 fftw/gcc/openmpi-4.0.4/3.3.8 anaconda ddt
module load cmake

#srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../EXAMPLE/big.rua
srun -n 1 ./EXAMPLE/pddrive -r 1 -c 1 ../../matrix/HTS/copter2.mtx
