#!/bin/bash
#
# Frank / saturn compute node (NVIDIA A100, CUDA).
# Copied from batch_script_mpi_runit_frank_instinct_3dsolve.sh (AMD/ROCm) and
# adapted for NVIDIA: load cuda instead of rocm, point LD_LIBRARY_PATH at the
# CUDA-built MAGMA / OpenBLAS / parmetis, and drop the instinct-only
# OPENBLAS_CORETYPE override (saturn's Intel Cooper Lake CPU has AVX-512).
#
#modules:
module load cmake
module load openmpi/4.1.8_gcc13.2.0
module load cuda/12.6
SAT=/home/users/tshi777/ECP_Tutorial_GPU_Capable_Sparse_Direct_Solvers/saturn/manual_install
export LD_LIBRARY_PATH=$SAT/magma-install/lib:$SAT/OpenBLAS/lib:$SAT/parmetis-4.0.3/install/lib:${CUDAROOT:-/packages/cuda/12.6.3}/lib64:$(mpicc --showme:libdirs 2>/dev/null | tr ' \n' '::' | sed 's/:*$//'):$LD_LIBRARY_PATH
ulimit -s unlimited
# NOTE: no OPENBLAS_CORETYPE override here.  On instinct the OpenBLAS is an
# AVX-512-only (skylakex) build running on an AVX2 EPYC CPU, so it forces ZEN.
# Saturn's OpenBLAS is a Cooper Lake (AVX-512) build and saturn's CPU
# (Xeon Platinum 8367HC) supports AVX-512, so no override is needed.
#MPI settings:
export MPICH_GPU_SUPPORT_ENABLED=1
echo MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED
#SUPERLU settings:


export SUPERLU_LBS=GD
export SUPERLU_ACC_OFFLOAD=1 # this can be 0 to do CPU tests on GPU nodes
export GPU3DVERSION=1
export ANC25D=0
export NEW3DSOLVE=1
export NEW3DSOLVETREECOMM=1
export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU
export SUPERLU_ACC_SOLVE=1  # 0=CPU trisolve (stable); set to 1 to try GPU trisolve (requires SLU_HAVE_LAPACK, now enabled)

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=10000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
export SUPERLU_NUM_GPU_STREAMS=1
export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
nmpipergpu=1
export SUPERLU_MPI_PROCESS_PER_GPU=$nmpipergpu # 2: this can better saturate GPU

export OMP_NUM_THREADS=1
# mpirun -np 1 ./EXAMPLE/pddrive3d /home/users/tshi777/ECP_Tutorial_GPU_Capable_Sparse_Direct_Solvers/instinct/manual_install/matrix/EMTrun1/EMTrun1_total.dat
mpirun -np 1 ./EXAMPLE/pddrive3d_vbatch -b 2 -f .dat /home/users/tshi777/ECP_Tutorial_GPU_Capable_Sparse_Direct_Solvers/instinct/manual_install/matrix/EMTrun1/EMTrun1_
