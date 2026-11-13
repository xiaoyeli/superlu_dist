#!/bin/bash
# Run the CI test suites (.ci_tests.sh, .ci_tests_gpu.sh, ctest) on Perlmutter,
# from the top-level directory, inside a GPU-node allocation, e.g.
#
#   salloc -N 1 -C gpu -G 4 -q interactive -t 01:00:00 -A <account>
#   bash example_scripts/run_ci_tests_perlmutter.sh cpu    # CPU tests (GPU offload off)
#   bash example_scripts/run_ci_tests_perlmutter.sh gpu    # GPU tests
#   bash example_scripts/run_ci_tests_perlmutter.sh gpu "2d batch"   # a selection
#
# The build directory is ./build (set SLU_CI_BUILD_DIR otherwise); the GPU
# build is the one of run_cmake_build_perlmutter_gcc_nvshmem_cuda12_9.sh
# (CUDA 12.9, NVSHMEM, MAGMA), with the whole "make", so that TEST/p?test_ci
# and the EXAMPLE drivers exist.

suite=${1:-gpu}
selection=${2:-all}

module load PrgEnv-gnu
module load cmake
module load cudatoolkit/12.9
module load nccl/2.24.3
module load nvshmem/3.2.5-1
module load cray-mpich/9.0.1
export SLURM_MPI_TYPE=cray_shasta
export LD_LIBRARY_PATH="$MPICH_DIR/lib:$CRAY_MPICH_ROOTDIR/gtl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
ulimit -s unlimited
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_MAX_THREAD_SAFETY=multiple
export OMP_NUM_THREADS=1

# NVSHMEM settings for Perlmutter (multi-process GPU triangular solves)
NVSHMEM_HOME=${NVSHMEM_ROOT}
export NVSHMEM_USE_GDRCOPY=1
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=${MPICH_DIR}
export NVSHMEM_LIBFABRIC_SUPPORT=1
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
module load libfabric/1.20.1
export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_CXI_OPTIMIZED_MRS=0
export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=libfabric

# Every rank sees the 4 GPUs of the node (needed by the GPU-aware MPI and by
# NVSHMEM); SUPERLU_BIND_MPI_GPU=1 assigns one GPU per rank.
export SUPERLU_BIND_MPI_GPU=1
export SLU_CI_MPIRUN="srun -c 2 --cpu-bind=cores --gpus=4 -n"

if [ "$suite" = cpu ]; then
    # CPU code on the GPU node: no offload, no GPU-aware MPI
    export MPICH_GPU_SUPPORT_ENABLED=0
    export SUPERLU_ACC_OFFLOAD=0
    ./.ci_tests.sh "$selection"
else
    export MPICH_GPU_SUPPORT_ENABLED=1
    export SLU_CI_GPU_AWARE_MPI=1      # cray-mpich is GPU-aware
    ./.ci_tests_gpu.sh "$selection"
fi

# The same tests through ctest, from the build directory:
#   ctest -L cpu          (or: ctest -L gpu, after configuring with
#                          -Denable_gpu_aware_mpi_tests=ON)
