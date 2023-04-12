#!/bin/bash

wget https://developer.download.nvidia.com/compute/redist/nvshmem/2.8.0/source/nvshmem_src_2.8.0-3.txz
tar -xvf nvshmem_src_2.8.0-3.txz
cd nvshmem_src_2.8.0-3

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
##export FI_LOG_LEVEL Warn
 
export NVSHMEM_HOME=${PWD}/build 
export NVSHMEM_USE_GDRCOPY=1
export GDRCOPY_HOME=/usr
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=${MPICH_DIR}
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
export NVSHMEM_DEFAULT_PMI2=1
export NVCUFLAGS="--allow-unsupported-compiler"
export MPICC=CC
export CC=cc
export CXX=CC
 
export NVSHMEM_LIBFABRIC_SUPPORT=1
export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.2.0
#export LD_LIBRARY_PATH /global/cfs/cdirs/m2956/nanding/software/MPI_Bootstrap_For_Nan/:$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
export NVSHMEM_BOOTSTRAP=MPI
#export NVSHMEM_BOOTSTRAP plugin
#export NVSHMEM_BOOTSTRAP_PLUGIN /global/cfs/cdirs/m2956/nanding/software/MPI_Bootstrap_For_Nan/nvshmem_bootstrap_mpich.so

export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_CXI_OPTIMIZED_MRS=false
export NVSHMEM_REMOTE_TRANSPORT=libfabric

export MPICH_GPU_SUPPORT_ENABLED=0
export CRAY_ACCEL_TARGET=nvidia80
export MPI_HOME=${MPICH_DIR}
export NVSHMEM_HOME=/global/cfs/cdirs/m2957/liuyangz/my_software/nvshmem_perlmutter/nvshmem_src_2.8.0-3/build
export NVSHMEM_LIBFABRIC_SUPPORT=1
export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.2.0

export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_CXI_OPTIMIZED_MRS=false

export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_DEBUG=0

export CUDA_HOME=${CUDA_HOME}
#cd ./examples
make -j16
