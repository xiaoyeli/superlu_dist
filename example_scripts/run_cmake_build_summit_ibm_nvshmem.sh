#!/bin/bash
#module load netlib-lapack/3.8.0
#module load gcc/6.4.0
#module swap xl gcc
module load xl
module load cmake/
module load cuda/10.1.168
module load essl

export CRAYPE_LINK_TYPE=dynamic
export PARMETIS_ROOT=/ccs/home/nanding/mysoftware/parmetis403install/
export ACC=GPU
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl

export CUDA_HOME=$OLCF_CUDA_ROOT
export MPI_HOME=$OLCF_SPECTRUM_MPI_ROOT
export SHMEM_HOME=$MPI_HOME
export NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so
export NVSHMEM_LMPI=-lmpi_ibm
export NVSHMEM_HOME=/ccs/home/nanding/mysoftware/EA025Release/nvshmem_0.2.5-0/install
export CUDA_INC=$CUDA_INC:$NVSHMEM_HOME/include

export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

CXX=mpiCC
##-qsmp=omp

cmake .. \
        -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${OLCF_CUDA_ROOT}/include" \
        -DTPL_PARMETIS_LIBRARIES="${PARMETIS_ROOT}/lib/libparmetis.so;${PARMETIS_ROOT}/lib/libmetis.so" \
        -DTPL_CUDA_LIBRARIES="${OLCF_CUDA_ROOT}/lib64/libcublas.so;${OLCF_CUDA_ROOT}/lib64/libcusparse.so;${OLCF_CUDA_ROOT}/lib64/libcudart.so" \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpiCC \
        -DTPL_ENABLE_CUDALIB=ON  \
        -DCMAKE_INSTALL_PREFIX=. \
        -DTPL_BLAS_LIBRARIES="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libessl.so;/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp.so" \
        -DTPL_LAPACK_LIBRARIES="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libessl.so" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
        -DCMAKE_CXX_FLAGS="-qsmp=omp -Ofast -DRELEASE ${INC_VTUNE}" \
        -DCMAKE_C_FLAGS="-qsmp=omp -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDxEBUGlevel=0 -DGPU_ACC -DHAVE_CUDA -fopenmp -I${NVSHMEM_HOME}/include/ " \
        -DCMAKE_CUDA_FLAGS="-ccbin ${CXX} -gencode=arch=compute_70,code=sm_70 -G -Xcompiler -rdynamic -I${NVSHMEM_HOME}/include/ -DPRNTlevel=1 -std=c++11 -DPROFlevel=0 -DEBUGlevel=0 -DGPU_ACC -DHAVE_CUDA --disable-warnings" \
        -DCMAKE_EXE_LINKER_FLAGS="-lcuda -L${CUDA_HOME}/lib64 -lcudart -libverbs -L${MPI_HOME}/lib ${MPI_LIBS} -L${NVSHMEM_HOME}/lib/ -lnvshmem"
make pddrive