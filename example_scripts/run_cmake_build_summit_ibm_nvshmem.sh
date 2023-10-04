#!/bin/bash
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

NVSHMEM_HOME=/ccs/home/nanding/mysoftware/nvshmem280_gdr23_cuda1103_20230127/
export CUDA_INC=$CUDA_INC:$NVSHMEM_HOME/include
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
CXX=mpiCC

cmake .. \
  -DCMAKE_C_FLAGS="-DGPU_SOLVE -D_USE_SUMMIT -qsmp=omp -DSLU_HAVE_LAPACK -fopenmp -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_ -I${NVSHMEM_HOME}/include" \
  -DCMAKE_CXX_COMPILER=mpiCC \
  -DCMAKE_C_COMPILER=mpicc \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DTPL_ENABLE_LAPACKLIB=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DTPL_ENABLE_CUDALIB=ON \
  -DCMAKE_CUDA_FLAGS="-I${NVSHMEM_HOME}/include -I${MPICH_DIR}/include -ccbin=${CXX}" \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_INSTALL_LIBDIR=./lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_ESSL_ROOT}/lib64/libesslsmp.so" \
  -DTPL_LAPACK_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so" \
  -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${OLCF_CUDA_ROOT}/include" \
  -DTPL_PARMETIS_LIBRARIES="${PARMETIS_ROOT}/lib/libparmetis.so;${PARMETIS_ROOT}/lib/libmetis.so" \
  -DTPL_ENABLE_COMBBLASLIB=OFF \
  -DTPL_ENABLE_NVSHMEM=ON \
  -DTPL_NVSHMEM_LIBRARIES="-lcuda -lcudadevrt -L${OLCF_CUDA_ROOT}/lib64 -lcudart_static -L${OLCF_CUDA_ROOT}/lib64/stubs/ -lnvidia-ml -libverbs -L${NVSHMEM_HOME}/lib/ -lnvshmem -L${OLCF_SPECTRUM_MPI_ROOT}/lib -lmpi_ibm"
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

make pddrive
