#!/bin/bash
module load xl
module load cmake
module load cuda
module load essl 

export CRAYPE_LINK_TYPE=dynamic
export PARMETIS_ROOT=/ccs/home/nanding/mysoftware/parmetis403install_upgrade/
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
export NVSHMEM_HOME=/ccs/home/nanding/mysoftware/nvshmem270_gdr23_cuda1102_11232022
#export NVSHMEM_HOME=/ccs/home/nanding/mysoftware/nvshmem203_gdr_cuda1103/
export CUDA_INC=$CUDA_INC:$NVSHMEM_HOME/include

export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
CXX=mpiCC
cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${OLCF_CUDA_ROOT}/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_ROOT}/lib/libparmetis.so;${PARMETIS_ROOT}/lib/libmetis.so" \
	-DTPL_CUDA_LIBRARIES="${OLCF_CUDA_ROOT}/lib64/libcublas.so;${OLCF_CUDA_ROOT}/lib64/libcusparse.so;${OLCF_CUDA_ROOT}/lib64/libcudart.so" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_C_COMPILER=mpicc \
	-DCMAKE_CXX_COMPILER=mpiCC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_ESSL_ROOT}/lib64/libesslsmp.so" \
	-DTPL_LAPACK_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so" \
	-DCMAKE_BUILD_TYPE=Release \
	-DTPL_ENABLE_CUDALIB=ON  \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_CXX_FLAGS="-qsmp=omp -Ofast -DRELEASE ${INC_VTUNE}" \
    	-DCMAKE_C_FLAGS="-qsmp=omp -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC -DSLU_HAVE_LAPACK -DGPU_SOLVE -D_USE_NVSHMEM -fopenmp -I${NVSHMEM_HOME}/include/ " \
    	-DCMAKE_CUDA_FLAGS="-ccbin ${CXX} -gencode arch=compute_70,code=sm_70 -I${NVSHMEM_HOME}/include/ -DENABLE_MPI_SUPPORT -DPRNTlevel=1 -std=c++11 -DPROFlevel=0 -DEBUGlevel=0 -DGPU_ACC -DSLU_HAVE_LAPACK -DGPU_SOLVE --disable-warnings" \
	-DCMAKE_EXE_LINKER_FLAGS="-lcuda -lcudadevrt -L${OLCF_CUDA_ROOT}/lib64 -lcudart_static -L${OLCF_CUDA_ROOT}/lib64/stubs/ -lnvidia-ml -libverbs -L${NVSHMEM_HOME}/lib/ -lnvshmem -L${OLCF_SPECTRUM_MPI_ROOT}/lib -lmpi_ibm" 
make pddrive		





