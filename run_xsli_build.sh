#!/bin/bash
THISHOST=`hostname -s`
echo "host: $THISHOST"

#export CRAYPE_LINK_TYPE=dynamic
rm -fr xsli-build; mkdir xsli-build; cd xsli-build;
export PARMETIS_ROOT=/ccs/home/anil/parmetis-4.0.3
#module load cuda/11.3.1
#export OLCF_CUDA_ROOT=/sw/summit/cuda/11.2.0
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-ppc64le
echo "ParMetis root: $PARMETIS_ROOT"
export ACC=GPU
#-DCMAKE_C_FLAGS="-I/usr/local/include_old -DGPU_ACC -std=c++11 -std=gnu99 -O3 -g -DPRNTlevel=1 -DDEBUGlevel=0" \
#-DCMAKE_CUDA_FLAGS="-DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC -I${OLCF_CUDA_ROOT}/include/" \
cmake .. \
    -DTPL_ENABLE_CUDALIB=TRUE \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include;" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a;" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_C_FLAGS="-I/usr/local/include_old -DGPU_ACC -std=c++11 -std=gnu99 -g -DPRNTlevel=1 -DDEBUGlevel=0 -Wall -pedantic" \
    -DCMAKE_CXX_COMPILER=mpiCC \
    -DCMAKE_CXX_FLAGS="-I/usr/local/include_old -std=c++11" \
    -DTPL_ENABLE_INTERNAL_BLASLIB=ON \
    -DTPL_ENABLE_COMBBLASLIB=OFF \
    -DTPL_ENABLE_LAPACKLIB=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DXSDK_ENABLE_Fortran=OFF \
    -DTPL_CUDA_LIBRARIES="${OLCF_CUDA_ROOT}/lib64/libcublas.so;${OLCF_CUDA_ROOT}/lib64/libcudart.so;" \
    -DCMAKE_CUDA_FLAGS="-DGPU_ACC -I${OLCF_CUDA_ROOT}/include/" \
    -Denable_openmp=TRUE \
    -DCMAKE_INSTALL_PREFIX=.

# make VERBOSE=1
# make test

