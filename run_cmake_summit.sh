#!/bin/bash

export CRAYPE_LINK_TYPE=dynamic
#export PARMETIS_ROOT=/ccs/home/xiaoye/dynamic-lib/parmetis-4.0.3
export PARMETIS_ROOT=/ccs/home/xiaoye/dynamic-lib/64/parmetis-4.0.3
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-ppc64le
export ACC=GPU

rm -fr 64bit-build; mkdir 64bit-build; cd 64bit-build

cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include;${OLCF_CUDA_ROOT}/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so;${LIB_VTUNE};${OLCF_CUDA_ROOT}/lib64/libcublas.so;${OLCF_CUDA_ROOT}/lib64/libcudart.so" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_C_COMPILER=mpicc \
	-DCMAKE_CXX_COMPILER=mpiCC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DTPL_BLAS_LIBRARIES="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libessl.so;/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp.so" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_CXX_FLAGS="-Ofast -DRELEASE ${INC_VTUNE}" \
        -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=2 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,${PARMETIS_BUILD_DIR}/libparmetis:${PARMETIS_BUILD_DIR}/libmetis" \
        -DTPL_ENABLE_LAPACKLIB=OFF \
        -DXSDK_INDEX_SIZE=64
#	-DTPL_LAPACK_LIBRARIES="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libessl.so" \

# make pddrive

