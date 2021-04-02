#!/bin/bash

rm -fr jlse-build; mkdir jlse-build; cd jlse-build;

cmake .. \
	  -DCMAKE_C_COMPILER=icx \
	  -DCMAKE_CXX_COMPILER=dpcpp \
	  -DCMAKE_Fortran_COMPILER=ifx \
	  -DTPL_ENABLE_SYCLLIB=TRUE \
	  -DCMAKE_C_FLAGS="-std=c99 -fPIC -DPRNTlevel=0" \
	  -DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so;${MKLROOT}/lib/intel64/libmkl_intel_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_sycl.so" \
	  -DTPL_LAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so;${MKLROOT}/lib/intel64/libmkl_intel_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;${MKLROOT}/lib/intel64/libmkl_sycl.so" \
	  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
	  -DTPL_ENABLE_COMBBLASLIB=FALSE \
	  -DTPL_ENABLE_PARMETISLIB=FALSE \
	  -DCMAKE_COLOR_MAKEFILE=TRUE \
	  -DCMAKE_VERBOSE_MAKEFILE=TRUE \
	  -Denable_openmp=FALSE \
	  -DCMAKE_INSTALL_PREFIX=.
#   -DTPL_ENABLE_CUDALIB=TRUE \
#    -DTPL_BLAS_LIBRARIES="-mkl" \
#    -DXSDK_INDEX_SIZE=64
#    -DXSDK_ENABLE_Fortran=TRUE \
#    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
#    -DTPL_ENABLE_COMBBLASLIB=FALSE \
#    -DTPL_ENABLE_LAPACKLIB=FALSE \
#    -DCMAKE_EXE_LINKER_FLAGS="-shared" \
#    -DCMAKE_CXX_FLAGS="-std=c++14" \
#    -DTPL_COMBBLAS_INCLUDE_DIRS="${COMBBLAS_ROOT}/_install/include;${COMBBLAS_ROOT}/_install/include/CombBLAS;${COMBBLAS_ROOT}/BipartiteMatchings"

#    -DXSDK_INDEX_SIZE=64 \
#    -DXSDK_ENABLE_Fortran=TRUE \
#   -DTPL_ENABLE_PARMETISLIB=OFF
#    -DCMAKE_CXX_FLAGS="-std=c++14"

# make VERBOSE=1
# make test
