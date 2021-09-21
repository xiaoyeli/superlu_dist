#!/bin/bash
#module load netlib-lapack/3.8.0
#module load gcc/6.4.0
module load xl
module load cmake
module load cuda
module load essl 

export CRAYPE_LINK_TYPE=dynamic
export PARMETIS_ROOT=/ccs/home/liuyangz/my_software/parmetis-4.0.3_ibm
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/static-build/Linux-ppc64le
export ACC=GPU
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl 




cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include;${OLCF_CUDA_ROOT}/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
	-DTPL_CUDA_LIBRARIES="${OLCF_CUDA_ROOT}/lib64/libcublas.so;${OLCF_CUDA_ROOT}/lib64/libcusparse.so;${OLCF_CUDA_ROOT}/lib64/libcudart.so" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_C_COMPILER=mpicc \
	-DCMAKE_CXX_COMPILER=mpiCC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_CUDA_HOST_COMPILER=mpicc \
	-DTPL_BLAS_LIBRARIES="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libessl.so;/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp.so" \
	-DTPL_LAPACK_LIBRARIES="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libessl.so" \
	-DCMAKE_BUILD_TYPE=Release \
	-DTPL_ENABLE_CUDALIB=ON \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_CXX_FLAGS="-qsmp=omp -Ofast -DRELEASE ${INC_VTUNE}" \
    -DCMAKE_C_FLAGS="-qsmp=omp  -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC -fopenmp" \
    -DCMAKE_CUDA_FLAGS="--disable-warnings -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC -gencode arch=compute_70,code=sm_70" 
make pddrive			
#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.so"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.so" \
#        -DCMAKE_CXX_FLAGS="-g -trace -Ofast -std=c++11 -DAdd_ -DRELEASE -tcollect -L$VT_LIB_DIR -lVT $VT_ADD_LIBS" \


#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_lapack95_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_blas95_lp64.a"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.a"  


#	-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE ${INC_VTUNE}" \
# DCMAKE_BUILD_TYPE=Release or Debug compiler options set in CMAKELIST.txt

#DCMAKE_CUDA_HOST_COMPILER
#        -DCMAKE_C_FLAGS="-g -O0 -std=c99 -DPRNTlevel=2 -DPROFlevel=1 -DDEBUGlevel=0" \
#	-DCMAKE_C_FLAGS="-g -O0 -std=c11 -DPRNTlevel=1 -DPROFlevel=1 -DDEBUGlevel=0" \
