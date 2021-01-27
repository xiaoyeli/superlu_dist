#!/bin/bash
# module load parmetis/4.0.3


# this is not working because cmake doesn't yet have good support for hipcc with HIP_PLATFORM=nvcc

module restore PrgEnv-cray
module load cray-mvapich2/2.3.4
module load gcc
module load cmake
module unload cray-libsci_acc
module load cray-libsci/20.03.1
module load craype-accel-nvidia70
module load cuda10.2/toolkit
module load rocm			 
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CRAYPE_LINK_TYPE=dynamic
export PARMETIS_ROOT=/home/users/coe0238/my_software/parmetis-4.0.3_cce_dynamic
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64

# export HIP_PLATFORM=nvcc
# export ACC=GPU
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl 
 
 


cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include;${CUDA_ROOT}/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_Fortran_COMPILER=ftn \
	-DCMAKE_C_COMPILER=cc \
	-DCMAKE_CXX_COMPILER=CC \
	-Denable_openmp=ON \
	-DTPL_BLAS_LIBRARIES="/opt/cray/pe/gcc-libs/libstdc++.so.6;/opt/cray/pe/libsci/20.03.1/CRAY/8.5/x86_64/lib/libsci_cray.so" \
	-DTPL_LAPACK_LIBRARIES="/opt/cray/pe/libsci/20.03.1/CRAY/8.5/x86_64/lib/libsci_cray.so" \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_CXX_FLAGS="-O3 -DRELEASE" \
	-DCMAKE_C_FLAGS="-DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0" \
	-DHIP_HIPCC_FLAGS="-gencode arch=compute_70,code=sm_70  -I/opt/cray/pe/cray-mvapich2/2.3.4/infiniband/crayclang/10.0/include/" \
	-DTPL_ENABLE_HIPLIB=ON
make pddrive 	



#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.so"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.so" \
#        -DCMAKE_CXX_FLAGS="-g -trace -Ofast -std=c++11 -DAdd_ -DRELEASE -tcollect -L$VT_LIB_DIR -lVT $VT_ADD_LIBS" \


#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_lapack95_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_blas95_lp64.a"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.a"  


#	-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE ${INC_VTUNE}" \
# DCMAKE_BUILD_TYPE=Release or Debug compiler options set in CMAKELIST.txt

#        -DCMAKE_C_FLAGS="-g -O0 -std=c99 -DPRNTlevel=2 -DPROFlevel=1 -DDEBUGlevel=0" \
#	-DCMAKE_C_FLAGS="-g -O0 -std=c11 -DPRNTlevel=1 -DPROFlevel=1 -DDEBUGlevel=0" \
