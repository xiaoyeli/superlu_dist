#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue


module purge
export ALLINEA_FORCE_CUDA_VERSION=20.0.1
module load cudatoolkit/11.2 pgi/20.4 openmpi/pgi-20.4/4.0.4/64 
module load hdf5/pgi-20.4/openmpi-4.0.4/1.10.6 fftw/gcc/openmpi-4.0.4/3.3.8 anaconda ddt 
#module load cmake


export PATH=/home/yl33/cmake-3.20.3/bin/:$PATH
cmake --version
export PARMETIS_ROOT=~/petsc_master/traverse-pgi-openmpi-199-gpucuda-branch/

export CUDA_ROOT=/usr/local/cuda-11.2
#export CUDA_PATH=${CUDA_ROOT}
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl 

cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_ROOT}/lib/libparmetis.a;${PARMETIS_ROOT}/lib/libmetis.a" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_C_COMPILER=mpicc \
	-DTPL_ENABLE_CUDALIB=TRUE \
	-DTPL_ENABLE_LAPACKLIB=TRUE \
	-Denable_openmp:BOOL=TRUE \
	-DCMAKE_CUDA_FLAGS="-ccbin pgc++ -D_PGIC_PRINCETON_OVERRIDE_" \
	-DCMAKE_CUDA_HOST_COMPILER=mpicc \
	-DCMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES="/usr/local/cuda-11.2/include" \
	-DCMAKE_INCLUDE_SYSTEM_FLAG_C="-I" \
	-DCMAKE_CXX_COMPILER=mpiCC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
        -DTPL_BLAS_LIBRARIES="${PARMETIS_ROOT}/lib/libflapack.a;${PARMETIS_ROOT}/lib/libfblas.a" \
        -DTPL_LAPACK_LIBRARIES="${PARMETIS_ROOT}/lib/libflapack.a;${PARMETIS_ROOT}/lib/libfblas.a" \
        -DCMAKE_CXX_FLAGS="-DRELEASE -pgf90libs" \
        -DCMAKE_C_FLAGS="-DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -pgf90libs"

make pddrive
make install

#	-DXSDK_ENABLE_Fortran=FALSE \ 


#	-DCUDAToolkit_LIBRARY_ROOT="${CUDA_ROOT}" \

#salloc -N 1 --qos=test -t 0:30:00 --gpus=2




#        -DCMAKE_CUDA_FLAGS="-DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC -gencode arch=compute_70,code=sm_70"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.so" \
#        -DCMAKE_CXX_FLAGS="-g -trace -Ofast -std=c++11 -DAdd_ -DRELEASE -tcollect -L$VT_LIB_DIR -lVT $VT_ADD_LIBS" \


#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_lapack95_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_blas95_lp64.a"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.a"  


# DCMAKE_BUILD_TYPE=Release or Debug compiler options set in CMAKELIST.txt

#        -DCMAKE_C_FLAGS="-g -O0 -std=c99 -DPRNTlevel=2 -DPROFlevel=1 -DDEBUGlevel=0" \
        #-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=1 -DPROFlevel=1 -DDEBUGlevel=0 ${INC_VTUNE}" \
