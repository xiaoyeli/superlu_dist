#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue








module load pgi/19.9/64
#module load hdf5/pgi-19.5/openmpi-4.0.2rc1/1.10.5
#module load git/2.18
module load openmpi/pgi-19.5/4.0.2rc1/64
#module load fftw/gcc/openmpi-4.0.1/3.3.8
module load cudatoolkit/10.0

module unload hdf5
module unload fftw


export ACC=GPU
export PARMETIS_ROOT=/home/yl33/parmetis-4.0.3-pgi
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build-shared/Linux-ppc64le

export CUDA_ROOT=/usr/local/cuda-10.0
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl 


cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so;${CUDA_ROOT}/lib64/libcublas.so;${CUDA_ROOT}/lib64/libcusparse.so;${CUDA_ROOT}/lib64/libcudart.so" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_C_COMPILER=mpicc \
	-Denable_openmp:BOOL=FALSE \
	-DCMAKE_CXX_COMPILER=mpiCC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
        -DTPL_BLAS_LIBRARIES="/usr/lib64/libblas.so" \
        -DTPL_LAPACK_LIBRARIES="/usr/lib64/liblapack.so" \
        -DCMAKE_CXX_FLAGS="-DRELEASE" \
        -DCMAKE_C_FLAGS="-DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC"

make pddrive



#salloc -N 1 --qos=test -t 0:30:00 --gpus=2




#        -DCMAKE_CUDA_FLAGS="-DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC -gencode arch=compute_70,code=sm_70"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.so" \
#        -DCMAKE_CXX_FLAGS="-g -trace -Ofast -std=c++11 -DAdd_ -DRELEASE -tcollect -L$VT_LIB_DIR -lVT $VT_ADD_LIBS" \


#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_lapack95_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_blas95_lp64.a"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.a"  


# DCMAKE_BUILD_TYPE=Release or Debug compiler options set in CMAKELIST.txt

#        -DCMAKE_C_FLAGS="-g -O0 -std=c99 -DPRNTlevel=2 -DPROFlevel=1 -DDEBUGlevel=0" \
        #-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=1 -DPROFlevel=1 -DDEBUGlevel=0 ${INC_VTUNE}" \
