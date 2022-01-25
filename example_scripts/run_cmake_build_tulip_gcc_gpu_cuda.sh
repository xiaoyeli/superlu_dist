#!/bin/bash
# module load parmetis/4.0.3


module load openmpi
module load cmake
module load cuda10.2/toolkit/10.2.89
module load craype-accel-nvidia70
module load rocm


export CRAY_CUDATOOLKIT_INCLUDE_OPTS="-I$CUDA_ROOT/include/ -I$CUDA_ROOT/extras/CUPTI/include/ -I$CUDA_ROOT/extras/Debugger/include/"
export CRAY_CUDATOOLKIT_POST_LINK_OPTS="-L$CUDA_ROOT/lib64/ -L$CUDA_ROOT/extras/CUPTI/lib64/ -Wl,--as-needed -Wl,-lcupti -Wl,-lcudart -Wl,--no-as-needed -L$CUDA_CMLOCAL_ROOT//lib64 -lcuda"


export CRAYPE_LINK_TYPE=dynamic
export PARMETIS_ROOT=/home/users/coe0238/my_software/parmetis-4.0.3_gnu_dynamic
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
export ACC=GPU
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl 




cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include;${CUDA_ROOT}/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so" \
	-DTPL_CUDA_LIBRARIES="${CUDA_ROOT}/lib64/libcublas.so;${CUDA_ROOT}/lib64/libcusparse.so;${CUDA_ROOT}/lib64/libcudart.so" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_C_COMPILER=mpicc \
	-DCMAKE_Fortran_COMPILER=mpif90 \
	-DCMAKE_CXX_COMPILER=mpicxx \
	-DCMAKE_INSTALL_PREFIX=. \
	-DTPL_BLAS_LIBRARIES="/home/users/coe0238/my_software/BLAS-3.8.0/libblas.so" \
	-DTPL_LAPACK_LIBRARIES="/home/users/coe0238/my_software/lapack-3.9.0/liblapack.so" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_CXX_FLAGS="-Ofast -DRELEASE" \
	-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC" \
	-DCMAKE_CUDA_FLAGS="--disable-warnings -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DGPU_ACC -gencode arch=compute_70,code=sm_70"
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
