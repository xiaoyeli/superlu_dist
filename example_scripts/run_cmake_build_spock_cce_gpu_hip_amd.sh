#!/bin/bash
# module load parmetis/4.0.3

module load cmake
module load rocm				 
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CRAYPE_LINK_TYPE=dynamic
export PARMETIS_ROOT=/ccs/home/liuyangz/my_software/parmetis-4.0.3_spock_cce
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
export ACC=GPU
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl 




cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so;/opt/rocm-4.1.0/lib/libroctx64.so;/opt/rocm-4.1.0/lib/libroctracer64.so" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_Fortran_COMPILER=ftn \
	-DCMAKE_C_COMPILER=cc \
	-DCMAKE_CXX_COMPILER=CC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DTPL_BLAS_LIBRARIES="/opt/cray/pe/libsci/21.04.1.1/CRAY/9.0/x86_64/lib/libsci_cray.so" \
	-DTPL_LAPACK_LIBRARIES="/opt/cray/pe/libsci/21.04.1.1/CRAY/9.0/x86_64/lib/libsci_cray.so" \
	-DCMAKE_BUILD_TYPE=Release \
	-DTPL_ENABLE_HIPLIB=ON \
	-DHIP_HIPCC_FLAGS="--amdgpu-target=gfx906,gfx908 -I/opt/cray/pe/mpich/8.1.4/ofi/crayclang/9.1/include" \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_CXX_FLAGS="-Wno-format -Wno-unused-value -Wno-return-type -Wno-unsequenced -Wno-switch -Wno-parentheses -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 " \
	-DCMAKE_C_FLAGS="-Wno-format -Wno-unused-value -Wno-return-type -Wno-unsequenced -Wno-switch -Wno-parentheses -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 "
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
