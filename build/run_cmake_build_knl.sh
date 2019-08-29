#!/bin/bash
# Bash script to submit many files to Cori/Edison/Queue



 module unload darshan
 module swap craype-haswell craype-mic-knl
# module load cray-fftw
# module swap intel/18.0.1.163 intel/17.0.3.191
 module load gsl
# module load cray-hdf5-parallel/1.10.0.3
 module load idl
 module load craype-hugepages2M
 module unload cray-libsci
 module load hpctoolkit

export LIBRARY_PATH=/global/cscratch1/sd/kz21/openmp-shared/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/global/cscratch1/sd/kz21/openmp-shared/lib:$LD_LIBRARY_PATH



Vtune=0
export CRAYPE_LINK_TYPE=dynamic
export PARMETIS_ROOT=/global/homes/l/liuyangz/Cori/my_software/parmetis-4.0.3_knl
#export PARMETIS_ROOT=/global/homes/x/xiaoye/Cori/KNL/lib/parmetis-4.0.3
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/shared-build-intel18/Linux-x86_64 
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl 
if [[ ${Vtune} == 1 ]]; then
INC_VTUNE="-g -DVTUNE=1 -I$VTUNE_AMPLIFIER_XE_2018_DIR/include"
LIB_VTUNE="$VTUNE_AMPLIFIER_XE_2018_DIR/lib64/libittnotify.a"
fi

cmake .. \
	-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
	-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so;${LIB_VTUNE}" \
	-Denable_blaslib=OFF \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_C_COMPILER=cc \
        -DCMAKE_CXX_COMPILER=CC \
	-DCMAKE_INSTALL_PREFIX=. \
	-DCMAKE_BUILD_TYPE=DebWithRelInfo \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DXSDK_INDEX_SIZE=32 \
    -DCMAKE_CXX_FLAGS="-g -Ofast -std=c++11 -DAdd_ -DRELEASE ${INC_VTUNE}" \
    -DCMAKE_C_FLAGS="-g -align -mkl -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 ${INC_VTUNE}" 
#    -DCMAKE_C_FLAGS="-align -mkl -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -qopt-report -qopt-report-phase=vec ${INC_VTUNE}" 
    # -DCMAKE_C_FLAGS="-align -mkl -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -qopt-report -qopt-report-phase=vec -vec-threshold0 ${INC_VTUNE}" 
#        -DCMAKE_C_FLAGS="-align -qopt-report=5 -fno-alias -mkl -std=c11 -DPRNTlevel=2 -DPROFlevel=2 -DDEBUGlevel=1 -DHAVE_PARMETIS ${INC_VTUNE}" 
 
#    -DCMAKE_C_FLAGS="-g -O0 -std=c11 -DPRNTlevel=1 -DPROFlevel=1 -DDEBUGlevel=0 -mkl -DHAVE_PARMETIS " \

make pddrive
#-lomp

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.so"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.so" \
#        -DCMAKE_CXX_FLAGS="-g -trace -Ofast -std=c++11 -DAdd_ -DRELEASE -tcollect -L$VT_LIB_DIR -lVT $VT_ADD_LIBS" \


#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_lapack95_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_blas95_lp64.a"

#	-DTPL_BLAS_LIBRARIES="/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_intel_lp64.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_sequential.a;/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/libmkl_core.a"  


# DCMAKE_BUILD_TYPE=Release or Debug compiler options set in CMAKELIST.txt

#        -DCMAKE_C_FLAGS="-g -O0 -std=c99 -DPRNTlevel=2 -DPROFlevel=1 -DDEBUGlevel=0" \
#	-DCMAKE_C_FLAGS="-g -O0 -std=c11 -DPRNTlevel=1 -DPROFlevel=1 -DDEBUGlevel=0" \
