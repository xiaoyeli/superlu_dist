#!/bin/bash

module load spack cmake


cmake .. \
  -DCMAKE_C_FLAGS="  -std=c99 -D_XOPEN_SOURCE -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_ -I${MKLROOT}/include -fopenmp " \
  -DCMAKE_CXX_FLAGS="-I${MKLROOT}/include -fopenmp " \
  -DCMAKE_EXE_LINKER_FLAGS="-lmpifort" \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DXSDK_ENABLE_Fortran=ON \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DTPL_ENABLE_LAPACKLIB=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DTPL_ENABLE_CUDALIB=OFF \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_INSTALL_LIBDIR=./lib \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTPL_BLAS_LIBRARIES="~/my_software/OpenBLAS/libopenblas.so" \
  -DTPL_LAPACK_LIBRARIES="~/my_software/OpenBLAS/libopenblas.so" \
  -DTPL_PARMETIS_INCLUDE_DIRS="/home/liuyangz/my_software/parmetis-4.0.3/include;/home/liuyangz/my_software/parmetis-4.0.3/metis/include" \
  -DTPL_PARMETIS_LIBRARIES="/home/liuyangz/my_software/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.so;/home/liuyangz/my_software/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so" \
  -DTPL_ENABLE_COMBBLASLIB=OFF \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON 
     
make pddrive -j16
make pddrive3d -j16
#make f_pddrive

## -DTPL_BLAS_LIBRARIES=/global/cfs/cdirs/m3894/ptlin/tpl/amd_blis/install/amd_blis-20211021-n9-gcc9.3.0/lib/libblis.a \
