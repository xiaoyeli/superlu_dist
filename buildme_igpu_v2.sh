#!/bin/bash

installdir=$PWD/install_igpu
echo $installdir

bdir=buiid_sycl

rm -rf make.inc $bdir; mkdir $bdir; cd $bdir
rm -rf ${installdir}
#export PARMETIS_ROOT=/gpfs/jlse-fs0/users/dguo/tests/Parmetis/parmetis-4.0.3

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DTPL_ENABLE_PARMETISLIB=FALSE \
      -DTPL_ENABLE_LAPACKLIB=OFF \
      -DTPL_ENABLE_SYCLLIB=TRUE \
      -Denable_openmp:BOOL=FALSE \
      -Denable_complex16:BOOL=FALSE \
      -DTPL_ENABLE_COMBBLASLIB=OFF \
      -DTPL_BLAS_LIBRARIES="-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -ldl" \
      -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_STATIC_LIBS=ON \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_Fortran_COMPILER=mpifort \
      -DCMAKE_CXX_FLAGS=" -DMKL_ILP64 -qmkl=sequential -D__STRICT_ANSI__ -DSUPERLU_USE_MKL -DPRNTlevel=1 -DDEBUGlevel=0 -Wno-format -Wno-deprecated-declarations -Wno-return-type -Wno-deprecated-declarations -Wno-writable-strings" \
      -DCMAKE_INSTALL_PREFIX=${installdir} \
      -DXSDK_INDEX_SIZE=64 \
      -DXSDK_ENABLE_Fortran=OFF \
      ..

