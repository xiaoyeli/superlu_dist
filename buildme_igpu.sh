#!/bin/bash

installdir=$PWD/install_igpu
echo $installdir

bdir=buiid_sycl

rm -rf make.inc $bdir; mkdir $bdir; cd $bdir
rm -rf ${installdir}
export PARMETIS_ROOT=/gpfs/jlse-fs0/users/dguo/tests/Parmetis/parmetis-4.0.3

export I_MPI_CC=icpx
export I_MPI_CXX=icpx

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DTPL_ENABLE_PARMETISLIB=FALSE \
      -DTPL_ENABLE_LAPACKLIB=OFF \
      -DTPL_ENABLE_SYCLLIB=TRUE \
      -Denable_openmp:BOOL=FALSE \
      -DTPL_ENABLE_COMBBLASLIB=OFF \
      -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
      -DTPL_BLAS_LIBRARIES="" \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_STATIC_LIBS=ON \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_Fortran_COMPILER=mpifort \
      -DCMAKE_Fortran_FLAGS="-I${MKLROOT}/include" \
      -DCMAKE_CXX_FLAGS="-D__STRICT_ANSI__ -I$MKLROOT/include -DSUPERLU_USE_MKL -DPRNTlevel=1 -DDEBUGlevel=0 -Wno-format -Wno-deprecated-declarations -Wno-return-type -Wno-deprecated-declarations" \
      -DCMAKE_EXE_LINKER_FLAGS="-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64" \
      -DCMAKE_INSTALL_PREFIX=${installdir} \
      -DXSDK_INDEX_SIZE=64 \
      ..

#       -DCMAKE_CXX_FLAGS="-g -fsycl -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -Wsycl-strict -sycl-std=2020 -D__STRICT_ANSI__ -I$MKLROOT/include -DSUPERLU_USE_MKL -DPRNTlevel=1 -DDEBUGlevel=0 -Wno-format -Wno-deprecated-declarations -Wno-return-type -Wno-deprecated-declarations" \

#      -DCMAKE_C_FLAGS="-g -D__STRICT_ANSI__ -I$MKLROOT/include -DSUPERLU_USE_MKL  -DPRNTlevel=0 -DDEBUGlevel=0 -Wno-format -Wno-deprecated-declarations" \

#      -DCMAKE_EXE_LINKER_FLAGS="-DMKL_ILP64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core" \
#      -DCMAKE_EXE_LINKER_FLAGS="-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64" \




#make -j16
#make install
