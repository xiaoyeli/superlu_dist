#!/bin/bash

installdir=$PWD/install_igpu
echo $installdir

bdir=buiid_sycl

rm -rf make.inc $bdir; mkdir $bdir; cd $bdir
rm -rf ${installdir}
export PARMETIS_ROOT=/gpfs/jlse-fs0/users/dguo/tests/Parmetis/parmetis-4.0.3

# using static linking of oneMKL
#      -DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_sycl.a ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a " \
# using dynamic linking of oneMKL
#-DTPL_BLAS_LIBRARIES="-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core" \

#

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
      -DCMAKE_CXX_FLAGS=" -DMKL_LP64 -qmkl=sequential -D__STRICT_ANSI__ -DSUPERLU_USE_MKL -DPRNTlevel=1 -DDEBUGlevel=0 -Wno-format -Wno-deprecated-declarations -Wno-return-type -Wno-deprecated-declarations -Wno-writable-strings" \
      -DCMAKE_INSTALL_PREFIX=${installdir} \
      -DXSDK_INDEX_SIZE=64 \
      -DXSDK_ENABLE_Fortran=OFF \
      ..

#            -DCMAKE_Fortran_FLAGS="-I${MKLROOT}/include" \

#      -DCMAKE_CXX_FLAGS="-D__STRICT_ANSI__ -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl -DSUPERLU_USE_MKL -DPRNTlevel=1 -DDEBUGlevel=0 -Wno-format -Wno-deprecated-declarations -Wno-return-type -Wno-deprecated-declarations -Denable_blaslib_DEFAULT=OFF" \

#       -DCMAKE_CXX_FLAGS="-g -fsycl -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -Wsycl-strict -sycl-std=2020 -D__STRICT_ANSI__ -I$MKLROOT/include -DSUPERLU_USE_MKL -DPRNTlevel=1 -DDEBUGlevel=0 -Wno-format -Wno-deprecated-declarations -Wno-return-type -Wno-deprecated-declarations" \

#      -DCMAKE_C_FLAGS="-g -D__STRICT_ANSI__ -I$MKLROOT/include -DSUPERLU_USE_MKL  -DPRNTlevel=0 -DDEBUGlevel=0 -Wno-format -Wno-deprecated-declarations" \

#      -DCMAKE_EXE_LINKER_FLAGS="-DMKL_ILP64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core" \
#      -DCMAKE_EXE_LINKER_FLAGS="-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64" \




#make -j16
#make install
