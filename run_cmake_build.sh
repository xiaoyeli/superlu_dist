#!/bin/bash

## if [ !$?NERSC_HOST ]
if [ -z $NERSC_HOST ]
then
    echo "NERSC_HOST undefined"
elif [ "$NERSC_HOST" == "edison" ]
then
    mkdir edison-build; cd edison-build;
#    export PARMETIS_ROOT=~/Edison/lib/parmetis-4.0.3_64
    export PARMETIS_ROOT=~/Edison/lib/parmetis-4.0.3
    export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
    cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DCMAKE_C_FLAGS="-std=c99 -fPIC -DPRNTlevel=1" \
    -DCMAKE_Fortran_COMPILER=ftn \
    -Denable_blaslib=OFF \
    -DTPL_BLAS_LIBRARIES="-mkl" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=.
#    -DXSDK_INDEX_SIZE=64 \
#    -DCMAKE_EXE_LINKER_FLAGS="-shared"
elif [ "$NERSC_HOST" == "cori" ]
then
    rm -fr cori-build; mkdir cori-build; cd cori-build;
    export PARMETIS_ROOT=~/Cori/lib/parmetis-4.0.3-64
#    export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/shared-build
    export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
    cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -Denable_blaslib=OFF \
    -DTPL_BLAS_LIBRARIES="-mkl" \
    -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_C_FLAGS="-std=c99 -fPIC -DPRNTlevel=1" \
    -DCMAKE_INSTALL_PREFIX=. \
    -DXSDK_INDEX_SIZE=64
#    -DCMAKE_EXE_LINKER_FLAGS="-shared" \
fi

THISHOST=`hostname -s`
echo "host: $THISHOST"
if [ "$THISHOST" == "ssg1" ]
then
  rm -fr ssg1-build; mkdir ssg1-build; cd ssg1-build;
#  export PARMETIS_ROOT=~/lib/static/64-bit/parmetis-4.0.3 
  export PARMETIS_ROOT=~/lib/static/parmetis-4.0.3 
  export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
  echo "ParMetis root: $PARMETIS_ROOT"
  export COMBBLAS_ROOT=~/Cori/KNL/combinatorial-blas-2.0/CombBLAS
  export COMBBLAS_BUILD_DIR=${COMBBLAS_ROOT}/_build
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DTPL_COMBBLAS_INCLUDE_DIRS="${COMBBLAS_ROOT}/_install/include;${COMBBLAS_R\
OOT}/Applications/BipartiteMatchings" \
    -DTPL_COMBBLAS_LIBRARIES="${COMBBLAS_BUILD_DIR}/libCombBLAS.a" \
    -DCMAKE_C_FLAGS="-std=c99 -g -DPRNTlevel=1 -DDEBUGlevel=0" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-std=c++11" \
    -Denable_blaslib=OFF \
    -Denable_combblaslib=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=.
fi
#   -Denable_parmetislib=OFF
#    -DXSDK_INDEX_SIZE=64 \

# make VERBOSE=1
# make test
