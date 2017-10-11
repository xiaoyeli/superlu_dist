#!/bin/bash

if [ !$?NERSC_HOST ]
then
    echo "NERSC_HOST undefined"
elif [ "$NERSC_HOST" == "edison" ]
then
    export PARMETIS_ROOT=~/Edison/lib/parmetis-4.0.3
#    setenv PARMETIS_BUILD_DIR ${PARMETIS_ROOT}/shared-build
    export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/static-build/Linux-x86_64
    cmake .. \
    -DUSE_XSDK_DEFAULTS=FALSE\
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DCMAKE_C_FLAGS="-std=c99 -fPIC" \
#    -DCMAKE_EXE_LINKER_FLAGS="-shared" \
    -DCMAKE_Fortran_COMPILER=ftn \
    -Denable_blaslib=OFF \
#    -DTPL_BLAS_LIBRARIES=" " \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=.
elif [ "$NERSC_HOST" == "cori" ]
then
    export PARMETIS_ROOT=~/Cori/lib/parmetis-4.0.3
#    export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/shared-build
    setenv PARMETIS_BUILD_DIR ${PARMETIS_ROOT}/static-build/Linux-x86_64
    cmake .. \
    -DUSE_XSDK_DEFAULTS=TRUE\
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -Denable_blaslib=OFF \
    -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_C_FLAGS="-std=c99 -fPIC" \
#    -DCMAKE_EXE_LINKER_FLAGS="-shared" \
    -DCMAKE_INSTALL_PREFIX=.
fi

THISHOST=`hostname -s`
echo "host: $THISHOST"
if [ "$THISHOST" == "ssg1" ]
then
  rm -fr ssg1-build; mkdir ssg1-build; cd ssg1-build;
  export PARMETIS_ROOT=~/lib/static/parmetis-4.0.3 
  export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
  echo "ParMetis root: $PARMETIS_ROOT"
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DCMAKE_C_FLAGS="-std=c99 -g -DPRNTlevel=0 -DDEBUGlevel=0" \
    -Denable_blaslib=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_INSTALL_PREFIX=.
fi

# make VERBOSE=1
# make test
