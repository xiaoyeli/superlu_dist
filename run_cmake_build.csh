#!/bin/csh

if ( ! $?NERSC_HOST ) then
    echo "NERSC_HOST undefined"
else
  if ( "$NERSC_HOST" == "edison" ) then
    setenv PARMETIS_ROOT ~/Edison/lib/parmetis-4.0.3
#    setenv PARMETIS_BUILD_DIR ${PARMETIS_ROOT}/shared-build
    setenv PARMETIS_BUILD_DIR ${PARMETIS_ROOT}/static-build/Linux-x86_64
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
    -DCMAKE_INSTALL_PREFIX=..
  endif

  if ( "$NERSC_HOST" == "cori" ) then
    setenv PARMETIS_ROOT ~/Cori/lib/parmetis-4.0.3
    setenv PARMETIS_BUILD_DIR ${PARMETIS_ROOT}/shared-build
#    setenv PARMETIS_BUILD_DIR ${PARMETIS_ROOT}/static-build/Linux-x86_64
    cmake .. \
    -DUSE_XSDK_DEFAULTS=TRUE\
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so" \
    -Denable_blaslib=OFF \
    -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_C_FLAGS="-std=c99 -fPIC" \
    -DCMAKE_EXE_LINKER_FLAGS="-shared" \
    -DCMAKE_INSTALL_PREFIX=..
  endif
endif

set THISHOST=`hostname -s`
#echo $THISHOST
if ( "$THISHOST" == "ssg1" ) then
  setenv PARMETIS_ROOT ~/lib/static/parmetis-4.0.3 
  setenv PARMETIS_BUILD_DIR ${PARMETIS_ROOT}/build/Linux-x86_64
    echo $PARMETIS_ROOT
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DCMAKE_C_FLAGS="-std=c99 -g" \
    -Denable_blaslib=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_INSTALL_PREFIX=..
endif

# make VERBOSE=1
# make test
