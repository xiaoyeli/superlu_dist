#!/bin/csh

if ( ! $?NERSC_HOST ) then
    echo "NERSC_HOST undefined"
else
  if ( "$NERSC_HOST" == "edison" ) then
  setenv PARMETIS_ROOT ~/Edison/lib/parmetis-4.0.3 
  setenv PARMETIS_BUILD_DIR ${PARMETIS_ROOT}/build/Linux-x86_64 
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DCMAKE_C_FLAGS="-std=c99 -g" \
  endif
endif

set THISHOST=`hostname -s`
echo $THISHOST
# if ( $(hostname -s) == "scg1" ) then
if ( "$THISHOST" == "scg1" ) then
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
