# Install script for directory: /global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so.6.4.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so.6"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/lib:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./lib" TYPE SHARED_LIBRARY FILES
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/SRC/libsuperlu_dist.so.6.4.0"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/SRC/libsuperlu_dist.so.6"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so.6.4.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so.6"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
           NEW_RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/lib:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so"
         RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/lib:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./lib" TYPE SHARED_LIBRARY FILES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/SRC/libsuperlu_dist.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so"
         OLD_RPATH "/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/lib:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libparmetis:/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit/build/Linux-x86_64/libmetis")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./lib/libsuperlu_dist.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./lib" TYPE STATIC_LIBRARY FILES "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/SRC/libsuperlu_dist.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/superlu_FCnames.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/gpublas_utils.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/dcomplex.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/machines.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/psymbfact.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/superlu_defs.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/superlu_enum_consts.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/supermatrix.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/util_dist.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/colamd.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/build_perlmutter_1122/SRC/superlu_dist_config.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/superlu_FortranCInterface.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/superlu_ddefs.h"
    "/global/cfs/cdirs/m2956/nanding/myprojects/multi-GPU/superlu_dist/SRC/superlu_zdefs.h"
    )
endif()

