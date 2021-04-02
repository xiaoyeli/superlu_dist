# Install script for directory: /gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
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
  set(CMAKE_OBJDUMP "/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/binutils-2.34-rnwhrdgiqluqiypg5palnxdxviv3mynt/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64" TYPE STATIC_LIBRARY FILES "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/SRC/libsuperlu_dist.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/superlu_FCnames.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/dcomplex.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/machines.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/psymbfact.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/superlu_defs.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/superlu_enum_consts.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/supermatrix.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/util_dist.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/colamd.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/TreeBcast_slu.hpp"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/TreeReduce_slu.hpp"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/TreeBcast_slu_impl.hpp"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/TreeReduce_slu_impl.hpp"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/jlse-build/SRC/superlu_dist_config.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/superlu_FortranCInterface.h"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/onemkl_utils.hpp"
    "/gpfs/jlse-fs0/users/abagusetty/tmp_superlu/superlu_dist/SRC/superlu_ddefs.h"
    )
endif()

