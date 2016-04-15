######################################################################
#
# CMakeLists.txt for SUPERLU_DIST
#
######################################################################

# Required version
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# Project version numbers
project(SuperLU_DIST NONE)
set(VERSION_MAJOR "5")
set(VERSION_MINOR "0")
set(VERSION_BugFix "0")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_BugFix})

######################################################################
#
# IDEAS: xSDK standards module
MESSAGE("\nProcess XSDK defaults ...")
# SET(USE_XSDK_DEFAULTS_DEFAULT TRUE) # Set to false if desired
INCLUDE("cmake/XSDKDefaults.cmake")
######################################################################

######################################################################
#
# Usual initialization stuff
#
######################################################################
# setup options
option(enable_blaslib   "Build the CBLAS library" ${enable_blaslib_DEFAULT})
option(enable_parmetislib   "Build the ParMETIS library" ON)
option(enable_doc       "Build doxygen documentation" OFF)
option(enable_double    "Enable double precision library" ON)
option(enable_complex16 "Enable complex16 precision library" ON)
option(enable_examples  "Build examples" ON)
option(TPL_PARMETIS_LIBRARIES "List of absolute paths to ParMETIS link libraries [].")
option(TPL_PARMETIS_INCLUDE_DIRS "List of absolute paths to ParMETIS include directories [].")

if (NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX /usr/local)
endif()


#---- For shared library

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

# the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#----

if (BUILD_SHARED_LIBS)
  message("-- SuperLU_DIST will be built as a shared library.")
  set(PROJECT_NAME_LIB_EXPORT libsuperlu_dist.so)
else()
  message("-- SuperLU_DIST will be built as a static library.")
  set(PROJECT_NAME_LIB_EXPORT libsuperlu_dist.a)
endif()

enable_language (C)
enable_language (Fortran)
set(NOFORTRAN FALSE)
set(SUPERLU_VERSION "${PROJECT_VERSION}")
set(SUPERLU_REV "${PROJECT_REV}")

# The XSDK standard does not allow using internally built BLAS
if (USE_XSDK_DEFAULTS)
  set(enable_blaslib_DEFAULT OFF)
else()
  set(enable_blaslib_DEFAULT ON)
endif()


# setup required compiler defines and options.
## get_directory_property( DirDefs COMPILE_DEFINITIONS )
set(CMAKE_C_FLAGS "-DDEBUGlevel=0 -DPRNTlevel=0 ${CMAKE_C_FLAGS}")
if(XSDK_INDEX_SIZE EQUAL 64)
    message("-- Using 64 bit integer for index size")
    set(CMAKE_C_FLAGS "-D_LONGINT ${CMAKE_C_FLAGS}")
endif()	
set(CMAKE_C_FLAGS_RELEASE "-O3" CACHE STRING "")

######################################################################
#
# Find packages
#
######################################################################
#
#--------------------- BLAS ---------------------
if(NOT enable_blaslib)
#  set(TPL_BLAS_LIBRARIES "" CACHE FILEPATH
#    "Override of list of absolute path to libs for BLAS.")
  if(TPL_BLAS_LIBRARIES)
    set(BLAS_FOUND TRUE)
  else()
    find_package(BLAS)
    if(BLAS_FOUND)
      set(TPL_BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE FILEPATH
        "Set from FindBLAS.cmake BLAS_LIBRARIES." FORCE)
    endif()
  endif()
endif()

if(BLAS_FOUND)
    message("-- Using TPL_BLAS_LIBRARIES='${TPL_BLAS_LIBRARIES}'")
    set(CMAKE_C_FLAGS "-DUSE_VENDOR_BLAS ${CMAKE_C_FLAGS}")
    set(BLAS_LIB ${TPL_BLAS_LIBRARIES})
    # fix up BLAS library name
    string (REPLACE ";" " " BLAS_LIB_STR "${BLAS_LIB}")
    set(BLAS_LIB_EXPORT ${BLAS_LIB_STR})
else()
    message("-- Did not find or specify BLAS, so configure to build internal CBLAS ...")
    add_subdirectory(CBLAS)
    set(BLAS_LIB blas)
    if (BUILD_SHARED_LIBS)  # export to be referenced by downstream makefile
        set(BLAS_LIB_EXPORT ${CMAKE_SOURCE_DIR}/build/CBLAS/libblas.so)
    else()
        set(BLAS_LIB_EXPORT ${CMAKE_SOURCE_DIR}/build/CBLAS/libblas.a)
    endif()
endif()

#--------------------- MPI ---------------------
find_package(MPI)
if(MPI_C_FOUND)
    set(CMAKE_C_FLAGS "${MPI_C_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
endif()

#--------------------- ParMETIS ---------------------
if (enable_parmetislib)   ## want to use parmetis
  if (NOT TPL_PARMETIS_LIBRARIES)
    message(FATAL_ERROR "TPL_PARMETIS_LIBRARIES option should be set for PARMETIS support to be enabled.")
  endif()

  if (NOT TPL_PARMETIS_INCLUDE_DIRS)
    message(FATAL_ERROR "TPL_PARMETIS_INCLUDE_DIRS option be set for PARMETIS support to be enabled.")
  endif()
  foreach(dir ${TPL_PARMETIS_INCLUDE_DIRS})
    if (NOT EXISTS ${dir})
      message(FATAL_ERROR "PARMETIS include directory not found: ${dir}")
    endif()
  endforeach()

  message("-- Enabled support for PARMETIS")
  set(PARMETIS_FOUND TRUE)

  set(PARMETIS_LIB ${TPL_PARMETIS_LIBRARIES})
  # fix up PARMETIS library names
  string (REPLACE ";" " " PARMETIS_LIB_STR "${PARMETIS_LIB}")
  set(PARMETIS_LIB_EXPORT ${PARMETIS_LIB_STR})

else()
  message("-- Will not link with ParMETIS.")
endif()

# if(NOT enable_parmetislib)
#  find_package(PARMETIS)    ## does not have this Module yet.
# endif()


######################################################################
#
# Include directories
#
######################################################################

include_directories(${CMAKE_SOURCE_DIR}/SRC)
include_directories(${TPL_PARMETIS_INCLUDE_DIRS})  ## parmetis
include_directories(${MPI_C_INCLUDE_PATH})

######################################################################
#
# Add subdirectories
#
######################################################################

add_subdirectory(SRC)

if(enable_tests)
  enable_testing()
  add_subdirectory(TESTING)
endif()

if(enable_doc)
  message(FATAL_ERROR "Documentation build requested but not implemented.")
  #implement doxygen
endif()

if(enable_examples)
  enable_testing()
  add_subdirectory(EXAMPLE)
endif()

# file(WRITE "make.defs" "# can be exposed to users" ${CMAKE_C_COMPILER})
# configure_file(${CMAKE_SOURCE_DIR}/make.inc.in ${CMAKE_BINARY_DIR}/make.inc)
configure_file(${SuperLU_DIST_SOURCE_DIR}/make.inc.in ${SuperLU_DIST_SOURCE_DIR}/make.inc)
