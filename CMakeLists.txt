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
set(VERSION_MINOR "3")
set(VERSION_BugFix "0")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_BugFix})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

######################################################################
#
# IDEAS: xSDK standards module
MESSAGE("\nProcess XSDK defaults ...")
# SET(USE_XSDK_DEFAULTS_DEFAULT TRUE) # Set to false if desired
INCLUDE("cmake/XSDKDefaults.cmake")
INCLUDE(CTest)

######################################################################

######################################################################
#
# Usual initialization stuff
#
######################################################################
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)    ## ????
set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

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

SET(BUILD_STATIC_LIBS TRUE CACHE BOOL "Include static libs when building shared")

if (BUILD_SHARED_LIBS)
  message("-- SuperLU_DIST will be built as a shared library.")
  set(PROJECT_NAME_LIB_EXPORT libsuperlu_dist.so)
  if (BUILD_STATIC_LIBS)
    message("-- SuperLU_DIST will also be built as a static library.")
  endif()
else()
  message("-- SuperLU_DIST will be built as a static library.")
  set(PROJECT_NAME_LIB_EXPORT libsuperlu_dist.a)
endif()

enable_language (C)
if (XSDK_ENABLE_Fortran)
  enable_language (Fortran)
  set(NOFORTRAN FALSE)
endif()
set(SUPERLU_VERSION "${PROJECT_VERSION}")
set(SUPERLU_REV "${PROJECT_REV}")

# The XSDK standard does not allow using internally built BLAS
if (USE_XSDK_DEFAULTS)
  set(enable_blaslib_DEFAULT OFF)
else()
  set(enable_blaslib_DEFAULT ON)
endif()

if (NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX /usr/local)
endif()


# setup options
option(enable_blaslib   "Build the CBLAS library" ${enable_blaslib_DEFAULT})
option(enable_parmetislib   "Build the ParMETIS library" ON)
option(enable_doc       "Build doxygen documentation" OFF)
option(enable_double    "Enable double precision library" ON)
option(enable_complex16 "Enable complex16 precision library" ON)
option(enable_tests  "Build tests" ON)
option(enable_examples  "Build examples" ON)
option(TPL_PARMETIS_LIBRARIES "List of absolute paths to ParMETIS link libraries [].")
option(TPL_PARMETIS_INCLUDE_DIRS "List of absolute paths to ParMETIS include directories [].")

if(NOT MSVC)
  include(GNUInstallDirs)
  set(default_install_inc_dir ${CMAKE_INSTALL_INCLUDEDIR})
  set(default_install_lib_dir ${CMAKE_INSTALL_LIBDIR})
  set(default_install_bin_dir ${CMAKE_INSTALL_BINDIR})
else()
  set(default_install_inc_dir "include")
  set(default_install_lib_dir "lib")
  set(default_install_bin_dir "bin")
endif()

set(INSTALL_INC_DIR "${default_install_inc_dir}" CACHE STRING "The folder where headers will be installed.")
set(INSTALL_LIB_DIR "${default_install_lib_dir}" CACHE STRING "The folder where libraries will be installed.")
set(INSTALL_BIN_DIR "${default_install_bin_dir}" CACHE STRING "The folder where runtime files will be installed.")


# Set up required compiler defines and options.
## get_directory_property( DirDefs COMPILE_DEFINITIONS )
# set(CMAKE_C_FLAGS "-DDEBUGlevel=0 -DPRNTlevel=0 ${CMAKE_C_FLAGS}")
if(XSDK_INDEX_SIZE EQUAL 64)
    message("-- Using 64 bit integer for index size")
endif()	
set(CMAKE_C_FLAGS_RELEASE "-O3" CACHE STRING "")

######################################################################
#
# Find packages
#
######################################################################
#
#--------------------- MPI ---------------------
find_package(MPI)
if(MPI_C_FOUND)
    set(CMAKE_C_FLAGS "${MPI_C_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}" )
endif()
#--------------------- OpenMP ---------------------
if (NOT DEFINED enable_openmp)
  set(enable_openmp TRUE)
endif ()
if (enable_openmp)
  find_package(OpenMP)
## include(FindOpenMP)  # Strumpack uses this
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
# On edison, OpenMP_EXE_LINKER_FLAGS is empty
#  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
    message("-- OpenMP_EXE_LINKER_FLAGS='${OpenMP_EXE_LINKER_FLAGS}'")
    message("-- CMAKE_EXE_LINKER_FLAGS='${CMAKE_EXE_LINKER_FLAGS}'")
  endif()
endif()
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
        set(BLAS_LIB_EXPORT ${CMAKE_INSTALL_PREFIX}/CBLAS/libblas.so)
    else()
        set(BLAS_LIB_EXPORT ${CMAKE_INSTALL_PREFIX}/CBLAS/libblas.a)
    endif()
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
    set(CMAKE_C_FLAGS "-I${dir} ${CMAKE_C_FLAGS}")
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

if(enable_parmetislib AND NOT PARMETIS_FOUND)
  find_package(ParMETIS)
  if(PARMETIS_FOUND)
    set(PARMETIS_LIB ParMETIS::ParMETIS)
    set(TPL_PARMETIS_INCLUDE_DIRS "")
  endif()
endif()

if(PARMETIS_FOUND)
  set(HAVE_PARMETIS TRUE)
endif()


######################################################################
#
# Include directories
#
######################################################################

include_directories(${CMAKE_BINARY_DIR}/SRC) # For superlu_dist_config.h
include_directories(${CMAKE_SOURCE_DIR}/SRC)
if (TPL_PARMETIS_INCLUDE_DIRS)
  include_directories(${TPL_PARMETIS_INCLUDE_DIRS})  ## parmetis
endif ()
include_directories(${MPI_C_INCLUDE_PATH})

######################################################################
#
# Add subdirectories
#
######################################################################

add_subdirectory(SRC)

if(enable_doc)
  message(FATAL_ERROR "Documentation build requested but not implemented.")
  #implement doxygen
endif()

if(enable_tests)
  enable_testing()
  add_subdirectory(TEST)
endif()

if(enable_examples)
  enable_testing()
  add_subdirectory(EXAMPLE)
endif()

# file(WRITE "make.defs" "# can be exposed to users" ${CMAKE_C_COMPILER})
# configure_file(${CMAKE_SOURCE_DIR}/make.inc.in ${CMAKE_BINARY_DIR}/make.inc)
configure_file(${SuperLU_DIST_SOURCE_DIR}/make.inc.in ${SuperLU_DIST_SOURCE_DIR}/make.inc)
configure_file(${SuperLU_DIST_SOURCE_DIR}/SRC/superlu_dist_config.h.in ${SuperLU_DIST_BINARY_DIR}/SRC/superlu_dist_config.h)

# Add pkg-config support
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/superlu_dist.pc.in ${CMAKE_CURRENT_BINARY_DIR}/superlu_dist.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/superlu_dist.pc
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
