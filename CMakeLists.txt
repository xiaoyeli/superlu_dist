######################################################################
#
# CMakeLists.txt for SUPERLU_DIST
#
######################################################################

# Required version
cmake_minimum_required(VERSION 3.18.1 FATAL_ERROR)

# Project version numbers
#project(SuperLU_DIST C CXX CUDA)
project(SuperLU_DIST C CXX)
set(VERSION_MAJOR "8")
set(VERSION_MINOR "0")
set(VERSION_BugFix "0")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_BugFix})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Set up options
option(enable_doc       "Build doxygen documentation" OFF)
option(enable_single    "Enable single precision library" ON)
option(enable_double    "Enable double precision library" ON)
option(enable_single    "Enable single precision library" OFF)
option(enable_complex16 "Enable complex16 precision library" ON)
option(enable_tests  "Build tests" ON)
option(enable_examples  "Build examples" ON)
option(XSDK_ENABLE_Fortran "Enable Fortran" ON)

#-- BLAS
option(TPL_ENABLE_INTERNAL_BLASLIB  "Build the CBLAS library" ${enable_blaslib_DEFAULT})
option(TPL_BLAS_LIBRARIES "List of absolute paths to blas libraries [].")
#-- ParMETIS
option(TPL_ENABLE_PARMETISLIB   "Build the ParMETIS library" ON)
option(TPL_PARMETIS_LIBRARIES "List of absolute paths to ParMETIS link libraries [].")
option(TPL_PARMETIS_INCLUDE_DIRS "List of absolute paths to ParMETIS include directories [].")
#-- LAPACK
option(TPL_ENABLE_LAPACKLIB "Enable LAPACK library" OFF)
option(TPL_LAPACK_LIBRARIES "List of absolute paths to LAPACK libraries [].")
#-- CombBLAS
option(TPL_ENABLE_COMBBLASLIB   "Build the CombBLAS library" OFF)
option(TPL_COMBBLAS_LIBRARIES "List of absolute paths to CombBLAS link libraries [].")
option(TPL_COMBBLAS_INCLUDE_DIRS "List of absolute paths to CombBLAS include directories [].")
#-- CUDA
option(TPL_ENABLE_CUDALIB   "Enable the CUDA libraries" OFF)
option(TPL_ENABLE_HIPLIB   "Enable the HIP libraries" OFF)

######################################################################
#
# IDEAS: xSDK standards module
#MESSAGE("\nProcess XSDK defaults ...")
# SET(USE_XSDK_DEFAULTS_DEFAULT TRUE) # Set to false if desired
#INCLUDE("cmake/XSDKDefaults.cmake")
######################################################################

include(CTest)
include(CheckLanguage)

######################################################################
#
# Usual initialization stuff
#
######################################################################
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#message("!!!! top: cxx_implicit='${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}'")

if (XSDK_ENABLE_Fortran)
  enable_language (Fortran)
  set(NOFORTRAN FALSE)
endif()

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
  # set the position independent code property on all targets, so that
  # -fPIC is added in compiler flag.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(PROJECT_NAME_LIB_EXPORT libsuperlu_dist.so)
  SET(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${CMAKE_INSTALL_PREFIX}/SRC")

  if (BUILD_STATIC_LIBS)
    message("-- SuperLU_DIST will also be built as a static library.")
  endif()
  set(SHARED_C_FLAGS_EXPORT ${CMAKE_SHARED_LIBRARY_C_FLAGS})
else()
  message("-- SuperLU_DIST will be built as a static library.")
  set(PROJECT_NAME_LIB_EXPORT libsuperlu_dist.a)
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

if(NOT MSVC)
  include(GNUInstallDirs)
  set(default_install_inc_dir ${CMAKE_INSTALL_INCLUDEDIR})
  set(default_install_lib_dir ${CMAKE_INSTALL_LIBDIR})
  set(default_install_bin_dir ${CMAKE_INSTALL_BINDIR})
else() # for Windows
  set(default_install_inc_dir "include")
  set(default_install_lib_dir "lib")
  set(default_install_bin_dir "bin")
endif()

set(INSTALL_INC_DIR "${default_install_inc_dir}" CACHE STRING "The folder where headers will be installed.")
set(INSTALL_LIB_DIR "${default_install_lib_dir}" CACHE STRING "The folder where libraries will be installed.")
set(INSTALL_BIN_DIR "${default_install_bin_dir}" CACHE STRING "The folder where runtime files will be installed.")

# Set up required compiler defines and options.
## get_directory_property( DirDefs COMPILE_DEFINITIONS )

if(XSDK_INDEX_SIZE EQUAL 64)
    message("-- Using 64 bit integer for index size.")
endif()	
set(CMAKE_C_FLAGS_RELEASE "-O3 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g" CACHE STRING "")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -g" CACHE STRING "")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

######################################################################
#
# Find packages
#
######################################################################
#
#--------------------- MPI ---------------------
find_package(MPI REQUIRED)
if(MPI_C_FOUND)
    set(CMAKE_C_FLAGS "${MPI_C_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${MPI_CXX_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}" )
#    message("MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}")
endif()
if (XSDK_ENABLE_Fortran)
  if(MPI_Fortran_FOUND)
    set(CMAKE_Fortran_FLAGS "${MPI_Fortran_COMPILE_FLAGS} ${CMAKE_Fortran_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_Fortran_LINK_FLAGS}")
    include_directories(${MPI_Fortran_INCLUDE_PATH})
  endif()
endif()  

#---- CUDA libraries
if (TPL_ENABLE_CUDALIB)   ## want to use cuda
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    message("-- Enabled support for CUDA.")
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    if (CUDA_FOUND)
      if (NOT CMAKE_CUDA_FLAGS)
        cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
      endif()
      set(HAVE_CUDA TRUE)
      set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --expt-relaxed-constexpr -DNDEBUG" CACHE STRING "")
      set(CMAKE_CUDA_FLAGS_DDEBUG "-O0 --expt-relaxed-constexpr -DDEBUG -g" CACHE STRING "")
    endif()

#     find_package(CUB REQUIRED)

    find_package(CUDAToolkit REQUIRED)
# The following appears in SRC/CMakeLists.txt
#    if(CUDAToolkit_FOUND)
#      target_link_libraries(superlu_dist PUBLIC CUDA::cudart CUDA::cusolver CUDA::cublas)
#    endif()
    message("-- CUDAToolkit_LIBRARY_ROOT='${CUDAToolkit_LIBRARY_ROOT}'")
    if (NOT "${CUDAToolkit_LIBRARY_ROOT}" STREQUAL "")
       set(CUDA_LIBRARIES "${CUDAToolkit_LIBRARY_ROOT}/lib64/libcudart.so")
       # expose the following to make.inc, but location may not be reliable
       # set(CUDA_CUBLAS_LIBRARIES "${CUDAToolkit_LIBRARY_ROOT}/lib64/libcublas.so")
    else()
	message("-- CUDAToolkit_LIBRARY_ROOT empty, not setting CUDA_LIBRARIES")
    endif()

#     # The following make.inc exporting does not work
#     set(CUDA_LIB CUDA::cudart CUDA::cublas CUDA::cusolver)
# # fix up CUDA library names    
#     string (REPLACE ";" " " CUDA_LIB_STR "${CUDA_LIB}")
#     set(CUDA_LIB_EXPORT ${CUDA_LIB_STR}) 
#     set(HAVE_CUDA TRUE)
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_CUDA")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DHAVE_CUDA")
  else()
    message("-- CUDA libraries not found.")
  endif()
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
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# The following causes problem with cmake/3.20.+
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
    message("-- OpenMP_EXE_LINKER_FLAGS='${OpenMP_EXE_LINKER_FLAGS}'")
    message("-- CMAKE_EXE_LINKER_FLAGS='${CMAKE_EXE_LINKER_FLAGS}'")
  endif()  
endif()


include(CheckLanguage)

#--------------------- CUDA libraries ---------------------
if (TPL_ENABLE_CUDALIB)   ## want to use cuda
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    message("-- Enabled support for CUDA.")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # make.inc exporting not yet tested
    set(CUDA_LIB CUDA::cudart CUDA::cublas CUDA::cusolver)
    string (REPLACE ";" " " CUDA_LIB_STR "${CUDA_LIB}")
    set(CUDA_LIB_EXPORT ${CUDA_LIB_STR}) 
    set(HAVE_CUDA TRUE)
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_CUDA")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DHAVE_CUDA")
  else()
    message("-- CUDA libraries not found.")
  endif()
endif()



#--------------------- HIP libraries ---------------------
if (TPL_ENABLE_HIPLIB)   ## want to use hip
  find_package(HIP MODULE)
  if(HIP_FOUND)
    message("-- Enabled support for HIP.")
    find_package(hipblas REQUIRED)
    find_package(rocsolver REQUIRED)
    find_package(rocblas REQUIRED)
    set(HIP_LIB roc::hipblas roc::rocblas roc::rocsolver)
    message("HIP_LIB ${HIP_LIB}")

    # make.inc exporting not yet tested
    string (REPLACE ";" " " HIP_LIB_STR "${HIP_LIB}")
    set(HIP_LIB_EXPORT ${HIP_LIB_STR}) 
    set(HAVE_HIP TRUE)
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_HIP")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_HIP")    
    # set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DHAVE_HIP")    
    if (BUILD_SHARED_LIBS)
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -fPIC")    
    endif()

    # if ("$ENV{HIP_PLATFORM}" STREQUAL "hcc")
    #   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_HIP -D__HIP_PLATFORM_HCC__")
    #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_HIP -D__HIP_PLATFORM_HCC__")
    # elseif ("$ENV{HIP_PLATFORM}" STREQUAL "nvcc")
    #   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_HIP -D__HIP_PLATFORM_NVCC__")
    #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_HIP -D__HIP_PLATFORM_NVCC__")
    # endif()

  else()
    message("-- HIP libraries not found.")
  endif()
endif()

#--------------------- BLAS ---------------------
if(NOT TPL_ENABLE_INTERNAL_BLASLIB)
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
    set(CMAKE_CUDA_FLAGS "-DUSE_VENDOR_BLAS ${CMAKE_CUDA_FLAGS}")
    set(BLAS_LIB ${TPL_BLAS_LIBRARIES})
    # fix up BLAS library name
    string (REPLACE ";" " " BLAS_LIB_STR "${BLAS_LIB}")
    set(BLAS_LIB_EXPORT ${BLAS_LIB_STR})
else()
    message("-- Did not find or specify BLAS, so configure to build internal CBLAS ...")
    add_subdirectory(CBLAS)
    set(BLAS_LIB blas)
    if (BUILD_SHARED_LIBS)  # export to be referenced by downstream makefile
        set(BLAS_LIB_EXPORT ${CMAKE_INSTALL_PREFIX}/${INSTALL_LIB_DIR}/libblas.so)
    else()
        set(BLAS_LIB_EXPORT ${CMAKE_INSTALL_PREFIX}/${INSTALL_LIB_DIR}/libblas.a)
    endif()
endif()

#--------------------- LAPACK ---------------------
if(TPL_ENABLE_LAPACKLIB)  ## want to use LAPACK
  if(TPL_LAPACK_LIBRARIES)
    set(LAPACK_FOUND TRUE)
  else()
    find_package(LAPACK)
    if(LAPACK_FOUND)
      set(TPL_LAPACK_LIBRARIES "${LAPACK_LIBRARIES}" CACHE FILEPATH
        "Set from FindLAPACK.cmake LAPACK_LIBRARIES." FORCE)
    endif()
  endif()
else()  ## not to use LAPACK
  set(LAPACK_FOUND FALSE)
  set(SLU_HAVE_LAPACK FALSE)
endif()

if(LAPACK_FOUND)
    message("-- Using TPL_LAPACK_LIBRARIES='${TPL_LAPACK_LIBRARIES}'")
    set(LAPACK_LIB ${TPL_LAPACK_LIBRARIES})
    # fix up LAPACK library name
    string (REPLACE ";" " " LAPACK_LIB_STR "${LAPACK_LIB}")
    set(LAPACK_LIB_EXPORT ${LAPACK_LIB_STR})
    set(SLU_HAVE_LAPACK TRUE)
else()
    message("-- Will not link with LAPACK.")
endif()

#--------------------- ParMETIS ---------------------
if (TPL_ENABLE_PARMETISLIB)   ## want to use parmetis
  if (NOT TPL_PARMETIS_LIBRARIES)
    message(FATAL_ERROR "TPL_PARMETIS_LIBRARIES option should be set for PARMETIS support to be enabled.")
  endif()

  if (NOT TPL_PARMETIS_INCLUDE_DIRS)
    message(FATAL_ERROR "TPL_PARMETIS_INCLUDE_DIRS option be set for PARMETIS support to be enabled.")
  endif()
  foreach(dir ${TPL_PARMETIS_INCLUDE_DIRS})
    if (NOT MINGW AND NOT EXISTS ${dir})
      message(FATAL_ERROR "PARMETIS include directory not found: ${dir}")
    endif()
    set(CMAKE_C_FLAGS "-I${dir} ${CMAKE_C_FLAGS}")
  endforeach()
  


  message("-- Enabled support for PARMETIS.")
  set(PARMETIS_FOUND TRUE)

  set(PARMETIS_LIB ${TPL_PARMETIS_LIBRARIES})
  # fix up PARMETIS library names
  string (REPLACE ";" " " PARMETIS_LIB_STR "${PARMETIS_LIB}")
  set(PARMETIS_LIB_EXPORT ${PARMETIS_LIB_STR})
else()
  message("-- Will not link with ParMETIS.")
endif()

if(TPL_ENABLE_PARMETISLIB AND NOT PARMETIS_FOUND)
  find_package(ParMETIS)
  if(PARMETIS_FOUND)
    set(PARMETIS_LIB ParMETIS::ParMETIS)
    set(TPL_PARMETIS_INCLUDE_DIRS "")
  endif()
endif()

if(PARMETIS_FOUND)
  set(HAVE_PARMETIS TRUE)
endif()

#---------------------- Additional C linker library ---------
SET(_c_libs ${CMAKE_C_IMPLICIT_LINK_LIBRARIES})
FOREACH(_lib ${_c_libs})
  set(EXTRA_LIB "-l${_lib} ${EXTRA_LIB}")
ENDFOREACH()
string (REPLACE ";" " " EXTRA_LIB_STR "${EXTRA_LIB}")
set(EXTRA_LIB_EXPORT ${EXTRA_LIB_STR}) 
message("-- EXTRA_LIB_EXPORT='${EXTRA_LIB_EXPORT}'")

#---------------------- Additional Fortran linker library ---------
if (XSDK_ENABLE_Fortran)
   SET(_f_libs ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
   FOREACH(_lib ${_f_libs})
       set(EXTRA_FLIB "${EXTRA_FLIB} -l${_lib}")
   ENDFOREACH()
   string (REPLACE ";" " " EXTRA_FLIB_STR "${EXTRA_FLIB}")
   set(EXTRA_FLIB_EXPORT ${EXTRA_FLIB_STR})
   message("-- EXTRA_FLIB_EXPORT='${EXTRA_FLIB_EXPORT}'")
   
   if (BUILD_SHARED_LIBS)
      message("-- superlu_dist_fortran will be built as a dynamic library.")
      set(PROJECT_NAME_LIB_FORTRAN libsuperlu_dist_fortran.so)
      SET(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
   else()
      message("-- superlu_dist_fortran will be built as a static library.")
      set(PROJECT_NAME_LIB_FORTRAN libsuperlu_dist_fortran.a)
   endif()
endif()

#--------------------- CombBLAS ---------------------
if (TPL_ENABLE_COMBBLASLIB)   ## want to use CombBLAS
  if (NOT TPL_COMBBLAS_LIBRARIES)
    message(FATAL_ERROR "TPL_COMBBLAS_LIBRARIES option should be set for COMBBLAS support to be enabled.")
  endif()

  if (NOT TPL_COMBBLAS_INCLUDE_DIRS)
    message(FATAL_ERROR "TPL_COMBBLAS_INCLUDE_DIRS option be set for COMBBLAS support to be enabled.")
  endif()
  foreach(dir ${TPL_COMBBLAS_INCLUDE_DIRS})
    if (NOT MINGW AND NOT EXISTS ${dir})
      message(FATAL_ERROR "COMBBLAS include directory not found: ${dir}")
    endif()
    set(CMAKE_CXX_FLAGS "-I${dir} ${CMAKE_CXX_FLAGS}")
  endforeach()

  message("-- Enabled support for COMBBLAS")
  set(COMBBLAS_FOUND TRUE)
  set(CMAKE_CXX_STANDARD 14)  # CombBLAS requires c++14

  set(COMBBLAS_LIB ${TPL_COMBBLAS_LIBRARIES})
  # fix up COMBBLAS library names
  string (REPLACE ";" " " COMBBLAS_LIB_STR "${COMBBLAS_LIB}")
  set(COMBBLAS_LIB_EXPORT ${COMBBLAS_LIB_STR})

else()
  message("-- Will not link with CombBLAS.")
endif()

if(COMBBLAS_FOUND)
  set(HAVE_COMBBLAS TRUE)
  set(LOADER $(CXX))
else()
  set(LOADER $(CC))
endif()

######################################################################
#
# Fortran-C name mangling 
#
######################################################################
if (XSDK_ENABLE_Fortran)
  include(FortranCInterface)
  FortranCInterface_HEADER(${SuperLU_DIST_SOURCE_DIR}/SRC/superlu_FortranCInterface.h
                            MACRO_NAMESPACE "FC_")
  FortranCInterface_VERIFY(CXX)
  SET(MPI_Fortran_LINK_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

  #---------------------------------------------------------------------
  # Set default fortran format to free and preprocess all files
  #---------------------------------------------------------------------
  set(CMAKE_Fortran_FORMAT FREE)
  set(CMAKE_Fortran_PREPROCESS ON)
endif()

######################################################################
#
# Include directories
#
######################################################################

include_directories(${CMAKE_BINARY_DIR}/SRC) # For superlu_dist_config.h
# include_directories(${CMAKE_SOURCE_DIR}/SRC)
if (TPL_PARMETIS_INCLUDE_DIRS)
  include_directories(${TPL_PARMETIS_INCLUDE_DIRS})  ## parmetis
endif ()
if (TPL_COMBBLAS_INCLUDE_DIRS)
  include_directories(${TPL_COMBBLAS_INCLUDE_DIRS})  ## CombBLAS
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

if (XSDK_ENABLE_Fortran)
  add_subdirectory(FORTRAN)
endif()

# superlu_dist uses c++11. PUBLIC means that the other codes linking to it need c++11
#target_compile_features(SuperLU_DIST PUBLIC cxx_std_11)

# Generate various configure files with proper definitions

# Add pkg-config support
if(IS_ABSOLUTE ${CMAKE_INSTALL_LIBDIR})
  set(pkgconfig_libdir ${CMAKE_INSTALL_LIBDIR})
else()
  set(pkgconfig_libdir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/superlu_dist.pc.in ${CMAKE_CURRENT_BINARY_DIR}/superlu_dist.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/superlu_dist.pc
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)


# configure_file(${CMAKE_SOURCE_DIR}/make.inc.in ${CMAKE_BINARY_DIR}/make.inc)
configure_file(${SuperLU_DIST_SOURCE_DIR}/make.inc.in ${SuperLU_DIST_SOURCE_DIR}/make.inc)

configure_file(${SuperLU_DIST_SOURCE_DIR}/SRC/superlu_dist_config.h.in ${SuperLU_DIST_BINARY_DIR}/SRC/superlu_dist_config.h)
configure_file(${SuperLU_DIST_SOURCE_DIR}/SRC/superlu_dist_config.h.in ${SuperLU_DIST_SOURCE_DIR}/SRC/superlu_dist_config.h)

# Following is to configure a file for FORTRAN code
configure_file(${SuperLU_DIST_SOURCE_DIR}/SRC/superlu_dist_config.h.in ${SuperLU_DIST_BINARY_DIR}/FORTRAN/superlu_dist_config.h)


#message("MPI_Fortran_LINK_FLAGS '${MPI_Fortran_LINK_FLAGS}'")
