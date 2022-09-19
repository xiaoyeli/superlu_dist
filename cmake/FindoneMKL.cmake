# - Find Intel MKL
# Find the MKL libraries
#
# Options:
#
#   MKL_STATIC       :   use static linking
#   MKL_MULTI_THREADED:   use multi-threading
#   MKL_SDL           :   Single Dynamic Library interface
#   MKL_ARCH : architecture to link (possible values: intel64, mic)
#
# This module defines the following variables:
#
#   MKL_FOUND            : True if MKL_INCLUDE_DIR are found
#   MKL_INCLUDE_DIR      : where to find mkl.h, etc.
#   MKL_LINK_FLAGS       : flags to use when linking with mkl
#   MKL_INCLUDE_DIRS     : set when MKL_INCLUDE_DIR found
#   MKL_LIBRARIES        : the library to link against.

# $(MKLROOT)/lib/intel64/libmkl_intel.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a -ldl -lpthread -lm
# $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a -ldl -lpthread -lm
# $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a -ldl -lpthread -lm

# interface layer, either lp64 or ilp64 (only for 64 bit build)
if(NOT MKL_ROOT)
    set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "Folder contains MKL")
endif(NOT MKL_ROOT)
message("-- Looking for MKL installation, MKL_ROOT = ${MKL_ROOT}")

if(MKL_ROOT)

if(NOT MKL_INTERFACE_LAYER)
    set(MKL_INTERFACE_LAYER "_lp64")
endif(NOT MKL_INTERFACE_LAYER)

if(NOT MKL_ARCH)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(MKL_ARCH "intel64")
  else(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(MKL_ARCH "ia32")
  endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
endif(NOT MKL_ARCH)

if(MKL_ARCH STREQUAL "ia32")
  if(WIN32)
    set(MKL_INTERFACE_LAYER "_c")
  else(WIN32)
    set(MKL_INTERFACE_LAYER "")
  endif(WIN32)
endif(MKL_ARCH STREQUAL "ia32")

include(FindPackageHandleStandardArgs)


# Find include dir
find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT}/include)

# Find libraries

# Handle suffix
set(_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(MKL_STATIC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
else(MKL_STATIC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
endif(MKL_STATIC)


# MKL is composed by four layers: Interface, Threading, Computational and RTL

######################### Interface layer #######################
set(MKL_INTERFACE_LIBNAME "mkl_intel${MKL_INTERFACE_LAYER}")

find_library(MKL_INTERFACE_LIBRARY ${MKL_INTERFACE_LIBNAME}
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)
get_filename_component(MKL_LIBRARY_DIR ${MKL_INTERFACE_LIBRARY} PATH)

######################## Threading layer ########################
if(MKL_MULTI_THREADED)
    set(MKL_THREADING_LIBNAME "mkl_intel_thread" "mkl_gnu_thread")
else(MKL_MULTI_THREADED)
    set(MKL_THREADING_LIBNAME "mkl_sequential")
endif(MKL_MULTI_THREADED)

find_library(MKL_THREADING_LIBRARY ${MKL_THREADING_LIBNAME}
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)

####################### Computational layer #####################
find_library(MKL_CORE_LIBRARY mkl_core
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)
find_library(MKL_FFT_LIBRARY mkl_cdft_core
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)
find_library(MKL_SCALAPACK_LIBRARY mkl_scalapack${MKL_INTERFACE_LAYER}
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)
if( NWPW_SYCL )
  find_library(MKL_ONEMKL_LIBRARY mkl_sycl
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)
endif(NWPW_SYCL)
find_library(MKL_BLACS_LIBRARY mkl_blacs_intelmpi${MKL_INTERFACE_LAYER}
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)

############################ RTL layer ##########################
find_library(MKL_FFT_LIBRARY mkl_rt
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)

if( NWPW_SYCL )

  set(MKL_LIBRARY "-Wl,--start-group\
      ${MKL_INTERFACE_LIBRARY}\
      ${MKL_THREADING_LIBRARY}\
      ${MKL_CORE_LIBRARY}\
      ${MKL_FFT_LIBRARY}\
      ${MKL_BLACS_LIBRARY}\
      ${MKL_SCALAPACK_LIBRARY}\
      ${MKL_RTL_LIBRARY}\
      ${MKL_ONEMKL_LIBRARY}\
      -Wl,--end-group -lsycl -lOpenCL -ldl -lpthread -lm")

  set(MKL_LINK_FLAGS "-Wl,--start-group\
      ${MKL_INTERFACE_LIBRARY}\
      ${MKL_THREADING_LIBRARY}\
      ${MKL_CORE_LIBRARY}\
      ${MKL_ONEMKL_LIBRARY}\
      -Wl,--end-group -lsycl -lOpenCL -ldl -lpthread -lm")

else(NWPW_SYCL)

  set(MKL_LIBRARY "-Wl,--start-group\
      ${MKL_INTERFACE_LIBRARY}\
      ${MKL_THREADING_LIBRARY}\
      ${MKL_CORE_LIBRARY}\
      ${MKL_FFT_LIBRARY}\
      ${MKL_BLACS_LIBRARY}\
      ${MKL_SCALAPACK_LIBRARY}\
      ${MKL_RTL_LIBRARY}\
      -Wl,--end-group -ldl -lpthread -lm")

  set(MKL_LINK_FLAGS "-Wl,--start-group\
      ${MKL_INTERFACE_LIBRARY}\
      ${MKL_THREADING_LIBRARY}\
      ${MKL_CORE_LIBRARY}\
      -Wl,--end-group -ldl -lpthread -lm")

endif(NWPW_SYCL)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

find_package_handle_standard_args(MKL DEFAULT_MSG
    MKL_INCLUDE_DIR MKL_LIBRARY )

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_LIBRARY})

    add_definitions(-DNWPW_INTEL_MKL)
    message("***************************************************MKL LIBS:\n\n${MKL_LIBRARIES}")

    message("**** MKL CMAKE VARABLES  *****, MKL_ROOT = ${MKL_ROOT}")
    message("**** -- MKL_INTERFACE_LAYER = ${MKL_INTERFACE_LAYER}")
    message("**** -- MKL_ARCH = ${MKL_ARCH}")
    message("**** -- _MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES = ${_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES}")
    message("**** -- CMAKE_FIND_LIBRARY_SUFFIXES = ${CMAKE_FIND_LIBRARY_SUFFIXES}")
    message("**** -- MKL_INTERFACE_LIBNAME = ${MKL_INTERFACE_LIBNAME}")
    message("**** -- MKL_MULTI_THREADED    = ${MKL_MULTI_THREADED}")
    message("**** -- MKL_THREADING_LIBRARY = ${MKL_THREADING_LIBRARY}")
    message("**** -- MKL_CORE_LIBRARY      = ${MKL_CORE_LIBRARY}")
    message("**** -- MKL_SCALAPACK_LIBRARY = ${MKL_SCALAPACK_LIBRARY}")
    message("**** -- MKL_BLACS_LIBRARY     = ${MKL_BLACS_LIBRARY}")
    message("**** -- MKL_LIBRARIES         = ${MKL_LIBRARIES}")
    message("**** -- MKL_INCLUDE_DIRS      = ${MKL_INCLUDE_DIRS}")
    message("**** -- MKL_FFTLIBRARY        = ${MKL_FFT_LIBRARY}")
    message("**** -- MKL_LINK_FLAGS        = ${MKL_LINK_FLAGS}")

    if(NWPW_SYCL)
      message("**** -- MKL_ONEMKL_FLAGS        = ${MKL_ONEMKL_LIBRARY}")
    endif(NWPW_SYCL)

    message("**** END MKL CMAKE VARABLES  *****")
endif(MKL_FOUND)

endif(MKL_ROOT)
