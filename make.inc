############################################################################
#
#  Program:         SuperLU_DIST
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   March 1, 2016	version 5.0.0
#
#  Modified:	    
#		    
#
############################################################################
#
#  The name of the libraries to be created/linked to
#
SuperLUroot	= /global/homes/x/xiaoye/Release_Codes/superlu_dist.git/build
SUPERLULIB   	= $(SuperLUroot)/SRC/libsuperlu_dist.a

# BLASDEF 	= -DUSE_VENDOR_BLAS

LIBS		= $(SUPERLULIB) /opt/intel/parallel_studio_xe_2016.0.047/compilers_and_libraries_2016.0.109/linux/mkl/lib/intel64/libmkl_intel_lp64.so /opt/intel/parallel_studio_xe_2016.0.047/compilers_and_libraries_2016.0.109/linux/mkl/lib/intel64/libmkl_intel_thread.so /opt/intel/parallel_studio_xe_2016.0.047/compilers_and_libraries_2016.0.109/linux/mkl/lib/intel64/libmkl_core.so /opt/intel/parallel_studio_xe_2016.0.047/compilers_and_libraries_2016.0.109/linux/compiler/lib/intel64/libiomp5.so /global/homes/x/xiaoye/Edison/lib/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a /global/homes/x/xiaoye/Edison/lib/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /usr/bin/ar
ARCHFLAGS    = cr
RANLIB       = /usr/bin/ranlib

CC           = /opt/cray/craype/2.4.2/bin/cc
CFLAGS 	     = -O3 -DNDEBUG  -DUSE_VENDOR_BLAS -DPRNTlevel=0 -DDEBUGlevel=0 -DAdd_ -std=c99 
# CFLAGS       += -D
# CFLAGS       +=  
NOOPTS       = -O0
FORTRAN	     = /opt/intel/parallel_studio_xe_2016.0.047/compilers_and_libraries_2016.0.109/linux/bin/intel64/ifort

LOADER       = $(CC)
LOADOPTS     =
