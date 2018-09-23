############################################################################
#
#  Program:         SuperLU_DIST
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   February 4, 1999   version alpha
#
#  Modified:	    September 1, 1999  version 1.0
#                   March 15, 2003     version 2.0
#		    November 1, 2007   version 2.1
#
############################################################################
#
#  The machine (platform) identifier to append to the library names
#
PLAT		= _sp

#
#  The name of the libraries to be created/linked to
#
SuperLUroot 	= ${HOME}/Release_Codes/SuperLU_DIST_4.2
DSUPERLULIB   	= $(SuperLUroot)/lib/libsuperlu_dist_4.2.a
INCLUDEDIR   	= $(SuperLUroot)/SRC
#
BLASDEF	     	= -DUSE_VENDOR_BLAS
BLASLIB      	= -lessl
LAPACKLIB	= 
#MPILIB		= -L/usr/lpp/ppe.poe/lib -lmpi
#PERFLIB     	= -L/vol1/VAMPIR/lib -lVT

############################################################################
## parmetis 4.x.x, 32-bit integer
PARMETIS_DIR	:= ${HOME}/Carver/lib/parmetis-4.0.3
## parmetis 4.x.x, 64-bit integer
# PARMETIS_DIR	:= ${HOME}/Carver/lib/parmetis-4.0.3_64

METISLIB := -L${PARMETIS_DIR}/build/Linux-x86_64/libmetis -lmetis
PARMETISLIB := -L${PARMETIS_DIR}/build/Linux-x86_64/libparmetis -lparmetis
I_PARMETIS := -I${PARMETIS_DIR}/include -I${PARMETIS_DIR}/metis/include
############################################################################

# Define the required Fortran libraries, if you use C compiler to link
FLIBS	 	=

# Define all the libraries
LIBS            = $(DSUPERLULIB) $(BLASLIB) $(PARMETISLIB) $(METISLIB) \
		  $(LAPACKLIB) $(FLIBS)

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         	= ar
ARCHFLAGS    	= cr
RANLIB       	= ranlib

############################################################################
CC           	= mpcc
# CFLAGS should be set to be the C flags that include optimization
CFLAGS          = -D_SP -O3 -qarch=PWR3 -qalias=allptrs \
		  -DDEBUGlevel=0 -DPRNTlevel=0 $(I_PARMETIS)
#
# NOOPTS should be set to be the C flags that turn off any optimization
# This must be enforced to compile the two routines: slamch.c and dlamch.c.
NOOPTS		=
############################################################################
FORTRAN         = mpxlf90
F90FLAGS        = -WF,-Dsp -O3 -Q -qstrict -qfixed -qinit=f90ptr -qarch=pwr3
############################################################################
LOADER	        = mpxlf90
#LOADOPTS	= -bmaxdata:0x80000000
LOADOPTS	= -bmaxdata:0x70000000
#
############################################################################
#  C preprocessor defs for compilation (-DNoChange, -DAdd_, or -DUpCase)
#
#  Need follow the convention of how C calls a Fortran routine.
#
CDEFS        = -DNoChange

