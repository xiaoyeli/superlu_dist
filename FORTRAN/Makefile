#######################################################################
#
#  This makefile creates the Fortran example programs for the
#  linear equation routines in SuperLU_DIST.
#
#  Creation date:   July 29, 2003   version 2.0
#  Modified:        Oct. 22, 2012   version 3.2
#
#######################################################################
.SUFFIXES: 
.SUFFIXES: .f90 .c .o
include ../make.inc
INCLUDEDIR = -I../SRC

#F90FLAGS	= $(FFLAGS) -qfree -qsuffix=f=f90  -qflag=w:w

F_MOD	= superlupara.o superlu_mod.o
C_DWRAP	= dcreate_dist_matrix.o superlu_c2f_dwrap.o
C_ZWRAP	= zcreate_dist_matrix.o superlu_c2f_zwrap.o

F_DEXM	= $(F_MOD) dhbcode1.o f_pddrive.o
F_ZEXM	= $(F_MOD) zhbcode1.o f_pzdrive.o
F_5x5 	= $(F_MOD) f_5x5.o sp_ienv.o 

all: f_pddrive f_pzdrive f_5x5

f_pddrive: $(F_DEXM) $(C_DWRAP) $(DSUPERLULIB)
	$(FORTRAN) $(LOADOPTS) $(F_DEXM) $(C_DWRAP) $(LIBS) -o $@

f_5x5: $(F_5x5) $(C_DWRAP) $(DSUPERLULIB)
	$(FORTRAN) $(LOADOPTS) $(F_5x5) $(C_DWRAP) $(LIBS) -o $@

f_pzdrive: $(F_ZEXM) $(C_ZWRAP) $(DSUPERLULIB)
	$(FORTRAN) $(LOADOPTS) $(F_ZEXM) $(C_ZWRAP) $(LIBS) -o $@

.c.o:
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c $< $(VERBOSE)

.f90.o:
	$(FORTRAN) $(F90FLAGS) -c $< $(VERBOSE)

.f.o:
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

clean:	
	rm -f *.o *.mod f_*drive f_5x5


