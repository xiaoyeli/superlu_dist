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

#F90FLAGS	= $(FFLAGS) -qfree -qsuffix=f=f90  -qflag=w:w

F_MOD	= superlupara.o superlu_mod.o
C_DWRAP	= c2f_dcreate_matrix_x_b.o superlu_c2f_dwrap.o #dcreate_dist_matrix.o
C_ZWRAP	= c2f_zcreate_matrix_x_b.o superlu_c2f_zwrap.o

F_DEXM	= $(F_MOD) f_pddrive.o
F_DEXM3D= $(F_MOD) f_pddrive3d.o
F_ZEXM	= $(F_MOD) f_pzdrive.o
F_ZEXM3D= $(F_MOD) f_pzdrive3d.o
F_5x5 	= $(F_MOD) f_5x5.o sp_ienv.o


all: f_pddrive f_pddrive3d f_pzdrive f_pzdrive3d f_5x5

f_5x5: $(F_5x5) $(C_DWRAP) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(F_5x5) $(C_DWRAP) $(LIBS) -o $@

f_pddrive: $(F_DEXM) $(C_DWRAP) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(F_DEXM) $(C_DWRAP) $(LIBS) -o $@

f_pddrive3d: $(F_DEXM3D) $(C_DWRAP) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(F_DEXM3D) $(C_DWRAP) $(LIBS) -o $@

f_pzdrive: $(F_ZEXM) $(C_ZWRAP) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(F_ZEXM) $(C_ZWRAP) $(LIBS) -o $@

f_pzdrive3d: $(F_ZEXM3D) $(C_ZWRAP) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(F_ZEXM3D) $(C_ZWRAP) $(LIBS) -o $@

.c.o:
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) -I$(INCLUDEDIR) -c $< $(VERBOSE)

.f90.o:
	$(FORTRAN) $(F90FLAGS) -c $< $(VERBOSE)

.f.o:
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

clean:	
	rm -f *.o *.mod f_*drive f_5x5


