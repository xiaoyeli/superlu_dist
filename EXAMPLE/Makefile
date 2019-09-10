#######################################################################
#
#  This makefile creates the example programs for the linear equation
#  routines in SuperLU_DIST.
#
#  The command
#       make
#  without any arguments creates all the example programs.
#  The command
# 	make double
#  creates double precision real example programs.
#  The command
#       make complex16
#  creates double precision complex example programs.
#
#  The executable files are called
#       double real:	pddrive pddrive_ABglobal pddrive1
#                       pddrive1_ABglobal pddrive2 pddrive3 pddrive4
#	double complex: pzdrive pzdrive_ABglobal pzdrive1
#                       pzdrive1_ABglobal pzdrive2 pzdrive3 pzdrive4 
#
#  Alternatively, you can create example programs individually by
#  typing the command (for example)
#	make pddrive
#
#  To remove the object files after the executable files have been
#  created, enter
#       make clean
#
#######################################################################
include ../make.inc

DEXM	= pddrive.o dcreate_matrix.o sp_ienv.o #pdgstrf2.o 
#pdgssvx.o
# pdgstrs_lsum_X1.o pdgstrf_X1.o
DEXM1	= pddrive1.o dcreate_matrix.o
DEXM2	= pddrive2.o dcreate_matrix.o dcreate_matrix_perturbed.o
DEXM3	= pddrive3.o dcreate_matrix.o
DEXM4	= pddrive4.o dcreate_matrix.o
DEXMG	= pddrive_ABglobal.o
DEXMG1	= pddrive1_ABglobal.o
DEXMG2	= pddrive2_ABglobal.o
DEXMG3	= pddrive3_ABglobal.o
DEXMG4	= pddrive4_ABglobal.o
ZEXM	= pzdrive.o zcreate_matrix.o
	#pzgstrf2.o pzgstrf_v3.3.o pzgstrf.o
ZEXM1	= pzdrive1.o zcreate_matrix.o
ZEXM2	= pzdrive2.o zcreate_matrix.o zcreate_matrix_perturbed.o
ZEXM3	= pzdrive3.o zcreate_matrix.o
ZEXM4	= pzdrive4.o zcreate_matrix.o
ZEXMG	= pzdrive_ABglobal.o
ZEXMG1	= pzdrive1_ABglobal.o
ZEXMG2	= pzdrive2_ABglobal.o
ZEXMG3	= pzdrive3_ABglobal.o
ZEXMG4	= pzdrive4_ABglobal.o


all: double complex16

double:    pddrive pddrive1 pddrive2 pddrive3 pddrive4 \
	   pddrive_ABglobal pddrive1_ABglobal pddrive2_ABglobal \
	   pddrive3_ABglobal pddrive4_ABglobal

complex16: pzdrive pzdrive1 pzdrive2 pzdrive3 pzdrive4 \
	   pzdrive_ABglobal pzdrive1_ABglobal pzdrive2_ABglobal \
	   pzdrive3_ABglobal pzdrive4_ABglobal

pddrive: $(DEXM) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM) $(LIBS) -lm -o $@

pddrive1: $(DEXM1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM1) $(LIBS) -lm -o $@

pddrive2: $(DEXM2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM2) $(LIBS) -lm -o $@

pddrive3: $(DEXM3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM3) $(LIBS) -lm -o $@

pddrive4: $(DEXM4) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM4) $(LIBS) -lm -o $@

pddrive_ABglobal: $(DEXMG) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXMG) $(LIBS) -lm -o $@

pddrive1_ABglobal: $(DEXMG1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXMG1) $(LIBS) -lm -o $@

pddrive2_ABglobal: $(DEXMG2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXMG2) $(LIBS) -lm -o $@

pddrive3_ABglobal: $(DEXMG3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXMG3) $(LIBS) -lm -o $@

pddrive4_ABglobal: $(DEXMG4) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXMG4) $(LIBS) -lm -o $@

pzdrive: $(ZEXM) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM) $(LIBS) -lm -o $@

pzdrive_triple: $(ZEXM) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM) $(LIBS) -lm -o $@

pzdrive1: $(ZEXM1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM1) $(LIBS) -lm -o $@

pzdrive2: $(ZEXM2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM2) $(LIBS) -lm -o $@

pzdrive3: $(ZEXM3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM3) $(LIBS) -lm -o $@

pzdrive4: $(ZEXM4) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM4) $(LIBS) -lm -o $@

pzdrive_ABglobal: $(ZEXMG) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXMG) $(LIBS) -lm -o $@

pzdrive1_ABglobal: $(ZEXMG1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXMG1) $(LIBS) -lm -o $@

pzdrive2_ABglobal: $(ZEXMG2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXMG2) $(LIBS) -lm -o $@

pzdrive3_ABglobal: $(ZEXMG3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXMG3) $(LIBS) -lm -o $@

pzdrive4_ABglobal: $(ZEXMG4) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXMG4) $(LIBS) -lm -o $@

#pdgstrf.o: dscatter.c dSchCompUdt-cuda.c pdgstrf.c
#	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) -I$(INCLUDEDIR) -c pdgstrf.c $(VERBOSE)
.c.o:
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c $< $(VERBOSE)

.cpp.o:
	$(CPP) $(CPPFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c $< $(VERBOSE)

.f.o:
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

clean:	
	rm -f *.o p[dz]drive p[dz]drive[1-9] \
	p[dz]drive_ABglobal p[dz]drive[1-9]_ABglobal


