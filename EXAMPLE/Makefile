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

SEXM	= psdrive.o screate_matrix.o dcreate_matrix.o screate_A_x_b.o \
	  psgssvx_tracking.o psgsrfs_tracking.o 
# psgssvx_d2.o psgsrfs_d2.o sutil_dist.o psgsequb.o

DEXM	= pddrive.o dcreate_matrix.o
	# pdgstrs.o pdgsrfs.o pdgstrs_lsum.o
DEXMRHS	= pddrive_rhs.o dcreate_matrix_rhs.o #sp_ienv.o 
       # pdgstrs.o pdgsrfs.o pdgstrs_lsum.o

SEXM1	= psdrive1.o screate_matrix.o
SEXM2	= psdrive2.o screate_matrix.o screate_matrix_perturbed.o
SEXM3	= psdrive3.o screate_matrix.o
SEXM4	= psdrive4.o screate_matrix.o
SEXMG	= psdrive_ABglobal.o
SEXMG1	= psdrive1_ABglobal.o
SEXMG2	= psdrive2_ABglobal.o
SEXMG3	= psdrive3_ABglobal.o
SEXMG4	= psdrive4_ABglobal.o

DEXM1	= pddrive1.o dcreate_matrix.o
DEXM2	= pddrive2.o dcreate_matrix.o dcreate_matrix_perturbed.o
DEXM3	= pddrive3.o dcreate_matrix.o
DEXM4	= pddrive4.o dcreate_matrix.o

DEXM3D	= pddrive3d.o dcreate_matrix.o dcreate_matrix3d.o 
DEXM3D1	= pddrive3d1.o dcreate_matrix.o dcreate_matrix3d.o 
DEXM3D2	= pddrive3d2.o dcreate_matrix.o dcreate_matrix3d.o
DEXM3D3	= pddrive3d3.o dcreate_matrix.o dcreate_matrix3d.o

#	   dtrfAux.o dtreeFactorization.o treeFactorization.o pd3dcomm.o superlu_grid3d.o pdgstrf3d.o
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
ZEXM3D	= pzdrive3d.o zcreate_matrix.o zcreate_matrix3d.o
ZEXM3D1	= pzdrive3d1.o zcreate_matrix.o zcreate_matrix3d.o 
ZEXM3D2	= pzdrive3d2.o zcreate_matrix.o zcreate_matrix3d.o
ZEXM3D3	= pzdrive3d3.o zcreate_matrix.o zcreate_matrix3d.o

ZEXMG	= pzdrive_ABglobal.o
ZEXMG1	= pzdrive1_ABglobal.o
ZEXMG2	= pzdrive2_ABglobal.o
ZEXMG3	= pzdrive3_ABglobal.o
ZEXMG4	= pzdrive4_ABglobal.o

all: single double complex16

single:   psdrive \
	psdrive1 psdrive2 psdrive3 psdrive4 \
	   psdrive_ABglobal psdrive1_ABglobal psdrive2_ABglobal \
	   psdrive3_ABglobal psdrive4_ABglobal

double:    pddrive pddrive1 pddrive2 pddrive3 pddrive4 \
	   pddrive3d pddrive3d1 pddrive3d2 pddrive3d3 \
	   pddrive_ABglobal pddrive1_ABglobal pddrive2_ABglobal \
	   pddrive3_ABglobal pddrive4_ABglobal

complex16: pzdrive pzdrive1 pzdrive2 pzdrive3 pzdrive4 \
	   pzdrive3d pzdrive3d1 pzdrive3d2 pzdrive3d3 \
	   pzdrive_ABglobal pzdrive1_ABglobal pzdrive2_ABglobal \
	   pzdrive3_ABglobal pzdrive4_ABglobal

psdrive: $(SEXM) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXM) $(LIBS) -lm -o $@

psdrive1: $(SEXM1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXM1) $(LIBS) -lm -o $@

psdrive2: $(SEXM2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXM2) $(LIBS) -lm -o $@

psdrive3: $(SEXM3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXM3) $(LIBS) -lm -o $@

psdrive4: $(SEXM4) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXM4) $(LIBS) -lm -o $@

psdrive_ABglobal: $(SEXMG) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXMG) $(LIBS) -lm -o $@

psdrive1_ABglobal: $(SEXMG1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXMG1) $(LIBS) -lm -o $@

psdrive2_ABglobal: $(SEXMG2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXMG2) $(LIBS) -lm -o $@

psdrive3_ABglobal: $(SEXMG3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXMG3) $(LIBS) -lm -o $@

psdrive4_ABglobal: $(SEXMG4) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(SEXMG4) $(LIBS) -lm -o $@

pddrive: $(DEXM) $(DSUPERLULIB)

	$(LOADER) $(LOADOPTS) $(DEXM) $(LIBS) -lm -o $@

pddrive_rhs: $(DEXMRHS) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXMRHS) $(LIBS) -lm -o $@

pddrive1: $(DEXM1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM1) $(LIBS) -lm -o $@

pddrive2: $(DEXM2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM2) $(LIBS) -lm -o $@

pddrive3: $(DEXM3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM3) $(LIBS) -lm -o $@

pddrive4: $(DEXM4) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM4) $(LIBS) -lm -o $@

pddrive3d: $(DEXM3D) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM3D) $(LIBS) -lm -o $@

pddrive3d1: $(DEXM3D1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM3D1) $(LIBS) -lm -o $@

pddrive3d2: $(DEXM3D2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM3D2) $(LIBS) -lm -o $@

pddrive3d3: $(DEXM3D3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(DEXM3D3) $(LIBS) -lm -o $@

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

pzdrive3d: $(ZEXM3D) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM3D) $(LIBS) -lm -o $@

pzdrive3d1: $(ZEXM3D1) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM3D1) $(LIBS) -lm -o $@

pzdrive3d2: $(ZEXM3D2) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM3D2) $(LIBS) -lm -o $@

pzdrive3d3: $(ZEXM3D3) $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) $(ZEXM3D3) $(LIBS) -lm -o $@

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
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) -I$(INCLUDEDIR) -c $< $(VERBOSE)

.cpp.o:
	$(CPP) $(CPPFLAGS) $(CDEFS) $(BLASDEF) -I$(INCLUDEDIR) -c $< $(VERBOSE)

.f.o:
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

clean:	
	rm -f *.o p[dsz]drive p[dsz]drive[1-9] p[dsz]drive3d p[dsz]drive3d[1-9] \
	p[dsz]drive_ABglobal p[dsz]drive[1-9]_ABglobal


