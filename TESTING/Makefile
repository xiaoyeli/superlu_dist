#######################################################################
#  This makefile creates the test programs for the linear equation
#  routines in SuperLU_DIST.  The test files are grouped as follows:
#
#       ALINTST -- Auxiliary test routines
#       DLINTST -- Double precision real test routines
#       ZLINTST -- Double precision complex test routines
#
#  Test programs can be generated for all or some of the two different
#  precisions.  Enter make followed by one or more of the data types
#  desired.  Some examples:
#       make complex16
#  Alternatively, the command
#       make
#  without any arguments creates all two test programs.
#  The executable files are called
#       dtest
#       ztest
#
#  To remove the object files after the executable files have been
#  created, enter
#       make clean
#  On some systems, you can force the source files to be recompiled by
#  entering (for example)
#       make double FRC=FRC
#
#
#  Creation date:	March 16, 2017
#  Modified: 	
#######################################################################

include ../make.inc
HEADER  = ../SRC

DLINTST = pdtest.o dcreate_matrix.o pdcompute_resid.o
#sp_dconvert.o dgst01.o dgst02.o dgst04.o dgst07.o 

ZLINTST = zdrive.o #sp_zconvert.o zgst01.o zgst02.o zgst04.o zgst07.o

all: pdtest #testmat double complex16

testmat:
	(cd MATGEN; $(MAKE))

double: ./dtest #dtest.out

./pdtest: $(DLINTST) $(DSUPERLULIB) $(TMGLIB)
	$(LOADER) $(LOADOPTS) $(DLINTST) $(TMGLIB) $(LIBS) -lm -o $@

dtest.out: dtest dtest.csh
	@echo Testing DOUBLE PRECISION linear equation routines 
	csh dtest.csh

complex: ./ctest ctest.out

./ctest: $(CLINTST) $(ALINTST) $(SUPERLULIB) $(TMGLIB)
	$(LOADER) $(LOADOPTS) $(CLINTST) $(ALINTST) \
	$(TMGLIB) $(LIBS) -lm -o $@

ctest.out: ctest ctest.csh
	@echo Testing SINGLE COMPLEX linear equation routines 
	csh ctest.csh

complex16: ./ztest ztest.out

./ztest: $(ZLINTST) $(ALINTST) $(SUPERLULIB) $(TMGLIB)
	$(LOADER) $(LOADOPTS) $(ZLINTST) $(ALINTST) \
	$(TMGLIB) $(LIBS) -lm -o $@

ztest.out: ztest ztest.csh
	@echo Testing DOUBLE COMPLEX linear equation routines 
	csh ztest.csh

##################################
# Do not optimize this routine   #
##################################
dlamch.o: dlamch.c ; $(CC) -c $<

timer.o:  timer.c ; $(CC) -O -c $<

.c.o:
	$(CC) $(CFLAGS) $(CDEFS) -I$(HEADER) -c $< $(VERBOSE)

clean:	
	rm -f *.o *test *.out

