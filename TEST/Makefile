#######################################################################
#  This makefile creates the test programs for the linear equation
#  routines in SuperLU_DIST.  The test files are grouped as follows:
#
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
#       pdtest
#       pztest
#  plus the self-checking drivers used by the CI scripts (see ptest_ci.c),
#  built from one source for each precision:
#       pstest_ci  pdtest_ci  pztest_ci        (make ci)
#
#  To remove the object files after the executable files have been
#  created, enter
#       make clean
#  On some systems, you can force the source files to be recompiled by
#  entering (for example)
#       make double FRC=FRC
#
#  Creation date:	March 16, 2017
#  Modified: 	
#######################################################################

include ../make.inc

DLINTST = pdtest.o dcreate_matrix.o pdcompute_resid.o

ZLINTST = pztest.o zcreate_matrix.o pzcompute_resid.o

all: double complex16

testmat:
	(cd MATGEN; $(MAKE))

./pdtest: $(DLINTST) $(DSUPERLULIB) $(TMGLIB)
	$(LOADER) $(LOADOPTS) $(DLINTST) $(TMGLIB) $(LIBS) -lm -o $@

./pztest: $(ZLINTST) $(DSUPERLULIB) $(TMGLIB)
	$(LOADER) $(LOADOPTS) $(ZLINTST) $(TMGLIB) $(LIBS) -lm -o $@

double: ./pdtest
complex16: ./pztest

# Self-checking CI drivers: one source, one program per precision
ci: ./pstest_ci ./pdtest_ci ./pztest_ci

pstest_ci.o: ptest_ci.c
	$(CC) $(CFLAGS) $(CDEFS) -DSLU_CI_SINGLE -I$(INCLUDEDIR) -c ptest_ci.c -o $@ $(VERBOSE)
pdtest_ci.o: ptest_ci.c
	$(CC) $(CFLAGS) $(CDEFS) -DSLU_CI_DOUBLE -I$(INCLUDEDIR) -c ptest_ci.c -o $@ $(VERBOSE)
pztest_ci.o: ptest_ci.c
	$(CC) $(CFLAGS) $(CDEFS) -DSLU_CI_COMPLEX16 -I$(INCLUDEDIR) -c ptest_ci.c -o $@ $(VERBOSE)

./pstest_ci: pstest_ci.o $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) pstest_ci.o $(LIBS) -lm -o $@
./pdtest_ci: pdtest_ci.o $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) pdtest_ci.o $(LIBS) -lm -o $@
./pztest_ci: pztest_ci.o $(DSUPERLULIB)
	$(LOADER) $(LOADOPTS) pztest_ci.o $(LIBS) -lm -o $@

.c.o:
	$(CC) $(CFLAGS) $(CDEFS) -I$(INCLUDEDIR) -c $< $(VERBOSE)

clean:	
	rm -f *.o *test *test_ci *.out

