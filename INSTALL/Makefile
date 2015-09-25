include ../make.inc

all:  testdmach testsmach testtimer #install.out

testdmach: dmach.o lsame.o dmachtst.o
	$(LOADER) $(LOADOPTS) -o testdmach dmach.o lsame.o dmachtst.o

testsmach: smach.o lsame.o smachtst.o
	$(LOADER) $(LOADOPTS) -o testsmach smach.o lsame.o smachtst.o

testtimer: superlu_timer.o timertst.o
	$(LOADER) $(LOADOPTS) -o testtimer superlu_timer.o timertst.o \
	$(MPILIB)

install.out:
	@echo Testing machines parameters and timer
	csh install.csh

smach.o: ../SRC/smach.c ; $(CC) $(NOOPTS) -c $<
dmach.o: ../SRC/dmach.c ; $(CC) $(NOOPTS) -c $<
superlu_timer.o: superlu_timer.c; $(CC) $(NOOPTS) -c $<

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o test* *.out
