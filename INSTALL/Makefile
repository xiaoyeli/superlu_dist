include ../make.inc

all:  testdmach testsmach testtimer #install.out

testdmach: dmach_dist.o dmachtst.o
	$(LOADER) $(LOADOPTS) -o testdmach dmach_dist.o dmachtst.o

testsmach: smach_dist.o smachtst.o
	$(LOADER) $(LOADOPTS) -o testsmach smach_dist.o smachtst.o

testtimer: superlu_timer.o timertst.o
	$(LOADER) $(LOADOPTS) -o testtimer superlu_timer.o timertst.o

install.out:
	@echo Testing machines parameters and timer
	csh install.csh

smach_dist.o: ../SRC/smach_dist.c ; $(CC) -c $<
dmach_dist.o: ../SRC/dmach_dist.c ; $(CC) -c $<
superlu_timer.o: ../SRC/superlu_timer.c ; $(CC) -c $<

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o test* *.out
