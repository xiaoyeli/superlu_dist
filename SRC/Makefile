#######################################################################
#
#  This makefile creates a library for distributed SuperLU.
#  The files are organized as follows:
#
#       ALLAUX  -- Auxiliary routines called from all precisions
#       DSLUSRC -- Double precision real serial SuperLU routines
#       DPLUSRC -- Double precision real parallel SuperLU routines
#       ZSLUSRC -- Double precision complex serial SuperLU routines
#       ZPLUSRC -- Double precision complex parallel SuperLU routines
#
#  The library can be set up to include routines for any combination
#  of the two precisions.  To create or add to the library, enter make
#  followed by one or more of the precisions desired.  Some examples:
#       make double
#       make double complex16
#  Alternatively, the command
#       make
#  without any arguments creates a library of all two precisions.
#  The library is called
#       superlu.a
#  and is created at the next higher directory level.
#
#  To remove the object files after the library is created, enter
#       make clean
#
#######################################################################
include ../make.inc
#
# Precision independent routines
#
ALLAUX 	= sp_ienv.o etree.o sp_colorder.o get_perm_c.o \
	  colamd.o mmd.o comm.o memory.o util.o superlu_grid.o \
	  pxerr_dist.o superlu_timer.o symbfact.o \
	  psymbfact.o psymbfact_util.o get_perm_c_parmetis.o mc64ad_dist.o \
	  xerr_dist.o smach_dist.o dmach_dist.o \
	  superlu_dist_version.o TreeInterface.o

ifeq ($(HAVE_CUDA),TRUE)
ALLAUX += gpublas_utils.o pdgstrs_lsum_cuda.o pdgstrs_lsum_cuda_dlink.o
endif

#
# Routines literally taken from SuperLU, but renamed with suffix _dist
#
DSLUSRC	= dlangs_dist.o dgsequ_dist.o dlaqgs_dist.o dutil_dist.o \
	  dmemory_dist.o dmyblas2_dist.o dsp_blas2_dist.o dsp_blas3_dist.o
ZSLUSRC	= dcomplex_dist.o zlangs_dist.o zgsequ_dist.o zlaqgs_dist.o \
	  zutil_dist.o zmemory_dist.o zmyblas2_dist.o \
	  zsp_blas2_dist.o zsp_blas3_dist.o

#
# Routines for double precision parallel SuperLU
DPLUSRC = pdgssvx.o pdgssvx_ABglobal.o \
	  dreadhb.o dreadrb.o dreadtriple.o dreadMM.o dbinary_io.o \
	  pdgsequ.o pdlaqgs.o dldperm_dist.o pdlangs.o pdutil.o \
	  pdsymbfact_distdata.o ddistribute.o pddistribute.o \
	  pdgstrf.o dstatic_schedule.o pdgstrf2.o pdGetDiagU.o \
	  pdgstrs.o pdgstrs1.o pdgstrs_lsum.o pdgstrs_Bglobal.o \
	  pdgsrfs.o pdgsmv.o pdgsrfs_ABXglobal.o pdgsmv_AXglobal.o \
	  dreadtriple_noheader.o
#
# Routines for double complex parallel SuperLU
ZPLUSRC = pzgssvx.o pzgssvx_ABglobal.o \
	  zreadhb.o zreadrb.o zreadtriple.o zreadMM.o zbinary_io.o\
	  pzgsequ.o pzlaqgs.o zldperm_dist.o pzlangs.o pzutil.o \
	  pzsymbfact_distdata.o zdistribute.o pzdistribute.o \
	  pzgstrf.o zstatic_schedule.o pzgstrf2.o pzGetDiagU.o \
	  pzgstrs.o pzgstrs1.o pzgstrs_lsum.o pzgstrs_Bglobal.o \
	  pzgsrfs.o pzgsmv.o pzgsrfs_ABXglobal.o pzgsmv_AXglobal.o \
	  zreadtriple_noheader.o

ifeq ($(HAVE_COMBBLAS),TRUE)
DPLUSRC += d_c2cpp_GetHWPM.o
ZPLUSRC += z_c2cpp_GetHWPM.o
endif

all:  double complex16

config_h:
	rm -f superlu_dist_config.h
ifeq ($(XSDK_INDEX_SIZE),64)
		printf "#define XSDK_INDEX_SIZE 64\n" >> superlu_dist_config.h
else
		printf "/* #define XSDK_INDEX_SIZE 64 */\n" >> superlu_dist_config.h
endif
ifeq ($(SLU_HAVE_LAPACK),TRUE)
		printf "#define SLU_HAVE_LAPACK TRUE\n" >> superlu_dist_config.h
else
		printf "/* #define SLU_HAVE_LAPACK TRUE */\n" >> superlu_dist_config.h
endif
ifeq ($(HAVE_PARMETIS),TRUE)
		printf "#define HAVE_PARMETIS TRUE\n" >> superlu_dist_config.h
else
		printf "/* #define HAVE_PARMETIS TRUE */\n" >> superlu_dist_config.h
endif
ifeq ($(HAVE_COMBBLAS),TRUE)
		printf "#define HAVE_COMBBLAS TRUE\n" >> superlu_dist_config.h
else
		printf "/* #define HAVE_COMBBLAS TRUE */\n" >> superlu_dist_config.h
endif
	printf "#if (XSDK_INDEX_SIZE == 64)\n#define _LONGINT 1\n#endif\n" >> superlu_dist_config.h


#config_h:
#	cp -f superlu_dist_config_for_makefile.h superlu_dist_config.h

double: config_h $(DSLUSRC) $(DPLUSRC) $(ALLAUX)
	$(ARCH) $(ARCHFLAGS) $(DSUPERLULIB) \
		$(DSLUSRC) $(DPLUSRC) $(ALLAUX)
	$(RANLIB) $(DSUPERLULIB)

complex16: config_h $(ZSLUSRC) $(ZPLUSRC) $(ALLAUX)
	$(ARCH) $(ARCHFLAGS) $(DSUPERLULIB) \
		$(ZSLUSRC) $(ZPLUSRC) $(ALLAUX)
	$(RANLIB) $(DSUPERLULIB)

pdgstrf.o: dscatter.c dlook_ahead_update.c dSchCompUdt-2Ddynamic.c pdgstrf.c
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c pdgstrf.c $(VERBOSE)

pzgstrf.o: zscatter.c zlook_ahead_update.c zSchCompUdt-2Ddynamic.c pzgstrf.c
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c pzgstrf.c $(VERBOSE)

.c.o:
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c $< $(VERBOSE)

pdgstrs_lsum_cuda.o: pdgstrs_lsum_cuda.cu
	$(NVCC) $(CUDACFLAGS) -dc $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c pdgstrs_lsum_cuda.cu $(VERBOSE)

pdgstrs_lsum_cuda_dlink.o:	
	$(NVCC) $(CUDACFLAGS) -dlink $(CUDALIBS) -o pdgstrs_lsum_cuda_dlink.o pdgstrs_lsum_cuda.o $(VERBOSE)	
	
	
.cpp.o:
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c $< $(VERBOSE)

.f.o:
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

clean:	
	rm -f *.o $(DSUPERLULIB) superlu_dist_config.h
