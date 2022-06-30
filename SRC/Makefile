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

# FACT3D = 	scatter.o
#	pdgstrs_vecpar.o ancFactorization.o

#  pddrive_params.o

#
# Precision independent routines
#
ALLAUX 	= sp_ienv.o etree.o sp_colorder.o get_perm_c.o \
	  colamd.o mmd.o comm.o memory.o util.o gpu_api_utils.o superlu_grid.o \
	  pxerr_dist.o superlu_timer.o symbfact.o psymbfact.o psymbfact_util.o \
	  get_perm_c_parmetis.o mc64ad_dist.o xerr_dist.o smach_dist.o dmach_dist.o \
	  superlu_dist_version.o comm_tree.o

# Following are from 3D code
ALLAUX += superlu_grid3d.o supernodal_etree.o supernodalForest.o \
	trfAux.o communication_aux.o treeFactorization.o sec_structs.o
#
# Routines literally taken from SuperLU, but renamed with suffix _dist
#
SSLUSRC	= slangs_dist.o sgsequ_dist.o slaqgs_dist.o sutil_dist.o \
	  smemory_dist.o smyblas2_dist.o ssp_blas2_dist.o ssp_blas3_dist.o
DSLUSRC	= dlangs_dist.o dgsequ_dist.o dlaqgs_dist.o dutil_dist.o \
	  dmemory_dist.o dmyblas2_dist.o dsp_blas2_dist.o dsp_blas3_dist.o
ZSLUSRC	= dcomplex_dist.o zlangs_dist.o zgsequ_dist.o zlaqgs_dist.o \
	  zutil_dist.o zmemory_dist.o zmyblas2_dist.o zsp_blas2_dist.o zsp_blas3_dist.o
#
# Routines for single precision parallel SuperLU
SPLUSRC = psgssvx.o psgssvx_d2.o psgssvx_ABglobal.o \
	  sreadhb.o sreadrb.o sreadtriple.o sreadtriple_noheader.o sreadMM.o sbinary_io.o \
	  psgsequ.o pslaqgs.o sldperm_dist.o pslangs.o psutil.o \
	  pssymbfact_distdata.o sdistribute.o psdistribute.o \
	  psgstrf.o sstatic_schedule.o psgstrf2.o psGetDiagU.o \
	  psgstrs.o psgstrs1.o psgstrs_lsum.o psgstrs_Bglobal.o \
	  psgsrfs.o psgsmv.o psgsrfs_ABXglobal.o psgsmv_AXglobal.o ssuperlu_blas.o \
	  psgsrfs_d2.o psgsmv_d2.o psgsequb.o
# from 3D code
SPLUSRC += psgssvx3d.o snrformat_loc3d.o psgstrf3d.o streeFactorization.o \
	streeFactorizationGPU.o sscatter3d.o sgather.o ps3dcomm.o strfAux.o \
	scommunication_aux.o strfCommWrapper.o
	  
#
# Routines for double precision parallel SuperLU
DPLUSRC = pdgssvx.o pdgssvx_ABglobal.o \
	  dreadhb.o dreadrb.o dreadtriple.o dreadtriple_noheader.o dreadMM.o dbinary_io.o \
	  pdgsequ.o pdlaqgs.o dldperm_dist.o pdlangs.o pdutil.o \
	  pdsymbfact_distdata.o ddistribute.o pddistribute.o \
	  pdgstrf.o dstatic_schedule.o pdgstrf2.o pdGetDiagU.o \
	  pdgstrs.o pdgstrs1.o pdgstrs_lsum.o pdgstrs_Bglobal.o \
	  pdgsrfs.o pdgsmv.o pdgsrfs_ABXglobal.o pdgsmv_AXglobal.o dsuperlu_blas.o
# from 3D code
DPLUSRC += pdgssvx3d.o dnrformat_loc3d.o pdgstrf3d.o dtreeFactorization.o \
	dtreeFactorizationGPU.o dscatter3d.o dgather.o pd3dcomm.o dtrfAux.o \
	dcommunication_aux.o dtrfCommWrapper.o

#
# Routines for double complex parallel SuperLU
ZPLUSRC = pzgssvx.o pzgssvx_ABglobal.o \
	  zreadhb.o zreadrb.o zreadtriple.o zreadMM.o zreadtriple_noheader.o zbinary_io.o\
	  pzgsequ.o pzlaqgs.o zldperm_dist.o pzlangs.o pzutil.o \
	  pzsymbfact_distdata.o zdistribute.o pzdistribute.o \
	  pzgstrf.o zstatic_schedule.o pzgstrf2.o pzGetDiagU.o \
	  pzgstrs.o pzgstrs1.o pzgstrs_lsum.o pzgstrs_Bglobal.o \
	  pzgsrfs.o pzgsmv.o pzgsrfs_ABXglobal.o pzgsmv_AXglobal.o zsuperlu_blas.o
# from 3D code
ZPLUSRC += pzgssvx3d.o znrformat_loc3d.o pzgstrf3d.o ztreeFactorization.o \
	ztreeFactorizationGPU.o zscatter3d.o zgather.o pz3dcomm.o ztrfAux.o \
	zcommunication_aux.o ztrfCommWrapper.o

ifeq ($(HAVE_CUDA),TRUE)
ALLAUX += superlu_gpu_utils.o 
SPLUSRC += ssuperlu_gpu.o
DPLUSRC += dsuperlu_gpu.o pdgstrs_lsum_cuda.o
ZPLUSRC += zsuperlu_gpu.o
endif

ifeq ($(HAVE_COMBBLAS),TRUE)
SPLUSRC += s_c2cpp_GetHWPM.o sHWPM_CombBLAS.hpp
DPLUSRC += d_c2cpp_GetHWPM.o dHWPM_CombBLAS.hpp
ZPLUSRC += z_c2cpp_GetHWPM.o zHWPM_CombBLAS.hpp
endif

all:  single double complex16

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

single: config_h $(SSLUSRC) $(SPLUSRC) $(ALLAUX)
	$(ARCH) $(ARCHFLAGS) $(DSUPERLULIB) \
		$(SSLUSRC) $(SPLUSRC) $(ALLAUX)
	$(RANLIB) $(DSUPERLULIB)

double: config_h $(DSLUSRC) $(DPLUSRC) $(ALLAUX)
	$(ARCH) $(ARCHFLAGS) $(DSUPERLULIB) \
		$(DSLUSRC) $(DPLUSRC) $(ALLAUX)
	$(RANLIB) $(DSUPERLULIB)

complex16: config_h $(ZSLUSRC) $(ZPLUSRC) $(ALLAUX)
	$(ARCH) $(ARCHFLAGS) $(DSUPERLULIB) \
		$(ZSLUSRC) $(ZPLUSRC) $(ALLAUX)
	$(RANLIB) $(DSUPERLULIB)

psgstrf.o: sscatter.c slook_ahead_update.c sSchCompUdt-2Ddynamic.c psgstrf.c
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) -I$(INCLUDEDIR) -c psgstrf.c $(VERBOSE)

pdgstrf.o: dscatter.c dlook_ahead_update.c dSchCompUdt-2Ddynamic.c pdgstrf.c
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c pdgstrf.c $(VERBOSE)

pzgstrf.o: zscatter.c zlook_ahead_update.c zSchCompUdt-2Ddynamic.c pzgstrf.c
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c pzgstrf.c $(VERBOSE)

.c.o:
	$(CC) $(CFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c $< $(VERBOSE)

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(CDEFS) $(BLASDEF) -I$(INCLUDEDIR) -c $< $(VERBOSE)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CDEFS) $(BLASDEF) $(INCLUDEDIR) -c $< $(VERBOSE)

.f.o:
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

clean:	
	rm -f *.o $(DSUPERLULIB) superlu_dist_config.h
