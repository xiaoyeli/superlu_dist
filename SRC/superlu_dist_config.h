/* superlu_dist_config.h.in */

/* Enable CUDA */
#define HAVE_CUDA TRUE

/* Enable parmetis */
#define HAVE_PARMETIS TRUE

/* Enable LAPACK */
#define SLU_HAVE_LAPACK TRUE

/* Enable CombBLAS */
/* #undef HAVE_COMBBLAS */

/* enable 64bit index mode */
#define XSDK_INDEX_SIZE 64

#if (XSDK_INDEX_SIZE == 64)
#define _LONGINT 1
#endif
