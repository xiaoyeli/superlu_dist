/* superlu_dist_config.h.in */

/* Enable CUDA */
/* #undef HAVE_CUDA */

/* Enable parmetis */
/* #undef HAVE_PARMETIS */

/* Enable LAPACK */
/* #undef SLU_HAVE_LAPACK */

/* Enable CombBLAS */
/* #undef HAVE_COMBBLAS */

/* enable 64bit index mode */
#define XSDK_INDEX_SIZE 64

#if (XSDK_INDEX_SIZE == 64)
#define _LONGINT 1
#endif
