/* superlu_dist_config.h.in */

/* Enable CUDA */
#define HAVE_CUDA TRUE

/* Enable NVSHMEM */
/* #undef HAVE_NVSHMEM */

/* Enable HIP */
/* #undef HAVE_HIP */

/* Enable parmetis */
#define HAVE_PARMETIS TRUE

/* Enable colamd */
/* #undef HAVE_COLAMD */

/* Enable LAPACK */
#define SLU_HAVE_LAPACK TRUE

/* Enable CombBLAS */
/* #undef HAVE_COMBBLAS */

/* Enable MAGMA */
/* #undef HAVE_MAGMA */

/* enable 64bit index mode */
/* #undef XSDK_INDEX_SIZE */

#if (XSDK_INDEX_SIZE == 64)
#define _LONGINT 1
#endif
