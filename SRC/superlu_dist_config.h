/* superlu_dist_config.h.in */

/* Enable CUDA */
/* #undef HAVE_CUDA */

/* Enable NVSHMEM */
/* #undef HAVE_NVSHMEM */

/* Enable HIP */
#define HAVE_HIP TRUE

/* Enable parmetis */
#define HAVE_PARMETIS TRUE

/* Enable colamd */
/* #undef HAVE_COLAMD */

/* Enable LAPACK */
/* #undef SLU_HAVE_LAPACK */

/* Enable CombBLAS */
/* #undef HAVE_COMBBLAS */

/* Enable MAGMA */
/* #undef HAVE_MAGMA */

/* enable 64bit index mode */
/* #undef XSDK_INDEX_SIZE */

#if (XSDK_INDEX_SIZE == 64)
#define _LONGINT 1
#endif
