/* superlu_dist_config.h.in */

/* Enable CUDA */
/* #undef HAVE_CUDA */

/* Enable HIP */
/* #undef HAVE_HIP */

/* Enable SYCL */
#define HAVE_SYCL TRUE

/* Enable DPCPP */
#define HAVE_DPCPP TRUE

/* Enable parmetis */
/* #undef HAVE_PARMETIS */

/* Enable colamd */
/* #undef HAVE_COLAMD */

/* Enable LAPACK */
/* #undef SLU_HAVE_LAPACK */

/* Enable CombBLAS */
/* #undef HAVE_COMBBLAS */

/* enable 64bit index mode */
#define XSDK_INDEX_SIZE 64

#if defined(XSDK_INDEX_SIZE) && (XSDK_INDEX_SIZE == 64)
#define _LONGINT 1
#endif
