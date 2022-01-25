/* superlu_dist_config.h.in */

/* Enable CUDA */
<<<<<<< HEAD
/* #undef HAVE_CUDA */

/* Enable parmetis */
/* #undef HAVE_PARMETIS */
=======
#define HAVE_CUDA TRUE

/* Enable HIP */
/* #undef HAVE_HIP */

/* Enable parmetis */
#define HAVE_PARMETIS TRUE
>>>>>>> master

/* Enable LAPACK */
/* #undef SLU_HAVE_LAPACK */

/* Enable CombBLAS */
/* #undef HAVE_COMBBLAS */

/* enable 64bit index mode */
<<<<<<< HEAD
#define XSDK_INDEX_SIZE 64
=======
/* #undef XSDK_INDEX_SIZE */
>>>>>>> master

#if (XSDK_INDEX_SIZE == 64)
#define _LONGINT 1
#endif
