/* superlu_dist_config.h.in */

/* Enable parmetis */
#define HAVE_PARMETIS TRUE

/* Enable CombBLAS */
/* #undef HAVE_COMBBLAS */

/* enable 64bit index mode */
/* #undef XSDK_INDEX_SIZE */

#if (XSDK_INDEX_SIZE == 64)
#define _LONGINT 1

#endif
