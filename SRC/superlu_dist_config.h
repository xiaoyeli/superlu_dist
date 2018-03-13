/* superlu_dist_config.h */

/* Enable parmetis */
#define HAVE_PARMETIS TRUE

/* enable 64bit index mode */
/* #define XSDK_INDEX_SIZE 64 */

#if (XSDK_INDEX_SIZE == 64)
#define _LONGINT 1
#endif

