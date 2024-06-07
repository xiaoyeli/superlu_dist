#ifndef __SUPERLU_BATCH_WRAPPERS_H__
#define __SUPERLU_BATCH_WRAPPERS_H__

#include "superlu_sdefs.h"
#include "superlu_ddefs.h"
#include "superlu_zdefs.h"

#ifdef HAVE_MAGMA
#include "magma.h"
#define BatchDim_t magma_int_t

inline void convertType(magmaDoubleComplex& a, doublecomplex& b)
{ a.x = b.r; a.y = b.i; }

////////////////////////////////////////////////////////////////////////////////////
// MAGMA TRSM external defs
////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magmablas_strsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);

extern "C" void
magmablas_dtrsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);
    
extern "C" void
magmablas_ztrsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **dA_array, magma_int_t* ldda,
    magmaDoubleComplex **dB_array, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue );

////////////////////////////////////////////////////////////////////////////////////
// MAGMA TRSM wrappers
////////////////////////////////////////////////////////////////////////////////////
inline void magmablas_trsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue)
{ 
    magmablas_strsm_vbatched_nocheck(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue); 
}

inline void magmablas_trsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue)
{ 
    magmablas_dtrsm_vbatched_nocheck(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue); 
}

inline void magmablas_trsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    doublecomplex alpha,
    doublecomplex **dA_array, magma_int_t* ldda,
    doublecomplex **dB_array, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue )
{ 
    magmaDoubleComplex magma_alpha;
    convertType(magma_alpha, alpha);
    
    magmablas_ztrsm_vbatched_nocheck(
        side, uplo, transA, diag, m, n, magma_alpha, (magmaDoubleComplex**)dA_array, ldda, 
        (magmaDoubleComplex**)dB_array, lddb, batchCount, queue
    ); 
}

////////////////////////////////////////////////////////////////////////////////////
// MAGMA GETRF wrappers
////////////////////////////////////////////////////////////////////////////////////
inline magma_int_t
magma_getrf_nopiv_vbatched(
    magma_int_t* m, magma_int_t* n,
    float **dA_array, magma_int_t *ldda,
    float *dtol_array, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{ 
    return magma_sgetrf_nopiv_vbatched(m, n, dA_array, ldda, dtol_array, info_array, batchCount, queue); 
}

inline magma_int_t
magma_getrf_nopiv_vbatched(
    magma_int_t* m, magma_int_t* n,
    double **dA_array, magma_int_t *ldda,
    double *dtol_array, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{ 
    return magma_dgetrf_nopiv_vbatched(m, n, dA_array, ldda, dtol_array, info_array, batchCount, queue); 
}

inline magma_int_t
magma_getrf_nopiv_vbatched(
    magma_int_t* m, magma_int_t* n,
    doublecomplex **dA_array, magma_int_t *ldda,
    double *dtol_array, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{ 
    return magma_zgetrf_nopiv_vbatched(m, n, (magmaDoubleComplex **)dA_array, ldda, dtol_array, info_array, batchCount, queue); 
}

////////////////////////////////////////////////////////////////////////////////////
// MAGMA GEMM wrappers
////////////////////////////////////////////////////////////////////////////////////
inline void
magmablas_gemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue )
{ 
    magmablas_sgemm_vbatched_max_nocheck(
        transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, 
        dC_array, lddc, max_m, max_n, max_k, batchCount, queue
    ); 
}

inline void
magmablas_gemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue )
{ 
    magmablas_dgemm_vbatched_max_nocheck(
        transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, 
        dC_array, lddc, max_m, max_n, max_k, batchCount, queue
    ); 
}

inline void
magmablas_gemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    doublecomplex alpha,
    doublecomplex const * const * dA_array, magma_int_t* ldda,
    doublecomplex const * const * dB_array, magma_int_t* lddb,
    doublecomplex beta,
    doublecomplex **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue )
{ 
    magmaDoubleComplex magma_alpha, magma_beta;
    convertType(magma_alpha, alpha);
    convertType(magma_beta, beta);

    magmablas_zgemm_vbatched_max_nocheck(
        transA, transB, m, n, k, magma_alpha, (magmaDoubleComplex const * const * )dA_array, ldda, 
        (magmaDoubleComplex const * const * )dB_array, lddb, magma_beta, 
        (magmaDoubleComplex **)dC_array, lddc, max_m, max_n, max_k, batchCount, queue
    ); 
}

#else 
// Can define batched functions to be for loops over standard library calls when magma is not used
#define BatchDim_t int_t
#endif

inline void pconvert_flatten_skyline2UROWDATA(superlu_dist_options_t *options, gridinfo_t *grid, sLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{ psconvert_flatten_skyline2UROWDATA(options, grid, LUstruct, stat, n); }
inline void pconvert_flatten_skyline2UROWDATA(superlu_dist_options_t *options, gridinfo_t *grid, dLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{ pdconvert_flatten_skyline2UROWDATA(options, grid, LUstruct, stat, n); }
inline void pconvert_flatten_skyline2UROWDATA(superlu_dist_options_t *options, gridinfo_t *grid, zLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{ pzconvert_flatten_skyline2UROWDATA(options, grid, LUstruct, stat, n); }

inline void pconvertUROWDATA2skyline(superlu_dist_options_t *options, gridinfo_t *grid, sLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{ psconvertUROWDATA2skyline(options, grid, LUstruct, stat, n); }
inline void pconvertUROWDATA2skyline(superlu_dist_options_t *options, gridinfo_t *grid, dLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{ pdconvertUROWDATA2skyline(options, grid, LUstruct, stat, n); }
inline void pconvertUROWDATA2skyline(superlu_dist_options_t *options, gridinfo_t *grid, zLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{ pzconvertUROWDATA2skyline(options, grid, LUstruct, stat, n); }

#endif 
