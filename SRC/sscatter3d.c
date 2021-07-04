/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Scatter the computed blocks into LU destination.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 * May 12, 2021
 */

#include "superlu_sdefs.h"
//#include "scatter.h"
//#include "compiler.h"

//#include "cblas.h"

#define ISORT
#define SCATTER_U_CPU  scatter_u

static void scatter_u (int_t ib, int_t jb, int_t nsupc, int_t iukp, int_t *xsup,
                 int_t klst, int_t nbrow, int_t lptr, int_t temp_nbrow,
 		 int_t *lsub, int_t *usub, float *tempv,
		 int *indirect,
           	 int_t **Ufstnz_br_ptr, float **Unzval_br_ptr, gridinfo_t *grid);


#if 0 /**** Sherry: this routine is moved to util.c ****/
void
arrive_at_ublock (int_t j,      //block number
                  int_t *iukp,  // output
                  int_t *rukp, int_t *jb,   /* Global block number of block U(k,j). */
                  int_t *ljb,   /* Local block number of U(k,j). */
                  int_t *nsupc,     /*supernode size of destination block */
                  int_t iukp0,  //input
                  int_t rukp0, int_t *usub,     /*usub scripts */
                  int_t *perm_u,    /*permutation matrix */
                  int_t *xsup,  /*for SuperSize and LBj */
                  gridinfo_t *grid)
{
    int_t jj;
    *iukp = iukp0;
    *rukp = rukp0;

#ifdef ISORT
    for (jj = 0; jj < perm_u[j]; jj++)
#else
    for (jj = 0; jj < perm_u[2 * j + 1]; jj++)
#endif
    {

        *jb = usub[*iukp];      /* Global block number of block U(k,j). */
        *nsupc = SuperSize (*jb);
        *iukp += UB_DESCRIPTOR; /* Start fstnz of block U(k,j). */
        *rukp += usub[*iukp - 1];   /* Move to block U(k,j+1) */
        *iukp += *nsupc;
    }

    /* reinitilize the pointers to the begining of the */
    /* kth column/row of L/U factors                   */
    *jb = usub[*iukp];          /* Global block number of block U(k,j). */
    *ljb = LBj (*jb, grid);     /* Local block number of U(k,j). */
    *nsupc = SuperSize (*jb);
    *iukp += UB_DESCRIPTOR;     /* Start fstnz of block U(k,j). */
}
#endif
/*--------------------------------------------------------------*/

void
sblock_gemm_scatter( int_t lb, int_t j,
                    Ublock_info_t *Ublock_info,
                    Remain_info_t *Remain_info,
                    float *L_mat, int ldl,
                    float *U_mat, int ldu,
                    float *bigV,
                    // int_t jj0,
                    int_t knsupc,  int_t klst,
                    int_t *lsub, int_t *usub, int_t ldt,
                    int_t thread_id,
                    int *indirect,
                    int *indirect2,
                    int_t **Lrowind_bc_ptr, float **Lnzval_bc_ptr,
                    int_t **Ufstnz_br_ptr, float **Unzval_br_ptr,
                    int_t *xsup, gridinfo_t *grid,
                    SuperLUStat_t *stat
#ifdef SCATTER_PROFILE
                    , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                  )
{
    // return ;
#ifdef _OPENMP    
    thread_id = omp_get_thread_num();
#else    
    thread_id = 0;
#endif    
    int *indirect_thread = indirect + ldt * thread_id;
    int *indirect2_thread = indirect2 + ldt * thread_id;
    float *tempv1 = bigV + thread_id * ldt * ldt;

    /* Getting U block information */

    int_t iukp =  Ublock_info[j].iukp;
    int_t jb   =  Ublock_info[j].jb;
    int_t nsupc = SuperSize(jb);
    int_t ljb = LBj (jb, grid);
    int_t st_col;
    int ncols;

    // if (j > jj0)
    if (j > 0)
    {
        ncols  = Ublock_info[j].full_u_cols - Ublock_info[j - 1].full_u_cols;
        st_col = Ublock_info[j - 1].full_u_cols;
    }
    else
    {
        ncols  = Ublock_info[j].full_u_cols;
        st_col = 0;
    }

    /* Getting L block information */
    int_t lptr = Remain_info[lb].lptr;
    int_t ib   = Remain_info[lb].ib;
    int temp_nbrow = lsub[lptr + 1];
    lptr += LB_DESCRIPTOR;
    int cum_nrow = (lb == 0 ? 0 : Remain_info[lb - 1].FullRow);
    float alpha = 1.0, beta = 0.0;

    /* calling SGEMM */
    // printf(" m %d n %d k %d ldu %d ldl %d st_col %d \n",temp_nbrow,ncols,ldu,ldl,st_col );
    superlu_sgemm("N", "N", temp_nbrow, ncols, ldu, alpha,
                &L_mat[(knsupc - ldu)*ldl + cum_nrow], ldl,
                &U_mat[st_col * ldu], ldu,
                beta, tempv1, temp_nbrow);
    
    // printf("SCU update: (%d, %d)\n",ib,jb );
#ifdef SCATTER_PROFILE
    double ttx = SuperLU_timer_();
#endif
    
    /*Now scattering the block*/
    if (ib < jb)
    {
        SCATTER_U_CPU (
            ib, jb,
            nsupc, iukp, xsup,
            klst, temp_nbrow,
            lptr, temp_nbrow, lsub,
            usub, tempv1,
            indirect_thread,
            Ufstnz_br_ptr,
            Unzval_br_ptr,
            grid
        );
    }
    else
    {
        //scatter_l (    Sherry
        sscatter_l (
            ib, ljb, nsupc, iukp, xsup, klst, temp_nbrow, lptr,
            temp_nbrow, usub, lsub, tempv1,
            indirect_thread, indirect2_thread,
            Lrowind_bc_ptr, Lnzval_bc_ptr, grid
        );

    }

    // #pragma omp atomic
    // stat->ops[FACT] += 2*temp_nbrow*ncols*ldu + temp_nbrow*ncols;

#ifdef SCATTER_PROFILE
    double t_s = SuperLU_timer_() - ttx;
    Host_TheadScatterMOP[thread_id * ((192 / 8) * (192 / 8)) + ((CEILING(temp_nbrow, 8) - 1)   +  (192 / 8) * (CEILING(ncols, 8) - 1))]
    += 3.0 * (double ) temp_nbrow * (double ) ncols;
    Host_TheadScatterTimer[thread_id * ((192 / 8) * (192 / 8)) + ((CEILING(temp_nbrow, 8) - 1)   +  (192 / 8) * (CEILING(ncols, 8) - 1))]
    += t_s;
#endif
} /* sblock_gemm_scatter */

#ifdef _OPENMP
/*this version uses a lock to prevent multiple thread updating the same block*/
void
sblock_gemm_scatter_lock( int_t lb, int_t j,
                         omp_lock_t* lock,
                         Ublock_info_t *Ublock_info,
                         Remain_info_t *Remain_info,
                         float *L_mat, int_t ldl,
                         float *U_mat, int_t ldu,
                         float *bigV,
                         // int_t jj0,
                         int_t knsupc,  int_t klst,
                         int_t *lsub, int_t *usub, int_t ldt,
                         int_t thread_id,
                         int *indirect,
                         int *indirect2,
                         int_t **Lrowind_bc_ptr, float **Lnzval_bc_ptr,
                         int_t **Ufstnz_br_ptr, float **Unzval_br_ptr,
                         int_t *xsup, gridinfo_t *grid
#ifdef SCATTER_PROFILE
                         , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                       )
{
    int *indirect_thread = indirect + ldt * thread_id;
    int *indirect2_thread = indirect2 + ldt * thread_id;
    float *tempv1 = bigV + thread_id * ldt * ldt;

    /* Getting U block information */

    int_t iukp =  Ublock_info[j].iukp;
    int_t jb   =  Ublock_info[j].jb;
    int_t nsupc = SuperSize(jb);
    int_t ljb = LBj (jb, grid);
    int_t st_col = Ublock_info[j].StCol;
    int_t ncols = Ublock_info[j].ncols;


    /* Getting L block information */
    int_t lptr = Remain_info[lb].lptr;
    int_t ib   = Remain_info[lb].ib;
    int temp_nbrow = lsub[lptr + 1];
    lptr += LB_DESCRIPTOR;
    int cum_nrow =  Remain_info[lb].StRow;

    float alpha = 1.0;  double beta = 0.0;

    /* calling SGEMM */
    superlu_sgemm("N", "N", temp_nbrow, ncols, ldu, alpha,
           &L_mat[(knsupc - ldu)*ldl + cum_nrow], ldl,
           &U_mat[st_col * ldu], ldu, beta, tempv1, temp_nbrow);
    
    /*try to get the lock for the block*/
    if (lock)       /*lock is not null*/
        while (!omp_test_lock(lock))
        {
        }

#ifdef SCATTER_PROFILE
    double ttx = SuperLU_timer_();
#endif
    /*Now scattering the block*/
    if (ib < jb)
    {
        SCATTER_U_CPU (
            ib, jb,
            nsupc, iukp, xsup,
            klst, temp_nbrow,
            lptr, temp_nbrow, lsub,
            usub, tempv1,
            indirect_thread,
            Ufstnz_br_ptr,
            Unzval_br_ptr,
            grid
        );
    }
    else
    {
        //scatter_l (  Sherry
        sscatter_l ( 
            ib, ljb, nsupc, iukp, xsup, klst, temp_nbrow, lptr,
            temp_nbrow, usub, lsub, tempv1,
            indirect_thread, indirect2_thread,
            Lrowind_bc_ptr, Lnzval_bc_ptr, grid
        );

    }

    if (lock)
        omp_unset_lock(lock);

#ifdef SCATTER_PROFILE
    //double t_s = (double) __rdtsc() - ttx;
    double t_s = SuperLU_timer_() - ttx;
    Host_TheadScatterMOP[thread_id * ((192 / 8) * (192 / 8)) + ((CEILING(temp_nbrow, 8) - 1)   +  (192 / 8) * (CEILING(ncols, 8) - 1))]
    += 3.0 * (double ) temp_nbrow * (double ) ncols;
    Host_TheadScatterTimer[thread_id * ((192 / 8) * (192 / 8)) + ((CEILING(temp_nbrow, 8) - 1)   +  (192 / 8) * (CEILING(ncols, 8) - 1))]
    += t_s;
#endif
} /* sblock_gemm_scatter_lock */
#endif  // Only if _OPENMP is defined


// there are following three variations of block_gemm_scatter call
/*
+---------------------------------------+
|          ||                           |
|  CPU     ||          CPU+TopRight     |
|  Top     ||                           |
|  Left    ||                           |
|          ||                           |
+---------------------------------------+
+---------------------------------------+
|          ||        |                  |
|          ||        |                  |
|          ||        |                  |
|  CPU     ||  CPU   |Accelerator       |
|  Bottom  ||  Bottom|                  |
|  Left    ||  Right |                  |
|          ||        |                  |
|          ||        |                  |
+--------------------+------------------+
                  jj_cpu
*/

int_t sblock_gemm_scatterTopLeft( int_t lb, /* block number in L */
				 int_t j,  /* block number in U */
                                 float* bigV, int_t knsupc,  int_t klst,
				 int_t* lsub, int_t * usub, int_t ldt,
				 int* indirect, int* indirect2, HyP_t* HyP,
                                 sLUstruct_t *LUstruct,
                                 gridinfo_t* grid,
                                 SCT_t*SCT, SuperLUStat_t *stat
                               )
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    float** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    float** Unzval_br_ptr = Llu->Unzval_br_ptr;
#ifdef _OPENMP    
    volatile int_t thread_id = omp_get_thread_num();
#else    
    volatile int_t thread_id = 0;
#endif    
    
//    printf("Thread's ID %lld \n", thread_id);
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    sblock_gemm_scatter( lb, j, HyP->Ublock_info, HyP->lookAhead_info,
			HyP->lookAhead_L_buff, HyP->Lnbrow,
                        HyP->bigU_host, HyP->ldu,
                        bigV, knsupc,  klst, lsub,  usub, ldt, thread_id,
			indirect, indirect2,
                        Lrowind_bc_ptr, Lnzval_bc_ptr, Ufstnz_br_ptr, Unzval_br_ptr,
			xsup, grid, stat
#ifdef SCATTER_PROFILE
                        , SCT->Host_TheadScatterMOP, SCT->Host_TheadScatterTimer
#endif
                      );
    //unsigned long long t2 = _rdtsc();
    double t2 = SuperLU_timer_();
    SCT->SchurCompUdtThreadTime[thread_id * CACHE_LINE_SIZE] += (double) (t2 - t1);
    return 0;
} /* sgemm_scatterTopLeft */

int_t sblock_gemm_scatterTopRight( int_t lb,  int_t j,
                                  float* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                  int_t* usub, int_t ldt, int* indirect, int* indirect2,
                                  HyP_t* HyP,
                                  sLUstruct_t *LUstruct,
                                  gridinfo_t* grid,
                                  SCT_t*SCT, SuperLUStat_t *stat
                                )
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    float** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    float** Unzval_br_ptr = Llu->Unzval_br_ptr;
#ifdef _OPENMP    
    volatile  int_t thread_id = omp_get_thread_num();
#else    
    volatile  int_t thread_id = 0;
#endif    
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    sblock_gemm_scatter( lb, j, HyP->Ublock_info_Phi, HyP->lookAhead_info, HyP->lookAhead_L_buff, HyP->Lnbrow,
                        HyP->bigU_Phi, HyP->ldu_Phi,
                        bigV, knsupc,  klst, lsub,  usub, ldt, thread_id, indirect, indirect2,
                        Lrowind_bc_ptr, Lnzval_bc_ptr, Ufstnz_br_ptr, Unzval_br_ptr, xsup, grid, stat
#ifdef SCATTER_PROFILE
                        , SCT->Host_TheadScatterMOP, SCT->Host_TheadScatterTimer
#endif
                      );
    //unsigned long long t2 = _rdtsc();
    double t2 = SuperLU_timer_();
    SCT->SchurCompUdtThreadTime[thread_id * CACHE_LINE_SIZE] += (double) (t2 - t1);
    return 0;
} /* sblock_gemm_scatterTopRight */

int_t sblock_gemm_scatterBottomLeft( int_t lb,  int_t j,
                                    float* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                    int_t* usub, int_t ldt, int* indirect, int* indirect2,
                                    HyP_t* HyP,
                                    sLUstruct_t *LUstruct,
                                    gridinfo_t* grid,
                                    SCT_t*SCT, SuperLUStat_t *stat
                                  )
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    float** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    float** Unzval_br_ptr = Llu->Unzval_br_ptr;
#ifdef _OPENMP    
    volatile int_t thread_id = omp_get_thread_num();
#else    
    volatile int_t thread_id = 0;
#endif    
    //printf("Thread's ID %lld \n", thread_id);
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    sblock_gemm_scatter( lb, j, HyP->Ublock_info, HyP->Remain_info, HyP->Remain_L_buff, HyP->Rnbrow,
                        HyP->bigU_host, HyP->ldu,
                        bigV, knsupc,  klst, lsub,  usub, ldt, thread_id, indirect, indirect2,
                        Lrowind_bc_ptr, Lnzval_bc_ptr, Ufstnz_br_ptr, Unzval_br_ptr, xsup, grid, stat
#ifdef SCATTER_PROFILE
                        , SCT->Host_TheadScatterMOP, SCT->Host_TheadScatterTimer
#endif
                      );
    //unsigned long long t2 = _rdtsc();
    double t2 = SuperLU_timer_();
    SCT->SchurCompUdtThreadTime[thread_id * CACHE_LINE_SIZE] += (double) (t2 - t1);
    return 0;

} /* sblock_gemm_scatterBottomLeft */

int_t sblock_gemm_scatterBottomRight( int_t lb,  int_t j,
                                     float* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                     int_t* usub, int_t ldt, int* indirect, int* indirect2,
                                     HyP_t* HyP,
                                     sLUstruct_t *LUstruct,
                                     gridinfo_t* grid,
                                     SCT_t*SCT, SuperLUStat_t *stat
                                   )
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    float** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    float** Unzval_br_ptr = Llu->Unzval_br_ptr;
#ifdef _OPENMP    
    volatile  int_t thread_id = omp_get_thread_num();
#else    
    volatile  int_t thread_id = 0;
#endif    
   // printf("Thread's ID %lld \n", thread_id);
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    sblock_gemm_scatter( lb, j, HyP->Ublock_info_Phi, HyP->Remain_info, HyP->Remain_L_buff, HyP->Rnbrow,
                        HyP->bigU_Phi, HyP->ldu_Phi,
                        bigV, knsupc,  klst, lsub,  usub, ldt, thread_id, indirect, indirect2,
                        Lrowind_bc_ptr, Lnzval_bc_ptr, Ufstnz_br_ptr, Unzval_br_ptr, xsup, grid, stat
#ifdef SCATTER_PROFILE
                        , SCT->Host_TheadScatterMOP, SCT->Host_TheadScatterTimer
#endif
                      );

    //unsigned long long t2 = _rdtsc();
    double t2 = SuperLU_timer_();
    SCT->SchurCompUdtThreadTime[thread_id * CACHE_LINE_SIZE] += (double) (t2 - t1);
    return 0;

} /* sblock_gemm_scatterBottomRight */

/******************************************************************
 * SHERRY: scatter_l is the same as dscatter_l in dscatter.c
 *         scatter_u is ALMOST the same as dscatter_u in dscatter.c
 ******************************************************************/
#if 0
void
scatter_l (int_t ib,
           int_t ljb,
           int_t nsupc,
           int_t iukp,
           int_t *xsup,
           int_t klst,
           int_t nbrow,
           int_t lptr,
           int_t temp_nbrow,
           int_t *usub,
           int_t *lsub,
           double *tempv,
           int *indirect_thread, int *indirect2,
           int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr, gridinfo_t *grid)
{
    int_t rel, i, segsize, jj;
    double *nzval;
    int_t *index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the dest lusup. */
    int_t lptrj = BC_HEADER;
    int_t luptrj = 0;
    int_t ijb = index[lptrj];

    while (ijb != ib)
    {
        luptrj += index[lptrj + 1];
        lptrj += LB_DESCRIPTOR + index[lptrj + 1];
        ijb = index[lptrj];
    }


    /*
     * Build indirect table. This is needed because the
     * indices are not sorted for the L blocks.
     */
    int_t fnz = FstBlockC (ib);
    int_t dest_nbrow;
    lptrj += LB_DESCRIPTOR;
    dest_nbrow = index[lptrj - 1];

    for (i = 0; i < dest_nbrow; ++i)
    {
        rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;

    }

    /* can be precalculated */
    for (i = 0; i < temp_nbrow; ++i)
    {
        rel = lsub[lptr + i] - fnz;
        indirect2[i] = indirect_thread[rel];
    }


    nzval = Lnzval_bc_ptr[ljb] + luptrj;
    for (jj = 0; jj < nsupc; ++jj)
    {

        segsize = klst - usub[iukp + jj];
        if (segsize)
        {
            for (i = 0; i < temp_nbrow; ++i)
            {
                nzval[indirect2[i]] -= tempv[i];
            }
            tempv += nbrow;
        }
        nzval += ldv;
    }

} /* scatter_l */
#endif // comment out

static void   // SHERRY: ALMOST the same as dscatter_u in dscatter.c
scatter_u (int_t ib,
           int_t jb,
           int_t nsupc,
           int_t iukp,
           int_t *xsup,
           int_t klst,
           int_t nbrow,
           int_t lptr,
           int_t temp_nbrow,
           int_t *lsub,
           int_t *usub,
           float *tempv,
           int *indirect,
           int_t **Ufstnz_br_ptr, float **Unzval_br_ptr, gridinfo_t *grid)
{
#ifdef PI_DEBUG
    printf ("A(%d,%d) goes to U block \n", ib, jb);
#endif
    int_t jj, i, fnz;
    int_t segsize;
    float *ucol;
    int_t ilst = FstBlockC (ib + 1);
    int_t lib = LBi (ib, grid);
    int_t *index = Ufstnz_br_ptr[lib];

    /* reinitialize the pointer to each row of U */
    int_t iuip_lib, ruip_lib;
    iuip_lib = BR_HEADER;
    ruip_lib = 0;

    int_t ijb = index[iuip_lib];
    while (ijb < jb)            /* Search for dest block. */
    {
        ruip_lib += index[iuip_lib + 1];

        iuip_lib += UB_DESCRIPTOR + SuperSize (ijb);
        ijb = index[iuip_lib];
    }
    /* Skip descriptor.  Now point_t to fstnz index of
       block U(i,j). */

    for (i = 0; i < temp_nbrow; ++i)
    {
        indirect[i] = lsub[lptr + i] ;
    }

    iuip_lib += UB_DESCRIPTOR;

    ucol = &Unzval_br_ptr[lib][ruip_lib];
    for (jj = 0; jj < nsupc; ++jj)
    {
        segsize = klst - usub[iukp + jj];
        fnz = index[iuip_lib++];
        ucol -= fnz;
        if (segsize)            /* Nonzero segment in U(k.j). */
        {
            for (i = 0; i < temp_nbrow; ++i)
            {
                ucol[indirect[i]] -= tempv[i];
            }                   /* for i=0..temp_nbropw */
            tempv += nbrow;

        } /*if segsize */
        ucol += ilst ;

    } /*for jj=0:nsupc */

}


