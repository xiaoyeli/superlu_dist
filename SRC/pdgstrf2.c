/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Performs panel LU factorization.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * August 15, 2014
 *
 * Modified:
 *   September 30, 2017
 *   May 10, 2019  v7.0.0
 *   December 12, 2021  v7.2.0
 *
 * Purpose
 * =======
 *   Panel factorization -- block column k
 *
 *   Factor diagonal and subdiagonal blocks and test for exact singularity.
 *   Only the column processes that own block column *k* participate
 *   in the work.
 *
 * Arguments
 * =========
 * options (input) superlu_dist_options_t* (global)
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *
 * k0     (input) int (global)
 *        Counter of the next supernode to be factorized.
 *
 * k      (input) int (global)
 *        The column number of the block column to be factorized.
 *
 * thresh (input) double (global)
 *        The threshold value = s_eps * anorm.
 *
 * Glu_persist (input) Glu_persist_t*
 *        Global data structures (xsup, supno) replicated on all processes.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Llu    (input/output) dLocalLU_t*
 *        Local data structures to store distributed L and U matrices.
 *
 * U_diag_blk_send_req (input/output) MPI_Request*
 *        List of send requests to send down the diagonal block of U.
 *
 * tag_ub (input) int
 *        Upper bound of MPI tag values.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the factorization.
 *        See SuperLUStat_t structure defined in util.h.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 * </pre>
 */

#include <math.h>
#include "superlu_ddefs.h"
//#include "cblas.h"

/*****************************************************************************
 * The following pdgstrf2_trsm is in version 6 and earlier.
 *****************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Panel factorization -- block column k
 *
 *   Factor diagonal and subdiagonal blocks and test for exact singularity.
 *   Only the column processes that own block column *k* participate
 *   in the work.
 *
 * Arguments
 * =========
 * options (input) superlu_dist_options_t* (global)
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *
 * k0     (input) int (global)
 *        Counter of the next supernode to be factorized.
 *
 * k      (input) int (global)
 *        The column number of the block column to be factorized.
 *
 * thresh (input) double (global)
 *        The threshold value = s_eps * anorm.
 *
 * Glu_persist (input) Glu_persist_t*
 *        Global data structures (xsup, supno) replicated on all processes.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Llu    (input/output) dLocalLU_t*
 *        Local data structures to store distributed L and U matrices.
 *
 * U_diag_blk_send_req (input/output) MPI_Request*
 *        List of send requests to send down the diagonal block of U.
 *
 * tag_ub (input) int
 *        Upper bound of MPI tag values.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the factorization.
 *        See SuperLUStat_t structure defined in util.h.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 * </pre>
 */
/* This pdgstrf2 is based on TRSM function */
void
pdgstrf2_trsm
    (superlu_dist_options_t * options, int_t k0, int_t k, double thresh,
     Glu_persist_t * Glu_persist, gridinfo_t * grid, dLocalLU_t * Llu,
     MPI_Request * U_diag_blk_send_req, int tag_ub,
     SuperLUStat_t * stat, int *info)
{
    /* printf("entering pdgstrf2 %d \n", grid->iam); */
    int cols_left, iam, l, pkk, pr;
    int incx = 1, incy = 1;

    int nsupr;            /* number of rows in the block (LDA) */
    int nsupc;            /* number of columns in the block */
    int luptr;
    int_t i, myrow, krow, j, jfst, jlst, u_diag_cnt;
    int_t *xsup = Glu_persist->xsup;
    double *lusup, temp;
    double *ujrow, *ublk_ptr;   /* pointer to the U block */
    double alpha = -1, zero = 0.0;
    int_t Pr;
    MPI_Status status;
    MPI_Comm comm = (grid->cscp).comm;
    double t1, t2;

    /* Initialization. */
    iam = grid->iam;
    Pr = grid->nprow;
    myrow = MYROW (iam, grid);
    krow = PROW (k, grid);
    pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    j = LBj (k, grid);          /* Local block number */
    jfst = FstBlockC (k);
    jlst = FstBlockC (k + 1);
    lusup = Llu->Lnzval_bc_ptr[j];
    nsupc = SuperSize (k);
    if (Llu->Lrowind_bc_ptr[j])
        nsupr = Llu->Lrowind_bc_ptr[j][1];
    else
        nsupr = 0;
#ifdef PI_DEBUG
    printf ("rank %d  Iter %d  k=%d \t dtrsm nsuper %d \n",
            iam, k0, k, nsupr);
#endif
    ublk_ptr = ujrow = Llu->ujrow;

    luptr = 0;                  /* Point to the diagonal entries. */
    cols_left = nsupc;          /* supernode size */
    int ld_ujrow = nsupc;       /* leading dimension of ujrow */
    u_diag_cnt = 0;
    incy = ld_ujrow;

    if ( U_diag_blk_send_req &&
	 U_diag_blk_send_req[myrow] != MPI_REQUEST_NULL ) {
        /* There are pending sends - wait for all Isend to complete */
#if ( PROFlevel>=1 )
	TIC (t1);
#endif
        for (pr = 0; pr < Pr; ++pr) {
            if (pr != myrow) {
                MPI_Wait (U_diag_blk_send_req + pr, &status);
            }
	}
#if ( PROFlevel>=1 )
	TOC (t2, t1);
	stat->utime[COMM] += t2;
	stat->utime[COMM_DIAG] += t2;
#endif
	/* flag no more outstanding send request. */
	U_diag_blk_send_req[myrow] = MPI_REQUEST_NULL;
    }

    if (iam == pkk) {            /* diagonal process */
	/* ++++ First step compute diagonal block ++++++++++ */
        for (j = 0; j < jlst - jfst; ++j) {  /* for each column in panel */
            /* Diagonal pivot */
            i = luptr;
            /* May replace zero pivot.  */
            if (options->ReplaceTinyPivot == YES )  {
                if (fabs (lusup[i]) < thresh) {  /* Diagonal */

#if ( PRNTlevel>=2 )
                    printf ("(%d) .. col %d, tiny pivot %e  ",
                            iam, jfst + j, lusup[i]);
#endif
                    /* Keep the new diagonal entry with the same sign. */
                    if (lusup[i] < 0)  lusup[i] = -thresh;
                    else  lusup[i] = thresh;
#if ( PRNTlevel>=2 )
                    printf ("replaced by %e\n", lusup[i]);
#endif
                    ++(stat->TinyPivots);
                }
            }

#if 0
            for (l = 0; l < cols_left; ++l, i += nsupr, ++u_diag_cnt)
                 ublk_ptr[u_diag_cnt] = lusup[i]; /* copy one row of U */
#endif

            /* storing U in full form  */
            int st;
            for (l = 0; l < cols_left; ++l, i += nsupr, ++u_diag_cnt) {
                st = j * ld_ujrow + j;
                ublk_ptr[st + l * ld_ujrow] = lusup[i]; /* copy one row of U */
            }

            if ( ujrow[0] == zero ) { /* Test for singularity. */
                *info = j + jfst + 1;
            } else {              /* Scale the j-th column within diag. block. */
                temp = 1.0 / ujrow[0];
                for (i = luptr + 1; i < luptr - j + nsupc; ++i)
		    lusup[i] *= temp;
                stat->ops[FACT] += nsupc - j - 1;
            }

            /* Rank-1 update of the trailing submatrix within diag. block. */
            if (--cols_left) {
                /* l = nsupr - j - 1;  */
                l = nsupc - j - 1;  /* Piyush */
                dger_ (&l, &cols_left, &alpha, &lusup[luptr + 1], &incx,
                       &ujrow[ld_ujrow], &incy, &lusup[luptr + nsupr + 1],
                       &nsupr);
                stat->ops[FACT] += 2 * l * cols_left;
            }

            /* ujrow = ublk_ptr + u_diag_cnt;  */
            ujrow = ujrow + ld_ujrow + 1; /* move to next row of U */
            luptr += nsupr + 1; /* move to next column */

        }                       /* for column j ...  first loop */

	/* ++++ Second step compute off-diagonal block with communication  ++*/

        ublk_ptr = ujrow = Llu->ujrow;

        if (U_diag_blk_send_req && iam == pkk)  { /* Send the U block downward */
            /** ALWAYS SEND TO ALL OTHERS - TO FIX **/
#if ( PROFlevel>=1 )
	    TIC (t1);
#endif
            for (pr = 0; pr < Pr; ++pr) {
                if (pr != krow) {
                    /* tag = ((k0<<2)+2) % tag_ub;        */
                    /* tag = (4*(nsupers+k0)+2) % tag_ub; */
                    MPI_Isend (ublk_ptr, nsupc * nsupc, MPI_DOUBLE, pr,
                               SLU_MPI_TAG (4, k0) /* tag */ ,
                               comm, U_diag_blk_send_req + pr);

                }
            }
#if ( PROFlevel>=1 )
	    TOC (t2, t1);
	    stat->utime[COMM] += t2;
	    stat->utime[COMM_DIAG] += t2;
#endif

	    /* flag outstanding Isend */
            U_diag_blk_send_req[krow] = (MPI_Request) TRUE; /* Sherry */
        }

        /* pragma below would be changed by an MKL call */

        l = nsupr - nsupc;
        // n = nsupc;
        double alpha = 1.0;
#ifdef PI_DEBUG
        printf ("calling dtrsm\n");
        printf ("dtrsm diagonal param 11:  %d \n", nsupr);
#endif

#if defined (USE_VENDOR_BLAS)
        dtrsm_ ("R", "U", "N", "N", &l, &nsupc,
                &alpha, ublk_ptr, &ld_ujrow, &lusup[nsupc], &nsupr,
		1, 1, 1, 1);
#else
        dtrsm_ ("R", "U", "N", "N", &l, &nsupc,
                &alpha, ublk_ptr, &ld_ujrow, &lusup[nsupc], &nsupr);
#endif
	stat->ops[FACT] += (flops_t) nsupc * (nsupc+1) * l;
    } else {  /* non-diagonal process */
        /* ================================================================== *
         * Receive the diagonal block of U for panel factorization of L(:,k). *
         * Note: we block for panel factorization of L(:,k), but panel        *
	 * factorization of U(:,k) do not block                               *
         * ================================================================== */

        /* tag = ((k0<<2)+2) % tag_ub;        */
        /* tag = (4*(nsupers+k0)+2) % tag_ub; */
        // printf("hello message receiving%d %d\n",(nsupc*(nsupc+1))>>1,SLU_MPI_TAG(4,k0));
#if ( PROFlevel>=1 )
	TIC (t1);
#endif
        MPI_Recv (ublk_ptr, (nsupc * nsupc), MPI_DOUBLE, krow,
                  SLU_MPI_TAG (4, k0) /* tag */ ,
                  comm, &status);
#if ( PROFlevel>=1 )
	TOC (t2, t1);
	stat->utime[COMM] += t2;
	stat->utime[COMM_DIAG] += t2;
#endif
        if (nsupr > 0) {
            double alpha = 1.0;

#ifdef PI_DEBUG
            printf ("dtrsm non diagonal param 11:  %d \n", nsupr);
            if (!lusup)
                printf (" Rank :%d \t Empty block column occurred :\n", iam);
#endif
#if defined (USE_VENDOR_BLAS)
            dtrsm_ ("R", "U", "N", "N", &nsupr, &nsupc,
                    &alpha, ublk_ptr, &ld_ujrow, lusup, &nsupr, 1, 1, 1, 1);
#else
            dtrsm_ ("R", "U", "N", "N", &nsupr, &nsupc,
                    &alpha, ublk_ptr, &ld_ujrow, lusup, &nsupr);
#endif
	    stat->ops[FACT] += (flops_t) nsupc * (nsupc+1) * nsupr;
        }

    } /* end if pkk ... */

    /* printf("exiting pdgstrf2 %d \n", grid->iam);  */

}  /* PDGSTRF2_trsm */


/*****************************************************************************
 * The following functions are for the new pdgstrf2_dtrsm in the 3D code.
 *****************************************************************************/
static
int_t LpanelUpdate(int off0,  int nsupc, double* ublk_ptr, int ld_ujrow,
                   double* lusup, int nsupr, SCT_t* SCT)
{
    int_t l = nsupr - off0;
    double alpha = 1.0;
    double t1 = SuperLU_timer_();

#define GT  32
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < CEILING(l, GT); ++i)
    {
        int_t off = i * GT;
        int len = SUPERLU_MIN(GT, l - i * GT);
	
        superlu_dtrsm("R", "U", "N", "N", len, nsupc, alpha,
		      ublk_ptr, ld_ujrow, &lusup[off0 + off], nsupr);

    } /* for i = ... */

    t1 = SuperLU_timer_() - t1;

    SCT->trf2_flops += (double) l * (double) nsupc * (double)nsupc;
    SCT->trf2_time += t1;
    SCT->L_PanelUpdate_tl += t1;
    return 0;
}

#pragma GCC push_options
#pragma GCC optimize ("O0")
/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Factorize the diagonal block; called from process that owns the (k,k) block
 *
 * Arguments
 * =========
 * 
 * info   (output) int*
 *        = 0: successful exit
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 */
void Local_Dgstrf2(superlu_dist_options_t *options, int_t k, double thresh,
                   double *BlockUFactor, /*factored U is overwritten here*/
                   Glu_persist_t *Glu_persist, gridinfo_t *grid, dLocalLU_t *Llu,
                   SuperLUStat_t *stat, int *info, SCT_t* SCT)
{
    //double t1 = SuperLU_timer_();
    int_t *xsup = Glu_persist->xsup;
    double alpha = -1, zero = 0.0;

    // printf("Entering dgetrf2 %d \n", k);
    /* Initialization. */
    int_t lk = LBj (k, grid);          /* Local block number */
    int_t jfst = FstBlockC (k);
    int_t jlst = FstBlockC (k + 1);
    double *lusup = Llu->Lnzval_bc_ptr[lk];
    int nsupc = SuperSize (k);
    int nsupr;
    if (Llu->Lrowind_bc_ptr[lk])
        nsupr = Llu->Lrowind_bc_ptr[lk][1];
    else
        nsupr = 0;
    double *ublk_ptr = BlockUFactor;
    double *ujrow = BlockUFactor;
    int_t luptr = 0;                  /* Point_t to the diagonal entries. */
    int cols_left = nsupc;          /* supernode size */
    int_t u_diag_cnt = 0;
    int_t ld_ujrow = nsupc;       /* leading dimension of ujrow */
    int incx = 1;
    int incy = ld_ujrow;

    for (int_t j = 0; j < jlst - jfst; ++j)   /* for each column in panel */
    {
        /* Diagonal pivot */
        int_t i = luptr;
        /* Allow to replace zero pivot.  */
        //if (options->ReplaceTinyPivot == YES && lusup[i] != 0.0)
        if (options->ReplaceTinyPivot == YES)
        {
            if (fabs (lusup[i]) < thresh) {  /* Diagonal */

#if ( PRNTlevel>=2 )
                    printf ("(%d) .. col %d, tiny pivot %e  ",
                            iam, jfst + j, lusup[i]);
#endif
                /* Keep the new diagonal entry with the same sign. */
                if (lusup[i] < 0) lusup[i] = -thresh;
                else lusup[i] = thresh;
#if ( PRNTlevel>=2 )
                    printf ("replaced by %e\n", lusup[i]);
#endif
                ++(stat->TinyPivots);
            }
        }

        for (int_t l = 0; l < cols_left; ++l, i += nsupr, ++u_diag_cnt)
        {
            int_t st = j * ld_ujrow + j;
            ublk_ptr[st + l * ld_ujrow] = lusup[i]; /* copy one row of U */
        }

        if (ujrow[0] == zero)   /* Test for singularity. */
        {
            *info = j + jfst + 1;
        }
        else                /* Scale the j-th column. */
        {
            double temp;
            temp = 1.0 / ujrow[0];
            for (int_t i = luptr + 1; i < luptr - j + nsupc; ++i)
                lusup[i] *= temp;
            stat->ops[FACT] += nsupc - j - 1;
        }

        /* Rank-1 update of the trailing submatrix. */
        if (--cols_left)
        {
            /*following must be int*/
            int l = nsupc - j - 1;

	    /* Rank-1 update */
            superlu_dger(l, cols_left, alpha, &lusup[luptr + 1], incx,
                         &ujrow[ld_ujrow], incy, &lusup[luptr + nsupr + 1], nsupr);
	    
            stat->ops[FACT] += 2 * l * cols_left;
        }

        ujrow = ujrow + ld_ujrow + 1; /* move to next row of U */
        luptr += nsupr + 1;           /* move to next column */

    }                       /* for column j ...  first loop */


    //int_t thread_id = omp_get_thread_num();
    // SCT->Local_Dgstrf2_Thread_tl[thread_id * CACHE_LINE_SIZE] += (double) ( SuperLU_timer_() - t1);
} /* end Local_Dgstrf2 */

#pragma GCC pop_options
/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Panel factorization -- block column k
 *
 *   Factor diagonal and subdiagonal blocks and test for exact singularity.
 *   Only the column processes that own block column *k* participate
 *   in the work.
 *
 * Arguments
 * =========
 * options (input) superlu_dist_options_t* (global)
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *
 * nsupers (input) int_t (global)
 *         Number of supernodes.
 *
 * k0     (input) int (global)
 *        Counter of the next supernode to be factorized.
 *
 * k      (input) int (global)
 *        The column number of the block column to be factorized.
 *
 * thresh (input) double (global)
 *        The threshold value = s_eps * anorm.
 *
 * Glu_persist (input) Glu_persist_t*
 *        Global data structures (xsup, supno) replicated on all processes.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Llu    (input/output) dLocalLU_t*
 *        Local data structures to store distributed L and U matrices.
 *
 * U_diag_blk_send_req (input/output) MPI_Request*
 *        List of send requests to send down the diagonal block of U.
 *
 * tag_ub (input) int
 *        Upper bound of MPI tag values.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the factorization.
 *        See SuperLUStat_t structure defined in util.h.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 * 
 * SCT    (output) SCT_t*
 *        Additional statistics used in the 3D algorithm.
 *
 * </pre>
 */
void pdgstrf2_xtrsm
(superlu_dist_options_t *options, int_t nsupers,
 int_t k0, int_t k, double thresh, Glu_persist_t *Glu_persist,
 gridinfo_t *grid, dLocalLU_t *Llu, MPI_Request *U_diag_blk_send_req,
 int tag_ub, SuperLUStat_t *stat, int *info, SCT_t *SCT)
{
    int cols_left, iam, pkk;
    int incy = 1;

    int nsupr;                  /* number of rows in the block (LDA) */
    int luptr;
    int_t myrow, krow, j, jfst, jlst, u_diag_cnt;
    int_t nsupc;                /* number of columns in the block */
    int_t *xsup = Glu_persist->xsup;
    double *lusup;
    double *ujrow, *ublk_ptr;   /* pointer to the U block */
    int_t Pr;

    /* Quick return. */
    *info = 0;

    /* Initialization. */
    iam = grid->iam;
    Pr = grid->nprow;
    myrow = MYROW (iam, grid);
    krow = PROW (k, grid);
    pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    j = LBj (k, grid);          /* Local block number */
    jfst = FstBlockC (k);
    jlst = FstBlockC (k + 1);
    lusup = Llu->Lnzval_bc_ptr[j];
    nsupc = SuperSize (k);
    if (Llu->Lrowind_bc_ptr[j])
        nsupr = Llu->Lrowind_bc_ptr[j][1];
    else
        nsupr = 0;
    ublk_ptr = ujrow = Llu->ujrow;

    luptr = 0;                  /* Point to the diagonal entries. */
    cols_left = nsupc;          /* supernode size */
    int ld_ujrow = nsupc;       /* leading dimension of ujrow */
    u_diag_cnt = 0;
    incy = ld_ujrow;

    if (U_diag_blk_send_req && U_diag_blk_send_req[myrow])
    {
        /* There are pending sends - wait for all Isend to complete */
        Wait_UDiagBlockSend(U_diag_blk_send_req, grid, SCT);
    }

    if (iam == pkk)             /* diagonal process */
    {
        /*factorize the diagonal block*/
        Local_Dgstrf2(options, k, thresh, Llu->ujrow, Glu_persist,
                      grid, Llu, stat, info, SCT);
        ublk_ptr = ujrow = Llu->ujrow;

        if (U_diag_blk_send_req && iam == pkk)  /* Send the U block */
        {
            dISend_UDiagBlock(k0, ublk_ptr, nsupc * nsupc, U_diag_blk_send_req,
			     grid, tag_ub);
            U_diag_blk_send_req[krow] = (MPI_Request) TRUE; /* flag outstanding Isend */
        }

        LpanelUpdate(nsupc,  nsupc, ublk_ptr, ld_ujrow, lusup, nsupr, SCT);
    }
    else                        /* non-diagonal process */
    {
        /* ================================================ *
         * Receive the diagonal block of U                  *
         * for panel factorization of L(:,k)                *
         * note: we block for panel factorization of L(:,k) *
         * but panel factorization of U(:,k) don't          *
         * ================================================ */

        dRecv_UDiagBlock( k0, ublk_ptr, (nsupc * nsupc),  krow, grid, SCT, tag_ub);

        if (nsupr > 0)
        {
            LpanelUpdate(0,  nsupc, ublk_ptr, ld_ujrow, lusup, nsupr, SCT);
        }
    } /* end if pkk ... */

} /* pdgstrf2_xtrsm */

/*****************************************************************************
 * The following functions are for the new pdgstrs2_omp in the 3D code.
 *****************************************************************************/

/* PDGSTRS2 helping kernels*/

int_t dTrs2_GatherU(int_t iukp, int_t rukp, int_t klst,
		    int_t nsupc, int_t ldu,
		    int_t *usub,
		    double* uval, double *tempv)
{
    double zero = 0.0;
    int_t ncols = 0;
    for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
    {
        int_t segsize = klst - usub[jj];
        if ( segsize )
        {
            int_t lead_zero = ldu - segsize;
            for (int_t i = 0; i < lead_zero; ++i) tempv[i] = zero;
            tempv += lead_zero;
            for (int_t i = 0; i < segsize; ++i)
                tempv[i] = uval[rukp + i];
            rukp += segsize;
            tempv += segsize;
            ncols++;
        }
    }
    return ncols;
}

int_t dTrs2_ScatterU(int_t iukp, int_t rukp, int_t klst,
		     int_t nsupc, int_t ldu,
		     int_t *usub, double* uval, double *tempv)
{
    for (int_t jj = 0; jj < nsupc; ++jj)
    {
        int_t segsize = klst - usub[iukp + jj];
        if (segsize)
        {
            int_t lead_zero = ldu - segsize;
            tempv += lead_zero;
            for (int i = 0; i < segsize; ++i)
            {
                uval[rukp + i] = tempv[i];
            }
            tempv += segsize;
            rukp += segsize;
        }
    } /*for jj=0:nsupc */
    return 0;
}

int_t dTrs2_GatherTrsmScatter(int_t klst, int_t iukp, int_t rukp,
			      int_t *usub, double *uval, double *tempv,
			      int_t knsupc, int nsupr, double *lusup,
			      Glu_persist_t *Glu_persist)    /*glupersist for xsup for supersize*/
{
    double alpha = 1.0;
    int_t *xsup = Glu_persist->xsup;
    // int_t iukp = Ublock_info.iukp;
    // int_t rukp = Ublock_info.rukp;
    int_t gb = usub[iukp];
    int_t nsupc = SuperSize (gb);
    iukp += UB_DESCRIPTOR;

    // printf("klst inside task%d\n", );
    /*find ldu */
    int ldu = 0;
    for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
    {
        ldu = SUPERLU_MAX( klst - usub[jj], ldu) ;
    }

    /*pack U block into a dense Block*/
    int ncols = dTrs2_GatherU(iukp, rukp, klst, nsupc, ldu, usub,
    	                           uval, tempv);

    /*now call dtrsm on packed dense block*/
    int_t luptr = (knsupc - ldu) * (nsupr + 1);
    // if(ldu>nsupr) printf("nsupr %d ldu %d\n",nsupr,ldu );
    
    superlu_dtrsm("L", "L", "N", "U", ldu, ncols, alpha,
		  &lusup[luptr], nsupr, tempv, ldu);

    /*now scatter the output into sparse U block*/
    dTrs2_ScatterU(iukp, rukp, klst, nsupc, ldu, usub, uval, tempv);

    return 0;
}

/* END 3D CODE */
/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#if 1

/*****************************************************************************
 * The following pdgstrf2_omp is improved for KNL, since Version 5.2.0.
 *****************************************************************************/
void pdgstrs2_omp
(int_t k0, int_t k, Glu_persist_t * Glu_persist, gridinfo_t * grid,
 dLocalLU_t * Llu, Ublock_info_t *Ublock_info, SuperLUStat_t * stat)
{
#ifdef PI_DEBUG
    printf("====Entering pdgstrs2==== \n");
#endif
    int iam, pkk;
    int incx = 1;
    int nsupr;                /* number of rows in the block L(:,k) (LDA) */
    int segsize;
    int nsupc;                /* number of columns in the block */
    int_t luptr, iukp, rukp;
    int_t b, gb, j, klst, knsupc, lk, nb;
    int_t *xsup = Glu_persist->xsup;
    int_t *usub;
    double *lusup, *uval;

#if 0
    //#ifdef USE_VTUNE
    __SSC_MARK(0x111);// start SDE tracing, note uses 2 underscores
    __itt_resume(); // start VTune, again use 2 underscores
#endif

    /* Quick return. */
    lk = LBi (k, grid);         /* Local block number */
    if (!Llu->Unzval_br_ptr[lk]) return;

    /* Initialization. */
    iam = grid->iam;
    pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    //int k_row_cycle = k / grid->nprow;  /* for which cycle k exist (to assign rowwise thread blocking) */
    //int gb_col_cycle;  /* cycle through block columns  */
    klst = FstBlockC (k + 1);
    knsupc = SuperSize (k);
    usub = Llu->Ufstnz_br_ptr[lk];  /* index[] of block row U(k,:) */
    uval = Llu->Unzval_br_ptr[lk];
    if (iam == pkk) {
        lk = LBj (k, grid);
        nsupr = Llu->Lrowind_bc_ptr[lk][1]; /* LDA of lusup[] */
        lusup = Llu->Lnzval_bc_ptr[lk];
    } else {
        nsupr = Llu->Lsub_buf_2[k0 % (1 + stat->num_look_aheads)][1];   /* LDA of lusup[] */
        lusup = Llu->Lval_buf_2[k0 % (1 + stat->num_look_aheads)];
    }

    /////////////////////new-test//////////////////////////
    /* !! Taken from Carl/SuperLU_DIST_5.1.0/EXAMPLE/pdgstrf2_v3.c !! */

    /* Master thread: set up pointers to each block in the row */
    nb = usub[0];
    iukp = BR_HEADER;
    rukp = 0;

    /* Sherry: can use the existing  Ublock_info[] array, call
       Trs2_InitUblock_info();                                 */
#undef USE_Ublock_info
#ifdef USE_Ublock_info /** 4/19/2019 **/
    /* Loop through all the row blocks. to get the iukp and rukp*/
    Trs2_InitUblock_info(klst, nb, Ublock_info, usub, Glu_persist, stat );
#else
    int* blocks_index_pointers = SUPERLU_MALLOC (3 * nb * sizeof(int));
    int* blocks_value_pointers = blocks_index_pointers + nb;
    int* nsupc_temp = blocks_value_pointers + nb;
    for (b = 0; b < nb; b++) { /* set up pointers to each block */
	blocks_index_pointers[b] = iukp + UB_DESCRIPTOR;
	blocks_value_pointers[b] = rukp;
	gb = usub[iukp];
	rukp += usub[iukp+1];
	nsupc = SuperSize( gb );
	nsupc_temp[b] = nsupc;
	iukp += (UB_DESCRIPTOR + nsupc);  /* move to the next block */
    }
#endif

    // Sherry: this version is more NUMA friendly compared to pdgstrf2_v2.c
    // https://stackoverflow.com/questions/13065943/task-based-programming-pragma-omp-task-versus-pragma-omp-parallel-for
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(shared) \
    private(b,j,iukp,rukp,segsize)
#endif
    /* Loop through all the blocks in the row. */
    for (b = 0; b < nb; ++b) {
#ifdef USE_Ublock_info
	iukp = Ublock_info[b].iukp;
	rukp = Ublock_info[b].rukp;
#else
	iukp = blocks_index_pointers[b];
	rukp = blocks_value_pointers[b];
#endif

        /* Loop through all the segments in the block. */
#ifdef USE_Ublock_info
	gb = usub[iukp];
	nsupc = SuperSize( gb );
	iukp += UB_DESCRIPTOR;
        for (j = 0; j < nsupc; j++) {
#else	
        for (j = 0; j < nsupc_temp[b]; j++) {
#endif
            segsize = klst - usub[iukp++];
	    if (segsize) {
#ifdef _OPENMP
#pragma omp task default(shared) firstprivate(segsize,rukp) if (segsize > 30)
#endif
		{ /* Nonzero segment. */
		    int_t luptr = (knsupc - segsize) * (nsupr + 1);
		    //printf("[2] segsize %d, nsupr %d\n", segsize, nsupr);

#if defined (USE_VENDOR_BLAS)
                    dtrsv_ ("L", "N", "U", &segsize, &lusup[luptr], &nsupr,
                            &uval[rukp], &incx, 1, 1, 1);
#else
                    dtrsv_ ("L", "N", "U", &segsize, &lusup[luptr], &nsupr,
                            &uval[rukp], &incx);
#endif
		} /* end task */
		rukp += segsize;
#ifndef USE_Ublock_info
		stat->ops[FACT] += segsize * (segsize + 1);
#endif
	    } /* end if segsize > 0 */
	} /* end for j in parallel ... */
#ifdef _OPENMP    
/* #pragma omp taskwait */
#endif
    }  /* end for b ... */

#ifndef USE_Ublock_info
    /* Deallocate memory */
    SUPERLU_FREE(blocks_index_pointers);
#endif

#if 0
    //#ifdef USE_VTUNE
    __itt_pause(); // stop VTune
    __SSC_MARK(0x222); // stop SDE tracing
#endif

} /* pdgstrs2_omp */

#else  /*==== new version from Piyush ====*/

void pdgstrs2_omp(int_t k0, int_t k, int_t* Lsub_buf, 
		  double *Lval_buf, Glu_persist_t *Glu_persist,
		  gridinfo_t *grid, dLocalLU_t *Llu, SuperLUStat_t *stat,
		  Ublock_info_t *Ublock_info, double *bigV, int_t ldt, SCT_t *SCT)
{
    double t1 = SuperLU_timer_();
    int_t *xsup = Glu_persist->xsup;
    /* Quick return. */
    int_t lk = LBi (k, grid);         /* Local block number */

    if (!Llu->Unzval_br_ptr[lk]) return;

    /* Initialization. */
    int_t klst = FstBlockC (k + 1);
    int_t knsupc = SuperSize (k);
    int_t *usub = Llu->Ufstnz_br_ptr[lk];  /* index[] of block row U(k,:) */
    double *uval = Llu->Unzval_br_ptr[lk];
    int_t nb = usub[0];

    int_t nsupr = Lsub_buf[1];   /* LDA of lusup[] */
    double *lusup = Lval_buf;

    /* Loop through all the row blocks. to get the iukp and rukp*/
    Trs2_InitUbloc_info(klst, nb, Ublock_info, usub, Glu_persist, stat );

    /* Loop through all the row blocks. */
#ifdef _OPENMP    
#pragma omp parallel for schedule(dynamic,2)
#endif
    for (int_t b = 0; b < nb; ++b)
    {
#ifdef _OPENMP    
        int thread_id = omp_get_thread_num();
#else	
        int thread_id = 0;
#endif	
        double *tempv = bigV +  thread_id * ldt * ldt;
        dTrs2_GatherTrsmScatter(klst, Ublock_info[b].iukp, Ublock_info[b].rukp,
				usub, uval, tempv, knsupc, nsupr, lusup, Glu_persist);
    } /* for b ... */

    SCT->PDGSTRS2_tl += (double) ( SuperLU_timer_() - t1);
} /* pdgstrs2_omp new version from Piyush */

#endif /* there are 2 versions of pdgstrs2_omp */
