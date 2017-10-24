/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Perform local block modifications: lsum[i] -= L_i,k * X[k]
 *
 * <pre>
 * -- Distributed SuperLU routine (version 2.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 *
 * Modified:
 *     Feburary 7, 2001    use MPI_Isend/MPI_Irecv
 *     October 2, 2001     use MPI_Isend/MPI_Irecv with MPI_Test
 * </pre>
 */

#include "superlu_ddefs.h"
#include "superlu_defs.h"

#define ISEND_IRECV

/*
 * Function prototypes
 */
#ifdef _CRAY
fortran void STRSM(_fcd, _fcd, _fcd, _fcd, int*, int*, double*,
		   double*, int*, double*, int*);
fortran void SGEMM(_fcd, _fcd, int*, int*, int*, double*, double*, 
		   int*, double*, int*, double*, double*, int*);
_fcd ftcs1;
_fcd ftcs2;
_fcd ftcs3;
#endif

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
 * </pre>
 */
void dlsum_fmod
/************************************************************************/
(
 double *lsum,    /* Sum of local modifications.                        */
 double *x,       /* X array (local)                                    */
 double *xk,      /* X[k].                                              */
 double *rtemp,   /* Result of full matrix-vector multiply.             */
 int   nrhs,      /* Number of right-hand sides.                        */
 int   knsupc,    /* Size of supernode k.                               */
 int_t k,         /* The k-th component of X.                           */
 int_t *fmod,     /* Modification count for L-solve.                    */
 int_t nlb,       /* Number of L blocks.                                */
 int_t lptr,      /* Starting position in lsub[*].                      */
 int_t luptr,     /* Starting position in lusup[*].                     */
 int_t *xsup,
 gridinfo_t *grid,
 LocalLU_t *Llu,
 MPI_Request send_req[], /* input/output */
 SuperLUStat_t *stat
)
{
    // // // // double alpha = 1.0, beta = 0.0;
    // // // // double *lusup, *lusup1;
    // // // // double *dest;
	// // // // int    iam, iknsupc, myrow, nbrow, nsupr, nsupr1, p, pi;
    // // // // int_t  i, ii, ik, il, ikcol, irow, j, lb, lk, lib, rel;
    // // // // int_t  *lsub, *lsub1, nlb1, lptr1, luptr1;
    // // // // int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    // // // // int_t  *frecv = Llu->frecv;
    // // // // int_t  **fsendx_plist = Llu->fsendx_plist;
    // // // // MPI_Status status;
    // // // // int test_flag;
	
// // // // #if ( PROFlevel>=1 )
    // // // // double t1, t2;
    // // // // float msg_vol = 0, msg_cnt = 0;
// // // // #endif 

	
// // // // #if ( PROFlevel>=1 )
		// // // // TIC(t1);
// // // // #endif	
	
    // // // // iam = grid->iam;
    // // // // myrow = MYROW( iam, grid );
    // // // // lk = LBj( k, grid ); /* Local block number, column-wise. */
    // // // // lsub = Llu->Lrowind_bc_ptr[lk];
    // // // // lusup = Llu->Lnzval_bc_ptr[lk];
    // // // // nsupr = lsub[1];
	
    // // // // for (lb = 0; lb < nlb; ++lb) {
	// // // // ik = lsub[lptr]; /* Global block number, row-wise. */
	// // // // nbrow = lsub[lptr+1];
// // // // #ifdef _CRAY
	// // // // SGEMM( ftcs2, ftcs2, &nbrow, &nrhs, &knsupc,
	      // // // // &alpha, &lusup[luptr], &nsupr, xk,
	      // // // // &knsupc, &beta, rtemp, &nbrow );
// // // // #elif defined (USE_VENDOR_BLAS)
	// // // // dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
	       // // // // &alpha, &lusup[luptr], &nsupr, xk,
	       // // // // &knsupc, &beta, rtemp, &nbrow, 1, 1 );
// // // // #else
	// // // // dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
	       // // // // &alpha, &lusup[luptr], &nsupr, xk,
	       // // // // &knsupc, &beta, rtemp, &nbrow );
// // // // #endif
	// // // // stat->ops[SOLVE] += 2 * nbrow * nrhs * knsupc + nbrow * nrhs;
   
	// // // // lk = LBi( ik, grid ); /* Local block number, row-wise. */
	// // // // iknsupc = SuperSize( ik );
	// // // // il = LSUM_BLK( lk );
	// // // // dest = &lsum[il];
	// // // // lptr += LB_DESCRIPTOR;
	// // // // rel = xsup[ik]; /* Global row index of block ik. */
	// // // // for (i = 0; i < nbrow; ++i) {
	    // // // // irow = lsub[lptr++] - rel; /* Relative row. */
	    // // // // RHS_ITERATE(j)
		// // // // dest[irow + j*iknsupc] -= rtemp[i + j*nbrow];
	// // // // }
	// // // // luptr += nbrow;



// // // // #if ( PROFlevel>=1 )
		// // // // TOC(t2, t1);
		// // // // stat->utime[SOL_GEMM] += t2;
	
// // // // #endif	



	
	// // // // if ( (--fmod[lk])==0 ) { /* Local accumulation done. */
	    // // // // ikcol = PCOL( ik, grid );
	    // // // // p = PNUM( myrow, ikcol, grid );
	    // // // // if ( iam != p ) {
// // // // #ifdef ISEND_IRECV
		// // // // MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   // // // // MPI_DOUBLE, p, LSUM, grid->comm,
                           // // // // &send_req[Llu->SolveMsgSent++] );
// // // // #else
// // // // #ifdef BSEND
		// // // // MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   // // // // MPI_DOUBLE, p, LSUM, grid->comm );
// // // // #else
		// // // // MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			 // // // // MPI_DOUBLE, p, LSUM, grid->comm );
// // // // #endif
// // // // #endif
// // // // #if ( DEBUGlevel>=2 )
		// // // // printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
		       // // // // iam, lsum[il-LSUM_H], iknsupc*nrhs+LSUM_H, p);
// // // // #endif
	    // // // // } else { /* Diagonal process: X[i] += lsum[i]. */
		// // // // ii = X_BLK( lk );
		// // // // RHS_ITERATE(j)
		    // // // // for (i = 0; i < iknsupc; ++i)
			// // // // x[i + ii + j*iknsupc] += lsum[i + il + j*iknsupc];
		// // // // if ( frecv[lk]==0 ) { /* Becomes a leaf node. */
		    // // // // fmod[lk] = -1; /* Do not solve X[k] in the future. */
		    // // // // lk = LBj( ik, grid );/* Local block number, column-wise. */
		    // // // // lsub1 = Llu->Lrowind_bc_ptr[lk];
		    // // // // lusup1 = Llu->Lnzval_bc_ptr[lk];
		    // // // // nsupr1 = lsub1[1];
			
			
// // // // #if ( PROFlevel>=1 )
		// // // // TIC(t1);
// // // // #endif				
			
// // // // #ifdef _CRAY
		    // // // // STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
			  // // // // lusup1, &nsupr1, &x[ii], &iknsupc);
// // // // #elif defined (USE_VENDOR_BLAS)
		   // // // // dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha, 
			// // // // lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);		   
// // // // #else
		    // // // // dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha, 
			   // // // // lusup1, &nsupr1, &x[ii], &iknsupc);
// // // // #endif


// // // // #if ( PROFlevel>=1 )
		// // // // TOC(t2, t1);
		// // // // stat->utime[SOL_TRSM] += t2;
	
// // // // #endif	


		    // // // // stat->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;
// // // // #if ( DEBUGlevel>=2 )
		    // // // // printf("(%2d) Solve X[%2d]\n", iam, ik);
// // // // #endif
		
		    // // // // /*
		     // // // // * Send Xk to process column Pc[k].
		     // // // // */			 
		    // // // // for (p = 0; p < grid->nprow; ++p) {
			// // // // if ( fsendx_plist[lk][p] != EMPTY ) {
			    // // // // pi = PNUM( p, ikcol, grid );
// // // // #ifdef ISEND_IRECV
			    // // // // MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       // // // // MPI_DOUBLE, pi, Xk, grid->comm,
				       // // // // &send_req[Llu->SolveMsgSent++] );
// // // // #else
// // // // #ifdef BSEND
			    // // // // MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       // // // // MPI_DOUBLE, pi, Xk, grid->comm );
// // // // #else
			    // // // // MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				     // // // // MPI_DOUBLE, pi, Xk, grid->comm );
// // // // #endif
// // // // #endif
// // // // #if ( DEBUGlevel>=2 )
			    // // // // printf("(%2d) Sent X[%2.0f] to P %2d\n",
				   // // // // iam, x[ii-XK_H], pi);
// // // // #endif
			// // // // }
                    // // // // }
		    // // // // /*
		     // // // // * Perform local block modifications.
		     // // // // */
		    // // // // nlb1 = lsub1[0] - 1;
		    // // // // lptr1 = BC_HEADER + LB_DESCRIPTOR + iknsupc;
		    // // // // luptr1 = iknsupc; /* Skip diagonal block L(I,I). */

		    // // // // dlsum_fmod(lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
			       // // // // fmod, nlb1, lptr1, luptr1, xsup,
			       // // // // grid, Llu, send_req, stat);
		// // // // } /* if frecv[lk] == 0 */
	    // // // // } /* if iam == p */
	// // // // } /* if fmod[lk] == 0 */

    // // // // } /* for lb ... */
} /* dLSUM_FMOD */


/************************************************************************/
void dlsum_bmod
/************************************************************************/
(
 double *lsum,        /* Sum of local modifications.                    */
 double *x,           /* X array (local).                               */
 double *xk,          /* X[k].                                          */
 int    nrhs,	      /* Number of right-hand sides.                    */
 int_t  k,            /* The k-th component of X.                       */
 int_t  *bmod,        /* Modification count for L-solve.                */
 int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
 Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
 int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
 int_t  *xsup,
 gridinfo_t *grid,
 LocalLU_t *Llu,
 MPI_Request send_req[], /* input/output */
 SuperLUStat_t *stat
 )
{
/*
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
 */
    // // // // double alpha = 1.0, beta = 0.0;
    // // // // int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
    // // // // int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
           // // // // j, jj, lk, lk1, nub, ub, uptr;
    // // // // int_t  *usub;
    // // // // double *uval, *dest, *y;
    // // // // int_t  *lsub;
    // // // // double *lusup;
    // // // // int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    // // // // int_t  *brecv = Llu->brecv;
    // // // // int_t  **bsendx_plist = Llu->bsendx_plist;
    // // // // MPI_Status status;
    // // // // int test_flag;

    // // // // iam = grid->iam;
    // // // // myrow = MYROW( iam, grid );
    // // // // knsupc = SuperSize( k );
    // // // // lk = LBj( k, grid ); /* Local block number, column-wise. */
    // // // // nub = Urbs[lk];      /* Number of U blocks in block column lk */

    // // // // for (ub = 0; ub < nub; ++ub) {
	// // // // ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
	// // // // usub = Llu->Ufstnz_br_ptr[ik];
	// // // // uval = Llu->Unzval_br_ptr[ik];
	// // // // i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
	// // // // i += UB_DESCRIPTOR;
	// // // // il = LSUM_BLK( ik );
	// // // // gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
	// // // // iknsupc = SuperSize( gik );
	// // // // ikfrow = FstBlockC( gik );
	// // // // iklrow = FstBlockC( gik+1 );

	// // // // RHS_ITERATE(j) {
	    // // // // dest = &lsum[il + j*iknsupc];
	    // // // // y = &xk[j*knsupc];
	    // // // // uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
	    // // // // for (jj = 0; jj < knsupc; ++jj) {
		// // // // fnz = usub[i + jj];
		// // // // if ( fnz < iklrow ) { /* Nonzero segment. */
		    // // // // /* AXPY */
		    // // // // for (irow = fnz; irow < iklrow; ++irow)
			// // // // dest[irow - ikfrow] -= uval[uptr++] * y[jj];
		    // // // // stat->ops[SOLVE] += 2 * (iklrow - fnz);
		// // // // }
	    // // // // } /* for jj ... */
	// // // // }

	// // // // if ( (--bmod[ik]) == 0 ) { /* Local accumulation done. */
	    // // // // gikcol = PCOL( gik, grid );
	    // // // // p = PNUM( myrow, gikcol, grid );
	    // // // // if ( iam != p ) {
// // // // #ifdef ISEND_IRECV
		// // // // MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   // // // // MPI_DOUBLE, p, LSUM, grid->comm,
                           // // // // &send_req[Llu->SolveMsgSent++] );
// // // // #else
// // // // #ifdef BSEND
		// // // // MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   // // // // MPI_DOUBLE, p, LSUM, grid->comm );
// // // // #else
		// // // // MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			  // // // // MPI_DOUBLE, p, LSUM, grid->comm );
// // // // #endif
// // // // #endif
// // // // #if ( DEBUGlevel>=2 )
		// // // // printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
		       // // // // iam, lsum[il-LSUM_H], iknsupc*nrhs+LSUM_H, p);
// // // // #endif
	    // // // // } else { /* Diagonal process: X[i] += lsum[i]. */
		// // // // ii = X_BLK( ik );
		// // // // dest = &x[ii];
		// // // // RHS_ITERATE(j)
		    // // // // for (i = 0; i < iknsupc; ++i)
			// // // // dest[i + j*iknsupc] += lsum[i + il + j*iknsupc];
		// // // // if ( !brecv[ik] ) { /* Becomes a leaf node. */
		    // // // // bmod[ik] = -1; /* Do not solve X[k] in the future. */
		    // // // // lk1 = LBj( gik, grid ); /* Local block number. */
		    // // // // lsub = Llu->Lrowind_bc_ptr[lk1];
		    // // // // lusup = Llu->Lnzval_bc_ptr[lk1];
		    // // // // nsupr = lsub[1];
// // // // #ifdef _CRAY
		    // // // // STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &iknsupc, &nrhs, &alpha,
			  // // // // lusup, &nsupr, &x[ii], &iknsupc);
// // // // #elif defined (USE_VENDOR_BLAS)
		// // // // dtrsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha, 
		       // // // // lusup, &nsupr, &x[ii], &iknsupc, 1, 1, 1, 1);	
// // // // #else
		    // // // // dtrsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha, 
			   // // // // lusup, &nsupr, &x[ii], &iknsupc);
// // // // #endif
		    // // // // stat->ops[SOLVE] += iknsupc * (iknsupc + 1) * nrhs;
// // // // #if ( DEBUGlevel>=2 )
		    // // // // printf("(%2d) Solve X[%2d]\n", iam, gik);
// // // // #endif

		    // // // // /*
		     // // // // * Send Xk to process column Pc[k].
		     // // // // */
		    // // // // for (p = 0; p < grid->nprow; ++p) {
			// // // // if ( bsendx_plist[lk1][p] != EMPTY ) {
			    // // // // pi = PNUM( p, gikcol, grid );
// // // // #ifdef ISEND_IRECV
			    // // // // MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       // // // // MPI_DOUBLE, pi, Xk, grid->comm,
				       // // // // &send_req[Llu->SolveMsgSent++] );
// // // // #else
// // // // #ifdef BSEND
			    // // // // MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       // // // // MPI_DOUBLE, pi, Xk, grid->comm );
// // // // #else
			    // // // // MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				     // // // // MPI_DOUBLE, pi, Xk, grid->comm );
// // // // #endif
// // // // #endif
// // // // #if ( DEBUGlevel>=2 )
			    // // // // printf("(%2d) Sent X[%2.0f] to P %2d\n",
				   // // // // iam, x[ii-XK_H], pi);
// // // // #endif
			// // // // }
                     // // // // }
		    // // // // /*
		     // // // // * Perform local block modifications.
		     // // // // */
		    // // // // if ( Urbs[lk1] )
			// // // // dlsum_bmod(lsum, x, &x[ii], nrhs, gik, bmod, Urbs,
				   // // // // Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
				   // // // // send_req, stat);
		// // // // } /* if brecv[ik] == 0 */
	    // // // // }
	// // // // } /* if bmod[ik] == 0 */

    // // // // } /* for ub ... */
	
} /* dlSUM_BMOD */




// /************************************************************************/
// /*! \brief
 // *
 // * <pre>
 // * Purpose
 // * =======
 // *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
 // * </pre>
 // */
// void dlsum_fmod_inv
// /************************************************************************/
// (
 // double *lsum,    /* Sum of local modifications.                        */
 // double *x,       /* X array (local)                                    */
 // double *xk,      /* X[k].                                              */
 // double *rtemp,   /* Result of full matrix-vector multiply.             */
 // int   nrhs,      /* Number of right-hand sides.                        */
 // int   knsupc,    /* Size of supernode k.                               */
 // int_t k,         /* The k-th component of X.                           */
 // int_t *fmod,     /* Modification count for L-solve.                    */
 // int_t nlb,       /* Number of L blocks.                                */
 // int_t lptr,      /* Starting position in lsub[*].                      */
 // int_t luptr,     /* Starting position in lusup[*].                     */
 // int_t *xsup,
 // gridinfo_t *grid,
 // LocalLU_t *Llu,
 // MPI_Request send_req[], /* input/output */
 // SuperLUStat_t *stat
// )
// {
    // double alpha = 1.0, beta = 0.0;
    // double *lusup, *lusup1;
    // double *dest;
    // double *Linv;/* Inverse of diagonal block */    
	// int    iam, iknsupc, myrow, nbrow, nsupr, nsupr1, p, pi;
    // int_t  i, ii, ik, il, ikcol, irow, j, lb, lk, rel, lib;
    // int_t  *lsub, *lsub1, nlb1, lptr1, luptr1;
    // int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    // int_t  *frecv = Llu->frecv;
    // int_t  **fsendx_plist = Llu->fsendx_plist;
    // MPI_Status status;
    // int test_flag;
	// yes_no_t done;
	// BcTree  *LBtree_ptr = Llu->LBtree_ptr;
	// RdTree  *LRtree_ptr = Llu->LRtree_ptr;
	
	
// #if ( PROFlevel>=1 )
    // double t1, t2;
    // float msg_vol = 0, msg_cnt = 0;
// #endif 

	
    // iam = grid->iam;
    // myrow = MYROW( iam, grid );
    // lk = LBj( k, grid ); /* Local block number, column-wise. */
    // lsub = Llu->Lrowind_bc_ptr[lk];
    // lusup = Llu->Lnzval_bc_ptr[lk];
    // nsupr = lsub[1];
	
	// // printf("nlb: %5d\n",nlb);
	// // fflush(stdout);
	
	
	
	
	
    // for (lb = 0; lb < nlb; ++lb) {
		
// #if ( PROFlevel>=1 )
		// TIC(t1);
// #endif	
		
	// ik = lsub[lptr]; /* Global block number, row-wise. */
	// nbrow = lsub[lptr+1];
// #ifdef _CRAY
	// SGEMM( ftcs2, ftcs2, &nbrow, &nrhs, &knsupc,
	      // &alpha, &lusup[luptr], &nsupr, xk,
	      // &knsupc, &beta, rtemp, &nbrow );
// #elif defined (USE_VENDOR_BLAS)
	// dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
	       // &alpha, &lusup[luptr], &nsupr, xk,
	       // &knsupc, &beta, rtemp, &nbrow, 1, 1 );
// #else
	// dgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
	       // &alpha, &lusup[luptr], &nsupr, xk,
	       // &knsupc, &beta, rtemp, &nbrow );
// #endif
	// stat->ops[SOLVE] += 2 * nbrow * nrhs * knsupc + nbrow * nrhs;
   
	// lk = LBi( ik, grid ); /* Local block number, row-wise. */
	// iknsupc = SuperSize( ik );
	// il = LSUM_BLK( lk );
	// dest = &lsum[il];
	// lptr += LB_DESCRIPTOR;
	// rel = xsup[ik]; /* Global row index of block ik. */
	// RHS_ITERATE(j)	
	// for (i = 0; i < nbrow; ++i) {
	    // irow = lsub[lptr++] - rel; /* Relative row. */
		// dest[irow + j*iknsupc] -= rtemp[i + j*nbrow];
	// }
	// luptr += nbrow;



// #if ( PROFlevel>=1 )
		// TOC(t2, t1);
		// stat->utime[SOL_GEMM] += t2;
	
// #endif	
	
	// if ( (--fmod[lk])==0 ) { /* Local accumulation done. */
	    // ikcol = PCOL( ik, grid );
	    // p = PNUM( myrow, ikcol, grid );
	    // if ( iam != p ) {
			// if(frecv[lk]==0){
			// fmod[lk] = -1;
			// RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il - LSUM_H]);
			// }
			
// // #ifdef ISEND_IRECV
		// // MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   // // MPI_DOUBLE, p, LSUM, grid->comm,
                           // // &send_req[Llu->SolveMsgSent++] );
// // #else
// // #ifdef BSEND
		// // MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   // // MPI_DOUBLE, p, LSUM, grid->comm );
// // #else
		// // MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			 // // MPI_DOUBLE, p, LSUM, grid->comm );
// // #endif
// // #endif
// // #if ( DEBUGlevel>=2 )
		// // printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
		       // // iam, lsum[il-LSUM_H], iknsupc*nrhs+LSUM_H, p);
// // #endif




	    // } else { /* Diagonal process: X[i] += lsum[i]. */
		// ii = X_BLK( lk );
		// RHS_ITERATE(j)
		    // for (i = 0; i < iknsupc; ++i)
			// x[i + ii + j*iknsupc] += lsum[i + il + j*iknsupc];
		// if ( frecv[lk]==0 ) { /* Becomes a leaf node. */
		    // fmod[lk] = -1; /* Do not solve X[k] in the future. */
		    // lk = LBj( ik, grid );/* Local block number, column-wise. */
		    // lsub1 = Llu->Lrowind_bc_ptr[lk];
		    // lusup1 = Llu->Lnzval_bc_ptr[lk];
		    // nsupr1 = lsub1[1];
			
			
// #if ( PROFlevel>=1 )
		// TIC(t1);
// #endif				
		
		// if(Llu->inv == 1){
		  // Linv = Llu->Linv_bc_ptr[lk];
// #ifdef _CRAY
		  // SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
			  // &alpha, Linv, &iknsupc, &x[ii],
			  // &iknsupc, &beta, rtemp, &iknsupc );
// #elif defined (USE_VENDOR_BLAS)
		  // dgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
			   // &alpha, Linv, &iknsupc, &x[ii],
			   // &iknsupc, &beta, rtemp, &iknsupc, 1, 1 );
// #else
		  // dgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
			   // &alpha, Linv, &iknsupc, &x[ii],
			   // &iknsupc, &beta, rtemp, &iknsupc );
// #endif   
		  // for (i=0 ; i<iknsupc ; i++){
			// x[ii+i] = rtemp[i];
		  // }		
		// }else{
// #ifdef _CRAY
		    // STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
			  // lusup1, &nsupr1, &x[ii], &iknsupc);
// #elif defined (USE_VENDOR_BLAS)
		   // dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha, 
			// lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);		   
// #else
		    // dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha, 
			   // lusup1, &nsupr1, &x[ii], &iknsupc);
// #endif
		// }
		
// #if ( PROFlevel>=1 )
		// TOC(t2, t1);
		// stat->utime[SOL_TRSM] += t2;
	
// #endif	


		    // stat->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;
// #if ( DEBUGlevel>=2 )
		    // printf("(%2d) Solve X[%2d]\n", iam, ik);
// #endif
		
		    // /*
		     // * Send Xk to process column Pc[k].
		     // */

			// if(LBtree_ptr[lk]!=NULL)
			// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H]);
	
	
		    // // for (p = 0; p < grid->nprow; ++p) {
			// // if ( fsendx_plist[lk][p] != EMPTY ) {
			    // // pi = PNUM( p, ikcol, grid );
// // #ifdef ISEND_IRECV
			    // // MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       // // MPI_DOUBLE, pi, Xk, grid->comm,
				       // // &send_req[Llu->SolveMsgSent++] );
// // #else
// // #ifdef BSEND
			    // // MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       // // MPI_DOUBLE, pi, Xk, grid->comm );
// // #else
			    // // MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				     // // MPI_DOUBLE, pi, Xk, grid->comm );
// // #endif
// // #endif
// // #if ( DEBUGlevel>=2 )
			    // // printf("(%2d) Sent X[%2.0f] to P %2d\n",
				   // // iam, x[ii-XK_H], pi);
// // #endif
			// // }
                    // // }
					
					
					
					
					
		    // /*
		     // * Perform local block modifications.
		     // */
		    // nlb1 = lsub1[0] - 1;
		    // lptr1 = BC_HEADER + LB_DESCRIPTOR + iknsupc;
		    // luptr1 = iknsupc; /* Skip diagonal block L(I,I). */

		    // dlsum_fmod_inv(lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
			       // fmod, nlb1, lptr1, luptr1, xsup,
			       // grid, Llu, send_req, stat);
		// } /* if frecv[lk] == 0 */
	    // } /* if iam == p */
	// } /* if fmod[lk] == 0 */

    // } /* for lb ... */
	
// } /* dLSUM_FMOD_inv */










/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
 * </pre>
 */
void dlsum_fmod_inv
/************************************************************************/
(
 double *lsum,    /* Sum of local modifications.                        */
 double *x,       /* X array (local)                                    */
 double *xk,      /* X[k].                                              */
 double *rtemp,   /* Result of full matrix-vector multiply.             */
 int   nrhs,      /* Number of right-hand sides.                        */
 int   knsupc,    /* Size of supernode k.                               */
 int_t k,         /* The k-th component of X.                           */
 int_t *fmod,     /* Modification count for L-solve.                    */
 int_t nlb,       /* Number of L blocks.                                */
 int_t lptr,      /* Starting position in lsub[*].                      */
 int_t luptr,     /* Starting position in lusup[*].                     */
 int_t *xsup,
 gridinfo_t *grid,
 LocalLU_t *Llu,
 MPI_Request send_req[], /* input/output */
 SuperLUStat_t *stat
)
{
    double alpha = 1.0, beta = 0.0,malpha=-1.0;
    double *lusup, *lusup1;
    double *dest;
    double *Linv;/* Inverse of diagonal block */    
	int    iam, iknsupc, myrow, krow, nbrow, nsupr, nsupr1, p, pi, idx_r;
    int_t  i, ii, ik, il, ikcol, irow, j, lb, lk, rel, lib;
    int_t  *lsub, *lsub1, nlb1, lptr1, luptr1,*lloc;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int_t  *frecv = Llu->frecv;
    int_t  **fsendx_plist = Llu->fsendx_plist;
	int_t  luptr_tmp,lptr1_tmp,maxrecvsz, idx_i, idx_v,idx_n, m, idx_l;
	int thread_id;
	flops_t ops_loc=0.0;         
    MPI_Status status;
    int test_flag;
	yes_no_t done;
	BcTree  *LBtree_ptr = Llu->LBtree_ptr;
	RdTree  *LRtree_ptr = Llu->LRtree_ptr;
	int_t* idx_lsum,idx_lsum1;
	
#if ( PROFlevel>=1 )
    double t1, t2;
    float msg_vol = 0, msg_cnt = 0;
#endif 


if(nlb>0){


#if ( PROFlevel>=1 )
		TIC(t1);
#endif	

    maxrecvsz = sp_ienv_dist(3) * nrhs + SUPERLU_MAX( XK_H, LSUM_H );
    	
    iam = grid->iam;
    myrow = MYROW( iam, grid );
    lk = LBj( k, grid ); /* Local block number, column-wise. */
    lsub = Llu->Lrowind_bc_ptr[lk];
    lusup = Llu->Lnzval_bc_ptr[lk];
    lloc = Llu->Lindval_loc_bc_ptr[lk];
	idx_lsum = Llu->Lrowind_bc_2_lsum[lk];
	
    nsupr = lsub[1];
	 	  
	// printf("nlb: %5d lk: %5d\n",nlb,lk);
	// fflush(stdout);



	krow = PROW( k, grid );
	if(myrow==krow){
		idx_n = 1;
		idx_i = nlb+2;
		idx_v = 2*nlb+3;
		luptr_tmp = lloc[idx_v];
		m = nsupr-knsupc;
	}else{
		idx_n = 0;
		idx_i = nlb;
		idx_v = 2*nlb;
		luptr_tmp = lloc[idx_v];
		m = nsupr;
	}
	  
		// printf("m %5d k %5d \n",m,k);
		// fflush(stdout);	

#ifdef _CRAY
	SGEMM( ftcs2, ftcs2, &m, &nrhs, &knsupc,
	      &alpha, &lusup[luptr_tmp], &nsupr, xk,
	      &knsupc, &beta, rtemp, &m );
#elif defined (USE_VENDOR_BLAS)
	dgemm_( "N", "N", &m, &nrhs, &knsupc,
	       &alpha, &lusup[luptr_tmp], &nsupr, xk,
	       &knsupc, &beta, rtemp, &m, 1, 1 );
#else
	dgemm_( "N", "N", &m, &nrhs, &knsupc,
	       &alpha, &lusup[luptr_tmp], &nsupr, xk,
	       &knsupc, &beta, rtemp, &m );
#endif   


	stat->ops[SOLVE] += 2 * m * nrhs * knsupc;	
		

	
	for (i = 0; i < m*nrhs; ++i) {
		lsum[idx_lsum[i]] -=rtemp[i];
	}

	
	
	
	
#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat->utime[SOL_GEMM] += t2;
	
#endif		
	
			
	
	 

	
	// idx_r=0;
    // for (lb = 0; lb < nlb; ++lb) {
	
	// // printf("ind: %5d val: %5d\n",(lb+1)+(nlb+1), (lb+1)+2*(nlb+1));
	// // fflush(stdout);

	// lptr1_tmp = lloc[lb+idx_i];	

	// ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
	// nbrow = lsub[lptr1_tmp+1];
	


	// lk = LBi( ik, grid ); /* Local block number, row-wise. */
	// iknsupc = SuperSize( ik );
	// il = LSUM_BLK( lk );
	// dest = &lsum[il];
	// // lptr1_tmp += LB_DESCRIPTOR; 
	// rel = xsup[ik]; /* Global row index of block ik. */
	// RHS_ITERATE(j)
	// for (i = 0; i < nbrow; ++i) {
	    // irow = lsub[lptr1_tmp+LB_DESCRIPTOR+i] - rel; /* Relative row. */
		
		// // RHS_ITERATE(j)
		// dest[irow + j*iknsupc] -= rtemp[i + j*m + idx_r];
		
		// // daxpy_(&nrhs,&malpha,&rtemp[i + idx_r],&m,&dest[irow],&iknsupc);
		
		// // lsum[il+irow + j*iknsupc] -= rtemp[i + j*m + idx_r];
	// }
	// idx_r +=nbrow;

		
		

	
	// }

	
// #if ( PROFlevel>=1 )
		// TOC(t2, t1);
		// stat->utime[SOL_GEMM] += t2;
	
// #endif		
	
	
    for (lb = 0; lb < nlb; ++lb) {	
	lk = lloc[lb+idx_n];
	if ( (--fmod[lk])==0 ) { /* Local accumulation done. */
	
		lptr1_tmp = lloc[lb+idx_i];	
		// luptr_tmp = lloc[lb+idx_v];
		
		ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
		lk = LBi( ik, grid ); /* Local block number, row-wise. */	
	
		iknsupc = SuperSize( ik );
		il = LSUM_BLK( lk );

		nbrow = lsub[lptr1_tmp+1];
	
	    ikcol = PCOL( ik, grid );
	    p = PNUM( myrow, ikcol, grid );
	    if ( iam != p ) {
			if(frecv[lk]==0){
			fmod[lk] = -1;
			RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il - LSUM_H]);
			}


	    } else { /* Diagonal process: X[i] += lsum[i]. */

		ii = X_BLK( lk );
		RHS_ITERATE(j)
		    for (i = 0; i < iknsupc; ++i)
			x[i + ii + j*iknsupc] += lsum[i + il + j*iknsupc];
		
		
		if ( frecv[lk]==0 ) { /* Becomes a leaf node. */
		    fmod[lk] = -1; /* Do not solve X[k] in the future. */
		    lk = LBj( ik, grid );/* Local block number, column-wise. */
		    lsub1 = Llu->Lrowind_bc_ptr[lk];
		    lusup1 = Llu->Lnzval_bc_ptr[lk];
		    nsupr1 = lsub1[1];
			
			
#if ( PROFlevel>=1 )
		TIC(t1);
#endif				
		
		if(Llu->inv == 1){
		  Linv = Llu->Linv_bc_ptr[lk];
#ifdef _CRAY
		  SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
			  &alpha, Linv, &iknsupc, &x[ii],
			  &iknsupc, &beta, rtemp, &iknsupc );
#elif defined (USE_VENDOR_BLAS)
		  dgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
			   &alpha, Linv, &iknsupc, &x[ii],
			   &iknsupc, &beta, rtemp, &iknsupc, 1, 1 );
#else
		  dgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
			   &alpha, Linv, &iknsupc, &x[ii],
			   &iknsupc, &beta, rtemp, &iknsupc );
#endif   
		  for (i=0 ; i<iknsupc*nrhs ; i++){
			x[ii+i] = rtemp[i];
		  }		
		}else{
#ifdef _CRAY
		    STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
			  lusup1, &nsupr1, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
		   dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha, 
			lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);		   
#else
		    dtrsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha, 
			   lusup1, &nsupr1, &x[ii], &iknsupc);
#endif
		}
		
#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat->utime[SOL_TRSM] += t2;
	
#endif	


		    stat->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;
#if ( DEBUGlevel>=2 )
		    printf("(%2d) Solve X[%2d]\n", iam, ik);
#endif
		
		    /*
		     * Send Xk to process column Pc[k].
		     */

			if(LBtree_ptr[lk]!=NULL)
			BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H]);
	
		    /*
		     * Perform local block modifications.
		     */
		    nlb1 = lsub1[0] - 1;
		    lptr1 = BC_HEADER + LB_DESCRIPTOR + iknsupc;
		    luptr1 = iknsupc; /* Skip diagonal block L(I,I). */

			
		    dlsum_fmod_inv(lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
			       fmod, nlb1, lptr1, luptr1, xsup,
			       grid, Llu, send_req, stat);

			   
				   
		} /* if frecv[lk] == 0 */
	    } /* if iam == p */
	} /* if fmod[lk] == 0 */

    } /* for lb ... */
	
	} /* if nlb>0*/
} /* dLSUM_FMOD_inv */





/************************************************************************/
void dlsum_bmod_inv
/************************************************************************/
(
 double *lsum,        /* Sum of local modifications.                    */
 double *x,           /* X array (local).                               */
 double *xk,          /* X[k].                                          */
 double *rtemp,   /* Result of full matrix-vector multiply.             */
 int    nrhs,	      /* Number of right-hand sides.                    */
 int_t  k,            /* The k-th component of X.                       */
 int_t  *bmod,        /* Modification count for L-solve.                */
 int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
 Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
 int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
 int_t  *xsup,
 gridinfo_t *grid,
 LocalLU_t *Llu,
 MPI_Request send_req[], /* input/output */
 SuperLUStat_t *stat
 )
{
/*
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
 */
    double alpha = 1.0, beta = 0.0;
    int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
    int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
           j, jj, lk, lk1, nub, ub, uptr;
    int_t  *usub;
    double *uval, *dest, *y;
    int_t  *lsub;
    double *lusup;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int_t  *brecv = Llu->brecv;
    int_t  **bsendx_plist = Llu->bsendx_plist;
    MPI_Status status;
    int test_flag;

    double *Uinv;/* Inverse of diagonal block */    
	
    iam = grid->iam;
    myrow = MYROW( iam, grid );
    knsupc = SuperSize( k );
    lk = LBj( k, grid ); /* Local block number, column-wise. */
    nub = Urbs[lk];      /* Number of U blocks in block column lk */

    for (ub = 0; ub < nub; ++ub) {
	ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
	usub = Llu->Ufstnz_br_ptr[ik];
	uval = Llu->Unzval_br_ptr[ik];
	i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
	i += UB_DESCRIPTOR;
	il = LSUM_BLK( ik );
	gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
	iknsupc = SuperSize( gik );
	ikfrow = FstBlockC( gik );
	iklrow = FstBlockC( gik+1 );

	RHS_ITERATE(j) {
	    dest = &lsum[il + j*iknsupc];
	    y = &xk[j*knsupc];
	    uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
	    for (jj = 0; jj < knsupc; ++jj) {
		fnz = usub[i + jj];
		if ( fnz < iklrow ) { /* Nonzero segment. */
		    /* AXPY */
		    for (irow = fnz; irow < iklrow; ++irow)
			dest[irow - ikfrow] -= uval[uptr++] * y[jj];
		    stat->ops[SOLVE] += 2 * (iklrow - fnz);
		}
	    } /* for jj ... */
	}

	if ( (--bmod[ik]) == 0 ) { /* Local accumulation done. */
	    gikcol = PCOL( gik, grid );
	    p = PNUM( myrow, gikcol, grid );
	    if ( iam != p ) {
#ifdef ISEND_IRECV
		MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   MPI_DOUBLE, p, LSUM, grid->comm,
                           &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
		MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   MPI_DOUBLE, p, LSUM, grid->comm );
#else
		MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			  MPI_DOUBLE, p, LSUM, grid->comm );
#endif
#endif
#if ( DEBUGlevel>=2 )
		printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
		       iam, lsum[il-LSUM_H], iknsupc*nrhs+LSUM_H, p);
#endif
	    } else { /* Diagonal process: X[i] += lsum[i]. */
		ii = X_BLK( ik );
		dest = &x[ii];
		RHS_ITERATE(j)
		    for (i = 0; i < iknsupc; ++i)
			dest[i + j*iknsupc] += lsum[i + il + j*iknsupc];
		if ( !brecv[ik] ) { /* Becomes a leaf node. */
		    bmod[ik] = -1; /* Do not solve X[k] in the future. */
		    lk1 = LBj( gik, grid ); /* Local block number. */
		    lsub = Llu->Lrowind_bc_ptr[lk1];
		    lusup = Llu->Lnzval_bc_ptr[lk1];
		    nsupr = lsub[1];

			if(Llu->inv == 1){
				Uinv = Llu->Uinv_bc_ptr[lk1];  
#ifdef _CRAY
				SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
				  &alpha, Uinv, &iknsupc, &x[ii],
				  &iknsupc, &beta, rtemp, &iknsupc );
#elif defined (USE_VENDOR_BLAS)
			  dgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
				   &alpha, Uinv, &iknsupc, &x[ii],
				   &iknsupc, &beta, rtemp, &iknsupc, 1, 1 );
#else
			  dgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
				   &alpha, Uinv, &iknsupc, &x[ii],
				   &iknsupc, &beta, rtemp, &iknsupc );
#endif	   
			  for (i=0 ; i<iknsupc*nrhs ; i++){
				x[ii+i] = rtemp[i];
			  }		
			}else{
#ifdef _CRAY
				STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &iknsupc, &nrhs, &alpha,
				  lusup, &nsupr, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
				dtrsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha, 
				   lusup, &nsupr, &x[ii], &iknsupc, 1, 1, 1, 1);	
#else
				dtrsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha, 
				   lusup, &nsupr, &x[ii], &iknsupc);
#endif
			}
		    stat->ops[SOLVE] += iknsupc * (iknsupc + 1) * nrhs;
#if ( DEBUGlevel>=2 )
		    printf("(%2d) Solve X[%2d]\n", iam, gik);
#endif

		    /*
		     * Send Xk to process column Pc[k].
		     */
		    for (p = 0; p < grid->nprow; ++p) {
			if ( bsendx_plist[lk1][p] != EMPTY ) {
			    pi = PNUM( p, gikcol, grid );
#ifdef ISEND_IRECV
			    MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       MPI_DOUBLE, pi, Xk, grid->comm,
				       &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
			    MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       MPI_DOUBLE, pi, Xk, grid->comm );
#else
			    MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				     MPI_DOUBLE, pi, Xk, grid->comm );
#endif
#endif
#if ( DEBUGlevel>=2 )
			    printf("(%2d) Sent X[%2.0f] to P %2d\n",
				   iam, x[ii-XK_H], pi);
#endif
			}
                     }
		    /*
		     * Perform local block modifications.
		     */
		    if ( Urbs[lk1] )
			dlsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, gik, bmod, Urbs,
				   Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
				   send_req, stat);
		} /* if brecv[ik] == 0 */
	    }
	} /* if bmod[ik] == 0 */

    } /* for ub ... */

	
} /* dlSUM_BMOD_inv */
