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
 * -- Distributed SuperLU routine (version 7.1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 *
 * Modified:
 *     Feburary 7, 2001    use MPI_Isend/MPI_Irecv
 *     October 2, 2001     use MPI_Isend/MPI_Irecv with MPI_Test
 *     February 8, 2019  version 6.1.1
 *     October 5, 2021   version 7.1.0  disable a few 'omp simd'
 * </pre>
 */

#include "superlu_sdefs.h"
#include "superlu_defs.h"

#ifndef CACHELINE
#define CACHELINE 64  /* bytes, Xeon Phi KNL, Cori haswell, Edision */
#endif

#define ISEND_IRECV

/*
 * Function prototypes
 */
#ifdef _CRAY
fortran void STRSM(_fcd, _fcd, _fcd, _fcd, int*, int*, float*,
		   float*, int*, float*, int*);
fortran void SGEMM(_fcd, _fcd, int*, int*, int*, float*, float*,
		   int*, float*, int*, float*, float*, int*);
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
void slsum_fmod
/************************************************************************/
(
 float *lsum,    /* Sum of local modifications.                        */
 float *x,       /* X array (local)                                    */
 float *xk,      /* X[k].                                              */
 float *rtemp,   /* Result of full matrix-vector multiply.             */
 int   nrhs,      /* Number of right-hand sides.                        */
 int   knsupc,    /* Size of supernode k.                               */
 int_t k,         /* The k-th component of X.                           */
 int *fmod,     /* Modification count for L-solve.                    */
 int_t nlb,       /* Number of L blocks.                                */
 int_t lptr,      /* Starting position in lsub[*].                      */
 int_t luptr,     /* Starting position in lusup[*].                     */
 int_t *xsup,
 gridinfo_t *grid,
 sLocalLU_t *Llu,
 MPI_Request send_req[], /* input/output */
 SuperLUStat_t *stat
)
{
    float alpha = 1.0, beta = 0.0;
    float *lusup, *lusup1;
    float *dest;
    int    iam, iknsupc, myrow, nbrow, nsupr, nsupr1, p, pi;
    int_t  i, ii, ik, il, ikcol, irow, j, lb, lk, lib, rel;
    int_t  *lsub, *lsub1, nlb1, lptr1, luptr1;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int  *frecv = Llu->frecv;
    int  **fsendx_plist = Llu->fsendx_plist;
    MPI_Status status;
    int test_flag;

#if ( PROFlevel>=1 )
	double t1, t2;
	float msg_vol = 0, msg_cnt = 0;
#endif
#if ( PROFlevel>=1 )
	TIC(t1);
#endif

    iam = grid->iam;
    myrow = MYROW( iam, grid );
    lk = LBj( k, grid ); /* Local block number, column-wise. */
    lsub = Llu->Lrowind_bc_ptr[lk];
    lusup = Llu->Lnzval_bc_ptr[lk];
    nsupr = lsub[1];

    for (lb = 0; lb < nlb; ++lb) {
	ik = lsub[lptr]; /* Global block number, row-wise. */
	nbrow = lsub[lptr+1];
#ifdef _CRAY
	SGEMM( ftcs2, ftcs2, &nbrow, &nrhs, &knsupc,
	      &alpha, &lusup[luptr], &nsupr, xk,
	      &knsupc, &beta, rtemp, &nbrow );
#elif defined (USE_VENDOR_BLAS)
	sgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
	       &alpha, &lusup[luptr], &nsupr, xk,
	       &knsupc, &beta, rtemp, &nbrow, 1, 1 );
#else
	sgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
	       &alpha, &lusup[luptr], &nsupr, xk,
	       &knsupc, &beta, rtemp, &nbrow );
#endif
	stat->ops[SOLVE] += 2 * nbrow * nrhs * knsupc + nbrow * nrhs;

	lk = LBi( ik, grid ); /* Local block number, row-wise. */
	iknsupc = SuperSize( ik );
	il = LSUM_BLK( lk );
	dest = &lsum[il];
	lptr += LB_DESCRIPTOR;
	rel = xsup[ik]; /* Global row index of block ik. */
	for (i = 0; i < nbrow; ++i) {
	    irow = lsub[lptr++] - rel; /* Relative row. */
	    RHS_ITERATE(j)
		dest[irow + j*iknsupc] -= rtemp[i + j*nbrow];
	}
	luptr += nbrow;

#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat->utime[SOL_GEMM] += t2;
#endif

	if ( (--fmod[lk])==0 ) { /* Local accumulation done. */
	    ikcol = PCOL( ik, grid );
	    p = PNUM( myrow, ikcol, grid );
	    if ( iam != p ) {
#ifdef ISEND_IRECV
		MPI_Isend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   MPI_FLOAT, p, LSUM, grid->comm,
                           &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
		MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   MPI_FLOAT, p, LSUM, grid->comm );
#else
		MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			 MPI_FLOAT, p, LSUM, grid->comm );
#endif
#endif
#if ( DEBUGlevel>=2 )
		printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
		       iam, lsum[il-LSUM_H], iknsupc*nrhs+LSUM_H, p);
#endif
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
#ifdef _CRAY
		    STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
			  lusup1, &nsupr1, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
		    strsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
			   lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
		    strsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
			   lusup1, &nsupr1, &x[ii], &iknsupc);
#endif
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
		    for (p = 0; p < grid->nprow; ++p) {
			if ( fsendx_plist[lk][p] != EMPTY ) {
			    pi = PNUM( p, ikcol, grid );
#ifdef ISEND_IRECV
			    MPI_Isend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       MPI_FLOAT, pi, Xk, grid->comm,
				       &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
			    MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       MPI_FLOAT, pi, Xk, grid->comm );
#else
			    MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				     MPI_FLOAT, pi, Xk, grid->comm );
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
		    nlb1 = lsub1[0] - 1;
		    lptr1 = BC_HEADER + LB_DESCRIPTOR + iknsupc;
		    luptr1 = iknsupc; /* Skip diagonal block L(I,I). */

		    slsum_fmod(lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
			       fmod, nlb1, lptr1, luptr1, xsup,
			       grid, Llu, send_req, stat);
		} /* if frecv[lk] == 0 */
	    } /* if iam == p */
	} /* if fmod[lk] == 0 */

    } /* for lb ... */

} /* sLSUM_FMOD */


/************************************************************************/
void slsum_bmod
/************************************************************************/
(
 float *lsum,        /* Sum of local modifications.                    */
 float *x,           /* X array (local).                               */
 float *xk,          /* X[k].                                          */
 int    nrhs,	      /* Number of right-hand sides.                    */
 int_t  k,            /* The k-th component of X.                       */
 int  *bmod,        /* Modification count for L-solve.                */
 int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
 Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
 int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
 int_t  *xsup,
 gridinfo_t *grid,
 sLocalLU_t *Llu,
 MPI_Request send_req[], /* input/output */
 SuperLUStat_t *stat
 )
{
/*
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
 */
    float alpha = 1.0, beta = 0.0;
    int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
    int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
           j, jj, lk, lk1, nub, ub, uptr;
    int_t  *usub;
    float *uval, *dest, *y;
    int_t  *lsub;
    float *lusup;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int  *brecv = Llu->brecv;
    int    **bsendx_plist = Llu->bsendx_plist;
    MPI_Status status;
    int test_flag;

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
			   MPI_FLOAT, p, LSUM, grid->comm,
                           &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
		MPI_Bsend( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			   MPI_FLOAT, p, LSUM, grid->comm );
#else
		MPI_Send( &lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H,
			  MPI_FLOAT, p, LSUM, grid->comm );
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
#ifdef _CRAY
		    STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &iknsupc, &nrhs, &alpha,
			  lusup, &nsupr, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
		    strsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
			   lusup, &nsupr, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
		    strsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
			   lusup, &nsupr, &x[ii], &iknsupc);
#endif
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
				       MPI_FLOAT, pi, Xk, grid->comm,
				       &send_req[Llu->SolveMsgSent++] );
#else
#ifdef BSEND
			    MPI_Bsend( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				       MPI_FLOAT, pi, Xk, grid->comm );
#else
			    MPI_Send( &x[ii - XK_H], iknsupc * nrhs + XK_H,
				     MPI_FLOAT, pi, Xk, grid->comm );
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
			slsum_bmod(lsum, x, &x[ii], nrhs, gik, bmod, Urbs,
				   Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
				   send_req, stat);
		} /* if brecv[ik] == 0 */
	    }
	} /* if bmod[ik] == 0 */

    } /* for ub ... */

} /* slSUM_BMOD */



/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
 * </pre>
 */
void slsum_fmod_inv
/************************************************************************/
(
 float *lsum,    /* Sum of local modifications.                        */
 float *x,       /* X array (local)                                    */
 float *xk,      /* X[k].                                              */
 float *rtemp,   /* Result of full matrix-vector multiply.             */
 int   nrhs,      /* Number of right-hand sides.                        */
 int_t k,         /* The k-th component of X.                           */
 int *fmod,     /* Modification count for L-solve.                    */
 int_t *xsup,
 gridinfo_t *grid,
 sLocalLU_t *Llu,
 SuperLUStat_t **stat,
 int_t *leaf_send,
 int_t *nleaf_send,
 int_t sizelsum,
 int_t sizertemp,
 int_t recurlevel,
 int_t maxsuper,
 int thread_id,
 int num_thread
)
{
    float alpha = 1.0, beta = 0.0,malpha=-1.0;
    float *lusup, *lusup1;
    float *dest;
	float *Linv;/* Inverse of diagonal block */
	int    iam, iknsupc, myrow, krow, nbrow, nbrow1, nbrow_ref, nsupr, nsupr1, p, pi, idx_r,m;
	int_t  i, ii,jj, ik, il, ikcol, irow, j, lb, lk, rel, lib,lready;
	int_t  *lsub, *lsub1, nlb1, lptr1, luptr1,*lloc;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int  *frecv = Llu->frecv;
    int  **fsendx_plist = Llu->fsendx_plist;
	int_t  luptr_tmp,luptr_tmp1,lptr1_tmp,maxrecvsz, idx_i, idx_v,idx_n,  idx_l, fmod_tmp, lbstart,lbend,nn,Nchunk,nlb_loc,remainder;
	int thread_id1;
	flops_t ops_loc=0.0;
    MPI_Status status;
    int test_flag;
	yes_no_t done;
	C_Tree  *LBtree_ptr = Llu->LBtree_ptr;
	C_Tree  *LRtree_ptr = Llu->LRtree_ptr;
	int_t* idx_lsum,idx_lsum1;
	float *rtemp_loc;
	int_t ldalsum;
	int_t nleaf_send_tmp;
	int_t lptr;      /* Starting position in lsub[*].                      */
	int_t luptr;     /* Starting position in lusup[*].                     */
	int_t iword = sizeof(int_t);
	int_t dword = sizeof (float);
	int aln_d,aln_i;
	aln_d = 1; //ceil(CACHELINE/(double)dword);
	aln_i = 1; //ceil(CACHELINE/(double)iword);
	int   knsupc;    /* Size of supernode k.                               */
	int_t nlb;       /* Number of L blocks.                                */


	knsupc = SuperSize( k );

	lk = LBj( k, grid ); /* Local block number, column-wise. */
	lsub = Llu->Lrowind_bc_ptr[lk];
	nlb = lsub[0] - 1;


	ldalsum=Llu->ldalsum;

	rtemp_loc = &rtemp[sizertemp* thread_id];

	// #if ( PROFlevel>=1 )
	double t1, t2, t3, t4;
	float msg_vol = 0, msg_cnt = 0;
	// #endif

	if(nlb>0){

		iam = grid->iam;
		myrow = MYROW( iam, grid );

		lusup = Llu->Lnzval_bc_ptr[lk];
		lloc = Llu->Lindval_loc_bc_ptr[lk];

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

		assert(m>0);

		if(m>8*maxsuper){
		// if(0){

			// Nchunk=floor(num_thread/2.0)+1;
			Nchunk=SUPERLU_MIN(num_thread,nlb);
			// Nchunk=1;
			nlb_loc = floor(((double)nlb)/Nchunk);
			remainder = nlb % Nchunk;

#ifdef _OPENMP
#pragma	omp	taskloop private (lptr1,luptr1,nlb1,thread_id1,lsub1,lusup1,nsupr1,Linv,nn,lbstart,lbend,luptr_tmp1,nbrow,lb,lptr1_tmp,rtemp_loc,nbrow_ref,lptr,nbrow1,ik,rel,lk,iknsupc,il,i,irow,fmod_tmp,ikcol,p,ii,jj,t1,t2,j,nleaf_send_tmp) untied nogroup
#endif
			for (nn=0;nn<Nchunk;++nn){

#ifdef _OPENMP
			    thread_id1 = omp_get_thread_num ();
#else
			    thread_id1 = 0;
#endif
			    rtemp_loc = &rtemp[sizertemp* thread_id1];

				if(nn<remainder){
					lbstart = nn*(nlb_loc+1);
					lbend = (nn+1)*(nlb_loc+1);
				}else{
					lbstart = remainder+nn*nlb_loc;
					lbend = remainder + (nn+1)*nlb_loc;
				}

				if(lbstart<lbend){

#if ( PROFlevel>=1 )
					TIC(t1);
#endif
					luptr_tmp1 = lloc[lbstart+idx_v];
					nbrow=0;
					for (lb = lbstart; lb < lbend; ++lb){
						lptr1_tmp = lloc[lb+idx_i];
						nbrow += lsub[lptr1_tmp+1];
					}

#ifdef _CRAY
					SGEMM( ftcs2, ftcs2, &nbrow, &nrhs, &knsupc,
						  &alpha, &lusup[luptr_tmp1], &nsupr, xk,
						  &knsupc, &beta, rtemp_loc, &nbrow );
#elif defined (USE_VENDOR_BLAS)
					sgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
						   &alpha, &lusup[luptr_tmp1], &nsupr, xk,
						   &knsupc, &beta, rtemp_loc, &nbrow, 1, 1 );
#else
					sgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
						   &alpha, &lusup[luptr_tmp1], &nsupr, xk,
						   &knsupc, &beta, rtemp_loc, &nbrow );
#endif

					nbrow_ref=0;
					for (lb = lbstart; lb < lbend; ++lb){
					    lptr1_tmp = lloc[lb+idx_i];
					    lptr= lptr1_tmp+2;
					    nbrow1 = lsub[lptr1_tmp+1];
					    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
					    rel = xsup[ik]; /* Global row index of block ik. */

					    lk = LBi( ik, grid ); /* Local block number, row-wise. */

					    iknsupc = SuperSize( ik );
					    il = LSUM_BLK( lk );

					    RHS_ITERATE(j)
		#ifdef _OPENMP
		#pragma omp simd
		#endif
						for (i = 0; i < nbrow1; ++i) {
					   	    irow = lsub[lptr+i] - rel; /* Relative row. */
						    lsum[il+irow + j*iknsupc+sizelsum*thread_id1] -= rtemp_loc[nbrow_ref+i + j*nbrow];
						}
						nbrow_ref+=nbrow1;
					} /* endd for lb ... */

#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat[thread_id1]->utime[SOL_GEMM] += t2;
#endif

					for (lb=lbstart;lb<lbend;lb++){
					    lk = lloc[lb+idx_n];
#ifdef _OPENMP
#pragma omp atomic capture
#endif
					    fmod_tmp=--fmod[lk*aln_i];

					    if ( fmod_tmp==0 ) { /* Local accumulation done. */

						lptr1_tmp = lloc[lb+idx_i];

						ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
						lk = LBi( ik, grid ); /* Local block number, row-wise. */

						iknsupc = SuperSize( ik );
						il = LSUM_BLK( lk );

						ikcol = PCOL( ik, grid );
						p = PNUM( myrow, ikcol, grid );
						if ( iam != p ) {
						    for (ii=1;ii<num_thread;ii++)
						#ifdef _OPENMP
						#pragma omp simd
						#endif
							for (jj=0;jj<iknsupc*nrhs;jj++)
								lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

#ifdef _OPENMP
#pragma omp atomic capture
#endif
							nleaf_send_tmp = ++nleaf_send[0];
							leaf_send[(nleaf_send_tmp-1)*aln_i] = -lk-1;
							// RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il - LSUM_H ],'s');

						} else { /* Diagonal process: X[i] += lsum[i]. */

#if ( PROFlevel>=1 )
							TIC(t1);
#endif
							for (ii=1;ii<num_thread;ii++)
						#ifdef _OPENMP
						#pragma omp simd
						#endif
							    for (jj=0;jj<iknsupc*nrhs;jj++)
								lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

							ii = X_BLK( lk );
							RHS_ITERATE(j)
						#ifdef _OPENMP
						#pragma omp simd
						#endif
							    for (i = 0; i < iknsupc; ++i)
								x[i + ii + j*iknsupc] += lsum[i + il + j*iknsupc ];

							// fmod[lk] = -1; /* Do not solve X[k] in the future. */
							lk = LBj( ik, grid );/* Local block number, column-wise. */
							lsub1 = Llu->Lrowind_bc_ptr[lk];
							lusup1 = Llu->Lnzval_bc_ptr[lk];
							nsupr1 = lsub1[1];

							if(Llu->inv == 1){
								Linv = Llu->Linv_bc_ptr[lk];


#ifdef _CRAY
								SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
										&alpha, Linv, &iknsupc, &x[ii],
										&iknsupc, &beta, rtemp_loc, &iknsupc );
#elif defined (USE_VENDOR_BLAS)
								sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
										&alpha, Linv, &iknsupc, &x[ii],
										&iknsupc, &beta, rtemp_loc, &iknsupc, 1, 1 );
#else
								sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
										&alpha, Linv, &iknsupc, &x[ii],
										&iknsupc, &beta, rtemp_loc, &iknsupc );
#endif
							#ifdef _OPENMP
							#pragma omp simd
							#endif
								for (i=0 ; i<iknsupc*nrhs ; i++){
									x[ii+i] = rtemp_loc[i];
								}

							}else{
#ifdef _CRAY
								STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
											lusup1, &nsupr1, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
								strsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
										lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
								strsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
										lusup1, &nsupr1, &x[ii], &iknsupc);

#endif
							} /* end else */

#if ( PROFlevel>=1 )
							TOC(t2, t1);
							stat[thread_id1]->utime[SOL_TRSM] += t2;

#endif

							stat[thread_id1]->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;

#if ( DEBUGlevel>=2 )
							printf("(%2d) Solve X[%2d]\n", iam, ik);

#endif

							/*
							 * Send Xk to process column Pc[k].
							 */

							if(LBtree_ptr[lk].empty_==NO){
#ifdef _OPENMP
#pragma omp atomic capture
#endif
								nleaf_send_tmp = ++nleaf_send[0];
								leaf_send[(nleaf_send_tmp-1)*aln_i] = lk;
							}

							/*
							 * Perform local block modifications.
							 */

// #ifdef _OPENMP
// #pragma	omp	task firstprivate (Llu,sizelsum,iknsupc,ii,ik,lsub1,x,rtemp,fmod,lsum,stat,nrhs,grid,xsup,recurlevel) private(lptr1,luptr1,nlb1,thread_id1) untied priority(1)
// #endif
							{

								slsum_fmod_inv(lsum, x, &x[ii], rtemp, nrhs, ik,
										fmod, xsup,
										grid, Llu, stat, leaf_send, nleaf_send ,sizelsum,sizertemp,1+recurlevel,maxsuper,thread_id1,num_thread);
							}

							// } /* if frecv[lk] == 0 */
						} /* end if iam == p */
					} /* if fmod[lk] == 0 */
				}

			} /* end tasklook for nn ... */
		}

		}else{

#if ( PROFlevel>=1 )
			TIC(t1);
#endif

#ifdef _CRAY
			SGEMM( ftcs2, ftcs2, &m, &nrhs, &knsupc,
					&alpha, &lusup[luptr_tmp], &nsupr, xk,
					&knsupc, &beta, rtemp_loc, &m );
#elif defined (USE_VENDOR_BLAS)
			sgemm_( "N", "N", &m, &nrhs, &knsupc,
					&alpha, &lusup[luptr_tmp], &nsupr, xk,
					&knsupc, &beta, rtemp_loc, &m, 1, 1 );
#else
			sgemm_( "N", "N", &m, &nrhs, &knsupc,
					&alpha, &lusup[luptr_tmp], &nsupr, xk,
					&knsupc, &beta, rtemp_loc, &m );
#endif

			nbrow=0;
			for (lb = 0; lb < nlb; ++lb){
				lptr1_tmp = lloc[lb+idx_i];
				nbrow += lsub[lptr1_tmp+1];
			}
			nbrow_ref=0;
			for (lb = 0; lb < nlb; ++lb){
				lptr1_tmp = lloc[lb+idx_i];
				lptr= lptr1_tmp+2;
				nbrow1 = lsub[lptr1_tmp+1];
				ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
				rel = xsup[ik]; /* Global row index of block ik. */

				lk = LBi( ik, grid ); /* Local block number, row-wise. */

				iknsupc = SuperSize( ik );
				il = LSUM_BLK( lk );

				RHS_ITERATE(j)
		#ifdef _OPENMP
		#pragma omp simd
		#endif
				    for (i = 0; i < nbrow1; ++i) {
					irow = lsub[lptr+i] - rel; /* Relative row. */

					lsum[il+irow + j*iknsupc+sizelsum*thread_id] -= rtemp_loc[nbrow_ref+i + j*nbrow];
				    }
				nbrow_ref+=nbrow1;
			} /* end for lb ... */

			// TOC(t3, t1);

#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat[thread_id]->utime[SOL_GEMM] += t2;
#endif

			for (lb=0;lb<nlb;lb++){
				lk = lloc[lb+idx_n];
#ifdef _OPENMP
#pragma omp atomic capture
#endif
				fmod_tmp=--fmod[lk*aln_i];

				if ( fmod_tmp==0 ) { /* Local accumulation done. */

				    lptr1_tmp = lloc[lb+idx_i];

				    ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
				    lk = LBi( ik, grid ); /* Local block number, row-wise. */

				    iknsupc = SuperSize( ik );
				    il = LSUM_BLK( lk );
				    ikcol = PCOL( ik, grid );
				    p = PNUM( myrow, ikcol, grid );
				    if ( iam != p ) {
					for (ii=1;ii<num_thread;ii++)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
				    	    for (jj=0;jj<iknsupc*nrhs;jj++)
						lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

#ifdef _OPENMP
#pragma omp atomic capture
#endif
					nleaf_send_tmp = ++nleaf_send[0];
						leaf_send[(nleaf_send_tmp-1)*aln_i] = -lk-1;

				    } else { /* Diagonal process: X[i] += lsum[i]. */

#if ( PROFlevel>=1 )
					TIC(t1);
#endif
					for (ii=1;ii<num_thread;ii++)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
					    for (jj=0;jj<iknsupc*nrhs;jj++)
						lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

					ii = X_BLK( lk );
					RHS_ITERATE(j)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
				 	    for (i = 0; i < iknsupc; ++i)
					        x[i + ii + j*iknsupc] += lsum[i + il + j*iknsupc];

					lk = LBj( ik, grid );/* Local block number, column-wise. */
					lsub1 = Llu->Lrowind_bc_ptr[lk];
					lusup1 = Llu->Lnzval_bc_ptr[lk];
					nsupr1 = lsub1[1];

					if(Llu->inv == 1){
					    Linv = Llu->Linv_bc_ptr[lk];
#ifdef _CRAY
					    SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
							&alpha, Linv, &iknsupc, &x[ii],
							&iknsupc, &beta, rtemp_loc, &iknsupc );
#elif defined (USE_VENDOR_BLAS)
					    sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
							&alpha, Linv, &iknsupc, &x[ii],
							&iknsupc, &beta, rtemp_loc, &iknsupc, 1, 1 );
#else
					    sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
							&alpha, Linv, &iknsupc, &x[ii],
							&iknsupc, &beta, rtemp_loc, &iknsupc );
#endif
					#ifdef _OPENMP
					#pragma omp simd
					#endif
					    for (i=0 ; i<iknsupc*nrhs ; i++){
						x[ii+i] = rtemp_loc[i];
					    }
					}else{
#ifdef _CRAY
					    STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
							lusup1, &nsupr1, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
					    strsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
							lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
					    strsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
									lusup1, &nsupr1, &x[ii], &iknsupc);
#endif
					} /* end else */

#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat[thread_id]->utime[SOL_TRSM] += t2;
#endif

					stat[thread_id]->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;

#if ( DEBUGlevel>=2 )
					printf("(%2d) Solve X[%2d]\n", iam, ik);
#endif

					/*
					 * Send Xk to process column Pc[k].
					 */

					if(LBtree_ptr[lk].empty_==NO){
#ifdef _OPENMP
#pragma omp atomic capture
#endif
						nleaf_send_tmp = ++nleaf_send[0];
						// printf("nleaf_send_tmp %5d lk %5d\n",nleaf_send_tmp);
						leaf_send[(nleaf_send_tmp-1)*aln_i] = lk;
						// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],'s');
					}

					/*
					 * Perform local block modifications.
					 */

// #ifdef _OPENMP
// #pragma	omp	task firstprivate (Llu,sizelsum,iknsupc,ii,ik,lsub1,x,rtemp,fmod,lsum,stat,nrhs,grid,xsup,recurlevel) private(lptr1,luptr1,nlb1) untied priority(1)
// #endif
					{
						slsum_fmod_inv(lsum, x, &x[ii], rtemp, nrhs, ik,
							fmod, xsup,
							grid, Llu, stat, leaf_send, nleaf_send ,sizelsum,sizertemp,1+recurlevel,maxsuper,thread_id,num_thread);
					}

						// } /* if frecv[lk] == 0 */
				} /* end else iam == p */
			} /* if fmod[lk] == 0 */
		}
		// }
}

	stat[thread_id]->ops[SOLVE] += 2 * m * nrhs * knsupc;


} /* if nlb>0*/
} /* sLSUM_FMOD_INV */

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
 * </pre>
 */
void slsum_fmod_inv_master
/************************************************************************/
(
 float *lsum,    /* Sum of local modifications.                        */
 float *x,       /* X array (local)                                    */
 float *xk,      /* X[k].                                              */
 float *rtemp,   /* Result of full matrix-vector multiply.             */
 int   nrhs,      /* Number of right-hand sides.                        */
 int   knsupc,    /* Size of supernode k.                               */
 int_t k,         /* The k-th component of X.                           */
 int *fmod,     /* Modification count for L-solve.                    */
 int_t nlb,       /* Number of L blocks.                                */
 int_t *xsup,
 gridinfo_t *grid,
 sLocalLU_t *Llu,
 SuperLUStat_t **stat,
 int_t sizelsum,
 int_t sizertemp,
 int_t recurlevel,
 int_t maxsuper,
 int thread_id,
 int num_thread
)
{
    float alpha = 1.0, beta = 0.0, malpha=-1.0;
    float *lusup, *lusup1;
    float *dest;
	float *Linv;/* Inverse of diagonal block */
	int    iam, iknsupc, myrow, krow, nbrow, nbrow1, nbrow_ref, nsupr, nsupr1, p, pi, idx_r;
	int_t  i, ii,jj, ik, il, ikcol, irow, j, lb, lk, rel, lib,lready;
	int_t  *lsub, *lsub1, nlb1, lptr1, luptr1,*lloc;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int  *frecv = Llu->frecv;
    int  **fsendx_plist = Llu->fsendx_plist;
	int_t  luptr_tmp,luptr_tmp1,lptr1_tmp,maxrecvsz, idx_i, idx_v,idx_n,  idx_l, fmod_tmp, lbstart,lbend,nn,Nchunk,nlb_loc,remainder;
	int thread_id1;
	int m;
	flops_t ops_loc=0.0;
    MPI_Status status;
    int test_flag;
	yes_no_t done;
	C_Tree  *LBtree_ptr = Llu->LBtree_ptr;
	C_Tree  *LRtree_ptr = Llu->LRtree_ptr;
	int_t* idx_lsum,idx_lsum1;
	float *rtemp_loc;
	int_t ldalsum;
	int_t nleaf_send_tmp;
	int_t lptr;      /* Starting position in lsub[*].                      */
	int_t luptr;     /* Starting position in lusup[*].                     */
	int_t iword = sizeof(int_t);
	int_t dword = sizeof (float);
	int aln_d,aln_i;
	aln_d = 1; //ceil(CACHELINE/(double)dword);
	aln_i = 1; //ceil(CACHELINE/(double)iword);

	ldalsum=Llu->ldalsum;

	rtemp_loc = &rtemp[sizertemp* thread_id];

	// #if ( PROFlevel>=1 )
	double t1, t2, t3, t4;
	float msg_vol = 0, msg_cnt = 0;
	// #endif

	if(nlb>0){

		iam = grid->iam;
		myrow = MYROW( iam, grid );
		lk = LBj( k, grid ); /* Local block number, column-wise. */

		// printf("ya1 %5d k %5d lk %5d\n",thread_id,k,lk);
		// fflush(stdout);

		lsub = Llu->Lrowind_bc_ptr[lk];

		// printf("ya2 %5d k %5d lk %5d\n",thread_id,k,lk);
		// fflush(stdout);

		lusup = Llu->Lnzval_bc_ptr[lk];
		lloc = Llu->Lindval_loc_bc_ptr[lk];
		// idx_lsum = Llu->Lrowind_bc_2_lsum[lk];

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

		assert(m>0);

		if(m>4*maxsuper || nrhs>10){
			// if(m<1){
			// TIC(t1);
			Nchunk=num_thread;
			nlb_loc = floor(((double)nlb)/Nchunk);
			remainder = nlb % Nchunk;

#ifdef _OPENMP
#pragma	omp	taskloop private (lptr1,luptr1,nlb1,thread_id1,lsub1,lusup1,nsupr1,Linv,nn,lbstart,lbend,luptr_tmp1,nbrow,lb,lptr1_tmp,rtemp_loc,nbrow_ref,lptr,nbrow1,ik,rel,lk,iknsupc,il,i,irow,fmod_tmp,ikcol,p,ii,jj,t1,t2,j) untied
#endif
			for (nn=0;nn<Nchunk;++nn){

#ifdef _OPENMP
				thread_id1 = omp_get_thread_num ();
#else
				thread_id1 = 0;
#endif
				rtemp_loc = &rtemp[sizertemp* thread_id1];

				if(nn<remainder){
					lbstart = nn*(nlb_loc+1);
					lbend = (nn+1)*(nlb_loc+1);
				}else{
					lbstart = remainder+nn*nlb_loc;
					lbend = remainder + (nn+1)*nlb_loc;
				}

				if(lbstart<lbend){

#if ( PROFlevel>=1 )
					TIC(t1);
#endif
					luptr_tmp1 = lloc[lbstart+idx_v];
					nbrow=0;
					for (lb = lbstart; lb < lbend; ++lb){
						lptr1_tmp = lloc[lb+idx_i];
						nbrow += lsub[lptr1_tmp+1];
					}

				#ifdef _CRAY
					SGEMM( ftcs2, ftcs2, &nbrow, &nrhs, &knsupc,
						  &alpha, &lusup[luptr_tmp1], &nsupr, xk,
						  &knsupc, &beta, rtemp_loc, &nbrow );
				#elif defined (USE_VENDOR_BLAS)
					sgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
						   &alpha, &lusup[luptr_tmp1], &nsupr, xk,
						   &knsupc, &beta, rtemp_loc, &nbrow, 1, 1 );
				#else
					sgemm_( "N", "N", &nbrow, &nrhs, &knsupc,
						   &alpha, &lusup[luptr_tmp1], &nsupr, xk,
						   &knsupc, &beta, rtemp_loc, &nbrow );
				#endif

					nbrow_ref=0;
					for (lb = lbstart; lb < lbend; ++lb){
						lptr1_tmp = lloc[lb+idx_i];
						lptr= lptr1_tmp+2;
						nbrow1 = lsub[lptr1_tmp+1];
						ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
						rel = xsup[ik]; /* Global row index of block ik. */

						lk = LBi( ik, grid ); /* Local block number, row-wise. */

						iknsupc = SuperSize( ik );
						il = LSUM_BLK( lk );

						RHS_ITERATE(j)
					#ifdef _OPENMP
					#pragma omp simd lastprivate(irow)
					#endif
							for (i = 0; i < nbrow1; ++i) {
								irow = lsub[lptr+i] - rel; /* Relative row. */
								lsum[il+irow + j*iknsupc] -= rtemp_loc[nbrow_ref+i + j*nbrow];
							}
						nbrow_ref+=nbrow1;
					} /* end for lb ... */

#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat[thread_id1]->utime[SOL_GEMM] += t2;
#endif
			} /* end if (lbstart<lbend) ... */

		} /* end taskloop for nn = ... */

		}else{

#if ( PROFlevel>=1 )
			TIC(t1);
#endif

#ifdef _CRAY
			SGEMM( ftcs2, ftcs2, &m, &nrhs, &knsupc,
					&alpha, &lusup[luptr_tmp], &nsupr, xk,
					&knsupc, &beta, rtemp_loc, &m );
#elif defined (USE_VENDOR_BLAS)
			sgemm_( "N", "N", &m, &nrhs, &knsupc,
					&alpha, &lusup[luptr_tmp], &nsupr, xk,
					&knsupc, &beta, rtemp_loc, &m, 1, 1 );
#else
			sgemm_( "N", "N", &m, &nrhs, &knsupc,
					&alpha, &lusup[luptr_tmp], &nsupr, xk,
					&knsupc, &beta, rtemp_loc, &m );
#endif

			nbrow=0;
			for (lb = 0; lb < nlb; ++lb){
				lptr1_tmp = lloc[lb+idx_i];
				nbrow += lsub[lptr1_tmp+1];
			}
			nbrow_ref=0;
			for (lb = 0; lb < nlb; ++lb){
				lptr1_tmp = lloc[lb+idx_i];
				lptr= lptr1_tmp+2;
				nbrow1 = lsub[lptr1_tmp+1];
				ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
				rel = xsup[ik]; /* Global row index of block ik. */

				lk = LBi( ik, grid ); /* Local block number, row-wise. */

				iknsupc = SuperSize( ik );
				il = LSUM_BLK( lk );

				RHS_ITERATE(j)
			#ifdef _OPENMP
			#pragma omp simd lastprivate(irow)
			#endif
					for (i = 0; i < nbrow1; ++i) {
						irow = lsub[lptr+i] - rel; /* Relative row. */

						lsum[il+irow + j*iknsupc+sizelsum*thread_id] -= rtemp_loc[nbrow_ref+i + j*nbrow];
					}
				nbrow_ref+=nbrow1;
			} /* end for lb ... */
#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat[thread_id]->utime[SOL_GEMM] += t2;
#endif
		} /* end else ... */
		// TOC(t3, t1);
		rtemp_loc = &rtemp[sizertemp* thread_id];

		for (lb=0;lb<nlb;lb++){
			lk = lloc[lb+idx_n];

			// #ifdef _OPENMP
			// #pragma omp atomic capture
			// #endif
			fmod_tmp=--fmod[lk*aln_i];


			if ( fmod_tmp==0 ) { /* Local accumulation done. */
				// --fmod[lk];


				lptr1_tmp = lloc[lb+idx_i];
				// luptr_tmp = lloc[lb+idx_v];

				ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
				lk = LBi( ik, grid ); /* Local block number, row-wise. */

				iknsupc = SuperSize( ik );
				il = LSUM_BLK( lk );

				// nbrow = lsub[lptr1_tmp+1];

				ikcol = PCOL( ik, grid );
				p = PNUM( myrow, ikcol, grid );
				if ( iam != p ) {
					// if(frecv[lk]==0){
					// fmod[lk] = -1;

					for (ii=1;ii<num_thread;ii++)
						// if(ii!=thread_id)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
						for (jj=0;jj<iknsupc*nrhs;jj++)
							lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

					// RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(LRtree_ptr[lk],'s')*nrhs+LSUM_H,'s');
					C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
					// }


				} else { /* Diagonal process: X[i] += lsum[i]. */

					// if ( frecv[lk]==0 ) { /* Becomes a leaf node. */
#if ( PROFlevel>=1 )
					TIC(t1);
#endif
					for (ii=1;ii<num_thread;ii++)
						// if(ii!=thread_id)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
						for (jj=0;jj<iknsupc*nrhs;jj++)
							lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

					ii = X_BLK( lk );
					// for (jj=0;jj<num_thread;jj++)
					RHS_ITERATE(j)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
						for (i = 0; i < iknsupc; ++i)
							x[i + ii + j*iknsupc] += lsum[i + il + j*iknsupc ];

					// fmod[lk] = -1; /* Do not solve X[k] in the future. */
					lk = LBj( ik, grid );/* Local block number, column-wise. */
					lsub1 = Llu->Lrowind_bc_ptr[lk];
					lusup1 = Llu->Lnzval_bc_ptr[lk];
					nsupr1 = lsub1[1];

					if(Llu->inv == 1){
						Linv = Llu->Linv_bc_ptr[lk];
#ifdef _CRAY
						SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
								&alpha, Linv, &iknsupc, &x[ii],
								&iknsupc, &beta, rtemp_loc, &iknsupc );
#elif defined (USE_VENDOR_BLAS)
						sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
								&alpha, Linv, &iknsupc, &x[ii],
								&iknsupc, &beta, rtemp_loc, &iknsupc, 1, 1 );
#else
						sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
								&alpha, Linv, &iknsupc, &x[ii],
								&iknsupc, &beta, rtemp_loc, &iknsupc );
#endif
				#ifdef _OPENMP
				#pragma omp simd
				#endif
						for (i=0 ; i<iknsupc*nrhs ; i++){
							x[ii+i] = rtemp_loc[i];
						}
					}else{
#ifdef _CRAY
						STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &iknsupc, &nrhs, &alpha,
								lusup1, &nsupr1, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
						strsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
								lusup1, &nsupr1, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
						strsm_("L", "L", "N", "U", &iknsupc, &nrhs, &alpha,
								lusup1, &nsupr1, &x[ii], &iknsupc);
#endif
					}

#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat[thread_id]->utime[SOL_TRSM] += t2;
#endif

					stat[thread_id]->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;

#if ( DEBUGlevel>=2 )
					printf("(%2d) Solve X[%2d]\n", iam, ik);
#endif

					/*
					 * Send Xk to process column Pc[k].
					 */

					if(LBtree_ptr[lk].empty_==NO) {
						//BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'s')*nrhs+XK_H,'s');
						C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);
					}

					/*
					 * Perform local block modifications.
					 */

// #ifdef _OPENMP
// #pragma	omp	task firstprivate (Llu,sizelsum,iknsupc,ii,ik,lsub1,x,rtemp,fmod,lsum,stat,nrhs,grid,xsup,recurlevel) private(lptr1,luptr1,nlb1,thread_id1) untied priority(1)
// #endif
					{
						nlb1 = lsub1[0] - 1;

						slsum_fmod_inv_master(lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
								fmod, nlb1, xsup,
								grid, Llu, stat,sizelsum,sizertemp,1+recurlevel,maxsuper,thread_id,num_thread);
					}

					// } /* if frecv[lk] == 0 */
				} /* if iam == p */
			} /* if fmod[lk] == 0 */
		}
		// }
		stat[thread_id]->ops[SOLVE] += 2 * m * nrhs * knsupc;
	} /* end if nlb>0*/
} /* end slsum_fmod_inv_master */



/************************************************************************/
void slsum_bmod_inv
/************************************************************************/
(
 float *lsum,        /* Sum of local modifications.                    */
 float *x,           /* X array (local).                               */
 float *xk,          /* X[k].                                          */
 float *rtemp,   /* Result of full matrix-vector multiply.             */
 int    nrhs,	      /* Number of right-hand sides.                    */
 int_t  k,            /* The k-th component of X.                       */
 int *bmod,        /* Modification count for L-solve.                */
 int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
 Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
 int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
 int_t  *xsup,
 gridinfo_t *grid,
 sLocalLU_t *Llu,
 SuperLUStat_t **stat,
 int_t* root_send,
 int_t* nroot_send,
 int_t sizelsum,
 int_t sizertemp,
 int thread_id,
 int num_thread
 )
{
	/*
	 * Purpose
	 * =======
	 *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
	 */
    float alpha = 1.0, beta = 0.0;
	int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
	int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
	       j, jj, lk, lk1, nub, ub, uptr;
	int_t  *usub;
	float *uval, *dest, *y;
	int_t  *lsub;
	float *lusup;
	int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
	int  *brecv = Llu->brecv;
	int    **bsendx_plist = Llu->bsendx_plist;
	C_Tree  *UBtree_ptr = Llu->UBtree_ptr;
	C_Tree  *URtree_ptr = Llu->URtree_ptr;
	MPI_Status status;
	int test_flag;
	int bmod_tmp;
	int thread_id1;
	float *rtemp_loc;
	int_t nroot_send_tmp;
	float *Uinv;/* Inverse of diagonal block */
	float temp;
	double t1, t2;
	float msg_vol = 0, msg_cnt = 0;
	int_t Nchunk, nub_loc,remainder,nn,lbstart,lbend;
	int_t iword = sizeof(int_t);
	int_t dword = sizeof(float);
	int aln_d,aln_i;
	aln_d = 1; //ceil(CACHELINE/(double)dword);
	aln_i = 1; //ceil(CACHELINE/(double)iword);

	iam = grid->iam;
	myrow = MYROW( iam, grid );
	knsupc = SuperSize( k );
	lk = LBj( k, grid ); /* Local block number, column-wise. */
	nub = Urbs[lk];      /* Number of U blocks in block column lk */

	if(Llu->Unnz[lk]>knsupc*64 || nub>16){
	// if(nub>num_thread){
	// if(nub>16){
	// // // // if(Urbs2[lk]>num_thread){
	// if(Urbs2[lk]>0){
		Nchunk=SUPERLU_MIN(num_thread,nub);
		nub_loc = floor(((double)nub)/Nchunk);
		remainder = nub % Nchunk;
		// printf("Unnz: %5d nub: %5d knsupc: %5d\n",Llu->Unnz[lk],nub,knsupc);
#ifdef _OPENMP
#pragma	omp	taskloop firstprivate (stat) private (thread_id1,Uinv,nn,lbstart,lbend,ub,temp,rtemp_loc,ik,lk1,gik,gikcol,usub,uval,lsub,lusup,iknsupc,il,i,irow,bmod_tmp,p,ii,jj,t1,t2,j,ikfrow,iklrow,dest,y,uptr,fnz,nsupr) untied nogroup
#endif
		for (nn=0;nn<Nchunk;++nn){

#ifdef _OPENMP
			thread_id1 = omp_get_thread_num ();
#else
			thread_id1 = 0;
#endif
			rtemp_loc = &rtemp[sizertemp* thread_id1];

			if(nn<remainder){
				lbstart = nn*(nub_loc+1);
				lbend = (nn+1)*(nub_loc+1);
			}else{
				lbstart = remainder+nn*nub_loc;
				lbend = remainder + (nn+1)*nub_loc;
			}
			for (ub = lbstart; ub < lbend; ++ub){
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

#if ( PROFlevel>=1 )
				TIC(t1);
#endif

				RHS_ITERATE(j) {
					dest = &lsum[il + j*iknsupc+sizelsum*thread_id1];
					y = &xk[j*knsupc];
					uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
					for (jj = 0; jj < knsupc; ++jj) {
						fnz = usub[i + jj];
						if ( fnz < iklrow ) { /* Nonzero segment. */
							/* AXPY */
//#ifdef _OPENMP  
//#pragma omp simd // In complex case, this SIMD loop has 2 instructions, the compiler may generate incoreect code, so need to disable this omp simd
//#endif
							for (irow = fnz; irow < iklrow; ++irow)
								dest[irow - ikfrow] -= uval[uptr++] * y[jj];
								stat[thread_id1]->ops[SOLVE] += 2 * (iklrow - fnz);
						}
					} /* end for jj ... */
				}

#if ( PROFlevel>=1 )
				TOC(t2, t1);
				stat[thread_id1]->utime[SOL_GEMM] += t2;
#endif

		#ifdef _OPENMP
		#pragma omp atomic capture
		#endif
				bmod_tmp=--bmod[ik*aln_i];

				if ( bmod_tmp == 0 ) { /* Local accumulation done. */
					gikcol = PCOL( gik, grid );
					p = PNUM( myrow, gikcol, grid );
					if ( iam != p ) {
						for (ii=1;ii<num_thread;ii++)
							// if(ii!=thread_id1)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
							for (jj=0;jj<iknsupc*nrhs;jj++)
								lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

#ifdef _OPENMP
#pragma omp atomic capture
#endif
						nroot_send_tmp = ++nroot_send[0];
						root_send[(nroot_send_tmp-1)*aln_i] = -ik-1;
						// RdTree_forwardMessageSimple(URtree_ptr[ik],&lsum[il - LSUM_H ],'s');

		#if ( DEBUGlevel>=2 )
						printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
								iam, lsum[il-LSUM_H], iknsupc*nrhs+LSUM_H, p);
		#endif
					} else { /* Diagonal process: X[i] += lsum[i]. */

#if ( PROFlevel>=1 )
						TIC(t1);
#endif
						for (ii=1;ii<num_thread;ii++)
							// if(ii!=thread_id1)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
							for (jj=0;jj<iknsupc*nrhs;jj++)
								lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

						ii = X_BLK( ik );
						dest = &x[ii];

						RHS_ITERATE(j)
				#ifdef _OPENMP
				#pragma omp simd
				#endif
							for (i = 0; i < iknsupc; ++i)
							    dest[i + j*iknsupc] += lsum[i + il + j*iknsupc];

						// if ( !brecv[ik] ) { /* Becomes a leaf node. */
							// bmod[ik] = -1; /* Do not solve X[k] in the future. */
							lk1 = LBj( gik, grid ); /* Local block number. */
							lsub = Llu->Lrowind_bc_ptr[lk1];
							lusup = Llu->Lnzval_bc_ptr[lk1];
							nsupr = lsub[1];

							if(Llu->inv == 1){
								Uinv = Llu->Uinv_bc_ptr[lk1];
		#ifdef _CRAY
								SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
										&alpha, Uinv, &iknsupc, &x[ii],
										&iknsupc, &beta, rtemp_loc, &iknsupc );
		#elif defined (USE_VENDOR_BLAS)
								sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
										&alpha, Uinv, &iknsupc, &x[ii],
										&iknsupc, &beta, rtemp_loc, &iknsupc, 1, 1 );
		#else
								sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
										&alpha, Uinv, &iknsupc, &x[ii],
										&iknsupc, &beta, rtemp_loc, &iknsupc );
		#endif
								#ifdef _OPENMP
								#pragma omp simd
								#endif
								for (i=0 ; i<iknsupc*nrhs ; i++){
									x[ii+i] = rtemp_loc[i];
								}
							}else{
		#ifdef _CRAY
								STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &iknsupc, &nrhs, &alpha,
										lusup, &nsupr, &x[ii], &iknsupc);
		#elif defined (USE_VENDOR_BLAS)
								strsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
										lusup, &nsupr, &x[ii], &iknsupc, 1, 1, 1, 1);
		#else
								strsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
										lusup, &nsupr, &x[ii], &iknsupc);
		#endif
							}

		#if ( PROFlevel>=1 )
							TOC(t2, t1);
							stat[thread_id1]->utime[SOL_TRSM] += t2;
		#endif
							stat[thread_id1]->ops[SOLVE] += iknsupc * (iknsupc + 1) * nrhs;

		#if ( DEBUGlevel>=2 )
							printf("(%2d) Solve X[%2d]\n", iam, gik);
		#endif

							/*
							 * Send Xk to process column Pc[k].
							 */

							 // for (i=0 ; i<iknsupc*nrhs ; i++){
								// printf("xre: %f\n",x[ii+i]);
								// fflush(stdout);
							// }
							if(UBtree_ptr[lk1].empty_==NO){
#ifdef _OPENMP
#pragma omp atomic capture
#endif
							    nroot_send_tmp = ++nroot_send[0];
							    root_send[(nroot_send_tmp-1)*aln_i] = lk1;
							// BcTree_forwardMessageSimple(UBtree_ptr[lk1],&x[ii - XK_H],'s');
							}

							/*
							 * Perform local block modifications.
							 */
							if ( Urbs[lk1] ){
// #ifdef _OPENMP
// #pragma	omp	task firstprivate (Ucb_indptr,Ucb_valptr,Llu,sizelsum,ii,gik,x,rtemp,bmod,Urbs,lsum,stat,nrhs,grid,xsup) untied
// #endif
								{
								slsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, gik, bmod, Urbs,
										Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
										stat, root_send, nroot_send, sizelsum,sizertemp,thread_id1,num_thread);
								}
							}
						// } /* if brecv[ik] == 0 */
					}
				} /* if bmod[ik] == 0 */
			} /* end for ub = ... */
		} /* end for taskloop nn = ... */

	} else {

		rtemp_loc = &rtemp[sizertemp* thread_id];

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

#if ( PROFlevel>=1 )
		TIC(t1);
#endif
			RHS_ITERATE(j) {
				dest = &lsum[il + j*iknsupc+sizelsum*thread_id];
				y = &xk[j*knsupc];
				uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
				for (jj = 0; jj < knsupc; ++jj) {
					fnz = usub[i + jj];
					if ( fnz < iklrow ) { /* Nonzero segment. */
						/* AXPY */
//#ifdef _OPENMP
//#pragma omp simd // In complex case, this SIMD loop has 2 instructions, the compiler may generate incoreect code, so need to disable this omp simd
//#endif
						for (irow = fnz; irow < iklrow; ++irow)
						    dest[irow - ikfrow] -= uval[uptr++] * y[jj];
						stat[thread_id]->ops[SOLVE] += 2 * (iklrow - fnz);
					}
				} /* for jj ... */
			}

#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat[thread_id]->utime[SOL_GEMM] += t2;
#endif

	#ifdef _OPENMP
	#pragma omp atomic capture
	#endif
			bmod_tmp=--bmod[ik*aln_i];

			if ( bmod_tmp == 0 ) { /* Local accumulation done. */
				gikcol = PCOL( gik, grid );
				p = PNUM( myrow, gikcol, grid );
				if ( iam != p ) {
					for (ii=1;ii<num_thread;ii++)
						// if(ii!=thread_id)
			#ifdef _OPENMP
			#pragma omp simd
			#endif
						for (jj=0;jj<iknsupc*nrhs;jj++)
							lsum[il + jj ] += lsum[il + jj + ii*sizelsum];
#ifdef _OPENMP
#pragma omp atomic capture
#endif
					nroot_send_tmp = ++nroot_send[0];
					root_send[(nroot_send_tmp-1)*aln_i] = -ik-1;
					// RdTree_forwardMessageSimple(URtree_ptr[ik],&lsum[il - LSUM_H ],'s');

	#if ( DEBUGlevel>=2 )
					printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
							iam, lsum[il-LSUM_H], iknsupc*nrhs+LSUM_H, p);
	#endif
				} else { /* Diagonal process: X[i] += lsum[i]. */

#if ( PROFlevel>=1 )
					TIC(t1);
#endif

					for (ii=1;ii<num_thread;ii++)
						// if(ii!=thread_id)
			#ifdef _OPENMP
			#pragma omp simd
			#endif
						for (jj=0;jj<iknsupc*nrhs;jj++)
								lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

					ii = X_BLK( ik );
					dest = &x[ii];

					RHS_ITERATE(j)
			#ifdef _OPENMP
			#pragma omp simd
			#endif
						for (i = 0; i < iknsupc; ++i)
							dest[i + j*iknsupc] += lsum[i + il + j*iknsupc];

					// if ( !brecv[ik] ) { /* Becomes a leaf node. */
						// bmod[ik] = -1; /* Do not solve X[k] in the future. */
						lk1 = LBj( gik, grid ); /* Local block number. */
						lsub = Llu->Lrowind_bc_ptr[lk1];
						lusup = Llu->Lnzval_bc_ptr[lk1];
						nsupr = lsub[1];

						if(Llu->inv == 1){
							Uinv = Llu->Uinv_bc_ptr[lk1];
	#ifdef _CRAY
							SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
									&alpha, Uinv, &iknsupc, &x[ii],
									&iknsupc, &beta, rtemp_loc, &iknsupc );
	#elif defined (USE_VENDOR_BLAS)
							sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
									&alpha, Uinv, &iknsupc, &x[ii],
									&iknsupc, &beta, rtemp_loc, &iknsupc, 1, 1 );
	#else
							sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
									&alpha, Uinv, &iknsupc, &x[ii],
									&iknsupc, &beta, rtemp_loc, &iknsupc );
	#endif
				#ifdef _OPENMP
				#pragma omp simd
				#endif
							for (i=0 ; i<iknsupc*nrhs ; i++){
								x[ii+i] = rtemp_loc[i];
							}
						}else{
	#ifdef _CRAY
							STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &iknsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &iknsupc);
	#elif defined (USE_VENDOR_BLAS)
							strsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &iknsupc, 1, 1, 1, 1);
	#else
							strsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &iknsupc);
	#endif
						}

	#if ( PROFlevel>=1 )
						TOC(t2, t1);
						stat[thread_id]->utime[SOL_TRSM] += t2;
	#endif
						stat[thread_id]->ops[SOLVE] += iknsupc * (iknsupc + 1) * nrhs;
	#if ( DEBUGlevel>=2 )
						printf("(%2d) Solve X[%2d]\n", iam, gik);
	#endif

						/*
						 * Send Xk to process column Pc[k].
						 */

						 // for (i=0 ; i<iknsupc*nrhs ; i++){
							// printf("xre: %f\n",x[ii+i]);
							// fflush(stdout);
						// }
						if(UBtree_ptr[lk1].empty_==NO){
#ifdef _OPENMP
#pragma omp atomic capture
#endif
						nroot_send_tmp = ++nroot_send[0];
						root_send[(nroot_send_tmp-1)*aln_i] = lk1;
						// BcTree_forwardMessageSimple(UBtree_ptr[lk1],&x[ii - XK_H],'s');
						}

						/*
						 * Perform local block modifications.
						 */
						if ( Urbs[lk1] )

// if(Urbs[lk1]>16){
// #ifdef _OPENMP
// #pragma omp	task firstprivate (Ucb_indptr,Ucb_valptr,Llu,sizelsum,ii,gik,x,rtemp,bmod,Urbs,lsum,stat,nrhs,grid,xsup) untied
// #endif
							// 	slsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, gik, bmod, Urbs,
									//	Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
									//	stat, root_send, nroot_send, sizelsum,sizertemp);
							//}else{
							slsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, gik, bmod, Urbs,
								Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
								stat, root_send, nroot_send, sizelsum,sizertemp,thread_id,num_thread);
							//}

					// } /* if brecv[ik] == 0 */
				}
			} /* if bmod[ik] == 0 */

		} /* end for ub ... */
	} /* end else ... */

} /* slSUM_BMOD_inv */



/************************************************************************/
void slsum_bmod_inv_master
/************************************************************************/
(
 float *lsum,        /* Sum of local modifications.                    */
 float *x,           /* X array (local).                               */
 float *xk,          /* X[k].                                          */
 float *rtemp,   /* Result of full matrix-vector multiply.             */
 int    nrhs,	      /* Number of right-hand sides.                    */
 int_t  k,            /* The k-th component of X.                       */
 int  *bmod,        /* Modification count for L-solve.                */
 int_t  *Urbs,        /* Number of row blocks in each block column of U.*/
 Ucb_indptr_t **Ucb_indptr,/* Vertical linked list pointing to Uindex[].*/
 int_t  **Ucb_valptr, /* Vertical linked list pointing to Unzval[].     */
 int_t  *xsup,
 gridinfo_t *grid,
 sLocalLU_t *Llu,
 SuperLUStat_t **stat,
 int_t sizelsum,
 int_t sizertemp,
 int thread_id,
 int num_thread
 )
{
	/*
	 * Purpose
	 * =======
	 *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
	 */
    float alpha = 1.0, beta = 0.0;
	int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
	int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
	       j, jj, lk, lk1, nub, ub, uptr;
	int_t  *usub;
	float *uval, *dest, *y;
	int_t  *lsub;
	float *lusup;
	int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
	int *brecv = Llu->brecv;
	int  **bsendx_plist = Llu->bsendx_plist;
	C_Tree  *UBtree_ptr = Llu->UBtree_ptr;
	C_Tree  *URtree_ptr = Llu->URtree_ptr;
	MPI_Status status;
	int test_flag;
	int_t bmod_tmp;
	int thread_id1;
	float *rtemp_loc;
	float temp;
	float *Uinv;/* Inverse of diagonal block */

	double t1, t2;
	float msg_vol = 0, msg_cnt = 0;
	int_t Nchunk, nub_loc,remainder,nn,lbstart,lbend;
	int_t iword = sizeof(int_t);
	int_t dword = sizeof (float);
	int aln_d,aln_i;
	aln_d = 1; //ceil(CACHELINE/(double)dword);
	aln_i = 1; //ceil(CACHELINE/(double)iword);


	rtemp_loc = &rtemp[sizertemp* thread_id];


	iam = grid->iam;
	myrow = MYROW( iam, grid );
	knsupc = SuperSize( k );
	lk = LBj( k, grid ); /* Local block number, column-wise. */
	nub = Urbs[lk];      /* Number of U blocks in block column lk */

	// printf("Urbs2[lk] %5d lk %5d nub %5d\n",Urbs2[lk],lk,nub);
	// fflush(stdout);

	if(nub>num_thread){
	// if(nub>0){
		Nchunk=num_thread;
		nub_loc = floor(((double)nub)/Nchunk);
		remainder = nub % Nchunk;

//#ifdef _OPENMP
//#pragma	omp	taskloop firstprivate (stat) private (thread_id1,nn,lbstart,lbend,ub,temp,rtemp_loc,ik,gik,usub,uval,iknsupc,il,i,irow,jj,t1,t2,j,ikfrow,iklrow,dest,y,uptr,fnz) untied
//#endif
		for (nn=0;nn<Nchunk;++nn){

#ifdef _OPENMP
			thread_id1 = omp_get_thread_num ();
#else
			thread_id1 = 0;
#endif
			rtemp_loc = &rtemp[sizertemp* thread_id1];

#if ( PROFlevel>=1 )
			TIC(t1);
#endif

			if(nn<remainder){
				lbstart = nn*(nub_loc+1);
				lbend = (nn+1)*(nub_loc+1);
			}else{
				lbstart = remainder+nn*nub_loc;
				lbend = remainder + (nn+1)*nub_loc;
			}
			for (ub = lbstart; ub < lbend; ++ub){
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
					dest = &lsum[il + j*iknsupc+sizelsum*thread_id1];
					y = &xk[j*knsupc];
					uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
					for (jj = 0; jj < knsupc; ++jj) {
						fnz = usub[i + jj];
						if ( fnz < iklrow ) { /* Nonzero segment. */
							/* AXPY */
//#ifdef _OPENMP
//#pragma omp simd // In complex case, this SIMD loop has 2 instructions, the compiler may generate incoreect code, so need to disable this omp simd
//#endif
							for (irow = fnz; irow < iklrow; ++irow)
								dest[irow - ikfrow] -= uval[uptr++] * y[jj];
							stat[thread_id1]->ops[SOLVE] += 2 * (iklrow - fnz);

						}
					} /* for jj ... */
				}
			}
#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat[thread_id1]->utime[SOL_GEMM] += t2;
#endif
		}

	}else{
		rtemp_loc = &rtemp[sizertemp* thread_id];
#if ( PROFlevel>=1 )
		TIC(t1);
#endif
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
				dest = &lsum[il + j*iknsupc+sizelsum*thread_id];
				y = &xk[j*knsupc];
				uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
				for (jj = 0; jj < knsupc; ++jj) {
					fnz = usub[i + jj];
					if ( fnz < iklrow ) { /* Nonzero segment. */
						/* AXPY */
//#ifdef _OPENMP
//#pragma omp simd // In complex case, this SIMD loop has 2 instructions, the compiler may generate incoreect code, so need to disable this omp simd
//#endif
						for (irow = fnz; irow < iklrow; ++irow)
							dest[irow - ikfrow] -= uval[uptr++] * y[jj];
						stat[thread_id]->ops[SOLVE] += 2 * (iklrow - fnz);

					}
				} /* for jj ... */
			}
		}
#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat[thread_id]->utime[SOL_GEMM] += t2;
#endif
	}


	rtemp_loc = &rtemp[sizertemp* thread_id];
	for (ub = 0; ub < nub; ++ub){
		ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
		il = LSUM_BLK( ik );
		gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
		iknsupc = SuperSize( gik );

	// #ifdef _OPENMP
	// #pragma omp atomic capture
	// #endif
		bmod_tmp=--bmod[ik*aln_i];

		if ( bmod_tmp == 0 ) { /* Local accumulation done. */
			gikcol = PCOL( gik, grid );
			p = PNUM( myrow, gikcol, grid );
			if ( iam != p ) {
				for (ii=1;ii<num_thread;ii++)
					// if(ii!=thread_id)
		#ifdef _OPENMP
		#pragma omp simd
		#endif
					for (jj=0;jj<iknsupc*nrhs;jj++)
						lsum[il + jj ] += lsum[il + jj + ii*sizelsum];
				//RdTree_forwardMessageSimple(URtree_ptr[ik],&lsum[il - LSUM_H ],RdTree_GetMsgSize(URtree_ptr[ik],'s')*nrhs+LSUM_H,'s');
				C_RdTree_forwardMessageSimple(&URtree_ptr[ik],&lsum[il - LSUM_H ],URtree_ptr[ik].msgSize_*nrhs+LSUM_H);

#if ( DEBUGlevel>=2 )
				printf("(%2d) Sent LSUM[%2.0f], size %2d, to P %2d\n",
						iam, lsum[il-LSUM_H], iknsupc*nrhs+LSUM_H, p);
#endif
			} else { /* Diagonal process: X[i] += lsum[i]. */

#if ( PROFlevel>=1 )
				TIC(t1);
#endif
				for (ii=1;ii<num_thread;ii++)
					// if(ii!=thread_id)
		#ifdef _OPENMP
		#pragma omp simd
		#endif
					for (jj=0;jj<iknsupc*nrhs;jj++)
						lsum[il + jj ] += lsum[il + jj + ii*sizelsum];

				ii = X_BLK( ik );
				dest = &x[ii];

				RHS_ITERATE(j)
		#ifdef _OPENMP
		#pragma omp simd
		#endif
					for (i = 0; i < iknsupc; ++i)
						dest[i + j*iknsupc] += lsum[i + il + j*iknsupc];

				// if ( !brecv[ik] ) { /* Becomes a leaf node. */
					// bmod[ik] = -1; /* Do not solve X[k] in the future. */
					lk1 = LBj( gik, grid ); /* Local block number. */
					lsub = Llu->Lrowind_bc_ptr[lk1];
					lusup = Llu->Lnzval_bc_ptr[lk1];
					nsupr = lsub[1];

					if(Llu->inv == 1){
						Uinv = Llu->Uinv_bc_ptr[lk1];
#ifdef _CRAY
						SGEMM( ftcs2, ftcs2, &iknsupc, &nrhs, &iknsupc,
								&alpha, Uinv, &iknsupc, &x[ii],
								&iknsupc, &beta, rtemp_loc, &iknsupc );
#elif defined (USE_VENDOR_BLAS)
						sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
								&alpha, Uinv, &iknsupc, &x[ii],
								&iknsupc, &beta, rtemp_loc, &iknsupc, 1, 1 );
#else
						sgemm_( "N", "N", &iknsupc, &nrhs, &iknsupc,
								&alpha, Uinv, &iknsupc, &x[ii],
								&iknsupc, &beta, rtemp_loc, &iknsupc );
#endif
			#ifdef _OPENMP
			#pragma omp simd
			#endif
						for (i=0 ; i<iknsupc*nrhs ; i++){
							x[ii+i] = rtemp_loc[i];
						}
					}else{
#ifdef _CRAY
						STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &iknsupc, &nrhs, &alpha,
								lusup, &nsupr, &x[ii], &iknsupc);
#elif defined (USE_VENDOR_BLAS)
						strsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
								lusup, &nsupr, &x[ii], &iknsupc, 1, 1, 1, 1);
#else
						strsm_("L", "U", "N", "N", &iknsupc, &nrhs, &alpha,
								lusup, &nsupr, &x[ii], &iknsupc);
#endif
					}

#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat[thread_id]->utime[SOL_TRSM] += t2;
#endif
					stat[thread_id]->ops[SOLVE] += iknsupc * (iknsupc + 1) * nrhs;
#if ( DEBUGlevel>=2 )
					printf("(%2d) Solve X[%2d]\n", iam, gik);
#endif

					/*
					 * Send Xk to process column Pc[k].
					 */

					 // for (i=0 ; i<iknsupc*nrhs ; i++){
						// printf("xre: %f\n",x[ii+i]);
						// fflush(stdout);
					// }
					if(UBtree_ptr[lk1].empty_==NO){
					  //BcTree_forwardMessageSimple(UBtree_ptr[lk1],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk1],'s')*nrhs+XK_H,'s');
					  C_BcTree_forwardMessageSimple(&UBtree_ptr[lk1], &x[ii - XK_H], UBtree_ptr[lk1].msgSize_*nrhs+XK_H);
					}

					/*
					 * Perform local block modifications.
					 */
					if ( Urbs[lk1] ){
// #ifdef _OPENMP
// #pragma	omp	task firstprivate (Ucb_indptr,Ucb_valptr,Llu,sizelsum,ii,gik,x,rtemp,bmod,Urbs,lsum,stat,nrhs,grid,xsup) untied
// #endif
						{
						slsum_bmod_inv_master(lsum, x, &x[ii], rtemp, nrhs, gik, bmod, Urbs,
								Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
								stat, sizelsum,sizertemp,thread_id,num_thread);
						}
					}
				// } /* if brecv[ik] == 0 */
			}
		} /* if bmod[ik] == 0 */
	}

} /* slsum_bmod_inv_master */
