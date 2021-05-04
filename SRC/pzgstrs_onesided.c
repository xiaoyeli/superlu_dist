/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Solves a system of distributed linear equations A*X = B with a
 * general N-by-N matrix A using the LU factors computed previously.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 15, 2008
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
 * </pre>
 */
#include <math.h>
#include "superlu_zdefs.h"
#ifndef CACHELINE
#define CACHELINE 64  /* bytes, Xeon Phi KNL, Cori haswell, Edision */
#endif



/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * PZGSTRS solves a system of distributed linear equations
 * A*X = B with a general N-by-N matrix A using the LU factorization
 * computed by PZGSTRF.
 * If the equilibration, and row and column permutations were performed,
 * the LU factorization was performed for A1 where
 *     A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 * and the linear system solved is
 *     A1 * Y = Pc*Pr*B1, where B was overwritten by B1 = diag(R)*B, and
 * the permutation to B1 by Pc*Pr is applied internally in this routine.
 *
 * Arguments
 * =========
 *
 * n      (input) int (global)
 *        The order of the system of linear equations.
 *
 * LUstruct (input) zLUstruct_t*
 *        The distributed data structures storing L and U factors.
 *        The L and U factors are obtained from PZGSTRF for
 *        the possibly scaled and permuted matrix A.
 *        See superlu_zdefs.h for the definition of 'zLUstruct_t'.
 *        A may be scaled and permuted into A1, so that
 *        A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh. It contains the MPI communicator, the number
 *        of process rows (NPROW), the number of process columns (NPCOL),
 *        and my process rank. It is an input argument to all the
 *        parallel routines.
 *        Grid can be initialized by subroutine SUPERLU_GRIDINIT.
 *        See superlu_defs.h for the definition of 'gridinfo_t'.
 *
 * B      (input/output) doublecomplex*
 *        On entry, the distributed right-hand side matrix of the possibly
 *        equilibrated system. That is, B may be overwritten by diag(R)*B.
 *        On exit, the distributed solution matrix Y of the possibly
 *        equilibrated system if info = 0, where Y = Pc*diag(C)^(-1)*X,
 *        and X is the solution of the original system.
 *
 * m_loc  (input) int (local)
 *        The local row dimension of matrix B.
 *
 * fst_row (input) int (global)
 *        The row number of B's first row in the global matrix.
 *
 * ldb    (input) int (local)
 *        The leading dimension of matrix B.
 *
 * nrhs   (input) int (global)
 *        Number of right-hand sides.
 *
 * SOLVEstruct (input) zSOLVEstruct_t* (global)
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the triangular solves.
 *        See util.h for the definition of 'SuperLUStat_t'.
 *
 * info   (output) int*
 * 	   = 0: successful exit
 *	   < 0: if info = -i, the i-th argument had an illegal value
 * </pre>
 */
#ifdef onesided
void
pzgstrs_onesided(int_t n, zLUstruct_t *LUstruct,
	zScalePermstruct_t *ScalePermstruct,
	gridinfo_t *grid, doublecomplex *B,
	int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
	zSOLVEstruct_t *SOLVEstruct,
	SuperLUStat_t *stat, int *info)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t *Llu = LUstruct->Llu;
    doublecomplex alpha = {1.0, 0.0};
    doublecomplex beta = {0.0, 0.0};
    doublecomplex zero = {0.0, 0.0};
    doublecomplex *lsum;  /* Local running sum of the updates to B-components */
    doublecomplex *x;     /* X component at step k. */
		    /* NOTE: x and lsum are of same size. */
    doublecomplex *lusup, *dest;
    doublecomplex *recvbuf, *recvbuf_on, *tempv,
            *recvbufall, *recvbuf_BC_fwd, *recvbuf0, *xin;
    doublecomplex *rtemp, *rtemp_loc; /* Result of full matrix-vector multiply. */
    doublecomplex *Linv; /* Inverse of diagonal block */
    doublecomplex *Uinv; /* Inverse of diagonal block */
    int *ipiv;
    int_t *leaf_send;
    int_t nleaf_send, nleaf_send_tmp;
    int_t *root_send;
    int_t nroot_send, nroot_send_tmp;
    int_t  **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
        /*-- Data structures used for broadcast and reduction trees. --*/
    BcTree  *LBtree_ptr = Llu->LBtree_ptr;
    RdTree  *LRtree_ptr = Llu->LRtree_ptr;
    BcTree  *UBtree_ptr = Llu->UBtree_ptr;
    RdTree  *URtree_ptr = Llu->URtree_ptr;
    int_t  *Urbs1; /* Number of row blocks in each block column of U. */
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t  kcol, krow, mycol, myrow;
    int_t  i, ii, il, j, jj, k, kk, lb, ljb, lk, lib, lptr, luptr, gb, nn;
    int_t  nb, nlb,nlb_nodiag, nub, nsupers, nsupers_j, nsupers_i,maxsuper;
    int_t  *xsup, *supno, *lsub, *usub;
    int_t  *ilsum;    /* Starting position of each supernode in lsum (LOCAL)*/
    int    Pc, Pr, iam;
    int    knsupc, nsupr, nprobe;
    int    nbtree, nrtree, outcount;
    int    ldalsum;   /* Number of lsum entries locally owned. */
    int    maxrecvsz, p, pi;
    int_t  **Lrowind_bc_ptr;
    doublecomplex **Lnzval_bc_ptr;
    doublecomplex **Linv_bc_ptr;
    doublecomplex **Uinv_bc_ptr;
    doublecomplex sum;
    MPI_Status status,status_on,statusx,statuslsum;
    pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;
    SuperLUStat_t **stat_loc;

    double tmax;
    	/*-- Counts used for L-solve --*/
    int_t  *fmod;         /* Modification count for L-solve --
    			 Count the number of local block products to
    			 be summed into lsum[lk]. */
    int_t fmod_tmp;
    int_t  **fsendx_plist = Llu->fsendx_plist;
    int_t  nfrecvx = Llu->nfrecvx; /* Number of X components to be recv'd. */
    int_t  nfrecvx_buf=0;
    int_t  *frecv;        /* Count of lsum[lk] contributions to be received
    			     from processes in this row.
    			     It is only valid on the diagonal processes. */
    int_t  frecv_tmp;
    int_t  nfrecvmod = 0; /* Count of total modifications to be recv'd. */
    int_t  nfrecv = 0; /* Count of total messages to be recv'd. */
    int_t  nbrecv = 0; /* Count of total messages to be recv'd. */
    int_t  nleaf = 0, nroot = 0;
    int_t  nleaftmp = 0, nroottmp = 0;
    int_t  msgsize;
        /*-- Counts used for U-solve --*/
    int_t  *bmod;         /* Modification count for U-solve. */
    int_t  bmod_tmp;
    int_t  **bsendx_plist = Llu->bsendx_plist;
    int_t  nbrecvx = Llu->nbrecvx; /* Number of X components to be recv'd. */
    int_t  nbrecvx_buf=0;
    int_t  *brecv;        /* Count of modifications to be recv'd from
    			     processes in this row. */
    int_t  nbrecvmod = 0; /* Count of total modifications to be recv'd. */
    int_t flagx,flaglsum,flag;
    int_t *LBTree_active, *LRTree_active, *LBTree_finish, *LRTree_finish, *leafsups, *rootsups;
    int_t TAG;
    double t1_sol, t2_sol, t;
#if ( DEBUGlevel>=2 )
    int_t Ublocks = 0;
#endif

    int_t gik,iklrow,fnz;

    int_t *mod_bit = Llu->mod_bit; /* flag contribution from each row block */
    int INFO, pad;
    int_t tmpresult;

    // #if ( PROFlevel>=1 )
    double t1, t2;
    float msg_vol = 0, msg_cnt = 0;
    // #endif

    int_t msgcnt[4]; /* Count the size of the message xfer'd in each buffer:
		      *     0 : transferred in Lsub_buf[]
		      *     1 : transferred in Lval_buf[]
		      *     2 : transferred in Usub_buf[]
		      *     3 : transferred in Uval_buf[]
		      */
    int iword = sizeof (int_t);
    int dword = sizeof (doublecomplex);
    int Nwork;
    int_t procs = grid->nprow * grid->npcol;
    yes_no_t done;
    yes_no_t startforward;
    int nbrow;
    int_t  ik, rel, idx_r, jb, nrbl, irow, pc,iknsupc;
    int_t  lptr1_tmp, idx_i, idx_v,m;
    int_t ready;
    static int thread_id = 0;
    yes_no_t empty;
    int_t sizelsum,sizertemp,aln_d,aln_i;
    aln_d = ceil(CACHELINE/(double)dword);
    aln_i = ceil(CACHELINE/(double)iword);
    int num_thread = 1;

    maxsuper = sp_ienv_dist(3);

#ifdef _OPENMP
#pragma omp threadprivate(thread_id)
#endif

#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
    	if (omp_get_thread_num () == 0) {
    		num_thread = omp_get_num_threads ();
    	}
	thread_id = omp_get_thread_num ();
    }
#endif

#if ( PRNTlevel>=2 )
    if( grid->iam==0 ) {
	printf("num_thread: %5d\n", num_thread);
	fflush(stdout);
    }
#endif

    MPI_Barrier( grid->comm );
    t1_sol = SuperLU_timer_();
    t = SuperLU_timer_();

    /* Test input parameters. */
    *info = 0;
    if ( n < 0 ) *info = -1;
    else if ( nrhs < 0 ) *info = -9;
    if ( *info ) {
	pxerr_dist("PZGSTRS", grid, -*info);
	return;
    }

    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW( iam, grid );
    mycol = MYCOL( iam, grid );
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n-1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    Linv_bc_ptr = Llu->Linv_bc_ptr;
    Uinv_bc_ptr = Llu->Uinv_bc_ptr;
    nlb = CEILING( nsupers, Pr ); /* Number of local block rows. */

    stat->utime[SOL_COMM] = 0.0;
    stat->utime[SOL_GEMM] = 0.0;
    stat->utime[SOL_TRSM] = 0.0;
    stat->utime[SOL_TOT] = 0.0;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter pzgstrs()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    /* Save the count to be altered so it can be used by
       subsequent call to PDGSTRS. */
    if ( !(fmod = intMalloc_dist(nlb*aln_i)) )
	ABORT("Malloc fails for fmod[].");
    for (i = 0; i < nlb; ++i) fmod[i*aln_i] = Llu->fmod[i];
    if ( !(frecv = intCalloc_dist(nlb)) )
	ABORT("Calloc fails for frecv[].");
    Llu->frecv = frecv;

    if ( !(leaf_send = intMalloc_dist((CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i)) )
	ABORT("Malloc fails for leaf_send[].");
    nleaf_send=0;
    if ( !(root_send = intMalloc_dist((CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i)) )
	ABORT("Malloc fails for root_send[].");
    nroot_send=0;

#ifdef _CRAY
    ftcs1 = _cptofcd("L", strlen("L"));
    ftcs2 = _cptofcd("N", strlen("N"));
    ftcs3 = _cptofcd("U", strlen("U"));
#endif


    /* Obtain ilsum[] and ldalsum for process column 0. */
    ilsum = Llu->ilsum;
    ldalsum = Llu->ldalsum;

    /* Allocate working storage. */
    knsupc = sp_ienv_dist(3);
    maxrecvsz = knsupc * nrhs + SUPERLU_MAX( XK_H, LSUM_H )+1;
    sizelsum = (((size_t)ldalsum)*nrhs + nlb*LSUM_H);
    sizelsum = ((sizelsum + (aln_d - 1)) / aln_d) * aln_d;

#ifdef _OPENMP
    if ( !(lsum = (doublecomplex*)SUPERLU_MALLOC(sizelsum*num_thread * sizeof(doublecomplex))))
	ABORT("Malloc fails for lsum[].");
#pragma omp parallel default(shared) private(ii)
    {
	for (ii=0; ii<sizelsum; ii++)
    	    lsum[thread_id*sizelsum+ii]=zero;
    }
#else
    if ( !(lsum = (doublecomplex*)SUPERLU_MALLOC(sizelsum*num_thread * sizeof(doublecomplex))))
  	    ABORT("Malloc fails for lsum[].");
    for ( ii=0; ii < sizelsum*num_thread; ii++ )
	lsum[ii]=zero;
#endif
    /* intermediate solution x[] vector has same structure as lsum[], see leading comment */
    if ( !(x = doublecomplexCalloc_dist(ldalsum * nrhs + nlb * XK_H)) )
	ABORT("Calloc fails for x[].");

    sizertemp=ldalsum * nrhs;
    sizertemp = ((sizertemp + (aln_d - 1)) / aln_d) * aln_d;
    if ( !(rtemp = (doublecomplex*)SUPERLU_MALLOC((sizertemp*num_thread + 1) * sizeof(doublecomplex))) )
	ABORT("Malloc fails for rtemp[].");
#ifdef _OPENMP
#pragma omp parallel default(shared) private(ii)
    {
	for ( ii=0; ii<sizertemp; ii++ )
		rtemp[thread_id*sizertemp+ii]=zero;
    }
#else
    for ( ii=0; ii<sizertemp*num_thread; ii++ )
	rtemp[ii]=zero;
#endif

    if ( !(stat_loc = (SuperLUStat_t**) SUPERLU_MALLOC(num_thread*sizeof(SuperLUStat_t*))) )
	ABORT("Malloc fails for stat_loc[].");

    for ( i=0; i<num_thread; i++) {
	stat_loc[i] = (SuperLUStat_t*)SUPERLU_MALLOC(sizeof(SuperLUStat_t));
	PStatInit(stat_loc[i]);
    }

#if ( DEBUGlevel>=2 )
    /* Dump the L factor using matlab triple-let format. */
    zDumpLblocks(iam, nsupers, grid, Glu_persist, Llu);
#endif

    /*---------------------------------------------------
     * Forward solve Ly = b.
     *---------------------------------------------------*/
    /* Redistribute B into X on the diagonal processes. */
    pzReDistribute_B_to_X(B, m_loc, nrhs, ldb, fst_row, ilsum, x,
			  ScalePermstruct, Glu_persist, grid, SOLVEstruct);

#if ( PRNTlevel>=2 )
    t = SuperLU_timer_() - t;
    if ( !iam) printf(".. B to X redistribute time\t%8.4f\n", t);
    fflush(stdout);
    t = SuperLU_timer_();
#endif

    /* Set up the headers in lsum[]. */
#ifdef _OPENMP
	#pragma omp simd lastprivate(krow,lk,il)
#endif
    for (k = 0; k < nsupers; ++k) {
	krow = PROW( k, grid );
	if ( myrow == krow ) {
	    lk = LBi( k, grid );   /* Local block number. */
	    il = LSUM_BLK( lk );
	    lsum[il - LSUM_H].r = k;/* Block number prepended in the header.*/
	    lsum[il - LSUM_H].i = 0;
	}
    }

	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */

	nbtree = 0;
	for (lk=0;lk<nsupers_j;++lk){
		if(LBtree_ptr[lk]!=NULL){
			// printf("LBtree_ptr lk %5d\n",lk);
			if(BcTree_IsRoot(LBtree_ptr[lk],'z')==NO){
				nbtree++;
				if(BcTree_getDestCount(LBtree_ptr[lk],'z')>0)nfrecvx_buf++;
			}
			BcTree_allocateRequest(LBtree_ptr[lk],'z');
		}
	}

	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	leafsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for leafsups.");

	nrtree = 0;
	nleaf=0;
	nfrecvmod=0;



if(procs==1){
	for (lk=0;lk<nsupers_i;++lk){
		gb = myrow+lk*grid->nprow;  /* not sure */
		if(gb<nsupers){
			if (fmod[lk*aln_i]==0){
				leafsups[nleaf]=gb;
				++nleaf;
			}
		}
	}
}else{
	for (lk=0;lk<nsupers_i;++lk){
		if(LRtree_ptr[lk]!=NULL){
			nrtree++;
			RdTree_allocateRequest(LRtree_ptr[lk],'z');
			frecv[lk] = RdTree_GetDestCount(LRtree_ptr[lk],'z');
			nfrecvmod += frecv[lk];
		}else{
			gb = myrow+lk*grid->nprow;  /* not sure */
			if(gb<nsupers){
				kcol = PCOL( gb, grid );
				if(mycol==kcol) { /* Diagonal process */
					if (fmod[lk*aln_i]==0){
						leafsups[nleaf]=gb;
						++nleaf;
					}
				}
			}
		}
	}
}


#ifdef _OPENMP
#pragma omp simd
#endif
	for (i = 0; i < nlb; ++i) fmod[i*aln_i] += frecv[i];
/*---------------------------------------------------
     * Setup onesided buffers
 *---------------------------------------------------*/
    int iam_col=MYROW( iam, grid );
    int iam_row=MYCOL( iam, grid );
    int *BCcount, *RDcount;
    long *BCbase, *RDbase; //BCsendoffset, RDsendoffset;
    double nfrecv1=0;
    int checkend=0;
    int ird=0, tidx=0, bcidx=0, rdidx=0, tmp_id=0;
    int *BCis_solved, *RDis_solved;
    int totalsolveBC=0, totalsolveRD=0;
    long* BC_taskbuf_offset;
    long *RD_taskbuf_offset;
    int BC_buffer_size=0; //= Pr * maxrecvsz*(nfrecvx+1) + Pr;
    int RD_buffer_size=0; //= Pc * maxrecvsz*(nfrecvmod+1) + Pc;
    int shift=0;
    int recvRankNum=-1;
    uint16_t crc_16_val;
    uint8_t crc_8_val;
    int *validBCQindex;
    int *validRDQindex;
    int *validBCQindex_u;
    int *validRDQindex_u;
    int my_adjust_num=2;

    BCcount = (int*)SUPERLU_MALLOC( Pr * sizeof(int));
    RDcount = (int*)SUPERLU_MALLOC( Pc * sizeof(int));
    memset(BCcount, 0, ( Pr * sizeof(int)));
    memset(RDcount, 0, ( Pc * sizeof(int)));

    BCbase = (long*)SUPERLU_MALLOC( Pr * sizeof(long));
    RDbase = (long*)SUPERLU_MALLOC( Pc * sizeof(long));
    memset(BCbase, 0, ( Pr * sizeof(long)));
    memset(RDbase, 0, ( Pc * sizeof(long)));

    if ( !(validBCQindex = (int*)SUPERLU_MALLOC( Pr * sizeof(int))) )
    ABORT("Malloc fails for validBCQindex[]");
    if ( !(validRDQindex = (int*)SUPERLU_MALLOC( Pc *sizeof(int))) )
    ABORT("Malloc fails for validRDQindex[]");
    if ( !(validBCQindex_u = (int*)SUPERLU_MALLOC( Pr * sizeof(int))) )
    ABORT("Malloc fails for validBCQindex_u[]");
    if ( !(validRDQindex_u = (int*)SUPERLU_MALLOC( Pc *sizeof(int))) )
    ABORT("Malloc fails for validRDQindex_u[]");

    if( Pr > 1){
        for (i=0;i<Pr;i++){
            BCbase[i] = recv_size_all[i]*maxrecvsz*my_adjust_num;
            validBCQindex[i]=keep_validBCQindex[i];
            validBCQindex_u[i]=keep_validBCQindex_u[i];
#if ( DEBUGlevel>=2 )
            printf("iam=%d,BCbase[%d]=%lu,validBCQindex[%d]=%d,validBCQindex_u[%d]=%d\n",iam,i,BCbase[i],i,validBCQindex[i],i,validBCQindex_u[i]);
                fflush(stdout);
#endif
        }
     }
    if(Pc > 1){
        for (i=0;i<Pc;i++){
            RDbase[i] = recv_size_all[Pr+i]*maxrecvsz*my_adjust_num;
            validRDQindex[i]=keep_validRDQindex[i];
            validRDQindex_u[i]=keep_validRDQindex_u[i];
#if ( DEBUGlevel>=2 )
            printf("iam=%d,RDbase[%d]=%lu,validRDQindex[%d]=%d,validRDQindex_u[%d]=%d\n",iam,i,RDbase[i],i,validRDQindex[i],i,validRDQindex_u[i]);
                    fflush(stdout);
#endif
        }
    }

    nfrecvx_buf=0;
    double checksum=0;

    BC_taskbuf_offset = (long*)SUPERLU_MALLOC( Pr * sizeof(long));   // this needs to be optimized for 1D row mapping
    RD_taskbuf_offset = (long*)SUPERLU_MALLOC( Pc * sizeof(long));   // this needs to be optimized for 1D row mapping
    memset(BC_taskbuf_offset, 0, Pr * sizeof(long));
    memset(RD_taskbuf_offset, 0, Pc * sizeof(long));

    for (bcidx=0;bcidx<Pr;bcidx++){
        for(int tmp=0;tmp<bcidx;tmp++){
            BC_taskbuf_offset[bcidx] += BufSize[tmp]*maxrecvsz*my_adjust_num;
        }
#if ( DEBUGlevel>=2 )
        printf("iam=%d, BC_taskbuf_offset[%d]=%lu\n",iam,bcidx,BC_taskbuf_offset[bcidx]);
        fflush(stdout);
#endif
    }
    for (rdidx=0;rdidx<Pc;rdidx++){
        for(int tmp=0;tmp<rdidx;tmp++){
            RD_taskbuf_offset[rdidx] += BufSize_rd[tmp]*maxrecvsz*my_adjust_num;
        }
#if ( DEBUGlevel>=2 )
        printf("iam=%d, RD_taskbuf_offset[%d]=%lu\n",iam,rdidx,RD_taskbuf_offset[rdidx]);
        fflush(stdout);
#endif
    }

    BCis_solved = (int*)SUPERLU_MALLOC( Pr * sizeof(int));   // this needs to be optimized for 1D row mapping
    RDis_solved = (int*)SUPERLU_MALLOC( Pc * sizeof(int));   // this needs to be optimized for 1D row mapping
    memset(BCis_solved, 0, Pr * sizeof(int));
    memset(RDis_solved, 0, Pc * sizeof(int));
#if ( DEBUGlevel>=2 )
    printf("iam=%d, End setup oneside L solve\n",iam);
	printf("(%2d) nfrecvx %4d,  nfrecvmod %4d,  nleaf %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nfrecvx, nfrecvmod, nleaf, nbtree, nrtree);
    fflush(stdout);
#endif


	log_memory(nlb*aln_i*iword+nlb*iword+(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*2.0*iword+ nsupers_i*iword + sizelsum*num_thread * dword*2.0 + (ldalsum * nrhs + nlb * XK_H) *dword*2.0 + (sizertemp*num_thread + 1)*dword*2.0+maxrecvsz*(nfrecvx+1)*dword*2.0, stat);	//account for fmod, frecv, leaf_send, root_send, leafsups, recvbuf_BC_fwd	, lsum, x, rtemp



#if ( DEBUGlevel>=2 )
	printf("(%2d) nfrecvx %4d,  nfrecvmod %4d,  nleaf %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nfrecvx, nfrecvmod, nleaf, nbtree, nrtree);
	fflush(stdout);
#endif

#if ( PRNTlevel>=2 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Setup L-solve time\t%8.4f\n", t);
	fflush(stdout);
	MPI_Barrier( grid->comm );
	t = SuperLU_timer_();
#endif

#if ( VAMPIR>=1 )
	// VT_initialize();
	VT_traceon();
#endif

#ifdef USE_VTUNE
	__SSC_MARK(0x111);// start SDE tracing, note uses 2 underscores
	__itt_resume(); // start VTune, again use 2 underscores
#endif

	/* ---------------------------------------------------------
	   Solve the leaf nodes first by all the diagonal processes.
	   --------------------------------------------------------- */
#if ( DEBUGlevel>=2 )
	printf("(%2d) nleaf %4d\n", iam, nleaf);
	fflush(stdout);
#endif


#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
		{

            if (Llu->inv == 1) { /* Diagonal is inverted. */

#ifdef _OPENMP
#pragma	omp	for firstprivate(nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Linv,i,lib,rtemp_loc,nleaf_send_tmp) nowait
#endif
		for (jj=0;jj<nleaf;jj++){
		    k=leafsups[jj];

// #ifdef _OPENMP
// #pragma omp task firstprivate (k,nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,knsupc,lk,luptr,lsub,nsupr,lusup,thread_id,t1,t2,Linv,i,lib,rtemp_loc)
// #endif
   		    {

#if ( PROFlevel>=1 )
					TIC(t1);
#endif
					rtemp_loc = &rtemp[sizertemp* thread_id];


					knsupc = SuperSize( k );
					lk = LBi( k, grid );

					ii = X_BLK( lk );
					lk = LBj( k, grid ); /* Local block number, column-wise. */
					lsub = Lrowind_bc_ptr[lk];
					lusup = Lnzval_bc_ptr[lk];

					nsupr = lsub[1];

					Linv = Linv_bc_ptr[lk];
#ifdef _CRAY
					CGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
					zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
					zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc );
#endif

				#ifdef _OPENMP
					#pragma omp simd
				#endif
					for (i=0 ; i<knsupc*nrhs ; i++){
						z_copy(&x[ii+i],&rtemp_loc[i]);
					}

					// for (i=0 ; i<knsupc*nrhs ; i++){
					// printf("x_l: %f %f\n",x[ii+i].r,x[ii+i].i);
					// fflush(stdout);
					// }


#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat_loc[thread_id]->utime[SOL_TRSM] += t2;

#endif

					stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc - 1) * nrhs
					+ 10 * knsupc * nrhs; /* complex division */


					// --nleaf;
#if ( DEBUGlevel>=2 )
					printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

					/*
					 * Send Xk to process column Pc[k].
					 */

					if(LBtree_ptr[lk]!=NULL){
						lib = LBi( k, grid ); /* Local block number, row-wise. */
						ii = X_BLK( lib );

#ifdef _OPENMP
#pragma omp atomic capture
#endif
						nleaf_send_tmp = ++nleaf_send;
						leaf_send[(nleaf_send_tmp-1)*aln_i] = lk;
						// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],'z');
					}
				}
			}
	} else { /* Diagonal is not inverted. */
#ifdef _OPENMP
#pragma	omp	for firstprivate (nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Linv,i,lib,rtemp_loc,nleaf_send_tmp) nowait
#endif
	    for (jj=0;jj<nleaf;jj++) {
		k=leafsups[jj];
		{

#if ( PROFlevel>=1 )
		    TIC(t1);
#endif
		    rtemp_loc = &rtemp[sizertemp* thread_id];

		    knsupc = SuperSize( k );
		    lk = LBi( k, grid );

		    ii = X_BLK( lk );
		    lk = LBj( k, grid ); /* Local block number, column-wise. */
		    lsub = Lrowind_bc_ptr[lk];
		    lusup = Lnzval_bc_ptr[lk];

		    nsupr = lsub[1];

#ifdef _CRAY
   		    CTRSM(ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
				lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
		    ztrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
				lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
 		    ztrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
					lusup, &nsupr, &x[ii], &knsupc);
#endif

#if ( PROFlevel>=1 )
		    TOC(t2, t1);
		    stat_loc[thread_id]->utime[SOL_TRSM] += t2;

#endif

		    stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc - 1) * nrhs
				+ 10 * knsupc * nrhs; /* complex division */

		    // --nleaf;
#if ( DEBUGlevel>=2 )
		    printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

		    /*
		     * Send Xk to process column Pc[k].
		     */

		    if (LBtree_ptr[lk]!=NULL) {
			lib = LBi( k, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lib );

#ifdef _OPENMP
#pragma omp atomic capture
#endif
			nleaf_send_tmp = ++nleaf_send;
			leaf_send[(nleaf_send_tmp-1)*aln_i] = lk;
		    }
		    } /* end a block */
		} /* end for jj ... */
	    } /* end else ... diagonal is not invedted */
	  }
	}

	jj=0;

#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
		{

#ifdef _OPENMP
#pragma omp master
#endif
		    {

#ifdef _OPENMP
#pragma	omp taskloop private (k,ii,lk) num_tasks(num_thread*8) nogroup
#endif

			for (jj=0;jj<nleaf;jj++){
			    k=leafsups[jj];

			    {
				/* Diagonal process */
				lk = LBi( k, grid );
				ii = X_BLK( lk );
				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				zlsum_fmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, fmod, xsup, grid, Llu, stat_loc, leaf_send, &nleaf_send,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);
			    }

			} /* for jj ... */
		    }

		}

	for (i=0;i<nleaf_send;i++){
		lk = leaf_send[i*aln_i];
		if(lk>=0){ // this is a bcast forwarding
			gb = mycol+lk*grid->npcol;  /* not sure */
			lib = LBi( gb, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lib );
			BcTree_forwardMessageOneSide(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'z')*nrhs+XK_H,'z', &iam_col, BCcount, BCbase, &maxrecvsz,Pc);
		}else{ // this is a reduce forwarding
			lk = -lk - 1;
			il = LSUM_BLK( lk );
			RdTree_forwardMessageOneSide(LRtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(LRtree_ptr[lk],'z')*nrhs+LSUM_H,'z', &iam_row, RDcount, RDbase, &maxrecvsz, Pc);
		}
	}



#ifdef USE_VTUNE
	__itt_pause(); // stop VTune
	__SSC_MARK(0x222); // stop SDE tracing
#endif

			/* -----------------------------------------------------------
			   Compute the internal nodes asynchronously by all processes.
			   ----------------------------------------------------------- */
    while( nfrecv1 < nfrecvx+nfrecvmod ){
        thread_id = 0;
        if (totalsolveBC < nfrecvx){
            shift=0;
            for (bcidx=0;bcidx<Pr && validBCQindex[bcidx]!=-1;bcidx++){

                recvRankNum=validBCQindex[bcidx];  //bcidx; //validBCQindex[bcidx];
                i=BC_taskbuf_offset[recvRankNum]+BCis_solved[recvRankNum]*maxrecvsz*my_adjust_num; //BCis_solved[bcidx];
                recvbuf0 = &BC_taskq[i];
	    		k = (*recvbuf0).r;
#if ( PROFlevel>=1 )
                TIC(t1);
#endif
                if (k < 0)  {
                    if(shift>0){
                        validBCQindex[bcidx-shift]=validBCQindex[bcidx];
                        validBCQindex[bcidx]=-1;
                        //printf("iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                    continue;
                }

                lk = LBj( k, grid );    /* local block number */

                checkend=BcTree_GetMsgSize(LBtree_ptr[lk],'z')*nrhs;
                crc_8_val=crc_8((unsigned char*)&recvbuf0[XK_H],sizeof(doublecomplex)*checkend);
                //crc_16_val=crc_16((unsigned char*)&recvbuf0[XK_H],sizeof(doublecomplex)*checkend);
                //myhash=calcul_hash(&recvbuf0[XK_H],sizeof(double)*checkend);
                if(crc_8_val!=(uint8_t)recvbuf0[XK_H-1].r) {
                //if(crc_16_val!=(uint16_t)recvbuf0[XK_H-1].r) {
                    if(shift>0){
                        validBCQindex[bcidx-shift]=validBCQindex[bcidx];
                        validBCQindex[bcidx]=-1;
                        //printf("iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                    continue;
                }
#if ( PROFlevel>=1 )
                TOC(t2, t1);
						stat_loc[thread_id]->utime[SOL_COMM] += t2;
#endif
                //t= SuperLU_timer_();

                totalsolveBC += 1; //BC_subtotal[bcidx] - BCis_solved[bcidx];
                BCis_solved[recvRankNum]++;


                if(BcTree_getDestCount(LBtree_ptr[lk],'d')>0){
                    //BcTree_forwardMessageOneSide(LBtree_ptr[lk],recvbuf0,checkend,'d', &iam_col, BCcount, BCbase, &maxrecvsz, Pc);
                    BcTree_forwardMessageOneSide(LBtree_ptr[lk],recvbuf0,BcTree_GetMsgSize(LBtree_ptr[lk],'z')*nrhs+XK_H,'z', &iam_col, BCcount, BCbase, &maxrecvsz, Pc);
                }
                lsub = Lrowind_bc_ptr[lk];
                //printf("In BC solve, iam %d, k=%d, lk=%d, lsub =%d,checksum=%u\n", iam, k, lk, lsub,crc_16_val);
                //fflush(stdout);

                if ( lsub ) {
                    krow = PROW( k, grid );
                    if(myrow==krow){
                        nb = lsub[0] - 1;
                        knsupc = SuperSize( k );
                        ii = X_BLK( LBi( k, grid ) );
                        xin = &x[ii];
                    }else{
                        nb   = lsub[0];
                        knsupc = SuperSize( k );
                        xin = &recvbuf0[XK_H] ;
                    }
                    zlsum_fmod_inv_master_onesided(lsum, x, xin, rtemp, nrhs, knsupc, k,
                                                fmod, nb, xsup, grid, Llu,
                                                stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread,
                                                &iam_row, RDcount, RDbase, &iam_col, BCcount, BCbase, Pc, maxrecvsz);
                } /* if lsub */

                if (BCis_solved[recvRankNum] == BufSize[recvRankNum]) {
                    validBCQindex[bcidx]=-1;
                    shift += 1;
                    //printf("iam=%d,shift=%d\n",iam,shift);
                    //fflush(stdout);
                }else{
                    if(shift>0){
                        validBCQindex[bcidx-shift]=validBCQindex[bcidx];
                        validBCQindex[bcidx]=-1;
                        //printf("iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                }
                //printf("iam=%d,BCis_solved[%d]=%d,BufSize[%d]=%d\n",iam,recvRankNum,BCis_solved[recvRankNum],recvRankNum,BufSize[recvRankNum]);
                //fflush(stdout);
            } // for bcidx

            //TOC(t2, t1);
            //onesidecomm_rd += t2;
        }

        if (totalsolveRD < nfrecvmod){
            shift=0;
            //for (rdidx=0;rdidx<Pc ;rdidx++){
            for (rdidx=0;rdidx<Pc && validRDQindex[rdidx]!=-1;rdidx++){
                //if (validRDQindex[rdidx]==-1) continue;
                //if (rdidx == iam_row) continue;
                //if (BufSize_rd[rdidx] == 0) continue;
                //if(RDis_solved[rdidx] == BufSize_rd[rdidx]) continue;

                recvRankNum=validRDQindex[rdidx];  //bcidx; //validBCQindex[bcidx];
                ird=RD_taskbuf_offset[recvRankNum]+RDis_solved[recvRankNum]*maxrecvsz*my_adjust_num;
                recvbuf0 = &RD_taskq[ird];
	    		k = (*recvbuf0).r;
#if ( PROFlevel>=1 )
                TIC(t1);
#endif
                if (k < 0)  {
                    if(shift>0){
                        validRDQindex[rdidx-shift]=validRDQindex[rdidx];
                        validRDQindex[rdidx]=-1;
                    }
                    continue;
                }
                lk = LBi( k, grid );

                checkend=RdTree_GetMsgSize(LRtree_ptr[lk],'z')*nrhs;
                //crc_16_val=crc_16((unsigned char*)&recvbuf0[LSUM_H],sizeof(doublecomplex)*checkend);
                crc_8_val=crc_8((unsigned char*)&recvbuf0[LSUM_H],sizeof(doublecomplex)*checkend);
                if(crc_8_val!=(uint8_t)recvbuf0[LSUM_H-1].r) {
                //if(crc_16_val!=(uint16_t)recvbuf0[LSUM_H-1].r) {
                    if(shift>0){
                        validRDQindex[rdidx-shift]=validRDQindex[rdidx];
                        validRDQindex[rdidx]=-1;
                    }
                    continue;
                }
#if ( PROFlevel>=1 )
                TOC(t2, t1);
						stat_loc[thread_id]->utime[SOL_COMM] += t2;
#endif
                //t = SuperLU_timer_();
                totalsolveRD += 1; //RD_subtotal[rdidx]-RDis_solved[rdidx];

                RDis_solved[recvRankNum] += 1 ;
                //printf("In RD solve, iam %d, k=%d, lk=%d,checksum=%u\n", iam, k, lk,crc_16_val);
                //fflush(stdout);

                knsupc = SuperSize( k );
                tempv = &recvbuf0[LSUM_H];
                il = LSUM_BLK( lk );
                RHS_ITERATE(j) {
                    for (i = 0; i < knsupc; ++i)
										z_add(&lsum[i + il + j*knsupc + thread_id*sizelsum],
											  &lsum[i + il + j*knsupc + thread_id*sizelsum],
											  &tempv[i + j*knsupc]);
                }

                fmod_tmp=--fmod[lk*aln_i];

                thread_id = 0;
                rtemp_loc = &rtemp[sizertemp* thread_id];
                //printf("5----iam=%d,k=%d\n",iam,k);
                //fflush(stdout);
                if ( fmod_tmp==0 ) {
                    //printf("6----iam=%d,k=%d\n",iam,k);
                    //fflush(stdout);
                    if(RdTree_IsRoot(LRtree_ptr[lk],'z')==YES){
                        knsupc = SuperSize( k );
                        //printf("7----iam=%d,k=%d\n",iam,k);
                        //fflush(stdout);
                        for (ii=1;ii<num_thread;ii++)
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (jj=0;jj<knsupc*nrhs;jj++)
													z_add(&lsum[il + jj ],
														  &lsum[il + jj ],
														  &lsum[il + jj + ii*sizelsum]);

                        ii = X_BLK( lk );
                        RHS_ITERATE(j){
#ifdef _OPENMP
#pragma omp simd
#endif
                            for (i = 0; i < knsupc; ++i)
                            z_add(&x[i + ii + j*knsupc],
                                  &x[i + ii + j*knsupc],
                                  &lsum[i + il + j*knsupc] );
                        }
                        lk = LBj( k, grid ); /* Local block number, column-wise. */
                        lsub = Lrowind_bc_ptr[lk];
                        lusup = Lnzval_bc_ptr[lk];
                        nsupr = lsub[1];

#if ( PROFlevel>=1 )
                        TIC(t1);
#endif

                        if(Llu->inv == 1){
                            Linv = Linv_bc_ptr[lk];
#ifdef _CRAY
												CGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
												zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
												zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc );
#endif
#ifdef _OPENMP
#pragma omp simd
#endif
												for (i=0 ; i<knsupc*nrhs ; i++){
													z_copy(&x[ii+i],&rtemp_loc[i]);
                                                }
                        }else{   //if(Llu->inv == 1)
#ifdef _CRAY
												CTRSM(ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
												ztrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
												ztrsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc);
#endif
                        } // end if (Llu->inv == 1)

#if ( PROFlevel>=1 )
                        TOC(t2, t1);
          				        stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif

											stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc - 1) * nrhs
											+ 10 * knsupc * nrhs; /* complex division */

                        //printf("8----iam=%d,k=%d\n",iam,k);
                        //fflush(stdout);

                        if(LBtree_ptr[lk]!=NULL){
                            //printf("9----iam=%d,k=%d\n",iam,k);
                            //fflush(stdout);
                            BcTree_forwardMessageOneSide(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'z')*nrhs+XK_H,'z',  &iam_col, BCcount, BCbase, &maxrecvsz,Pc);
                            //printf("10----iam=%d,k=%d\n",iam,k);
                            //fflush(stdout);
                        }

                        lk = LBj( k, grid ); /* Local block number, column-wise. */
                        lsub = Lrowind_bc_ptr[lk];
                        lusup = Lnzval_bc_ptr[lk];
                        if ( lsub ) {
                            krow = PROW( k, grid );
                            nb = lsub[0] - 1;
                            knsupc = SuperSize( k );
                            ii = X_BLK( LBi( k, grid ) );
                            xin = &x[ii];
                            //printf("11----iam=%d,k=%d\n",iam,k);
                            //fflush(stdout);
                            zlsum_fmod_inv_master_onesided(lsum, x, xin, rtemp, nrhs, knsupc, k,
                                                        fmod, nb, xsup, grid, Llu,
                                                        stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread,
                                                        &iam_row, RDcount, RDbase, &iam_col, BCcount, BCbase, Pc, maxrecvsz);
                            //printf("12----iam=%d,k=%d\n",iam,k);
                            //fflush(stdout);
                        } /* if lsub */
                    }else{ // RdTree Yes
                        //printf("13----iam=%d,k=%d\n",iam,k);
                        //fflush(stdout);
                        il = LSUM_BLK( lk );
                        knsupc = SuperSize( k );

                        for (ii=1;ii<num_thread;ii++)
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (jj=0;jj<knsupc*nrhs;jj++)
												z_add(&lsum[il + jj ],
													  &lsum[il + jj ],
													  &lsum[il + jj + ii*sizelsum]);
                        //printf("14----iam=%d,k=%d\n",iam,k);
                        //fflush(stdout);
                        RdTree_forwardMessageOneSide(LRtree_ptr[lk],&lsum[il-LSUM_H],RdTree_GetMsgSize(LRtree_ptr[lk],'z')*nrhs+LSUM_H,'z', &iam_row, RDcount, RDbase, &maxrecvsz, Pc);
                        //printf("15----iam=%d,k=%d\n",iam,k);
                        //fflush(stdout);
                    } // end if RD xxxx YES
                } // end of fmod_tmp=0
                if (RDis_solved[recvRankNum] == BufSize_rd[recvRankNum]) {
                    validRDQindex[rdidx]=-1;
                    shift += 1;
                    //printf("iam=%d,shift=%d\n",iam,shift);
                    //fflush(stdout);
                }else{
                    if(shift>0){
                        validRDQindex[rdidx-shift]=validRDQindex[rdidx];
                        validRDQindex[rdidx]=-1;
                        //printf("iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                }
                //printf("iam=%d,RDis_solved[%d]=%d,BufSize_rd[%d]=%d\n",iam,recvRankNum,RDis_solved[recvRankNum],recvRankNum,BufSize_rd[recvRankNum]);
                //fflush(stdout);
            }// for (rdidx=0;rdidx<Pc;rdidx++)
        }
        nfrecv1 = totalsolveBC + totalsolveRD;
    }// outer-most while


#if ( PRNTlevel>=2 )
		t = SuperLU_timer_() - t;
		stat->utime[SOL_TOT] += t;
		if ( !iam ) {
			printf(".. L-solve time\t%8.4f\n", t);
			fflush(stdout);
		}


		MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE,
				MPI_MAX, 0, grid->comm);
		if ( !iam ) {
			printf(".. L-solve time (MAX) \t%8.4f\n", tmax);
			fflush(stdout);
		}


		t = SuperLU_timer_();
#endif


#if ( DEBUGlevel==2 )
		{
			printf("(%d) .. After L-solve: y =\n", iam);
			for (i = 0, k = 0; k < nsupers; ++k) {
				krow = PROW( k, grid );
				kcol = PCOL( k, grid );
				if ( myrow == krow && mycol == kcol ) { /* Diagonal process */
					knsupc = SuperSize( k );
					lk = LBi( k, grid );
					ii = X_BLK( lk );
					for (j = 0; j < knsupc; ++j)
						printf("\t(%d)\t%4d\t%.10f\n", iam, xsup[k]+j, x[ii+j]);
					fflush(stdout);
				}
				MPI_Barrier( grid->comm );
			}
		}
#endif

		SUPERLU_FREE(fmod);
		SUPERLU_FREE(frecv);
		SUPERLU_FREE(leaf_send);
		SUPERLU_FREE(leafsups);
	    int tmp_size=maxrecvsz * ( (nfrecvx>nbrecvx?nfrecvx:nbrecvx) + 1 )*my_adjust_num;
        for(i=0;i<tmp_size;i++){
            BC_taskq[i]=(-1.0);
        }
        tmp_size=((nfrecvmod>nbrecvmod?nfrecvmod:nbrecvmod)+1)*maxrecvsz*my_adjust_num;
        for(i=0;i<tmp_size;i++){
            RD_taskq[i]=(-1.0);
        }
		log_memory(-nlb*aln_i*iword-nlb*iword-(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*iword- nsupers_i*iword -maxrecvsz*(nfrecvx+1)*dword*2.0, stat);	//account for fmod, frecv, leaf_send, leafsups, recvbuf_BC_fwd

		MPI_Barrier( grid->comm );

#if ( VAMPIR>=1 )
		VT_traceoff();
		VT_finalize();
#endif


		/*---------------------------------------------------
		 * Back solve Ux = y.
		 *
		 * The Y components from the forward solve is already
		 * on the diagonal processes.
	 *---------------------------------------------------*/


		/* Save the count to be altered so it can be used by
		   subsequent call to PDGSTRS. */
		if ( !(bmod = intMalloc_dist(nlb*aln_i)) )
			ABORT("Malloc fails for bmod[].");
		for (i = 0; i < nlb; ++i) bmod[i*aln_i] = Llu->bmod[i];
		if ( !(brecv = intCalloc_dist(nlb)) )
			ABORT("Calloc fails for brecv[].");
		Llu->brecv = brecv;

		k = SUPERLU_MAX( Llu->nfsendx, Llu->nbsendx ) + nlb;

		/* Re-initialize lsum to zero. Each block header is already in place. */

#ifdef _OPENMP

#pragma omp parallel default(shared) private(ii)
	{
		for(ii=0;ii<sizelsum;ii++)
			lsum[thread_id*sizelsum+ii]=zero;
	}
    /* Set up the headers in lsum[]. */
#ifdef _OPENMP
	#pragma omp simd lastprivate(krow,lk,il)
#endif
    for (k = 0; k < nsupers; ++k) {
	krow = PROW( k, grid );
	if ( myrow == krow ) {
	    lk = LBi( k, grid );   /* Local block number. */
	    il = LSUM_BLK( lk );
	    lsum[il - LSUM_H].r = k;/* Block number prepended in the header.*/
	    lsum[il - LSUM_H].i = 0;
	}
    }

#else
	for (k = 0; k < nsupers; ++k) {
		krow = PROW( k, grid );
		if ( myrow == krow ) {
			knsupc = SuperSize( k );
			lk = LBi( k, grid );
			il = LSUM_BLK( lk );
			dest = &lsum[il];

			for (jj = 0; jj < num_thread; ++jj) {
				RHS_ITERATE(j) {
					for (i = 0; i < knsupc; ++i) dest[i + j*knsupc + jj*sizelsum] = zero;
				}
			}
		}
	}
#endif

#if ( DEBUGlevel>=2 )
		for (p = 0; p < Pr*Pc; ++p) {
			if (iam == p) {
				printf("(%2d) .. Ublocks %d\n", iam, Ublocks);
				for (lb = 0; lb < nub; ++lb) {
					printf("(%2d) Local col %2d: # row blocks %2d\n",
							iam, lb, Urbs[lb]);
					if ( Urbs[lb] ) {
						for (i = 0; i < Urbs[lb]; ++i)
							printf("(%2d) .. row blk %2d:\
									lbnum %d, indpos %d, valpos %d\n",
									iam, i,
									Ucb_indptr[lb][i].lbnum,
									Ucb_indptr[lb][i].indpos,
									Ucb_valptr[lb][i]);
					}
				}
			}
			MPI_Barrier( grid->comm );
		}
		for (p = 0; p < Pr*Pc; ++p) {
			if ( iam == p ) {
				printf("\n(%d) bsendx_plist[][]", iam);
				for (lb = 0; lb < nub; ++lb) {
					printf("\n(%d) .. local col %2d: ", iam, lb);
					for (i = 0; i < Pr; ++i)
						printf("%4d", bsendx_plist[lb][i]);
				}
				printf("\n");
			}
			MPI_Barrier( grid->comm );
		}
#endif /* DEBUGlevel */




	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */

	nbtree = 0;
	for (lk=0;lk<nsupers_j;++lk){
		if(UBtree_ptr[lk]!=NULL){
			// printf("UBtree_ptr lk %5d\n",lk);
			if(BcTree_IsRoot(UBtree_ptr[lk],'z')==NO){
				nbtree++;
				if(BcTree_getDestCount(UBtree_ptr[lk],'z')>0)nbrecvx_buf++;
			}
			BcTree_allocateRequest(UBtree_ptr[lk],'z');
		}
	}

	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	rootsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for rootsups.");

	nrtree = 0;
	nroot=0;
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk]!=NULL){
			// printf("here lk %5d myid %5d\n",lk,iam);
			// fflush(stdout);
			nrtree++;
			RdTree_allocateRequest(URtree_ptr[lk],'z');
			brecv[lk] = RdTree_GetDestCount(URtree_ptr[lk],'z');
			nbrecvmod += brecv[lk];
		}else{
			gb = myrow+lk*grid->nprow;  /* not sure */
			if(gb<nsupers){
				kcol = PCOL( gb, grid );
				if(mycol==kcol) { /* Diagonal process */
					if (bmod[lk*aln_i]==0){
						rootsups[nroot]=gb;
						++nroot;
					}
				}
			}
		}
	}

	#ifdef _OPENMP
	#pragma omp simd
	#endif
	for (i = 0; i < nlb; ++i) bmod[i*aln_i] += brecv[i];
	// for (i = 0; i < nlb; ++i)printf("bmod[i]: %5d\n",bmod[i]);



/*---------------------------------------------------
* setup U solve onesided
*---------------------------------------------------*/
	double nbrecv1=0;
    totalsolveBC=0;
    totalsolveRD=0;
    memset(BCcount, 0, ( Pr * sizeof(int)));
    memset(RDcount, 0, ( Pc * sizeof(int)));
    memset(BCbase, 0, ( Pr * sizeof(long)));
    memset(RDbase, 0, ( Pc * sizeof(long)));
    memset(BC_taskbuf_offset, 0, Pr * sizeof(long));
    memset(RD_taskbuf_offset, 0, Pc * sizeof(long));
    memset(BCis_solved, 0, Pr * sizeof(int));
    memset(RDis_solved, 0, Pc * sizeof(int));

    if( Pr > 1){
        for (i=0;i<Pr;i++){
            BCbase[i] = recv_size_all_u[i]*maxrecvsz*my_adjust_num;
        }
    }
    if(Pc > 1){
        for (i=0;i<Pc;i++){
            RDbase[i] = recv_size_all_u[Pr+i]*maxrecvsz*my_adjust_num;
        }
    }


    for (bcidx=0;bcidx<Pr;bcidx++){
        for(int tmp=0;tmp<bcidx;tmp++){
            BC_taskbuf_offset[bcidx] += BufSize_u[tmp]*maxrecvsz*my_adjust_num;
        }
    }
    for (rdidx=0;rdidx<Pc;rdidx++){
        for(int tmp=0;tmp<rdidx;tmp++){
            RD_taskbuf_offset[rdidx] += BufSize_urd[tmp]*maxrecvsz*my_adjust_num;
        }
    }

    log_memory(nlb*aln_i*iword+nlb*iword + nsupers_i*iword + maxrecvsz*(nbrecvx+1)*dword, stat);	//account for bmod, brecv, rootsups, recvbuf_BC_fwd
    nbrecvx_buf=0;


	log_memory(nlb*aln_i*iword+nlb*iword + nsupers_i*iword + maxrecvsz*(nbrecvx+1)*dword*2.0, stat);	//account for bmod, brecv, rootsups, recvbuf_BC_fwd

#if ( DEBUGlevel>=2 )
	printf("(%2d) nbrecvx %4d,  nbrecvmod %4d,  nroot %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nbrecvx, nbrecvmod, nroot, nbtree, nrtree);
	fflush(stdout);
#endif


#if ( PRNTlevel>=2 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Setup U-solve time\t%8.4f\n", t);
	fflush(stdout);
	MPI_Barrier( grid->comm );
	t = SuperLU_timer_();
#endif

		/*
		 * Solve the roots first by all the diagonal processes.
		 */
#if ( DEBUGlevel>=2 )
		printf("(%2d) nroot %4d\n", iam, nroot);
		fflush(stdout);
#endif



#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
#ifdef _OPENMP
#pragma omp master
#endif
		{
#ifdef _OPENMP
#pragma	omp	taskloop firstprivate (nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,jj,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Uinv,i,lib,rtemp_loc,nroot_send_tmp) nogroup
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];

#if ( PROFlevel>=1 )
			TIC(t1);
#endif

			rtemp_loc = &rtemp[sizertemp* thread_id];



			knsupc = SuperSize( k );
			lk = LBi( k, grid ); /* Local block number, row-wise. */

			// bmod[lk] = -1;       /* Do not solve X[k] in the future. */
			ii = X_BLK( lk );
			lk = LBj( k, grid ); /* Local block number, column-wise */
			lsub = Lrowind_bc_ptr[lk];
			lusup = Lnzval_bc_ptr[lk];
			nsupr = lsub[1];


			if(Llu->inv == 1){

				Uinv = Uinv_bc_ptr[lk];
#ifdef _CRAY
				CGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
				zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
				zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#endif
				#ifdef _OPENMP
					#pragma omp simd
				#endif
				for (i=0 ; i<knsupc*nrhs ; i++){
					z_copy(&x[ii+i],&rtemp_loc[i]);
				}
			}else{
#ifdef _CRAY
				CTRSM(ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
				ztrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
				ztrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc);
#endif
			}

#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif
			stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc + 1) * nrhs
			+ 10 * knsupc * nrhs; /* complex division */

#if ( DEBUGlevel>=2 )
			printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

			/*
			 * Send Xk to process column Pc[k].
			 */

			if(UBtree_ptr[lk]!=NULL){
#ifdef _OPENMP
#pragma omp atomic capture
#endif
				nroot_send_tmp = ++nroot_send;
				root_send[(nroot_send_tmp-1)*aln_i] = lk;

			}
		} /* for k ... */
	}
}


#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
#ifdef _OPENMP
#pragma omp master
#endif
		{
#ifdef _OPENMP
#pragma	omp	taskloop private (ii,jj,k,lk) nogroup
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];
			lk = LBi( k, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lk );
			lk = LBj( k, grid ); /* Local block number, column-wise */

			/*
			 * Perform local block modifications: lsum[i] -= U_i,k * X[k]
			 */
			if ( Urbs[lk] )
				zlsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
						Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
						stat_loc, root_send, &nroot_send, sizelsum,sizertemp,thread_id,num_thread);

		} /* for k ... */

	}
}

for (i=0;i<nroot_send;i++){
	lk = root_send[(i)*aln_i];
	if(lk>=0){ // this is a bcast forwarding
		gb = mycol+lk*grid->npcol;  /* not sure */
		lib = LBi( gb, grid ); /* Local block number, row-wise. */
		ii = X_BLK( lib );
		BcTree_forwardMessageOneSide(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'z')*nrhs+XK_H,'z',&iam_col,BCcount, BCbase, &maxrecvsz,Pc);
	}else{ // this is a reduce forwarding
		lk = -lk - 1;
		il = LSUM_BLK( lk );
		RdTree_forwardMessageOneSide(URtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(URtree_ptr[lk],'z')*nrhs+LSUM_H,'z',&iam_row, RDcount, RDbase, &maxrecvsz, Pc);
	}
}


		/*
		 * Compute the internal nodes asychronously by all processes.
		 */
 recvRankNum=-1;
 shift=0;
 while(nbrecv1< nbrecvx+nbrecvmod){
     thread_id=0;
     if (totalsolveBC < nbrecvx){
         shift=0;
         for (bcidx=0;bcidx<Pr && validBCQindex_u[bcidx]!=-1;bcidx++){

             recvRankNum=validBCQindex_u[bcidx];  //bcidx; //validBCQindex[bcidx];
             i=BC_taskbuf_offset[recvRankNum]+BCis_solved[recvRankNum]*maxrecvsz*my_adjust_num; //BCis_solved[bcidx];
             recvbuf0 = &BC_taskq[i];
	    		k = (*recvbuf0).r;

             //printf("bcbc--111--iam=%d, bcidx=%d,k=%d\n",iam,bcidx,k);
             //fflush(stdout);
#if ( PROFlevel>=1 )
                TIC(t1);
#endif

                if (k < 0 ) {
                    if(shift>0){
                        validBCQindex_u[bcidx-shift]=validBCQindex_u[bcidx];
                        validBCQindex_u[bcidx]=-1;
                        //printf("iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                    continue;
                }

                lk = LBj( k, grid );    /* local block number */

                //if (totalsolveBC % 10 == 0){
                checkend=BcTree_GetMsgSize(UBtree_ptr[lk],'z')*nrhs;
                //crc_16_val=crc_16((unsigned char*)&recvbuf0[XK_H],sizeof(doublecomplex)*checkend);
                //if(crc_16_val!=(uint16_t)recvbuf0[XK_H-1].r) {
                crc_8_val=crc_8((unsigned char*)&recvbuf0[XK_H],sizeof(doublecomplex)*checkend);
                if(crc_8_val!=(uint8_t)recvbuf0[XK_H-1].r) {
                    if(shift>0){
                        validBCQindex_u[bcidx-shift]=validBCQindex_u[bcidx];
                        validBCQindex_u[bcidx]=-1;
                        //printf("1-iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                    continue;
                }
#if ( PROFlevel>=1 )
                TOC(t2, t1);
						stat_loc[thread_id]->utime[SOL_COMM] += t2;
#endif
                totalsolveBC += 1; //BC_subtotal[bcidx] - BCis_solved[bcidx];
                BCis_solved[recvRankNum]++;
                //printf("In U-BC solve, iam %d, k=%d, lk=%d, lsub =%d,checksum=%u\n", iam, k, lk, lsub,crc_16_val);
                //fflush(stdout);

                if(BcTree_getDestCount(UBtree_ptr[lk],'d')>0){
                    //printf("iam=%d,before BcTree_forwardMessageOneSide\n",iam);
                    //fflush(stdout);
                    BcTree_forwardMessageOneSide(UBtree_ptr[lk],recvbuf0,BcTree_GetMsgSize(UBtree_ptr[lk],'z')*nrhs+XK_H,'z',&iam_col, BCcount, BCbase, &maxrecvsz, Pc);
                    //printf("iam=%d,end BcTree_forwardMessageOneSide\n",iam);
                    //fflush(stdout);
                }
                //printf("iam=%d,before dlsum_bmod_inv_master debug_count=%d\n",iam,debug_count);
                //fflush(stdout);

                zlsum_bmod_inv_master_onesided(lsum, x, &recvbuf0[XK_H], rtemp, nrhs, k, bmod, Urbs,
						Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
						stat_loc, sizelsum,sizertemp,thread_id,num_thread,
						&iam_row, RDcount, RDbase, &iam_col, BCcount, BCbase, Pc, maxrecvsz);
                //printf("iam=%d,End dlsum_bmod_inv_master debug_count=%d,shift=%d\n",iam,debug_count,shift);
                //fflush(stdout);

                if (BCis_solved[recvRankNum] == BufSize_u[recvRankNum]) {
                    validBCQindex_u[bcidx]=-1;
                    shift += 1;
                    //printf("iam=%d,shift=%d\n",iam,shift);
                    //fflush(stdout);
                }else{
                    if(shift>0){
                        validBCQindex_u[bcidx-shift]=validBCQindex_u[bcidx];
                        validBCQindex_u[bcidx]=-1;
                        //printf("End-iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                }
                //printf("iam=%d,at end of iter, BCis_solved[%d]=%d, BufSize_u[%d]=%d\n",iam, recvRankNum,BCis_solved[recvRankNum],recvRankNum,BufSize_u[recvRankNum]);
                //fflush(stdout);
            } // for (bcidx=0;bcidx<Pr && validBCQindex_u[bcidx]!=-1;bcidx++)
        } // if (totalsolveBC < nbrecvx)

        if (totalsolveRD < nbrecvmod){
            shift=0;
            //foMPI_Win_flush_all(rd_winl);
            for (rdidx=0;rdidx<Pc && validRDQindex_u[rdidx]!=-1;rdidx++){

                recvRankNum=validRDQindex_u[rdidx];  //bcidx; //validBCQindex[bcidx];
                ird=RD_taskbuf_offset[recvRankNum]+RDis_solved[recvRankNum]*maxrecvsz*my_adjust_num;
                recvbuf0 = &RD_taskq[ird];
			k = (*recvbuf0).r;
                //printf("rdrd--111--iam=%d, rdidx=%d,k=%d\n",iam,rdidx,k);
                //fflush(stdout);
#if ( PROFlevel>=1 )
                TIC(t1);
#endif
                if (k < 0) {
                    if(shift>0){
                        validRDQindex_u[rdidx-shift]=validRDQindex_u[rdidx];
                        validRDQindex_u[rdidx]=-1;
                    }
                    continue;
                }
                lk = LBi( k, grid );
                //if (totalsolveRD %10 == 0){
                checkend=RdTree_GetMsgSize(URtree_ptr[lk],'z')*nrhs;
                //crc_16_val=crc_16((unsigned char*)&recvbuf0[LSUM_H],sizeof(doublecomplex)*checkend);
                //if(crc_16_val!=(uint16_t)recvbuf0[LSUM_H-1].r) {
                crc_8_val=crc_8((unsigned char*)&recvbuf0[LSUM_H],sizeof(doublecomplex)*checkend);
                if(crc_8_val!=(uint8_t)recvbuf0[LSUM_H-1].r) {
                    if(shift>0){
                        validRDQindex_u[rdidx-shift]=validRDQindex_u[rdidx];
                        validRDQindex_u[rdidx]=-1;
                    }
                    continue;
                }
#if ( PROFlevel>=1 )
                TOC(t2, t1);
						stat_loc[thread_id]->utime[SOL_COMM] += t2;
#endif
                //}
                //t = SuperLU_timer_();
                totalsolveRD += 1; //RD_subtotal[rdidx]-RDis_solved[rdidx];

                RDis_solved[recvRankNum] += 1 ;
                //printf("In U-RD solve, iam %d, k=%d, lk=%d,checksum=%u\n", iam, k, lk,crc_16_val);
                //fflush(stdout);

                knsupc = SuperSize( k );
                tempv = &recvbuf0[LSUM_H];
                il = LSUM_BLK( lk );
                RHS_ITERATE(j) {
#ifdef _OPENMP
#pragma omp simd
#endif
                    for (i = 0; i < knsupc; ++i)
						z_add(&lsum[i + il + j*knsupc + thread_id*sizelsum],
							  &lsum[i + il + j*knsupc + thread_id*sizelsum],
							  &tempv[i + j*knsupc]);
                }

                bmod_tmp=--bmod[lk*aln_i];
                thread_id = 0;
                rtemp_loc = &rtemp[sizertemp* thread_id];
                if ( bmod_tmp==0 ) {
                    if(RdTree_IsRoot(URtree_ptr[lk],'z')==YES){
                        knsupc = SuperSize( k );
                        for (ii=1;ii<num_thread;ii++)
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (jj=0;jj<knsupc*nrhs;jj++)
								    z_add(&lsum[il+ jj ],
								    	  &lsum[il+ jj ],
								    	  &lsum[il + jj + ii*sizelsum]);

                        ii = X_BLK( lk );
                        RHS_ITERATE(j)
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (i = 0; i < knsupc; ++i)
							        z_add(&x[i + ii + j*knsupc],
									  &x[i + ii + j*knsupc],
									  &lsum[i + il + j*knsupc] );

                        lk = LBj( k, grid ); /* Local block number, column-wise. */
                        lsub = Lrowind_bc_ptr[lk];
                        lusup = Lnzval_bc_ptr[lk];
                        nsupr = lsub[1];

                        if(Llu->inv == 1){

                            Uinv = Uinv_bc_ptr[lk];

#ifdef _CRAY
							CGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
							zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
							zgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc );
#endif

#ifdef _OPENMP
#pragma omp simd
#endif
                            for (i=0 ; i<knsupc*nrhs ; i++){
								z_copy(&x[ii+i],&rtemp_loc[i]);
                            }
                        }else{
#ifdef _CRAY
							CTRSM(ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
							ztrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
							ztrsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &knsupc);
#endif
                        }

#if ( PROFlevel>=1 )
                        TOC(t2, t1);
							stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif
                        stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc + 1) * nrhs;

#if ( DEBUGlevel>=2 )
                        printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

                        if(UBtree_ptr[lk]!=NULL){
                            BcTree_forwardMessageOneSide(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'z')*nrhs+XK_H,'z',&iam_col,BCcount, BCbase, &maxrecvsz,Pc);
                        }

                        if ( Urbs[lk] )
                            zlsum_bmod_inv_master_onesided(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
									Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
									stat_loc, sizelsum,sizertemp,thread_id,num_thread,
						            &iam_row, RDcount, RDbase, &iam_col, BCcount, BCbase, Pc, maxrecvsz);

                    }else{ // if(RdTree_IsRoot(URtree_ptr[lk],'d')==YES)
                        il = LSUM_BLK( lk );
                        knsupc = SuperSize( k );

                        for (ii=1;ii<num_thread;ii++)
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (jj=0;jj<knsupc*nrhs;jj++)
    								z_add(&lsum[il+ jj ],
    									  &lsum[il+ jj ],
    									  &lsum[il + jj + ii*sizelsum]);

                        RdTree_forwardMessageOneSide(URtree_ptr[lk],&lsum[il-LSUM_H],RdTree_GetMsgSize(URtree_ptr[lk],'z')*nrhs+LSUM_H,'z',&iam_row, RDcount,RDbase, &maxrecvsz, Pc);
                    }//if(RdTree_IsRoot(URtree_ptr[lk],'d')==YES)
                }//if ( bmod_tmp==0 )
                if (RDis_solved[recvRankNum] == BufSize_urd[recvRankNum]) {
                    validRDQindex_u[rdidx]=-1;
                    shift += 1;
                    //printf("iam=%d,shift=%d\n",iam,shift);
                    //fflush(stdout);
                }else{
                    if(shift>0){
                        validRDQindex_u[rdidx-shift]=validRDQindex_u[rdidx];
                        validRDQindex_u[rdidx]=-1;
                        //printf("iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                }
            }//for (rdidx=0;rdidx<Pc;rdidx++)
        }
        nbrecv1 = totalsolveBC + totalsolveRD;
    }


#if ( PRNTlevel>=2 )
		t = SuperLU_timer_() - t;
		stat->utime[SOL_TOT] += t;
		if ( !iam ) printf(".. U-solve time\t%8.4f\n", t);
		MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE,
				MPI_MAX, 0, grid->comm);
		if ( !iam ) {
			printf(".. U-solve time (MAX) \t%8.4f\n", tmax);
			fflush(stdout);
		}
		t = SuperLU_timer_();
#endif


#if ( DEBUGlevel>=2 )
		{
			doublecomplex *x_col;
			int diag;
			printf("\n(%d) .. After U-solve: x (ON DIAG PROCS) = \n", iam);
			ii = 0;
			for (k = 0; k < nsupers; ++k) {
				knsupc = SuperSize( k );
				krow = PROW( k, grid );
				kcol = PCOL( k, grid );
				diag = PNUM( krow, kcol, grid);
				if ( iam == diag ) { /* Diagonal process. */
					lk = LBi( k, grid );
					jj = X_BLK( lk );
					x_col = &x[jj];
					RHS_ITERATE(j) {
						for (i = 0; i < knsupc; ++i) { /* X stored in blocks */
							printf("\t(%d)\t%4d\t%.10f\n",
									iam, xsup[k]+i, x_col[i]);
						}
						x_col += knsupc;
					}
				}
				ii += knsupc;
			} /* for k ... */
		}
#endif

		pzReDistribute_X_to_B(n, B, m_loc, ldb, fst_row, nrhs, x, ilsum,
				ScalePermstruct, Glu_persist, grid, SOLVEstruct);


#if ( PRNTlevel>=2 )
		t = SuperLU_timer_() - t;
		if ( !iam) printf(".. X to B redistribute time\t%8.4f\n", t);
		t = SuperLU_timer_();
#endif


		double tmp1=0;
		double tmp2=0;
		double tmp3=0;
		double tmp4=0;
		for(i=0;i<num_thread;i++){
			tmp1 = SUPERLU_MAX(tmp1,stat_loc[i]->utime[SOL_TRSM]);
			tmp2 = SUPERLU_MAX(tmp2,stat_loc[i]->utime[SOL_GEMM]);
			tmp3 = SUPERLU_MAX(tmp3,stat_loc[i]->utime[SOL_COMM]);
			tmp4 += stat_loc[i]->ops[SOLVE];
#if ( PRNTlevel>=2 )
			if(iam==0)printf("thread %5d gemm %9.5f\n",i,stat_loc[i]->utime[SOL_GEMM]);
#endif
		}


		stat->utime[SOL_TRSM] += tmp1;
		stat->utime[SOL_GEMM] += tmp2;
		stat->utime[SOL_COMM] += tmp3;
		stat->ops[SOLVE]+= tmp4;


		/* Deallocate storage. */
		for(i=0;i<num_thread;i++){
			PStatFree(stat_loc[i]);
			SUPERLU_FREE(stat_loc[i]);
		}
		SUPERLU_FREE(stat_loc);
		SUPERLU_FREE(rtemp);
		SUPERLU_FREE(lsum);
		SUPERLU_FREE(x);


		SUPERLU_FREE(bmod);
		SUPERLU_FREE(brecv);
		SUPERLU_FREE(root_send);

		SUPERLU_FREE(rootsups);

		log_memory(-nlb*aln_i*iword-nlb*iword - nsupers_i*iword - (CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*iword - maxrecvsz*(nbrecvx+1)*dword*2.0 - sizelsum*num_thread * dword*2.0 - (ldalsum * nrhs + nlb * XK_H) *dword*2.0 - (sizertemp*num_thread + 1)*dword*2.0, stat);	//account for bmod, brecv, root_send, rootsups, recvbuf_BC_fwd,rtemp,lsum,x


		MPI_Barrier( grid->comm );


#if ( PROFlevel>=2 )
		{
			float msg_vol_max, msg_vol_sum, msg_cnt_max, msg_cnt_sum;

			MPI_Reduce (&msg_cnt, &msg_cnt_sum,
					1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
			MPI_Reduce (&msg_cnt, &msg_cnt_max,
					1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
			MPI_Reduce (&msg_vol, &msg_vol_sum,
					1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
			MPI_Reduce (&msg_vol, &msg_vol_max,
					1, MPI_FLOAT, MPI_MAX, 0, grid->comm);
			if (!iam) {
				printf ("\tPDGSTRS comm stat:"
						"\tAvg\tMax\t\tAvg\tMax\n"
						"\t\t\tCount:\t%.0f\t%.0f\tVol(MB)\t%.2f\t%.2f\n",
						msg_cnt_sum / Pr / Pc, msg_cnt_max,
						msg_vol_sum / Pr / Pc * 1e-6, msg_vol_max * 1e-6);
			}
		}
#endif

    stat->utime[SOLVE] = SuperLU_timer_() - t1_sol;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit pzgstrs()");
#endif


#if ( PRNTlevel>=2 )
	    float for_lu, total, max, avg, temp;
		superlu_dist_mem_usage_t num_mem_usage;

	    zQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);
	    temp = num_mem_usage.total;

	    MPI_Reduce( &temp, &max,
		       1, MPI_FLOAT, MPI_MAX, 0, grid->comm );
	    MPI_Reduce( &temp, &avg,
		       1, MPI_FLOAT, MPI_SUM, 0, grid->comm );
            if (!iam) {
		printf("\n** Memory Usage **********************************\n");
                printf("** Total highmark (MB):\n"
		       "    Sum-of-all : %8.2f | Avg : %8.2f  | Max : %8.2f\n",
		       avg * 1e-6,
		       avg / grid->nprow / grid->npcol * 1e-6,
		       max * 1e-6);
		printf("**************************************************\n");
		fflush(stdout);
            }
#endif


    return;
} /* PZGSTRS */

#endif