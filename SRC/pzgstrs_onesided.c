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
 *   Compute the inverse of the diagonal blocks of the L and U
 *   triangular matrices using one-sided MPI.
 * </pre>
 */
#ifdef one_sided
void
pzgstrs_onesided(superlu_dist_options_t *options, int_t n,
        zLUstruct_t *LUstruct,
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
            *recvbufall, *recvbuf0, *xin, *recvbuf_BC_gpu,*recvbuf_RD_gpu;
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
    C_Tree  *LBtree_ptr = Llu->LBtree_ptr;
    C_Tree  *LRtree_ptr = Llu->LRtree_ptr;
    C_Tree  *UBtree_ptr = Llu->UBtree_ptr;
    C_Tree  *URtree_ptr = Llu->URtree_ptr;
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
    int  *fmod;         /* Modification count for L-solve --
    			 Count the number of local block products to
    			 be summed into lsum[lk]. */
    int_t *fmod_sort;
	int_t *order;
	//int_t *order1;
	//int_t *order2;
    int  fmod_tmp;
    int  **fsendx_plist = Llu->fsendx_plist;
    int  nfrecvx = Llu->nfrecvx; /* Number of X components to be recv'd. */
    int  nfrecvx_buf=0;
    int  *frecv;        /* Count of lsum[lk] contributions to be received
    			     from processes in this row.
    			     It is only valid on the diagonal processes. */
    int  frecv_tmp;
    int  nfrecvmod = 0; /* Count of total modifications to be recv'd. */
    int  nfrecv = 0; /* Count of total messages to be recv'd. */
    int  nbrecv = 0; /* Count of total messages to be recv'd. */
    int  nleaf = 0, nroot = 0;
    int  nleaftmp = 0, nroottmp = 0;
    int_t  msgsize;
        /*-- Counts used for U-solve --*/
    int  *bmod;         /* Modification count for U-solve. */
    int  bmod_tmp;
    int  **bsendx_plist = Llu->bsendx_plist;
    int  nbrecvx = Llu->nbrecvx; /* Number of X components to be recv'd. */
    int  nbrecvx_buf=0;
    int  *brecv;        /* Count of modifications to be recv'd from
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

    int *mod_bit = Llu->mod_bit; /* flag contribution from each row block */
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
    int thread_id = 0;
    yes_no_t empty;
    int_t sizelsum,sizertemp,aln_d,aln_i;
    aln_d = 1; //ceil(CACHELINE/(double)dword);
    aln_i = 1; //ceil(CACHELINE/(double)iword);
    int num_thread = 1;
    int_t cnt1,cnt2;


#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK) && defined(GPU_SOLVE)  /* GPU trisolve*/

#if ( PRNTlevel>=1 )
	if ( !iam) printf(".. GPU trisolve\n");
	fflush(stdout);
#endif

#ifdef GPUREF

#ifdef HAVE_CUDA
	int_t *cooCols,*cooRows;
	doublecomplex *nzval;
	int_t *rowind, *colptr;
	int_t *colind, *rowptr, *rowptr1;
	doublecomplex *cooVals;
	int_t ntmp,nnzL;

    cusparseHandle_t handle = NULL;
    gpuStream_t stream = NULL;
    cusparseStatus_t status1 = CUSPARSE_STATUS_SUCCESS;
    cusparseStatus_t status2 = CUSPARSE_STATUS_SUCCESS;
    cusparseStatus_t status3 = CUSPARSE_STATUS_SUCCESS;
    cusparseStatus_t status4 = CUSPARSE_STATUS_SUCCESS;
    cusparseStatus_t status5 = CUSPARSE_STATUS_SUCCESS;
	gpuError_t gpuStat = gpuSuccess;
    cusparseMatDescr_t descrA = NULL;
    csrsm2Info_t info1 = NULL;
    csrsv2Info_t info2 = NULL;


	int_t *d_csrRowPtr = NULL;
	int_t *d_cooRows = NULL;
    int_t *d_cooCols = NULL;
    int_t *d_P       = NULL;
    doublecomplex *d_cooVals = NULL;
    doublecomplex *d_csrVals = NULL;
    doublecomplex *d_B = NULL;
    doublecomplex *d_X = NULL;
    doublecomplex *Btmp;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;
    int_t *perm_r = ScalePermstruct->perm_r;
    int_t *perm_c = ScalePermstruct->perm_c;
	int_t l;


    size_t lworkInBytes = 0;
    int lworkInBytes2 = 0;
    char *d_work = NULL;

    const int algo = 1; /* 0: non-block version 1: block version */
	const double h_one = 1.0;
	const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
#else
	printf("only cusparse is implemented\n");
	exit(0);
#endif /* end HAVE_CUDA */
#else

	const int nwrp_block = 1; /* number of warps in each block */
	const int warp_size = 32; /* number of threads per warp*/
	gpuStream_t sid=0;
	int gid=0;
	gridinfo_t *d_grid = NULL;
	doublecomplex *d_x = NULL;
	doublecomplex *d_lsum = NULL;
    int_t  *d_fmod = NULL;
#endif /* end GPU_REF */
#endif /* end GPU trisolve */

    maxsuper = sp_ienv_dist(3, options);


#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
    	if (omp_get_thread_num () == 0) {
    		num_thread = omp_get_num_threads ();
    	}
    }
#else
	num_thread=1;
#endif

#if ( PRNTlevel>=1 )
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
    CHECK_MALLOC(iam, "Enter pzgstrs_onesided()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    /* Save the count to be altered so it can be used by
       subsequent call to PZGSTRS. */
    if ( !(fmod = int32Malloc_dist(nlb*aln_i)) )
	ABORT("Malloc fails for fmod[].");

    for (i = 0; i < nlb; ++i) fmod[i*aln_i] = Llu->fmod[i];
#if 0
    if ( !(fmod_sort = intCalloc_dist(nlb*2)) )
    		ABORT("Calloc fails for fmod_sort[].");

    for (j=0;j<nlb;++j)fmod_sort[j]=0;
    for (j=0;j<nlb;++j)fmod_sort[j+nlb]=j;
    zComputeLevelsets(iam, nsupers, grid, Glu_persist, Llu,fmod_sort);

    quickSortM(fmod_sort,0,nlb-1,nlb,0,2);

    if ( !(order = intCalloc_dist(nlb)) )
    	ABORT("Calloc fails for order[].");
    for (j=0;j<nlb;++j)order[j]=fmod_sort[j+nlb];


    // if ( !(order1 = intCalloc_dist(nlb)) )
    // 	ABORT("Calloc fails for order1[].");
    // if ( !(order2 = intCalloc_dist(nlb)) )
    // 	ABORT("Calloc fails for order2[].");
    // cnt1=0;
    // cnt2=0;
    // for (j=0;j<nlb;++j){
    // 	if(Llu->fmod[j]==0){
    // 		order1[cnt1]=j;
    // 		cnt1++;
    // 	}else{
    // 		order2[cnt2]=j;
    // 		cnt2++;
    // 	}
    // }

    // for (j=0;j<cnt1;++j){
    // 	order[j]=order1[j];
    // }
    // for (j=0;j<cnt2;++j){
    // 	order[j+cnt1]=order2[j];
    // }
   	// SUPERLU_FREE(order1);
   	// SUPERLU_FREE(order2);

    // for (j=0;j<nlb;++j){
    // 	printf("%5d%5d\n",order[j],fmod_sort[j]);
    // 	fflush(stdout);
    // }

    	SUPERLU_FREE(fmod_sort);
    	SUPERLU_FREE(order);
#endif

    if ( !(frecv = int32Calloc_dist(nlb)) )
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
    knsupc = sp_ienv_dist(3, options);
    maxrecvsz = knsupc * nrhs + SUPERLU_MAX( XK_H, LSUM_H )+2;
    sizelsum = (((size_t)ldalsum)*nrhs + nlb*LSUM_H);
    sizelsum = ((sizelsum + (aln_d - 1)) / aln_d) * aln_d;

#ifdef _OPENMP
    if ( !(lsum = (doublecomplex*)SUPERLU_MALLOC(sizelsum*num_thread * sizeof(doublecomplex))))
	ABORT("Malloc fails for lsum[].");
#pragma omp parallel default(shared) private(ii)
    {
	int thread_id = omp_get_thread_num(); //mjc
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
	int thread_id=omp_get_thread_num();
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
		if(LBtree_ptr[lk].empty_==NO){
			// printf("LBtree_ptr lk %5d\n",lk);
			//if(BcTree_IsRoot(LBtree_ptr[lk],'z')==NO){
			if(C_BcTree_IsRoot(&LBtree_ptr[lk])==NO){
				nbtree++;
				//if(BcTree_getDestCount(LBtree_ptr[lk],'z')>0)nfrecvx_buf++;
				if(LBtree_ptr[lk].destCnt_>0)nfrecvx_buf++;
			}
			//BcTree_allocateRequest(LBtree_ptr[lk],'z');
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
		if(LRtree_ptr[lk].empty_==NO){
			nrtree++;
			//RdTree_allocateRequest(LRtree_ptr[lk],'z');
			//frecv[lk] = RdTree_GetDestCount(LRtree_ptr[lk],'z');
			frecv[lk] = LRtree_ptr[lk].destCnt_;
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
    int *validBCQindex;
    int *validRDQindex;
    int *validBCQindex_u;
    int *validRDQindex_u;


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
            BCbase[i] = recv_size_all[i]*maxrecvsz;
            validBCQindex[i]=keep_validBCQindex[i];
            validBCQindex_u[i]=keep_validBCQindex_u[i];
#if ( DEBUGlevel>=1 )
            printf("iam=%d,BCbase[%d]=%lu,validBCQindex[%d]=%d,validBCQindex_u[%d]=%d\n",iam,i,BCbase[i],i,validBCQindex[i],i,validBCQindex_u[i]);
            fflush(stdout);
#endif
        }
     }
    if(Pc > 1){
        for (i=0;i<Pc;i++){
            RDbase[i] = recv_size_all[Pr+i]*maxrecvsz;
            validRDQindex[i]=keep_validRDQindex[i];
            validRDQindex_u[i]=keep_validRDQindex_u[i];
#if ( DEBUGlevel>=1 )
            printf("iam=%d,RDbase[%d]=%lu,validRDQindex[%d]=%d,validRDQindex_u[%d]=%d\n",iam,i,RDbase[i],i,validRDQindex[i],i,validRDQindex_u[i]);
            fflush(stdout);
#endif
        }
    }

    nfrecvx_buf=0;

    BC_taskbuf_offset = (long*)SUPERLU_MALLOC( Pr * sizeof(long));   // this needs to be optimized for 1D row mapping
    RD_taskbuf_offset = (long*)SUPERLU_MALLOC( Pc * sizeof(long));   // this needs to be optimized for 1D row mapping
    memset(BC_taskbuf_offset, 0, Pr * sizeof(long));
    memset(RD_taskbuf_offset, 0, Pc * sizeof(long));

    for (bcidx=0;bcidx<Pr;bcidx++){
        for(int tmp=0;tmp<bcidx;tmp++){
            BC_taskbuf_offset[bcidx] += BufSize[tmp]*maxrecvsz;
        }
#if ( DEBUGlevel>=1 )
        printf("iam=%d, BC_taskbuf_offset[%d]=%lu\n",iam,bcidx,BC_taskbuf_offset[bcidx]);
        fflush(stdout);
#endif
    }
    for (rdidx=0;rdidx<Pc;rdidx++){
        for(int tmp=0;tmp<rdidx;tmp++){
            RD_taskbuf_offset[rdidx] += BufSize_rd[tmp]*maxrecvsz;
        }
#if ( DEBUGlevel>=1 )
        printf("iam=%d, RD_taskbuf_offset[%d]=%lu\n",iam,rdidx,RD_taskbuf_offset[rdidx]);
        fflush(stdout);
#endif
    }
    BCis_solved = (int*)SUPERLU_MALLOC( Pr * sizeof(int));   // this needs to be optimized for 1D row mapping
    RDis_solved = (int*)SUPERLU_MALLOC( Pc * sizeof(int));   // this needs to be optimized for 1D row mapping
    memset(BCis_solved, 0, Pr * sizeof(int));
    memset(RDis_solved, 0, Pc * sizeof(int));
#if ( DEBUGlevel>=1 )
    printf("iam=%d, End setup oneside L solve\n",iam);
	printf("(%2d) nfrecvx %4d,  nfrecvmod %4d,  nleaf %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nfrecvx, nfrecvmod, nleaf, nbtree, nrtree);
    if (Pr>1){
        for (bcidx=0;bcidx<Pr;bcidx++){
            if (iam==bcidx) continue;
            printf("iam %d recv from %d: %d/%d\n",iam, bcidx, BufSize[bcidx], nfrecvx);
            fflush(stdout);
        }
    }

    if (Pc>1){
        for (bcidx=0;bcidx<Pc;bcidx++){
            if (iam==bcidx) continue;
            printf("iam %d recv from %d: %d/%d\n",iam, bcidx, BufSize_rd[bcidx], nfrecvmod);
            fflush(stdout);
        }
    }
    fflush(stdout);
#endif


	nfrecvx_buf=0;

	log_memory(nlb*aln_i*iword+nlb*iword+(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*2.0*iword+ nsupers_i*iword + sizelsum*num_thread * dword*2.0 + (ldalsum * nrhs + nlb * XK_H) *dword*2.0 + (sizertemp*num_thread + 1)*dword*2.0+maxrecvsz*(nfrecvx+1)*dword*2.0, stat);	//account for fmod, frecv, leaf_send, root_send, leafsups, recvbuf_BC_fwd	, lsum, x, rtemp


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

#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK) && defined(GPU_SOLVE)  /* GPU trisolve*/
// #if 0 /* CPU trisolve*/

#ifdef GPUREF /* use cuSparse*/
#ifdef HAVE_CUDA
	if(procs>1){
	printf("procs>1 with GPU not implemented for trisolve using CuSparse\n");
	fflush(stdout);
	exit(1);
	}

t1 = SuperLU_timer_();

#if 0  // this will readin a matrix with only lower triangular part, note that this code block is only for benchmarking cusparse performance

	FILE *fp, *fopen();
	if ( !(fp = fopen("/gpfs/alpine/scratch/liuyangz/csc289/matrix/HTS/copter2.mtx", "r")) ) {
	// if ( !(fp = fopen("/gpfs/alpine/scratch/liuyangz/csc289/matrix/HTS/epb3.mtx", "r")) ) {
	// if ( !(fp = fopen("/gpfs/alpine/scratch/liuyangz/csc289/matrix/HTS/gridgena.mtx", "r")) ) {
	// if ( !(fp = fopen("/gpfs/alpine/scratch/liuyangz/csc289/matrix/HTS/vanbody.mtx", "r")) ) {
	// if ( !(fp = fopen("/gpfs/alpine/scratch/liuyangz/csc289/matrix/HTS/shipsec1.mtx", "r")) ) {
	// if ( !(fp = fopen("/gpfs/alpine/scratch/liuyangz/csc289/matrix/HTS/dawson5.mtx", "r")) ) {
	// if ( !(fp = fopen("/gpfs/alpine/scratch/liuyangz/csc289/matrix/HTS/gas_sensor.mtx", "r")) ) {
	// if ( !(fp = fopen("/gpfs/alpine/scratch/liuyangz/csc289/matrix/HTS/rajat16.mtx", "r")) ) {

			ABORT("File does not exist");
		}
	int mtmp;
	dreadMM_dist_intoL_CSR(fp, &mtmp, &ntmp, &nnzL,&nzval, &colind, &rowptr);
	if ( !(Btmp = (double*)SUPERLU_MALLOC((nrhs*ntmp) * sizeof(double))) )
		ABORT("Calloc fails for Btmp[].");
	for (i = 0; i < ntmp; ++i) {
		irow = i;
		RHS_ITERATE(j) {
		Btmp[i + j*ldb]=1.0;
		}
	}
#else

// //////// dGenCSCLblocks(iam, nsupers, grid,Glu_persist,Llu, &nzval, &rowind, &colptr, &ntmp, &nnzL);
	dGenCSRLblocks(iam, nsupers, grid,Glu_persist,Llu, &nzval, &colind, &rowptr, &ntmp, &nnzL);
	if ( !(Btmp = (double*)SUPERLU_MALLOC((nrhs*m_loc) * sizeof(double))) )
		ABORT("Calloc fails for Btmp[].");
	for (i = 0; i < ntmp; ++i) {
		irow = perm_c[perm_r[i+fst_row]]; /* Row number in Pc*Pr*B */
		RHS_ITERATE(j) {
		Btmp[irow + j*ldb]=B[i + j*ldb];
		// printf("%d %e\n",irow + j*ldb,Btmp[irow + j*ldb]);
		}
	}
#endif

    if ( !(rowptr1 = (int_t *) SUPERLU_MALLOC((ntmp+1) * sizeof(int_t))) )
        ABORT("Malloc fails for row[]");
	for (i=0;i<ntmp;i++)
		rowptr1[i]=rowptr[i];
	rowptr1[ntmp]=	nnzL; // cusparse requires n+1 elements in the row pointers, the last one is the nonzero count



	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. convert to CSR time\t%15.7f\n", t1);
		fflush(stdout);
	}

	t1 = SuperLU_timer_();
	checkGPU(gpuStreamCreateWithFlags(&stream, gpuStreamDefault));
	status1 = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status1);
    status1 = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status1);
	status1 = cusparseCreateMatDescr(&descrA);
    assert(CUSPARSE_STATUS_SUCCESS == status1);

	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. gpu initialize time\t%15.7f\n", t1);
		fflush(stdout);
	}
	t1 = SuperLU_timer_();


	checkGPU(gpuMalloc( (void**)&d_B, sizeof(double)*ntmp*nrhs));
	checkGPU(gpuMalloc( (void**)&d_X, sizeof(double)*ntmp*nrhs));
	checkGPU(gpuMalloc( (void**)&d_cooCols, sizeof(int)*nnzL));
	checkGPU(gpuMalloc( (void**)&d_csrVals, sizeof(double)*nnzL));
	checkGPU(gpuMalloc( (void**)&d_csrRowPtr,(ntmp+1)*sizeof(double)));


	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. gpuMalloc time\t%15.7f\n", t1);
		fflush(stdout);
	}
	t1 = SuperLU_timer_();

	checkGPU(gpuMemcpy(d_B, Btmp, sizeof(double)*nrhs*ntmp, gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_cooCols, colind, sizeof(int)*nnzL   , gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_csrRowPtr, rowptr1, sizeof(int)*(ntmp+1)   , gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_csrVals, nzval, sizeof(double)*nnzL, gpuMemcpyHostToDevice));


	// checkGPU(cudaDeviceSynchronize);
	checkGPU(gpuStreamSynchronize(stream));

	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. HostToDevice time\t%15.7f\n", t1);
		fflush(stdout);
	}
	t1 = SuperLU_timer_();

/* A is base-0*/
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
/* A is lower triangle */
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
/* A has unit diagonal */
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT);



#if 1  // this only works for 1 rhs
	assert(nrhs == 1);
    status1 = cusparseCreateCsrsv2Info(&info2);
    assert(CUSPARSE_STATUS_SUCCESS == status1);
    status1 = cusparseDcsrsv2_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        ntmp,
        nnzL,
        descrA,
        d_csrVals,
        d_csrRowPtr,
        d_cooCols,
        info2,
        &lworkInBytes2);
    assert(CUSPARSE_STATUS_SUCCESS == status1);
	printf("lworkInBytes  = %lld \n", (long long)lworkInBytes2);
    if (NULL != d_work) { gpuFree(d_work); }
	checkGPU(gpuMalloc( (void**)&d_work, lworkInBytes2));

	status1 = cusparseDcsrsv2_analysis(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        ntmp,
        nnzL,
        descrA,
        d_csrVals,
        d_csrRowPtr,
        d_cooCols,
        info2,
        policy,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status1);


	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. Cusparse analysis time\t%15.7f\n", t1);
		fflush(stdout);
	}

	t1 = SuperLU_timer_();
    status1 = cusparseDcsrsv2_solve(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        ntmp,
        nnzL,
		&h_one,
        descrA,
        d_csrVals,
        d_csrRowPtr,
        d_cooCols,
        info2,
        d_B,
		d_X,
        policy,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status1);
    // checkGPU(gpuDeviceSynchronize);
	checkGPU(gpuStreamSynchronize(stream));
	checkGPU(gpuMemcpy(d_B, d_X, sizeof(double)*nrhs*ntmp, cudaMemcpyDeviceToDevice));
	checkGPU(gpuStreamSynchronize(stream));

	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. Cusparse solve time\t%15.7f\n", t1);
		fflush(stdout);
	}


#else

    status1 = cusparseCreateCsrsm2Info(&info1);
    assert(CUSPARSE_STATUS_SUCCESS == status1);
    status1 = cusparseDcsrsm2_bufferSizeExt(
        handle,
        algo,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
        ntmp,
        nrhs,
        nnzL,
        &h_one,
        descrA,
        d_csrVals,
        d_csrRowPtr,
        d_cooCols,
        d_B,
        ntmp,   /* ldb */
        info1,
        policy,
        &lworkInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == status1);

	printf("lworkInBytes  = %lld \n", (long long)lworkInBytes);
    if (NULL != d_work) { gpuFree(d_work); }
	checkGPU(gpuMalloc( (void**)&d_work, lworkInBytes));

    status1 = cusparseDcsrsm2_analysis(
        handle,
        algo,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
        ntmp,
        nrhs,
        nnzL,
        &h_one,
        descrA,
        d_csrVals,
        d_csrRowPtr,
        d_cooCols,
        d_B,
        ntmp,   /* ldb */
        info1,
        policy,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status1);


	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. Cusparse analysis time\t%15.7f\n", t1);
		fflush(stdout);
	}

	t1 = SuperLU_timer_();
    status1 = cusparseDcsrsm2_solve(
        handle,
        algo,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
        ntmp,
        nrhs,
        nnzL,
        &h_one,
        descrA,
        d_csrVals,
        d_csrRowPtr,
        d_cooCols,
        d_B,
        ntmp,   /* ldb */
        info1,
        policy,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status1);
    // checkGPU(gpuDeviceSynchronize);
	checkGPU(gpuStreamSynchronize(stream));

	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. Cusparse solve time\t%15.7f\n", t1);
		fflush(stdout);
	}


#endif




	t1 = SuperLU_timer_();
	checkGPU(gpuMemcpy(Btmp, d_B, sizeof(double)*ntmp*nrhs, gpuMemcpyDeviceToHost));
	// checkGPU(gpuDeviceSynchronize);
	checkGPU(gpuStreamSynchronize(stream));
	t1 = SuperLU_timer_() - t1;
	if ( !iam ) {
		printf(".. DeviceToHost time\t%15.7f\n", t1);
		fflush(stdout);
	}


	for (i = 0; i < m_loc; ++i) {
		irow = i+fst_row;

		k = BlockNum( irow );
		knsupc = SuperSize( k );
		l = X_BLK( k );

		irow = irow - FstBlockC(k); /* Relative row number in X-block */
		RHS_ITERATE(j) {
		x[l + irow + j*knsupc] = Btmp[i + j*ldb];
		// printf("%d %e\n",l + irow + j*knsupc,x[l + irow + j*knsupc]);
		// fflush(stdout);
		}
	}
	SUPERLU_FREE(Btmp);

#endif

#else

// #if HAVE_CUDA
// cudaProfilerStart();
// #elif defined(HAVE_HIP)
// roctracer_mark("before HIP LaunchKernel");
// roctxMark("before hipLaunchKernel");
// roctxRangePush("hipLaunchKernel");
// #endif

	checkGPU(gpuMalloc( (void**)&d_grid, sizeof(gridinfo_t)));

	checkGPU(gpuMalloc( (void**)&recvbuf_BC_gpu, maxrecvsz*  CEILING( nsupers, grid->npcol) * sizeof(double))); // used for receiving and forwarding x on each thread
	checkGPU(gpuMalloc( (void**)&recvbuf_RD_gpu, 2*maxrecvsz*  CEILING( nsupers, grid->nprow) * sizeof(double))); // used for receiving and forwarding lsum on each thread
	checkGPU(gpuMalloc( (void**)&d_lsum, sizelsum*num_thread * sizeof(double)));
	checkGPU(gpuMalloc( (void**)&d_x, (ldalsum * nrhs + nlb * XK_H) * sizeof(double)));
	checkGPU(gpuMalloc( (void**)&d_fmod, (nlb*aln_i) * sizeof(int_t)));


	checkGPU(gpuMemcpy(d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_lsum, lsum, sizelsum*num_thread * sizeof(double), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_x, x, (ldalsum * nrhs + nlb * XK_H) * sizeof(double), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_fmod, fmod, (nlb*aln_i) * sizeof(int_t), gpuMemcpyHostToDevice));

	k = CEILING( nsupers, grid->npcol);/* Number of local block columns divided by #warps per block used as number of thread blocks*/
	knsupc = sp_ienv_dist(3, options);
	dlsum_fmod_inv_gpu_wrap(k,nlb,DIM_X,DIM_Y,d_lsum,d_x,nrhs,knsupc,nsupers,d_fmod,Llu->d_LBtree_ptr,Llu->d_LRtree_ptr,Llu->d_ilsum,Llu->d_Lrowind_bc_dat, Llu->d_Lrowind_bc_offset, Llu->d_Lnzval_bc_dat, Llu->d_Lnzval_bc_offset, Llu->d_Linv_bc_dat, Llu->d_Linv_bc_offset, Llu->d_Lindval_loc_bc_dat, Llu->d_Lindval_loc_bc_offset,Llu->d_xsup,d_grid,recvbuf_BC_gpu,recvbuf_RD_gpu,maxrecvsz);

	checkGPU(gpuMemcpy(x, d_x, (ldalsum * nrhs + nlb * XK_H) * sizeof(double), gpuMemcpyDeviceToHost));

	checkGPU (gpuFree (d_grid));
	checkGPU (gpuFree (recvbuf_BC_gpu));
	checkGPU (gpuFree (recvbuf_RD_gpu));
	checkGPU (gpuFree (d_x));
	checkGPU (gpuFree (d_lsum));
	checkGPU (gpuFree (d_fmod));

	stat_loc[0]->ops[SOLVE]+=Llu->Lnzval_bc_cnt*nrhs*2; // YL: this is a rough estimate
#endif

#else  /* CPU trisolve*/

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
	    int thread_id = omp_get_thread_num();
#else
	{
 	    thread_id=0;
#endif
		{

            if (Llu->inv == 1) { /* Diagonal is inverted. */

#ifdef _OPENMP
#pragma	omp for firstprivate(nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Linv,i,lib,rtemp_loc,nleaf_send_tmp) nowait
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

			if(LBtree_ptr[lk].empty_==NO){
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

		    if (LBtree_ptr[lk].empty_==NO) {
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
	    } /* end else ... diagonal is not inverted */
	  }
	} /* end parallel region */

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
#pragma	omp taskloop private (k,ii,lk,thread_id) num_tasks(num_thread*8) nogroup
#endif
		for (jj=0;jj<nleaf;jj++){
		    k=leafsups[jj];

		    {
#ifdef _OPENMP
                        thread_id=omp_get_thread_num();
#else
                        thread_id=0;
#endif
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
			//BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'z')*nrhs+XK_H,'z');
			//C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);
			C_BcTree_forwardMessage_onesided(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H, BCcount, BCbase, &maxrecvsz,Pc);

		}else{ // this is a reduce forwarding
			lk = -lk - 1;
			il = LSUM_BLK( lk );
			//RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(LRtree_ptr[lk],'z')*nrhs+LSUM_H,'z');
			//C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
			C_RdTree_forwardMessage_onesided(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H, RDcount, RDbase, &maxrecvsz, Pc);

		}
	}
#if ( DEBUGlevel>=1 )
	printf("(%2d) end sending nleaf_send %4d\n", iam, nleaf_send);
	fflush(stdout);
#endif

#ifdef USE_VTUNE
	__itt_pause(); // stop VTune
	__SSC_MARK(0x222); // stop SDE tracing
#endif

	/* -----------------------------------------------------------
	   Compute the internal nodes asynchronously by all processes.
	   ----------------------------------------------------------- */

double k_tmp;
while( nfrecv1 < nfrecvx+nfrecvmod ){
        thread_id = 0;
        if (totalsolveBC < nfrecvx){
            shift=0;
            for (bcidx=0;bcidx<Pr && validBCQindex[bcidx]!=-1;bcidx++){

                recvRankNum=validBCQindex[bcidx];  //bcidx; //validBCQindex[bcidx];
                i=BC_taskbuf_offset[recvRankNum]+BCis_solved[recvRankNum]*maxrecvsz; //BCis_solved[bcidx];
                recvbuf0 = &BC_taskq[i];
   	    		k_tmp = (*recvbuf0).r;

#if ( DEBUGlevel>=2 )
        printf("iam=%d, recvRankNum=%d,BC_taskbuf_offset=%lu,BCis_solved=%d/%d,maxrecvsz=%d\n",
               iam,recvRankNum,BC_taskbuf_offset[recvRankNum],BCis_solved[recvRankNum],nfrecvx,maxrecvsz);
        fflush(stdout);
        printf("iam=%d, recvRankNum=%d,sig=%lf,%lf\n",iam,recvRankNum,k_tmp,recvbuf0[1].r);
        fflush(stdout);

#endif

#if ( PROFlevel>=1 )
                TIC(t1);
#endif
                int kint= k_tmp;
                if (kint != 1)  {
                    if(shift>0){
                        validBCQindex[bcidx-shift]=validBCQindex[bcidx];
                        validBCQindex[bcidx]=-1;
                        //printf("iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                    continue;
                }
                k=recvbuf0[1].r;
                lk = LBj( k, grid );    /* local block number */


#if ( PROFlevel>=1 )
                TOC(t2, t1);
				stat_loc[thread_id]->utime[SOL_COMM] += t2;
#endif
                //t= SuperLU_timer_();

                totalsolveBC += 1; //BC_subtotal[bcidx] - BCis_solved[bcidx];
                BCis_solved[recvRankNum]++;

#if ( DEBUGlevel>=2 )
                printf("iam=%d,k=%d, lk=%d,destCnt=%d,here\n",iam,k,lk,LBtree_ptr[lk].destCnt_);
                fflush(stdout);
                for (int dn=0; dn<LBtree_ptr[lk].msgSize_*nrhs+XK_H+1;dn++){
                    printf("%d recv'd %d at %d,buffer[%d]=%lf\n",iam, recvRankNum, BCis_solved[recvRankNum]-1, dn,recvbuf0[dn].r);
                    fflush(stdout);
                }
#endif
                if (LBtree_ptr[lk].destCnt_>0){
                    C_BcTree_forwardMessage_onesided(&LBtree_ptr[lk], &recvbuf0[1], LBtree_ptr[lk].msgSize_*nrhs+XK_H, BCcount, BCbase, &maxrecvsz,Pc);
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
                        xin = &recvbuf0[XK_H+1] ;
                    }
#if ( DEBUGlevel>=2 )
                    printf("iam=%d,lk=%d,enter master\n",iam,lk);
                    fflush(stdout);
#endif
                    zlsum_fmod_inv_master_onesided(lsum, x, xin, rtemp, nrhs, knsupc, k,
                                                fmod, nb, xsup, grid, Llu,
                                                stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread,
                                                RDcount, RDbase, BCcount, BCbase, Pc, maxrecvsz);
#if ( DEBUGlevel>=2 )
                    printf("iam=%d,lk=%d, out master\n",iam,lk);
                    fflush(stdout);
#endif

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
#if ( DEBUGlevel>=2 )
                printf("iam=%d,BCis_solved[%d]=%d,BufSize[%d]=%d\n",iam,recvRankNum,BCis_solved[recvRankNum],recvRankNum,BufSize[recvRankNum]);
                fflush(stdout);
#endif
            } // for bcidx

            //TOC(t2, t1);
            //onesidecomm_rd += t2;
        }


        if (totalsolveRD < nfrecvmod){
            shift=0;
            //for (rdidx=0;rdidx<Pc ;rdidx++){
            for (rdidx=0;rdidx<Pc && validRDQindex[rdidx]!=-1;rdidx++){
                recvRankNum=validRDQindex[rdidx];  //bcidx; //validBCQindex[bcidx];
                ird=RD_taskbuf_offset[recvRankNum]+RDis_solved[recvRankNum]*maxrecvsz;
                recvbuf0 = &RD_taskq[ird];
   	    		k_tmp = (*recvbuf0).r;
#if ( DEBUGlevel>=2 )
                printf("iam=%d, RD_taskbuf_offset=%lu,RDis_solved=%d,maxrecvsz=%d\n",
                       iam,RD_taskbuf_offset[recvRankNum],RDis_solved[recvRankNum],maxrecvsz);
                fflush(stdout);
                printf("iam=%d, recvRankNum=%d,sig=%lf,%lf\n",iam,recvRankNum,k_tmp,recvbuf0[1].r);
                fflush(stdout);
#endif

#if ( PROFlevel>=1 )
                TIC(t1);
#endif
                int kint = k_tmp;
                if (kint != 1)  {
#if ( DEBUGlevel>=2 )
                    printf("iam=%d,kint=%d,k=%lf\n",iam,kint,k_tmp);
                    fflush(stdout);
#endif
                    if(shift>0){
                        validRDQindex[rdidx-shift]=validRDQindex[rdidx];
                        validRDQindex[rdidx]=-1;
                    }
                    continue;
                }
                k=recvbuf0[1].r;

                lk = LBi( k, grid );

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
                tempv = &recvbuf0[LSUM_H+1];
                il = LSUM_BLK( lk );
                RHS_ITERATE(j) {
                    for (i = 0; i < knsupc; ++i)
					    z_add(&lsum[i + il + j*knsupc + thread_id*sizelsum], &lsum[i + il + j*knsupc + thread_id*sizelsum], &tempv[i + j*knsupc]);
                }

                fmod_tmp=--fmod[lk*aln_i];
                {
                    thread_id = 0;
                    rtemp_loc = &rtemp[sizertemp* thread_id];
                    if ( fmod_tmp==0 ) {
                        if(C_RdTree_IsRoot(&LRtree_ptr[lk])==YES){
                            knsupc = SuperSize( k );
                            for (ii=1;ii<num_thread;ii++)
                            for (jj=0;jj<knsupc*nrhs;jj++)
	    					z_add(&lsum[il + jj ], &lsum[il + jj ],
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
                            }else{
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
						    }//Llu->inv == 1

#if ( PROFlevel>=1 )
							TOC(t2, t1);
							stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif
					        stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc - 1) * nrhs
					            + 10 * knsupc * nrhs; /* complex division */

                            /*
							* Send Xk to process column Pc[k].
							*/
							if(LBtree_ptr[lk].empty_==NO){
								//C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);
                                C_BcTree_forwardMessage_onesided(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H, BCcount, BCbase, &maxrecvsz,Pc);
							}

                            /*
							 * Perform local block modifications.
                            */
							lk = LBj( k, grid ); /* Local block number, column-wise. */
							lsub = Lrowind_bc_ptr[lk];
							lusup = Lnzval_bc_ptr[lk];
							if ( lsub ) {
								krow = PROW( k, grid );
								nb = lsub[0] - 1;
								knsupc = SuperSize( k );
								ii = X_BLK( LBi( k, grid ) );
								xin = &x[ii];
								zlsum_fmod_inv_master_onesided(lsum, x, xin, rtemp, nrhs, knsupc, k,
                                                        fmod, nb, xsup, grid, Llu,
                                                        stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread,
                                                        RDcount, RDbase, BCcount, BCbase, Pc, maxrecvsz);
							} /* if lsub */
                        }else{ // RdTree Yes
                            il = LSUM_BLK( lk );
							knsupc = SuperSize( k );
                            for (ii=1;ii<num_thread;ii++)
							    for (jj=0;jj<knsupc*nrhs;jj++)
					            	z_add(&lsum[il + jj ],
					            		  &lsum[il + jj ],
					            		  &lsum[il + jj + ii*sizelsum]);
                            //C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
                            C_RdTree_forwardMessage_onesided(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H, RDcount, RDbase, &maxrecvsz, Pc);
                        } //if(C_RdTree_IsRoot(&LRtree_ptr[lk])==YES)
                    } // if fmod_tmp==0
                }

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
#if ( DEBUGlevel>=1 )
                printf("iam=%d,RDis_solved[%d]=%d,BufSize_rd[%d]=%d\n",iam,recvRankNum,RDis_solved[recvRankNum],recvRankNum,BufSize_rd[recvRankNum]);
                fflush(stdout);
#endif
            }// for (rdidx=0;rdidx<Pc;rdidx++)
        }
        nfrecv1 = totalsolveBC + totalsolveRD;
    }// outer-most while

#endif /* CPU trisolve*/

#if ( PRNTlevel>=1 )
	t = SuperLU_timer_() - t;
	stat->utime[SOL_TOT] += t;
	if ( !iam ) {
		printf(".. L-solve time\t%8.4f\n", t);
		fflush(stdout);
	}

	MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, grid->comm);
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

	int tmp_size=maxrecvsz * ( (nfrecvx>nbrecvx?nfrecvx:nbrecvx) + 1 );
    for(i=0;i<tmp_size;i++){
        BC_taskq[i].r=(-1.0);
        BC_taskq[i].i=(-1.0);
    }
    tmp_size=((nfrecvmod>nbrecvmod?nfrecvmod:nbrecvmod)+1)*maxrecvsz;
    for(i=0;i<tmp_size;i++){
        RD_taskq[i].r=(-1.0);
        RD_taskq[i].i=(-1.0);
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
	   subsequent call to PZGSTRS. */
	if ( !(bmod = int32Malloc_dist(nlb*aln_i)) )
		ABORT("Malloc fails for bmod[].");
	for (i = 0; i < nlb; ++i) bmod[i*aln_i] = Llu->bmod[i];
	if ( !(brecv = int32Calloc_dist(nlb)) )
		ABORT("Calloc fails for brecv[].");
	Llu->brecv = brecv;

	k = SUPERLU_MAX( Llu->nfsendx, Llu->nbsendx ) + nlb;

	/* Re-initialize lsum to zero. Each block header is already in place. */

#ifdef _OPENMP

#pragma omp parallel default(shared) private(ii)
	{
        int thread_id = omp_get_thread_num();
		for(ii=0;ii<sizelsum;ii++)
			lsum[thread_id*sizelsum+ii]=zero;
	}
  	/* Set up the headers in lsum[]. */
//#pragma omp simd lastprivate(krow,lk,il)
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
        nub = CEILING( nsupers, Pc ); /* Number of local block columns. */
        for (p = 0; p < Pr*Pc; ++p) {
	    if (iam == p) {
		printf("(%2d) .. Ublocks %d, nub %d\n",iam,Ublocks,nub); fflush(stdout);
		for (lb = 0; lb < nub; ++lb) {
		    printf("(%2d) Local col %2d: # row blocks %2d\n",
				iam, lb, Urbs[lb]); fflush(stdout);
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
	    if(UBtree_ptr[lk].empty_==NO){
		// printf("UBtree_ptr lk %5d\n",lk);
		if(C_BcTree_IsRoot(&UBtree_ptr[lk])==NO){
			nbtree++;
			if(UBtree_ptr[lk].destCnt_>0)nbrecvx_buf++;
		}
		//BcTree_allocateRequest(UBtree_ptr[lk],'z');
	    }
	}

	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	rootsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for rootsups.");

	nrtree = 0;
	nroot=0;
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
			// printf("here lk %5d myid %5d\n",lk,iam);
			// fflush(stdout);
			nrtree++;
			//RdTree_allocateRequest(URtree_ptr[lk],'z');
			brecv[lk] = URtree_ptr[lk].destCnt_;
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
            BCbase[i] = recv_size_all_u[i]*maxrecvsz;
        }
    }
    if(Pc > 1){
        for (i=0;i<Pc;i++){
            RDbase[i] = recv_size_all_u[Pr+i]*maxrecvsz;
        }
    }


    for (bcidx=0;bcidx<Pr;bcidx++){
        for(int tmp=0;tmp<bcidx;tmp++){
            BC_taskbuf_offset[bcidx] += BufSize_u[tmp]*maxrecvsz;
        }
    }
    for (rdidx=0;rdidx<Pc;rdidx++){
        for(int tmp=0;tmp<rdidx;tmp++){
            RD_taskbuf_offset[rdidx] += BufSize_urd[tmp]*maxrecvsz;
        }
    }


/*---------------------------------------------------
* End setup U solve onesided
*---------------------------------------------------*/
	nbrecvx_buf=0;


	log_memory(nlb*aln_i*iword+nlb*iword + nsupers_i*iword + maxrecvsz*(nbrecvx+1)*dword*2.0, stat);	//account for bmod, brecv, rootsups, recvbuf_BC_fwd

#if ( DEBUGlevel>=1 )
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


#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK) && defined(GPU_SOLVE)  /* GPU trisolve*/
// #if 0 /* CPU trisolve*/

	d_grid = NULL;
	d_x = NULL;
	d_lsum = NULL;
    int_t  *d_bmod = NULL;

	checkGPU(gpuMalloc( (void**)&d_grid, sizeof(gridinfo_t)));
	checkGPU(gpuMalloc( (void**)&d_lsum, sizelsum*num_thread * sizeof(double)));
	checkGPU(gpuMalloc( (void**)&d_x, (ldalsum * nrhs + nlb * XK_H) * sizeof(double)));
	checkGPU(gpuMalloc( (void**)&d_bmod, (nlb*aln_i) * sizeof(int_t)));


	checkGPU(gpuMemcpy(d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_lsum, lsum, sizelsum*num_thread * sizeof(double), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_x, x, (ldalsum * nrhs + nlb * XK_H) * sizeof(double), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_bmod, bmod, (nlb*aln_i) * sizeof(int_t), gpuMemcpyHostToDevice));

	k = CEILING( nsupers, grid->npcol);/* Number of local block columns divided by #warps per block used as number of thread blocks*/
	knsupc = sp_ienv_dist(3, options);



	dlsum_bmod_inv_gpu_wrap(options, k,nlb,DIM_X,DIM_Y,d_lsum,d_x,nrhs,knsupc,nsupers,d_bmod,Llu->d_UBtree_ptr,Llu->d_URtree_ptr,Llu->d_ilsum,Llu->d_Ucolind_bc_dat,Llu->d_Ucolind_bc_offset,Llu->d_Unzval_bc_dat,Llu->d_Unzval_bc_offset,Llu->d_Uinv_bc_dat,Llu->d_Uinv_bc_offset,Llu->d_Uindval_loc_bc_dat,Llu->d_Uindval_loc_bc_offset,Llu->d_xsup,d_grid);
	checkGPU(gpuMemcpy(x, d_x, (ldalsum * nrhs + nlb * XK_H) * sizeof(double), gpuMemcpyDeviceToHost));

	checkGPU (gpuFree (d_grid));
	checkGPU (gpuFree (d_x));
	checkGPU (gpuFree (d_lsum));
	checkGPU (gpuFree (d_bmod));

	stat_loc[0]->ops[SOLVE]+=Llu->Unzval_br_cnt*nrhs*2; // YL: this is a rough estimate

#else  /* CPU trisolve*/

#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
#ifdef _OPENMP
#pragma omp master
#endif
	    {
#ifdef _OPENMP
#pragma	omp taskloop firstprivate (nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,jj,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Uinv,i,lib,rtemp_loc,nroot_send_tmp,thread_id) nogroup
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];
#if ( PROFlevel>=1 )
			TIC(t1);
#endif
#ifdef _OPENMP
			thread_id = omp_get_thread_num ();
#else
			thread_id = 0;
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

			if(UBtree_ptr[lk].empty_==NO){
#ifdef _OPENMP
#pragma omp atomic capture
#endif
				nroot_send_tmp = ++nroot_send;
				root_send[(nroot_send_tmp-1)*aln_i] = lk;

			}
		} /* for jj ... */
	    } /* omp master region */
	} /* omp parallel region */


#ifdef _OPENMP
#pragma omp parallel default (shared)
#endif
	{
#ifdef _OPENMP
#pragma omp master
#endif
	    {
#ifdef _OPENMP
#pragma	omp taskloop private (ii,jj,k,lk,thread_id) nogroup
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];
			lk = LBi( k, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lk );
			lk = LBj( k, grid ); /* Local block number, column-wise */
#ifdef _OPENMP
                        thread_id=omp_get_thread_num();
#else
                        thread_id=0;
#endif
			/*
			 * Perform local block modifications: lsum[i] -= U_i,k * X[k]
			 */
			if ( Urbs[lk] )
			    zlsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
					Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
					stat_loc, root_send, &nroot_send, sizelsum,sizertemp,thread_id,num_thread);

		} /* for jj ... */

	    } /* omp master region */
	} /* omp parallel */

for (i=0;i<nroot_send;i++){
	lk = root_send[(i)*aln_i];
	if(lk>=0){ // this is a bcast forwarding
		gb = mycol+lk*grid->npcol;  /* not sure */
		lib = LBi( gb, grid ); /* Local block number, row-wise. */
		ii = X_BLK( lib );
		//BcTree_forwardMessageSimple(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'z')*nrhs+XK_H,'z');
		//C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H);
		C_BcTree_forwardMessage_onesided(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H, BCcount, BCbase, &maxrecvsz,Pc);
	}else{ // this is a reduce forwarding
		lk = -lk - 1;
		il = LSUM_BLK( lk );
		//RdTree_forwardMessageSimple(URtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(URtree_ptr[lk],'z')*nrhs+LSUM_H,'z');
		//C_RdTree_forwardMessageSimple(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H);
		C_RdTree_forwardMessage_onesided(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H,RDcount, RDbase, &maxrecvsz, Pc);

	}
}

	/*
	 * Compute the internal nodes asychronously by all processes.
	 */
 recvRankNum=-1;
 shift=0;
 k_tmp=0;
 while(nbrecv1< nbrecvx+nbrecvmod){
     thread_id=0;
     if (totalsolveBC < nbrecvx){
         shift=0;
         for (bcidx=0;bcidx<Pr && validBCQindex_u[bcidx]!=-1;bcidx++){

             recvRankNum=validBCQindex_u[bcidx];  //bcidx; //validBCQindex[bcidx];
             i=BC_taskbuf_offset[recvRankNum]+BCis_solved[recvRankNum]*maxrecvsz; //BCis_solved[bcidx];
             recvbuf0 = &BC_taskq[i];
            k_tmp = (*recvbuf0).r;

#if ( PROFlevel>=1 )
                TIC(t1);
#endif

                int kint= k_tmp;
                if (kint != 1)  {
                    if(shift>0){
                        validBCQindex_u[bcidx-shift]=validBCQindex_u[bcidx];
                        validBCQindex_u[bcidx]=-1;
                        //printf("iam=%d,Now shift %d to %d\n",iam,bcidx,bcidx-shift);
                        //fflush(stdout);
                    }
                    continue;
                }
                k=recvbuf0[1].r;
                lk = LBj( k, grid );    /* local block number */

#if ( PROFlevel>=1 )
                TOC(t2, t1);
                stat_loc[thread_id]->utime[SOL_COMM] += t2;
#endif
                totalsolveBC += 1; //BC_subtotal[bcidx] - BCis_solved[bcidx];
                BCis_solved[recvRankNum]++;

#if ( DEBUGlevel>=2 )
                printf("iam=%d,k=%d, lk=%d,destCnt=%d,here\n",iam,k,lk,UBtree_ptr[lk].destCnt_);
                fflush(stdout);
                //for (int dn=0; dn<UBtree_ptr[lk].msgSize_*nrhs+XK_H+1;dn++){
                //    printf("%d recv'd %d at %d,buffer[%d]=%lf\n",iam, recvRankNum, BCis_solved[recvRankNum]-1, dn,recvbuf0[dn]);
                //    fflush(stdout);
                //}
#endif
                if(UBtree_ptr[lk].destCnt_>0){
					C_BcTree_forwardMessage_onesided(&UBtree_ptr[lk], &recvbuf0[1], UBtree_ptr[lk].msgSize_*nrhs+XK_H, BCcount, BCbase, &maxrecvsz, Pc);
				}
                /*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */

				zlsum_bmod_inv_master_onesided(lsum, x, &recvbuf0[XK_H+1], rtemp, nrhs, k, bmod, Urbs,
						Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
						stat_loc, sizelsum,sizertemp,thread_id,num_thread,
                        RDcount, RDbase, BCcount, BCbase, Pc, maxrecvsz);

                if (BCis_solved[recvRankNum] == BufSize_u[recvRankNum]) {
                    validBCQindex_u[bcidx]=-1;
                    shift += 1;
                }else{
                    if(shift>0){
                        validBCQindex_u[bcidx-shift]=validBCQindex_u[bcidx];
                        validBCQindex_u[bcidx]=-1;
                    }
                }
            } // for (bcidx=0;bcidx<Pr && validBCQindex_u[bcidx]!=-1;bcidx++)
        } // if (totalsolveBC < nbrecvx)

        if (totalsolveRD < nbrecvmod){
            shift=0;
            for (rdidx=0;rdidx<Pc && validRDQindex_u[rdidx]!=-1;rdidx++){
                recvRankNum=validRDQindex_u[rdidx];  //bcidx; //validBCQindex[bcidx];
                ird=RD_taskbuf_offset[recvRankNum]+RDis_solved[recvRankNum]*maxrecvsz;
                recvbuf0 = &RD_taskq[ird];
                k_tmp=(*recvbuf0).r;

#if ( DEBUGlevel>=2 )
    printf("U iam=%d, RD_taskbuf_offset=%lu,RDis_solved=%d,maxrecvsz=%d\n",
        iam,RD_taskbuf_offset[recvRankNum],RDis_solved[recvRankNum],maxrecvsz);
    printf("U iam=%d, recvRankNum=%d,sig=%lf,%lf\n",iam,recvRankNum,k_tmp,recvbuf0[1]);
    fflush(stdout);
#endif

                //printf("rdrd--111--iam=%d, rdidx=%d,k=%d\n",iam,rdidx,k);
                //fflush(stdout);
#if ( PROFlevel>=1 )
                TIC(t1);
#endif
                int kint = k_tmp;
                if (kint != 1)  {
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
                totalsolveRD += 1; //RD_subtotal[rdidx]-RDis_solved[rdidx];
                RDis_solved[recvRankNum] += 1 ;
                //printf("In U-RD solve, iam %d, k=%d, lk=%d,checksum=%u\n", iam, k, lk,crc_16_val);
                //fflush(stdout);
                k=recvbuf0[1].r;
                lk = LBi( k, grid );
                knsupc = SuperSize( k );
                tempv = &recvbuf0[LSUM_H+1];
                il = LSUM_BLK( lk );
                RHS_ITERATE(j) {
                    for (i = 0; i < knsupc; ++i)
			            z_add(&lsum[i + il + j*knsupc + thread_id*sizelsum],
			        	  &lsum[i + il + j*knsupc + thread_id*sizelsum],
			        	  &tempv[i + j*knsupc]);
                }

                bmod_tmp=--bmod[lk*aln_i];
                thread_id = 0;
                rtemp_loc = &rtemp[sizertemp* thread_id];
                if ( bmod_tmp==0 ) {
                    if(C_RdTree_IsRoot(&URtree_ptr[lk])==YES){
                        knsupc = SuperSize( k );
                        for (ii=1;ii<num_thread;ii++)
                            for (jj=0;jj<knsupc*nrhs;jj++)
				            	z_add(&lsum[il+ jj ],
				            	  	  &lsum[il+ jj ],
				            		  &lsum[il + jj + ii*sizelsum]);

                        ii = X_BLK( lk );
                        RHS_ITERATE(j)
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
		        	    stat_loc[thread_id]->ops[SOLVE] += 4 * knsupc * (knsupc + 1) * nrhs
				        	+ 10 * knsupc * nrhs; /* complex division */

#if ( DEBUGlevel>=2 )
                        printf("(%2d) Solve X[%2d]\n", iam, k);
#endif


						/*
						 * Send Xk to process column Pc[k].
                        */

                        if(UBtree_ptr[lk].empty_==NO){
							C_BcTree_forwardMessage_onesided(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H, BCcount, BCbase, &maxrecvsz,Pc);
						}

                        /*
						 * Perform local block modifications:
						 *         lsum[i] -= U_i,k * X[k]
                        */
                        if ( Urbs[lk] )
							zlsum_bmod_inv_master_onesided(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
									Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
									stat_loc, sizelsum,sizertemp,thread_id,num_thread,
						            RDcount, RDbase, BCcount, BCbase, Pc, maxrecvsz);

                    }else{ // if(RdTree_IsRoot(URtree_ptr[lk],'d')==YES)
                        il = LSUM_BLK( lk );
                        knsupc = SuperSize( k );

                        for (ii=1;ii<num_thread;ii++)
                                for (jj=0;jj<knsupc*nrhs;jj++)
   			                		z_add(&lsum[il+ jj ], &lsum[il+ jj ],
   						                &lsum[il + jj + ii*sizelsum]);

                        C_RdTree_forwardMessage_onesided(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H, RDcount,RDbase, &maxrecvsz, Pc);
                    }//if(RdTree_IsRoot(URtree_ptr[lk],'d')==YES)
                }//if ( bmod_tmp==0 )
                if (RDis_solved[recvRankNum] == BufSize_urd[recvRankNum]) {
                    validRDQindex_u[rdidx]=-1;
                    shift += 1;
                }else{
                    if(shift>0){
                        validRDQindex_u[rdidx-shift]=validRDQindex_u[rdidx];
                        validRDQindex_u[rdidx]=-1;
                    }
                }
#if ( DEBUGlevel>=1 )
                printf("U iam=%d,RDis_solved[%d]=%d,BufSize_rd[%d]=%d\n",iam,recvRankNum,RDis_solved[recvRankNum],recvRankNum,BufSize_rd[recvRankNum]);
                fflush(stdout);
#endif
            }//for (rdidx=0;rdidx<Pc;rdidx++)
        }
        nbrecv1 = totalsolveBC + totalsolveRD;
    }
#endif /* CPU trisolve */

#if ( PRNTlevel>=2 )
	t = SuperLU_timer_() - t;
	stat->utime[SOL_TOT] += t;
	if ( !iam ) printf(".. U-solve time\t%8.4f\n", t);
	MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, grid->comm);
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
			   printf("\t(%d)\t%4d\t%.10f\n", iam, xsup[k]+i, x_col[i]);
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
			printf ("\tPZGSTRS comm stat:"
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

    MPI_Reduce( &temp, &max, 1, MPI_FLOAT, MPI_MAX, 0, grid->comm );
    MPI_Reduce( &temp, &avg, 1, MPI_FLOAT, MPI_SUM, 0, grid->comm );
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
