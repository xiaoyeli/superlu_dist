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
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 15, 2008
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
 * </pre>
 */
#include <math.h>
#include "superlu_sdefs.h"
#define ISEND_IRECV

// Broadcast the RHS to all grids from grid 0
int_t strs_B_init3d(int_t nsupers, float* x, int nrhs, sLUstruct_t * LUstruct,
	gridinfo3d_t *grid3d)
{

	gridinfo_t * grid = &(grid3d->grid2d);
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	sLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );

	for (int_t k = 0; k < nsupers; ++k)
	{
		/* code */
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow && mycol == kcol)
		{
			int_t lk = LBi(k, grid);
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize(k);
			MPI_Bcast( &x[ii - XK_H], knsupc * nrhs + XK_H, MPI_FLOAT, 0, grid3d->zscp.comm);

		}
	}

	return 0;
}

// Broadcast the RHS to all grids from grid 0. Once received, every grid zeros out certain subvectors to allow for the new 3D solve.
int_t strs_B_init3d_newsolve(int_t nsupers, float* x, int nrhs, sLUstruct_t * LUstruct,
	gridinfo3d_t *grid3d, strf3Dpartition_t*  trf3Dpartition)
{

	gridinfo_t * grid = &(grid3d->grid2d);
    int_t myGrid = grid3d->zscp.Iam;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	sLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );
    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    float zero = 0.0;
    float* xtmp;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t Pr = grid->nprow;
    int_t nlb = CEILING (nsupers, Pr);    /* Number of local block rows. */

    if (!(xtmp = floatCalloc_dist (Llu->ldalsum * nrhs + nlb * XK_H)))
    ABORT ("Malloc fails for xtmp[].");

	for (int_t k = 0; k < nsupers; ++k)
	{
		/* code */
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow && mycol == kcol)
		{
			int_t lk = LBi(k, grid);
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize(k);
            MPI_Bcast( &x[ii - XK_H], knsupc * nrhs + XK_H, MPI_FLOAT, 0, grid3d->zscp.comm);
            for (int_t i=0; i<XK_H; ++i){
                xtmp[ii-XK_H+i] = x[ii - XK_H+i];
            }
            for (int_t i=0; i<knsupc * nrhs; ++i){
                xtmp[ii+i] = x[ii+i];
                x[ii+i] = zero;
            }
		}
	}


    // fill corresponding RHSs
    for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        // printf("gana grid3d->zscp.iam %5d ilvl %5d myZeroTrIdxs[ilvl] %5d myTreeIdxs[ilvl] %5d\n",grid3d->zscp.Iam, ilvl, myZeroTrIdxs[ilvl],myTreeIdxs[ilvl]);
        if (!myZeroTrIdxs[ilvl])
        {
            int_t tree = myTreeIdxs[ilvl];
            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];
            /*main loop over all the super nodes*/
            if (sforest)
            {
                int_t nnodes = sforest->nNodes ;
	            int_t *nodeList = sforest->nodeList ;
                for (int_t k0 = 0; k0 < nnodes; ++k0)
	            {
		            int_t k = nodeList[k0];
                    int_t krow = PROW (k, grid);
                    int_t kcol = PCOL (k, grid);

                    if (myrow == krow && mycol == kcol)
                    {
                        int_t lk = LBi(k, grid);
                        int_t ii = X_BLK (lk);
                        int_t knsupc = SuperSize(k);
                        for(int_t i=0; i<knsupc * nrhs; ++i)
                            x[ii +i]= xtmp[ii+i];
                    }
                }
            }
        }
    }
    SUPERLU_FREE (xtmp);
	return 0;
}

// #ifdef HAVE_NVSHMEM
/*global variables for nvshmem, is it safe to be put them here? */
float *sready_x, *sready_lsum;
// #endif

int strs_compute_communication_structure(superlu_dist_options_t *options, int_t n, sLUstruct_t * LUstruct,
                           sScalePermstruct_t * ScalePermstruct,
                           int* supernodeMask, gridinfo_t *grid, SuperLUStat_t * stat)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int kr,kc,nlb,nub;
    int nsupers = Glu_persist->supno[n - 1] + 1;
    int_t *rowcounts, *colcounts, **rowlists, **collists, *tmpglo;
    int_t  *lsub, *lloc;
    int_t idx_i, lptr1_tmp, ib, jb, jj;
    int   *displs, *recvcounts, count, nbg;

    kr = CEILING( nsupers, grid->nprow);/* Number of local block rows */
    kc = CEILING( nsupers, grid->npcol);/* Number of local block columns */
    int_t iam=grid->iam;
    int nprocs = grid->nprow * grid->npcol;
    int_t myrow = MYROW( iam, grid );
    int_t mycol = MYCOL( iam, grid );
    int_t *ActiveFlag;
    int *ranks;
    superlu_scope_t *rscp = &grid->rscp;
    superlu_scope_t *cscp = &grid->cscp;
    int rank_cnt,rank_cnt_ref,Root;
    int_t Iactive,gb,pr,pc,nb, idx_n;

	C_Tree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
	C_Tree  *LRtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
	C_Tree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
	C_Tree  *URtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
	int msgsize;

    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t *usub;
    float *lnzval;


    int_t len, len1, len2, len3, nrbl;


	float **Lnzval_bc_ptr=Llu->Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc) */
	float *Lnzval_bc_dat;  /* size sum of sizes of Lnzval_bc_ptr[lk])                 */
    long int *Lnzval_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

	int_t  **Lrowind_bc_ptr=Llu->Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc) */
	int_t *Lrowind_bc_dat;  /* size sum of sizes of Lrowind_bc_ptr[lk])                 */
    long int *Lrowind_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

	int_t  **Lindval_loc_bc_ptr=Llu->Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
	int_t *Lindval_loc_bc_dat;  /* size sum of sizes of Lindval_loc_bc_ptr[lk])                 */
    long int *Lindval_loc_bc_offset;  /* size ceil(NSUPERS/Pc)                 */

    float **Linv_bc_ptr=Llu->Linv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
	float *Linv_bc_dat;  /* size sum of sizes of Linv_bc_ptr[lk])                 */
    long int *Linv_bc_offset;  /* size ceil(NSUPERS/Pc)                 */
    float **Uinv_bc_ptr=Llu->Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
	float *Uinv_bc_dat;  /* size sum of sizes of Uinv_bc_ptr[lk])                 */
    long int *Uinv_bc_offset;  /* size ceil(NSUPERS/Pc) */


	float **Unzval_br_ptr=Llu->Unzval_br_ptr;  /* size ceil(NSUPERS/Pr) */
	float *Unzval_br_dat;  /* size sum of sizes of Unzval_br_ptr[lk])                 */
	long int *Unzval_br_offset;  /* size ceil(NSUPERS/Pr)    */
	int_t  **Ufstnz_br_ptr=Llu->Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr) */
    int_t   *Ufstnz_br_dat;  /* size sum of sizes of Ufstnz_br_ptr[lk])                 */
    long int *Ufstnz_br_offset;  /* size ceil(NSUPERS/Pr)    */

    Ucb_indptr_t *Ucb_inddat;
    long int *Ucb_indoffset;
    int_t  **Ucb_valptr = Llu->Ucb_valptr;
    int_t  *Ucb_valdat;
    long int *Ucb_valoffset;
    int *h_recv_cnt;
    int *h_recv_cnt_u;

    /* Reconstruct the global L structure and compute the communication metadata */

    if ( !(tmpglo = intCalloc_dist(nsupers)) )
		ABORT("Calloc fails for tmpglo[].");
    if (!(recvcounts = (int *) SUPERLU_MALLOC (SUPERLU_MAX (grid->npcol, grid->nprow) * sizeof(int))))
        ABORT ("SUPERLU_MALLOC fails for recvcounts.");
    if (!(displs = (int *) SUPERLU_MALLOC (SUPERLU_MAX (grid->npcol, grid->nprow) * sizeof(int))))
        ABORT ("SUPERLU_MALLOC fails for displs.");


    /* gather information about the global L structure */

	if ( !(rowcounts = intCalloc_dist(kc)) )
		ABORT("Calloc fails for rowcounts[].");
	if ( !(colcounts = intCalloc_dist(kr)) )
		ABORT("Calloc fails for colcounts[].");

	if ( !(rowlists = (int_t**)SUPERLU_MALLOC(kc * sizeof(int_t*))) )
		fprintf(stderr, "Malloc fails for rowlists[].");
	if ( !(collists = (int_t**)SUPERLU_MALLOC(kr * sizeof(int_t*))) )
		fprintf(stderr, "Malloc fails for collists[].");

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                lsub = Llu->Lrowind_bc_ptr[lk];
                lloc = Llu->Lindval_loc_bc_ptr[lk];
                if(lsub){
                    nlb = lsub[0];
                    idx_i = nlb;
                    for (int_t lb = 0; lb < nlb; ++lb){
                        lptr1_tmp = lloc[lb+idx_i];
                        ib = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                        if(supernodeMask[ib]>0){
                            rowcounts[lk]++;
                            int_t lib = LBi( ib, grid ); /* Local block number, row-wise. */
                            colcounts[lib]++;
                        }
                    }
                }
            }
		}
	}

    for (int_t j=0; j<kc; j++){
        if(rowcounts[j]>0){
            if ( !(rowlists[j] = intCalloc_dist(rowcounts[j])) )
                ABORT("Calloc fails for rowlists[j].");
        }else{
            rowlists[j] = NULL;
        }
        rowcounts[j]=0;
    }
    for (int_t i=0; i<kr; i++){
        if(colcounts[i]>0){
            if ( !(collists[i] = intCalloc_dist(colcounts[i])) )
                ABORT("Calloc fails for collists[i].");
        }else{
            collists[i] = NULL;
        }
        colcounts[i]=0;
    }

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                lsub = Llu->Lrowind_bc_ptr[lk];
                lloc = Llu->Lindval_loc_bc_ptr[lk];
                if(lsub){
                    nlb = lsub[0];
                    idx_i = nlb;
                    for (int_t lb = 0; lb < nlb; ++lb){
                        lptr1_tmp = lloc[lb+idx_i];
                        ib = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                        if(supernodeMask[ib]>0){
                            rowlists[lk][rowcounts[lk]++]=ib;
                            int_t lib = LBi( ib, grid ); /* Local block number, row-wise. */
                            collists[lib][colcounts[lib]++]=jb;
                        }
                    }
                }
            }
		}
	}

    /* broadcast tree for L*/

	if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
		ABORT("Calloc fails for ActiveFlag[].");
	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
		ABORT("Malloc fails for ranks[].");
	if ( !(LBtree_ptr = (C_Tree*)SUPERLU_MALLOC(kc * sizeof(C_Tree))) )
		ABORT("Malloc fails for LBtree_ptr[].");
	for (int_t lk = 0; lk <kc ; ++lk) {
		C_BcTree_Nullify(&LBtree_ptr[lk]);
	}

#ifdef GPU_ACC
#ifdef HAVE_NVSHMEM
    if ( !(mystatus = (int*)SUPERLU_MALLOC(kc * sizeof(int))) )
        ABORT("Malloc fails for mystatus[].");
    if ( !(h_nfrecv = (int*)SUPERLU_MALLOC( 3* sizeof(int))) )
        ABORT("Malloc fails for h_nfrecv[].");
    if ( !(h_nfrecvmod = (int*)SUPERLU_MALLOC( 4 * sizeof(int))) )
        ABORT("Malloc fails for h_nfrecvmod[].");
    h_nfrecvmod[3]=0;
    //printf("(%d)k=%d\n",iam,k);
	for (int i=0;i<kc;i++){
        mystatus[i]=1;
	}
#endif
#endif

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                // printf("iam %5d jb %5d \n",iam, jb);
                // fflush(stdout);
                pc = PCOL( jb, grid );
                count = rowcounts[lk];
                MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, cscp->comm);
                displs[0] = 0;
                nbg=0;
                for(int i=0; i<grid->nprow; ++i)
                {
                    nbg +=recvcounts[i];
                }
                if(nbg>0){
                    for(int i=0; i<grid->nprow-1; ++i)
                    {
                        displs[i+1] = displs[i] + recvcounts[i];
                    }
                    MPI_Allgatherv(rowlists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, cscp->comm);
                }
                for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j]=3*nsupers;
                for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
                for (int_t j=0;j<grid->nprow;++j)ranks[j]=-1;

                for (int_t i = 0; i < nbg; ++i) {
                    gb = tmpglo[i];
                    pr = PROW( gb, grid );
                    ActiveFlag[pr]=SUPERLU_MIN(ActiveFlag[pr],gb);
                } /* for i ... */

                Root=-1;
                Iactive = 0;
                for (int_t j=0;j<grid->nprow;++j){
                    if(ActiveFlag[j]!=3*nsupers){
                    gb = ActiveFlag[j];
                    pr = PROW( gb, grid );
                    if(gb==jb)Root=pr;
                    if(myrow==pr)Iactive=1;
                    }
                }

                quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,0,2);

                if(Iactive==1){
                    // printf("iam %5d jb %5d Root %5d \n",iam, jb,Root);
                    // fflush(stdout);
                    assert( Root>-1 );
                    rank_cnt = 1;
                    ranks[0]=Root;
                    for (int_t j = 0; j < grid->nprow; ++j){
                        if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
                            ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
                            ++rank_cnt;
                        }
                    }

                    if(rank_cnt>1){

                        for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                            ranks[ii] = PNUM( ranks[ii], pc, grid );

                        msgsize = SuperSize( jb );
                        int needrecv=0;
                        C_BcTree_Create_nv(&LBtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 's',&needrecv);
                        //C_BcTree_Create(&LBtree_ptr[ljb], grid->comm, ranks, rank_cnt, msgsize, 's');
                        //printf("(%d) HOST create:ljb=%d,msg=%d,needrecv=%d\n",iam,ljb,mysendmsg_num,needrecv);
                        #ifdef GPU_ACC
                        #ifdef HAVE_NVSHMEM
                        if (needrecv==1) {
                            mystatus[lk]=0;
                            //printf("(%d) Col %d need one msg %d\n",iam, ljb,mystatus[ljb]);
                            //fflush(stdout);
                        }
                        #endif
                        #endif
                        LBtree_ptr[lk].tag_=BC_L;

                        // printf("iam %5d btree rank_cnt %5d \n",iam,rank_cnt);
                        // fflush(stdout);

                    }
                }
            }
        }
    }
	SUPERLU_FREE(ActiveFlag);
	SUPERLU_FREE(ranks);


    /* reduction tree for L*/
	if ( !(LRtree_ptr = (C_Tree*)SUPERLU_MALLOC(kr * sizeof(C_Tree))) )
		ABORT("Malloc fails for LRtree_ptr[].");
	if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
		ABORT("Calloc fails for ActiveFlag[].");
	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
		ABORT("Malloc fails for ranks[].");
	for (int_t lk = 0; lk <kr ; ++lk) {
		C_RdTree_Nullify(&LRtree_ptr[lk]);
	}
#ifdef GPU_ACC
#ifdef HAVE_NVSHMEM
    if ( !(mystatusmod = (int*)SUPERLU_MALLOC(2*kr * sizeof(int))) )
        ABORT("Malloc fails for mystatusmod[].");
	if ( !(h_recv_cnt = (int*)SUPERLU_MALLOC(kr * sizeof(int))) )
        ABORT("Malloc fails for mystatusmod[].");

	int nfrecvmod=0;
	for (int i=0;i<kr;i++){
        h_recv_cnt[i]=0;
	}
	for (int i=0;i<2*kr;i++) mystatusmod[i]=1;
#endif
#endif

	for (int_t lk=0;lk<kr;++lk){
		ib = myrow+lk*grid->nprow;  /* not sure */
		if(ib<nsupers){
            if(supernodeMask[ib]>0){
                pr = PROW( ib, grid );

                count = colcounts[lk];
                MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, rscp->comm);
                displs[0] = 0;
                nbg=0;
                for(int i=0; i<grid->npcol; ++i)
                {
                    nbg +=recvcounts[i];
                }
                if(nbg>0){
                    for(int i=0; i<grid->npcol-1; ++i)
                    {
                        displs[i+1] = displs[i] + recvcounts[i];
                    }
                    MPI_Allgatherv(collists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, rscp->comm);
                }
                for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j]=-3*nsupers;
                for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
                for (int_t j=0;j<grid->npcol;++j)ranks[j]=-1;

                for (int_t j = 0; j < nbg; ++j) {
                    gb = tmpglo[j];
                    pc = PCOL( gb, grid );
                    ActiveFlag[pc]=SUPERLU_MAX(ActiveFlag[pc],gb);
                } /* for j ... */

                Root=-1;
                Iactive = 0;

                for (int_t j=0;j<grid->npcol;++j){
                    if(ActiveFlag[j]!=-3*nsupers){
                    jb = ActiveFlag[j];
                    pc = PCOL( jb, grid );
                    if(jb==ib)Root=pc;
                    if(mycol==pc)Iactive=1;
                    }
                }

                quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,1,2);

                if(Iactive==1){
                    assert( Root>-1 );
                    rank_cnt = 1;
                    ranks[0]=Root;
                    for (int_t j = 0; j < grid->npcol; ++j){
                        if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
                            ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
                            ++rank_cnt;
                        }
                    }
                    if(rank_cnt>1){
                        for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                            ranks[ii] = PNUM( pr, ranks[ii], grid );
                        msgsize = SuperSize( ib );
                        // C_RdTree_Create(&LRtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 's');

                        int needrecvrd=0;
                        int needsendrd=0;
                        C_RdTree_Create_nv(&LRtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 's', &needrecvrd, &needsendrd);
                        //C_RdTree_Create(&LRtree_ptr[lib], grid->comm, ranks, rank_cnt, msgsize, 's');
                        #ifdef GPU_ACC
                        #ifdef HAVE_NVSHMEM
                        h_nfrecvmod[3]+=needsendrd;
                        if (needrecvrd!=0) {
                            mystatusmod[lk*2]=0;
                            mystatusmod[lk*2+1]=0;
                            h_recv_cnt[lk]=needrecvrd;
                            //printf("(%d) on CPU, lib=%d, cnt=%d\n",iam,lib,LRtree_ptr[lib].destCnt_);
                            nfrecvmod+=needrecvrd;
                        }
                        #endif
                        #endif
                        LRtree_ptr[lk].tag_=RD_L;
                    }
                }
            }
        }
    }
	SUPERLU_FREE(ActiveFlag);
	SUPERLU_FREE(ranks);




    for (int_t j=0; j<kc; j++){
        if(rowlists[j]){
            SUPERLU_FREE(rowlists[j]);
        }
    }
    for (int_t i=0; i<kr; i++){
        if(collists[i]){
            SUPERLU_FREE(collists[i]);
        }
    }
    SUPERLU_FREE(rowcounts);
    SUPERLU_FREE(colcounts);
    SUPERLU_FREE(rowlists);
    SUPERLU_FREE(collists);


    /* gather information about the global U structure */

	if ( !(rowcounts = intCalloc_dist(kc)) )
		ABORT("Calloc fails for rowcounts[].");
	if ( !(colcounts = intCalloc_dist(kr)) )
		ABORT("Calloc fails for colcounts[].");

	if ( !(rowlists = (int_t**)SUPERLU_MALLOC(kc * sizeof(int_t*))) )
		fprintf(stderr, "Malloc fails for rowlists[].");
	if ( !(collists = (int_t**)SUPERLU_MALLOC(kr * sizeof(int_t*))) )
		fprintf(stderr, "Malloc fails for collists[].");

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                nub = Urbs[lk];      /* Number of U blocks in block column lk */
                for (int_t ub = 0; ub < nub; ++ub){
                    int_t lib = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
                    ib = lib * grid->nprow + myrow;/* Global block number, row-wise. */
                    if(supernodeMask[ib]>0){
                        rowcounts[lk]++;
                        colcounts[lib]++;
                    }
                }
            }
		}
	}

    for (int_t j=0; j<kc; j++){
        if(rowcounts[j]>0){
            if ( !(rowlists[j] = intCalloc_dist(rowcounts[j])) )
                ABORT("Calloc fails for rowlists[j].");
        }else{
            rowlists[j] = NULL;
        }
        rowcounts[j]=0;
    }
    for (int_t i=0; i<kr; i++){
        if(colcounts[i]>0){
            if ( !(collists[i] = intCalloc_dist(colcounts[i])) )
                ABORT("Calloc fails for collists[i].");
        }else{
            collists[i] = NULL;
        }
        colcounts[i]=0;
    }

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                nub = Urbs[lk];      /* Number of U blocks in block column lk */
                for (int_t ub = 0; ub < nub; ++ub){
                    int_t lib = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
                    ib = lib * grid->nprow + myrow;/* Global block number, row-wise. */
                    if(supernodeMask[ib]>0){
                        rowlists[lk][rowcounts[lk]++]=ib;
                        collists[lib][colcounts[lib]++]=jb;
                    }
                }
            }
		}
	}



    /* broadcast tree for U*/
	if ( !(UBtree_ptr = (C_Tree*)SUPERLU_MALLOC(kc * sizeof(C_Tree))) )
		ABORT("Malloc fails for UBtree_ptr[].");
	if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
		ABORT("Calloc fails for ActiveFlag[].");
	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
		ABORT("Malloc fails for ranks[].");
	for (int_t lk = 0; lk <kc ; ++lk) {
		C_BcTree_Nullify(&UBtree_ptr[lk]);
	}
#ifdef GPU_ACC
#ifdef HAVE_NVSHMEM
	if ( !(mystatus_u = (int*)SUPERLU_MALLOC(kc * sizeof(int))) )
        ABORT("Malloc fails for mystatus_u[].");
    if ( !(h_nfrecv_u = (int*)SUPERLU_MALLOC( 3* sizeof(int))) )
        ABORT("Malloc fails for h_nfrecv_u[].");
    if ( !(h_nfrecvmod_u = (int*)SUPERLU_MALLOC( 4* sizeof(int))) )
        ABORT("Malloc fails for h_nfrecvmod_u[].");
    h_nfrecvmod_u[3]=0;

	for (int i=0;i<kc;i++){
		mystatus_u[i]=1;
	}
#endif
#endif


    /* update bsendx_plist with the supernode mask. Note that fsendx_plist doesn't require updates */
    for (int_t lk=0;lk<kc;++lk){
        jb = mycol+lk*grid->npcol;  /* not sure */
        if(jb<nsupers){
        int_t krow = PROW(jb, grid);
        int_t kcol = PCOL(jb, grid);
        if (myrow == krow && mycol == kcol){
        for (int_t pr=0;pr<grid->nprow;++pr){
            Llu->bsendx_plist[lk][pr]=  SLU_EMPTY;
        }
        }
        }
    }

	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
		if(jb<nsupers){
            if(supernodeMask[jb]>0){
                pc = PCOL( jb, grid );
                count = rowcounts[lk];
                MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, cscp->comm);
                displs[0] = 0;
                nbg=0;
                for(int i=0; i<grid->nprow; ++i)
                {
                    nbg +=recvcounts[i];
                }
                if(nbg>0){
                    for(int i=0; i<grid->nprow-1; ++i)
                    {
                        displs[i+1] = displs[i] + recvcounts[i];
                    }
                    MPI_Allgatherv(rowlists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, cscp->comm);
                }

                for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j]=-3*nsupers;
                for (int_t j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
                for (int_t j=0;j<grid->nprow;++j)ranks[j]=-1;

                for (int_t i = 0; i < nbg; ++i) {
                    gb = tmpglo[i];
                    pr = PROW( gb, grid );
                    ActiveFlag[pr]=SUPERLU_MAX(ActiveFlag[pr],gb);
                } /* for i ... */

                pr = PROW( jb, grid ); // take care of diagonal node stored as L
                ActiveFlag[pr]=SUPERLU_MAX(ActiveFlag[pr],jb);

                Root=-1;
                Iactive = 0;
                for (int_t j=0;j<grid->nprow;++j){
                    if(ActiveFlag[j]!=-3*nsupers){
                    gb = ActiveFlag[j];
                    pr = PROW( gb, grid );
                    if(gb==jb)Root=pr;
                    if(myrow==pr)Iactive=1;
                    if(myrow!=pr && myrow == PROW(jb, grid)) /* update bsendx_plist with the supernode mask */
                        Llu->bsendx_plist[lk][pr]=YES;
                    }
                }

                quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,1,2);

                if(Iactive==1){
                    assert( Root>-1 );
                    rank_cnt = 1;
                    ranks[0]=Root;
                    for (int_t j = 0; j < grid->nprow; ++j){
                        if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
                            ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
                            ++rank_cnt;
                        }
                    }
                    if(rank_cnt>1){
                        for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                            ranks[ii] = PNUM( ranks[ii], pc, grid );
                        msgsize = SuperSize( jb );
                        // C_BcTree_Create(&UBtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 's');

                        int needrecv=0;
                        C_BcTree_Create_nv(&UBtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 's',&needrecv);
                        //C_BcTree_Create(&UBtree_ptr[ljb], grid->comm, ranks, rank_cnt, msgsize, 's');
                        #ifdef GPU_ACC
                        #ifdef HAVE_NVSHMEM
                        if (needrecv==1) {
                            mystatus_u[lk]=0;
                            //printf("(%d) Col %d need one msg %d\n",iam, ljb,mystatus[ljb]);
                        }
                        #endif
                        #endif
                        UBtree_ptr[lk].tag_=BC_U;
                    }
                }
            }
        }
    }
	SUPERLU_FREE(ActiveFlag);
	SUPERLU_FREE(ranks);


    /* reduction tree for U*/
	if ( !(URtree_ptr = (C_Tree*)SUPERLU_MALLOC(kr * sizeof(C_Tree))) )
		ABORT("Malloc fails for URtree_ptr[].");
	if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
		ABORT("Calloc fails for ActiveFlag[].");
	if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
		ABORT("Malloc fails for ranks[].");
	for (int_t lk = 0; lk <kr ; ++lk) {
		C_RdTree_Nullify(&URtree_ptr[lk]);
	}

    #ifdef GPU_ACC
    #ifdef HAVE_NVSHMEM
	if ( !(mystatusmod_u = (int*)SUPERLU_MALLOC(2*kr * sizeof(int))) )
        ABORT("Malloc fails for mystatusmod_u[].");
    if ( !(h_recv_cnt_u = (int*)SUPERLU_MALLOC(kr * sizeof(int))) )
        ABORT("Malloc fails for h_recv_cnt_u[].");

    int nbrecvmod=0;
	for (int i=0;i<kr;i++){
        h_recv_cnt_u[i]=0;
	}
	for (int i=0;i<2*kr;i++) mystatusmod_u[i]=1;
    #endif
    #endif

	for (int_t lk=0;lk<kr;++lk){
		ib = myrow+lk*grid->nprow;  /* not sure */
		if(ib<nsupers){
            if(supernodeMask[ib]>0){
                pr = PROW( ib, grid );

                count = colcounts[lk];
                MPI_Allgather(&count, 1, MPI_INT, recvcounts, 1, MPI_INT, rscp->comm);
                displs[0] = 0;
                nbg=0;
                for(int i=0; i<grid->npcol; ++i)
                {
                    nbg +=recvcounts[i];
                }
                if(nbg>0){
                    for(int i=0; i<grid->npcol-1; ++i)
                    {
                        displs[i+1] = displs[i] + recvcounts[i];
                    }
                    MPI_Allgatherv(collists[lk], count, mpi_int_t, tmpglo, recvcounts, displs, mpi_int_t, rscp->comm);
                }
                for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j]=3*nsupers;
                for (int_t j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
                for (int_t j=0;j<grid->npcol;++j)ranks[j]=-1;

                for (int_t j = 0; j < nbg; ++j) {
                    gb = tmpglo[j];
                    pc = PCOL( gb, grid );
                    ActiveFlag[pc]=SUPERLU_MIN(ActiveFlag[pc],gb);
                } /* for j ... */
                pc = PCOL( ib, grid ); // take care of diagonal node stored as L
                ActiveFlag[pc]=SUPERLU_MIN(ActiveFlag[pc],ib);

                Root=-1;
                Iactive = 0;

                for (int_t j=0;j<grid->npcol;++j){
                    if(ActiveFlag[j]!=3*nsupers){
                    jb = ActiveFlag[j];
                    pc = PCOL( jb, grid );
                    if(jb==ib)Root=pc;
                    if(mycol==pc)Iactive=1;
                    }
                }

                quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,0,2);

                if(Iactive==1){
                    assert( Root>-1 );
                    rank_cnt = 1;
                    ranks[0]=Root;
                    for (int_t j = 0; j < grid->npcol; ++j){
                        if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
                            ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
                            ++rank_cnt;
                        }
                    }
                    if(rank_cnt>1){
                        for (int_t ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                            ranks[ii] = PNUM( pr, ranks[ii], grid );
                        msgsize = SuperSize( ib );
                        // C_RdTree_Create(&URtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 's');

                        int needrecvrd=0;
                        int needsendrd=0;
                        C_RdTree_Create_nv(&URtree_ptr[lk], grid->comm, ranks, rank_cnt, msgsize, 's', &needrecvrd,&needsendrd);
                        //C_RdTree_Create(&URtree_ptr[lib], grid->comm, ranks, rank_cnt, msgsize, 's');
                        #ifdef GPU_ACC
                        #ifdef HAVE_NVSHMEM
                        h_nfrecvmod_u[3] +=needsendrd;
                        if (needrecvrd!=0) {
                            mystatusmod_u[lk*2]=0;
                            mystatusmod_u[lk*2+1]=0;
                            h_recv_cnt_u[lk]=needrecvrd;
                            //printf("(%d) on CPU, lib=%d, cnt=%d\n",iam,lib,LRtree_ptr[lib].destCnt_);
                            nbrecvmod+=needrecvrd;
                        }
                        #endif
                        #endif
                        URtree_ptr[lk].tag_=RD_U;
                    }
                }
            }
        }
    }
	SUPERLU_FREE(ActiveFlag);
	SUPERLU_FREE(ranks);

    for (int_t j=0; j<kc; j++){
        if(rowlists[j]){
            SUPERLU_FREE(rowlists[j]);
        }
    }
    for (int_t i=0; i<kr; i++){
        if(collists[i]){
            SUPERLU_FREE(collists[i]);
        }
    }
    SUPERLU_FREE(rowcounts);
    SUPERLU_FREE(colcounts);
    SUPERLU_FREE(rowlists);
    SUPERLU_FREE(collists);

    SUPERLU_FREE(tmpglo);
    SUPERLU_FREE(recvcounts);
    SUPERLU_FREE(displs);



    // ////////////////////////////////////////////////////
    // // use contignous memory for the L meta data
    // int_t k = kc;/* Number of local block columns */
    // long int Lnzval_bc_cnt=0;
    // long int Lrowind_bc_cnt=0;
    // long int Lindval_loc_bc_cnt=0;
	// long int Linv_bc_cnt=0;
	// long int Uinv_bc_cnt=0;

	// if ( !(Lnzval_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Lnzval_bc_offset[].");
	// }
	// Lnzval_bc_offset[k-1] = -1;

	// if ( !(Lrowind_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Lrowind_bc_offset[].");
	// }
	// Lrowind_bc_offset[k-1] = -1;
	// if ( !(Lindval_loc_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Lindval_loc_bc_offset[].");
	// }
	// Lindval_loc_bc_offset[k-1] = -1;
	// if ( !(Linv_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Linv_bc_offset[].");
	// }
	// Linv_bc_offset[k-1] = -1;
	// if ( !(Uinv_bc_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Uinv_bc_offset[].");
	// }
	// Uinv_bc_offset[k-1] = -1;


    // for (int_t lk=0;lk<k;++lk){
    //     jb = mycol+lk*grid->npcol;  /* not sure */
	//     lsub = Lrowind_bc_ptr[lk];
	//     lloc = Lindval_loc_bc_ptr[lk];
	//     lnzval = Lnzval_bc_ptr[lk];

    //     Linv_bc_offset[lk] = -1;
    //     Uinv_bc_offset[lk] = -1;
    //     Lrowind_bc_offset[lk]=-1;
    //     Lindval_loc_bc_offset[lk]=-1;
    //     Lnzval_bc_offset[lk]=-1;

    //     if(lsub){
    //         nrbl  =   lsub[0]; /*number of L blocks */
    //         len   = lsub[1];   /* LDA of the nzval[] */
    //         len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
    //         int_t nsupc = SuperSize(jb);
    //         len2  = nsupc * len;
    //         len3 = nrbl*3;
    //         Lnzval_bc_offset[lk]=len2;
    //         Lnzval_bc_cnt += Lnzval_bc_offset[lk];

    //         Lrowind_bc_offset[lk]=len1;
    //         Lrowind_bc_cnt += Lrowind_bc_offset[lk];

	// 		Lindval_loc_bc_offset[lk]=nrbl*3;
	// 		Lindval_loc_bc_cnt += Lindval_loc_bc_offset[lk];

    //         int_t krow = PROW( jb, grid );
	// 		if(myrow==krow){   /* diagonal block */
	// 			Linv_bc_offset[lk]=nsupc*nsupc;
	// 			Linv_bc_cnt += Linv_bc_offset[lk];
	// 			Uinv_bc_offset[lk]=nsupc*nsupc;
	// 			Uinv_bc_cnt += Uinv_bc_offset[lk];
	// 		}else{
	// 			Linv_bc_offset[lk] = -1;
	// 			Uinv_bc_offset[lk] = -1;
	// 		}

    //     }
    // }

	// Linv_bc_cnt +=1; // safe guard
	// Uinv_bc_cnt +=1;
	// Lrowind_bc_cnt +=1;
	// Lindval_loc_bc_cnt +=1;
	// Lnzval_bc_cnt +=1;
	// if ( !(Linv_bc_dat =
	// 			(float*)SUPERLU_MALLOC(Linv_bc_cnt * sizeof(float))) ) {
	// 	fprintf(stderr, "Malloc fails for Linv_bc_dat[].");
	// }
	// if ( !(Uinv_bc_dat =
	// 			(float*)SUPERLU_MALLOC(Uinv_bc_cnt * sizeof(float))) ) {
	// 	fprintf(stderr, "Malloc fails for Uinv_bc_dat[].");
	// }

	// if ( !(Lrowind_bc_dat =
	// 			(int_t*)SUPERLU_MALLOC(Lrowind_bc_cnt * sizeof(int_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Lrowind_bc_dat[].");
	// }
	// if ( !(Lindval_loc_bc_dat =
	// 			(int_t*)SUPERLU_MALLOC(Lindval_loc_bc_cnt * sizeof(int_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Lindval_loc_bc_dat[].");
	// }
	// if ( !(Lnzval_bc_dat =
	// 			(float*)SUPERLU_MALLOC(Lnzval_bc_cnt * sizeof(float))) ) {
	// 	fprintf(stderr, "Malloc fails for Lnzval_bc_dat[].");
	// }


	// /* use contingous memory for Linv_bc_ptr, Uinv_bc_ptr, Lrowind_bc_ptr, Lnzval_bc_ptr*/
	// Linv_bc_cnt=0;
	// Uinv_bc_cnt=0;
	// Lrowind_bc_cnt=0;
	// Lnzval_bc_cnt=0;
	// Lindval_loc_bc_cnt=0;
	// long int tmp_cnt;
	// for (jb = 0; jb < k; ++jb) { /* for each block column ... */
	// 	if(Linv_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Linv_bc_offset[jb]; ++jj) {
	// 			Linv_bc_dat[Linv_bc_cnt+jj]=Linv_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Linv_bc_ptr[jb]);
	// 		Linv_bc_ptr[jb]=&Linv_bc_dat[Linv_bc_cnt];
	// 		tmp_cnt = Linv_bc_offset[jb];
	// 		Linv_bc_offset[jb]=Linv_bc_cnt;
	// 		Linv_bc_cnt+=tmp_cnt;
	// 	}

	// 	if(Uinv_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Uinv_bc_offset[jb]; ++jj) {
	// 			Uinv_bc_dat[Uinv_bc_cnt+jj]=Uinv_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Uinv_bc_ptr[jb]);
	// 		Uinv_bc_ptr[jb]=&Uinv_bc_dat[Uinv_bc_cnt];
	// 		tmp_cnt = Uinv_bc_offset[jb];
	// 		Uinv_bc_offset[jb]=Uinv_bc_cnt;
	// 		Uinv_bc_cnt+=tmp_cnt;
	// 	}


	// 	if(Lrowind_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Lrowind_bc_offset[jb]; ++jj) {
	// 			Lrowind_bc_dat[Lrowind_bc_cnt+jj]=Lrowind_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Lrowind_bc_ptr[jb]);
	// 		Lrowind_bc_ptr[jb]=&Lrowind_bc_dat[Lrowind_bc_cnt];
	// 		tmp_cnt = Lrowind_bc_offset[jb];
	// 		Lrowind_bc_offset[jb]=Lrowind_bc_cnt;
	// 		Lrowind_bc_cnt+=tmp_cnt;
	// 	}

	// 	if(Lnzval_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Lnzval_bc_offset[jb]; ++jj) {
	// 			Lnzval_bc_dat[Lnzval_bc_cnt+jj]=Lnzval_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Lnzval_bc_ptr[jb]);
	// 		Lnzval_bc_ptr[jb]=&Lnzval_bc_dat[Lnzval_bc_cnt];
	// 		tmp_cnt = Lnzval_bc_offset[jb];
	// 		Lnzval_bc_offset[jb]=Lnzval_bc_cnt;
	// 		Lnzval_bc_cnt+=tmp_cnt;
	// 	}

	// 	if(Lindval_loc_bc_ptr[jb]!=NULL){
	// 		for (jj = 0; jj < Lindval_loc_bc_offset[jb]; ++jj) {
	// 			Lindval_loc_bc_dat[Lindval_loc_bc_cnt+jj]=Lindval_loc_bc_ptr[jb][jj];
	// 		}
	// 		SUPERLU_FREE(Lindval_loc_bc_ptr[jb]);
	// 		Lindval_loc_bc_ptr[jb]=&Lindval_loc_bc_dat[Lindval_loc_bc_cnt];
	// 		tmp_cnt = Lindval_loc_bc_offset[jb];
	// 		Lindval_loc_bc_offset[jb]=Lindval_loc_bc_cnt;
	// 		Lindval_loc_bc_cnt+=tmp_cnt;
	// 	}
	// }



    // // use contignous memory for the U meta data
    // k = kr;/* Number of local block rows */
    // long int Unzval_br_cnt=0;
    // long int Ufstnz_br_cnt=0;
    // long int Ucb_indcnt=0;
    // long int Ucb_valcnt=0;

	// if ( !(Unzval_br_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Unzval_br_offset[].");
	// }
	// Unzval_br_offset[k-1] = -1;
	// if ( !(Ufstnz_br_offset =
	// 			(long int*)SUPERLU_MALLOC(k * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Ufstnz_br_offset[].");
	// }
	// Ufstnz_br_offset[k-1] = -1;

    // int_t Pc = grid->npcol;
    // nub = CEILING (nsupers, Pc);
	// if ( !(Ucb_valoffset =
	// 			(long int*)SUPERLU_MALLOC(nub * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Ucb_valoffset[].");
	// }
	// Ucb_valoffset[nub-1] = -1;
	// if ( !(Ucb_indoffset =
	// 			(long int*)SUPERLU_MALLOC(nub * sizeof(long int))) ) {
	// 	fprintf(stderr, "Malloc fails for Ucb_indoffset[].");
	// }
	// Ucb_indoffset[nub-1] = -1;

    // for (int_t lk=0;lk<k;++lk){
    //     ib = myrow+lk*grid->nprow;  /* not sure */
	//     usub =  Ufstnz_br_ptr[lk];
    //     Unzval_br_offset[lk]=-1;
    //     Ufstnz_br_offset[lk]=-1;
    //     if(usub){
    //         int_t lenv = usub[1];
	// 	    int_t lens = usub[2];
    //         Unzval_br_offset[lk]=lenv;
	// 	    Unzval_br_cnt += Unzval_br_offset[lk];
    //         Ufstnz_br_offset[lk]=lens;
    //         Ufstnz_br_cnt += Ufstnz_br_offset[lk];
    //     }
    // }

	// /* Set up the vertical linked lists for the row blocks.
	//    One pass of the skeleton graph of U. */
	// for (int_t lb = 0; lb < kc; ++lb) {
	// 	if ( Urbs[lb] ) { /* Not an empty block column. */
	// 		Ucb_indoffset[lb]=Urbs[lb];
	// 		Ucb_indcnt += Ucb_indoffset[lb];
	// 		Ucb_valoffset[lb]=Urbs[lb];
	// 		Ucb_valcnt += Ucb_valoffset[lb];
	// 	}else{
	// 		Ucb_valoffset[lb]=-1;
	// 		Ucb_indoffset[lb]=-1;
	// 	}
	// }

	// Unzval_br_cnt +=1; // safe guard
	// Ufstnz_br_cnt +=1;
	// Ucb_valcnt +=1;
	// Ucb_indcnt +=1;
	// if ( !(Unzval_br_dat =
	// 			(float*)SUPERLU_MALLOC(Unzval_br_cnt * sizeof(float))) ) {
	// 	fprintf(stderr, "Malloc fails for Lnzval_bc_dat[].");
	// }
	// if ( !(Ufstnz_br_dat =
	// 			(int_t*)SUPERLU_MALLOC(Ufstnz_br_cnt * sizeof(int_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Ufstnz_br_dat[].");
	// }
	// if ( !(Ucb_valdat =
	// 			(int_t*)SUPERLU_MALLOC(Ucb_valcnt * sizeof(int_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Ucb_valdat[].");
	// }
	// if ( !(Ucb_inddat =
	// 			(Ucb_indptr_t*)SUPERLU_MALLOC(Ucb_indcnt * sizeof(Ucb_indptr_t))) ) {
	// 	fprintf(stderr, "Malloc fails for Ucb_inddat[].");
	// }


	// /* use contingous memory for Unzval_br_ptr, Ufstnz_br_ptr, Ucb_valptr */
	// k = CEILING( nsupers, grid->nprow );/* Number of local block rows */
	// Unzval_br_cnt=0;
	// Ufstnz_br_cnt=0;
	// for (int_t lb = 0; lb < k; ++lb) { /* for each block row ... */
	// 	if(Unzval_br_ptr[lb]!=NULL){
	// 		for (jj = 0; jj < Unzval_br_offset[lb]; ++jj) {
	// 			Unzval_br_dat[Unzval_br_cnt+jj]=Unzval_br_ptr[lb][jj];
	// 		}
	// 		SUPERLU_FREE(Unzval_br_ptr[lb]);
	// 		Unzval_br_ptr[lb]=&Unzval_br_dat[Unzval_br_cnt];
	// 		tmp_cnt = Unzval_br_offset[lb];
	// 		Unzval_br_offset[lb]=Unzval_br_cnt;
	// 		Unzval_br_cnt+=tmp_cnt;
	// 	}

	// 	if(Ufstnz_br_ptr[lb]!=NULL){
	// 		for (jj = 0; jj < Ufstnz_br_offset[lb]; ++jj) {
	// 			Ufstnz_br_dat[Ufstnz_br_cnt+jj]=Ufstnz_br_ptr[lb][jj];
	// 		}
	// 		SUPERLU_FREE(Ufstnz_br_ptr[lb]);
	// 		Ufstnz_br_ptr[lb]=&Ufstnz_br_dat[Ufstnz_br_cnt];
	// 		tmp_cnt = Ufstnz_br_offset[lb];
	// 		Ufstnz_br_offset[lb]=Ufstnz_br_cnt;
	// 		Ufstnz_br_cnt+=tmp_cnt;
	// 	}
	// }

	// k = CEILING( nsupers, grid->npcol );/* Number of local block columns */
	// Ucb_valcnt=0;
	// Ucb_indcnt=0;
	// for (int_t lb = 0; lb < k; ++lb) { /* for each block row ... */
	// 	if(Ucb_valptr[lb]!=NULL){
	// 		for (jj = 0; jj < Ucb_valoffset[lb]; ++jj) {
	// 			Ucb_valdat[Ucb_valcnt+jj]=Ucb_valptr[lb][jj];
	// 		}
	// 		SUPERLU_FREE(Ucb_valptr[lb]);
	// 		Ucb_valptr[lb]=&Ucb_valdat[Ucb_valcnt];
	// 		tmp_cnt = Ucb_valoffset[lb];
	// 		Ucb_valoffset[lb]=Ucb_valcnt;
	// 		Ucb_valcnt+=tmp_cnt;
	// 	}
	// 	if(Ucb_indptr[lb]!=NULL){
	// 		for (jj = 0; jj < Ucb_indoffset[lb]; ++jj) {
	// 			Ucb_inddat[Ucb_indcnt+jj]=Ucb_indptr[lb][jj];
	// 		}
	// 		SUPERLU_FREE(Ucb_indptr[lb]);
	// 		Ucb_indptr[lb]=&Ucb_inddat[Ucb_indcnt];
	// 		tmp_cnt = Ucb_indoffset[lb];
	// 		Ucb_indoffset[lb]=Ucb_indcnt;
	// 		Ucb_indcnt+=tmp_cnt;
	// 	}
	// }

	// Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
	// Llu->Lrowind_bc_dat = Lrowind_bc_dat;
	// Llu->Lrowind_bc_offset = Lrowind_bc_offset;
	// Llu->Lrowind_bc_cnt = Lrowind_bc_cnt;

	// Llu->Lindval_loc_bc_ptr = Lindval_loc_bc_ptr;
	// Llu->Lindval_loc_bc_dat = Lindval_loc_bc_dat;
	// Llu->Lindval_loc_bc_offset = Lindval_loc_bc_offset;
	// Llu->Lindval_loc_bc_cnt = Lindval_loc_bc_cnt;

	// Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
	// Llu->Lnzval_bc_dat = Lnzval_bc_dat;
	// Llu->Lnzval_bc_offset = Lnzval_bc_offset;
	// Llu->Lnzval_bc_cnt = Lnzval_bc_cnt;

	// Llu->Linv_bc_ptr = Linv_bc_ptr;
	// Llu->Linv_bc_dat = Linv_bc_dat;
	// Llu->Linv_bc_offset = Linv_bc_offset;
	// Llu->Linv_bc_cnt = Linv_bc_cnt;

	// Llu->Uinv_bc_ptr = Uinv_bc_ptr;
	// Llu->Uinv_bc_dat = Uinv_bc_dat;
	// Llu->Uinv_bc_offset = Uinv_bc_offset;
	// Llu->Uinv_bc_cnt = Uinv_bc_cnt;


	Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
    // Llu->Ufstnz_br_dat = Ufstnz_br_dat;
    // Llu->Ufstnz_br_offset = Ufstnz_br_offset;
    // Llu->Ufstnz_br_cnt = Ufstnz_br_cnt;

	Llu->Unzval_br_ptr = Unzval_br_ptr;
	// Llu->Unzval_br_dat = Unzval_br_dat;
	// Llu->Unzval_br_offset = Unzval_br_offset;
	// Llu->Unzval_br_cnt = Unzval_br_cnt;

	Llu->Ucb_indptr = Ucb_indptr;
	// Llu->Ucb_inddat = Ucb_inddat;
	// Llu->Ucb_indoffset = Ucb_indoffset;
	// Llu->Ucb_indcnt = Ucb_indcnt;
	Llu->Ucb_valptr = Ucb_valptr;
	// Llu->Ucb_valdat = Ucb_valdat;
	// Llu->Ucb_valoffset = Ucb_valoffset;
	// Llu->Ucb_valcnt = Ucb_valcnt;


	Llu->LRtree_ptr = LRtree_ptr;
	Llu->LBtree_ptr = LBtree_ptr;
	Llu->URtree_ptr = URtree_ptr;
	Llu->UBtree_ptr = UBtree_ptr;


    Llu->nbcol_masked=0;
	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
        if(jb<nsupers){
            if(supernodeMask[jb]==1){ // only record the columns performed on GPU
               Llu->nbcol_masked++;
            }
        }
    }
 	if ( !(Llu->bcols_masked =
				(int*)SUPERLU_MALLOC(Llu->nbcol_masked * sizeof(int))) ) {
		fprintf(stderr, "Malloc fails for nbcol_masked[].");
	}
    Llu->nbcol_masked=0;
	for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
		jb = mycol+lk*grid->npcol;  /* not sure */
        if(jb<nsupers){
            if(supernodeMask[jb]==1){ // only record the columns performed on GPU
               Llu->bcols_masked[Llu->nbcol_masked++]=lk;
            }
        }
    }
    // printf("Llu->nbcol_masked: %10d\n",Llu->nbcol_masked);
    // fflush(stdout);


#ifdef GPU_ACC
    if (get_acc_solve()){
	checkGPU(gpuMalloc( (void**)&Llu->d_bcols_masked, Llu->nbcol_masked * sizeof(int)));
	checkGPU(gpuMemcpy(Llu->d_bcols_masked, Llu->bcols_masked, Llu->nbcol_masked * sizeof(int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_xsup, (n+1) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_xsup, xsup, (n+1) * sizeof(int_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_LRtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_LBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_URtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_UBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree)));
	checkGPU(gpuMemcpy(Llu->d_LRtree_ptr, Llu->LRtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(Llu->d_LBtree_ptr, Llu->LBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(Llu->d_URtree_ptr, Llu->URtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(Llu->d_UBtree_ptr, Llu->UBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lrowind_bc_dat, (Llu->Lrowind_bc_cnt) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_Lrowind_bc_dat, Llu->Lrowind_bc_dat, (Llu->Lrowind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lindval_loc_bc_dat, (Llu->Lindval_loc_bc_cnt) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_Lindval_loc_bc_dat, Llu->Lindval_loc_bc_dat, (Llu->Lindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lrowind_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lrowind_bc_offset, Llu->Lrowind_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lindval_loc_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lindval_loc_bc_offset, Llu->Lindval_loc_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Lnzval_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lnzval_bc_offset, Llu->Lnzval_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
    checkGPU(gpuMalloc( (void**)&Llu->d_grid, sizeof(gridinfo_t)));
    checkGPU(gpuMemcpy(Llu->d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));

	// some dummy allocation to avoid checking whether they are null pointers later
#if 0
	checkGPU(gpuMalloc( (void**)&Llu->d_Ucolind_bc_dat, sizeof(int_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Ucolind_bc_offset, sizeof(int64_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Unzval_bc_dat, sizeof(float)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Unzval_bc_offset, sizeof(int64_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uindval_loc_bc_dat, sizeof(int_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uindval_loc_bc_offset, sizeof(int_t)));
#else
    Llu->d_Ucolind_bc_dat=NULL;
    Llu->d_Ucolind_br_dat=NULL;
    Llu->d_Ucolind_bc_offset=NULL;
    Llu->d_Ucolind_br_offset=NULL;
    Llu->d_Uind_br_dat=NULL;
    Llu->d_Uind_br_offset=NULL;
    Llu->d_Unzval_bc_dat=NULL;
    Llu->d_Unzval_bc_offset=NULL;
    Llu->d_Unzval_br_new_dat=NULL;
    Llu->d_Unzval_br_new_offset=NULL;
    Llu->d_Uindval_loc_bc_dat=NULL;
    Llu->d_Uindval_loc_bc_offset=NULL;
#endif


	checkGPU(gpuMalloc( (void**)&Llu->d_Linv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Linv_bc_offset, Llu->Linv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uinv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Uinv_bc_offset, Llu->Uinv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_ilsum, (CEILING( nsupers, grid->nprow )+1) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_ilsum, Llu->ilsum, (CEILING( nsupers, grid->nprow )+1) * sizeof(int_t), gpuMemcpyHostToDevice));


	/* gpuMemcpy for the following is performed in pxgssvx/pxgssvx3d */
	checkGPU(gpuMalloc( (void**)&Llu->d_Lnzval_bc_dat, (Llu->Lnzval_bc_cnt) * sizeof(float)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Linv_bc_dat, (Llu->Linv_bc_cnt) * sizeof(float)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uinv_bc_dat, (Llu->Uinv_bc_cnt) * sizeof(float)));


       /* nvshmem related*/
#ifdef HAVE_NVSHMEM
	checkGPU(gpuMalloc( (void**)&d_recv_cnt, CEILING(nsupers, grid->nprow) * sizeof(int)));
	checkGPU(gpuMemcpy(d_recv_cnt, h_recv_cnt,  CEILING(nsupers, grid->nprow) * sizeof(int), gpuMemcpyHostToDevice));
        checkGPU(gpuMalloc( (void**)&d_recv_cnt_u, CEILING(nsupers, grid->nprow) * sizeof(int)));
        checkGPU(gpuMemcpy(d_recv_cnt_u, h_recv_cnt_u,  CEILING(nsupers, grid->nprow) * sizeof(int), gpuMemcpyHostToDevice));
#endif
    }
#ifdef HAVE_NVSHMEM
    SUPERLU_FREE(h_recv_cnt);
    SUPERLU_FREE(h_recv_cnt_u);
#endif
#endif /* end ifdef GPU_ACC */

    // /* recompute fmod, bmod */
	// for (int_t i = 0; i < kc; ++i)
	// 	Llu->fmod[i] = 0;


	// for (int_t lk = 0; lk < kc; ++lk) { /* for each local block column ... */
	// 	jb = mycol+lk*grid->npcol;  /* not sure */
	// 	if(jb<nsupers){
    //         if(supernodeMask[jb]>0)
    //         {
    //             int_t krow = PROW (jb, grid);
    //             int_t kcol = PCOL (jb, grid);

    //             int_t* lsub = Lrowind_bc_ptr[lk];
    //             int_t* lloc = LUstruct->Llu->Lindval_loc_bc_ptr[lk];
    //             if(lsub){
    //             if(lsub[0]>0){
    //                 if(myrow==krow){
    //                     nb = lsub[0] - 1;
    //                     idx_n = 1;
    //                     idx_i = nb+2;
    //                 }else{
    //                     nb = lsub[0];
    //                     idx_n = 0;
    //                     idx_i = nb;
    //                 }
    //                 for (int_t lb=0;lb<nb;lb++){
    //                     int_t lik = lloc[lb+idx_n]; /* Local block number, row-wise. */
    //                     lptr1_tmp = lloc[lb+idx_i];
    //                     ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
    //                     if(supernodeMask[ik])
    //                         Llu->fmod[lik] +=1;
    //                 }
    //             }
    //             }
    //         }
    //     }
    // }

    return 0;
} // end strs_compute_communication_structure



int_t strs_x_reduction_newsolve(int_t nsupers, float* x, int nrhs, sLUstruct_t * LUstruct, gridinfo3d_t *grid3d, strf3Dpartition_t*  trf3Dpartition, float* recvbuf, xtrsTimer_t *xtrsTimer)

{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t myGrid = grid3d->zscp.Iam;
	int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
	int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;

	for (int_t ilvl = 1; ilvl < maxLvl ; ++ilvl)
	{
        if(!myZeroTrIdxs[ilvl-1]){ // this ensures the number of grids in communication is reduced by half every level down
            int_t sender, receiver;
            int_t tree = myTreeIdxs[ilvl];
            sForest_t** sForests = trf3Dpartition->sForests;
            sForest_t* sforest = sForests[tree];

            if ((myGrid % (1 << ilvl)) == 0)
            {
                sender = myGrid + (1 << (ilvl-1));
                receiver = myGrid;
            }
            else
            {
                sender = myGrid;
                receiver = myGrid - (1 << (ilvl-1));
            }
            int_t tr =  tree;
            for (int_t alvl = ilvl; alvl < maxLvl; alvl++)
            {
                /* code */
                // printf("myGrid %5d tr %5d sender %5d receiver %5d\n",myGrid,tr, sender, receiver);
                // fflush(stdout);
                sreduceSolvedX_newsolve(tr, sender, receiver, x, nrhs,  trf3Dpartition, LUstruct, grid3d, recvbuf, xtrsTimer);
                tr=(tr+1)/2-1;

            }
        }
	}

	return 0;
}



int_t strs_x_broadcast_newsolve(int_t nsupers, float* x, int nrhs, sLUstruct_t * LUstruct, gridinfo3d_t *grid3d, strf3Dpartition_t*  trf3Dpartition, float* recvbuf, xtrsTimer_t *xtrsTimer)

{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t myGrid = grid3d->zscp.Iam;
	int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
	int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;

	for (int_t ilvl = maxLvl-1; ilvl >0 ; --ilvl)
	{
        if(!myZeroTrIdxs[ilvl-1]){ // this ensures the number of grids in communication is doubled every level down
            int_t sender, receiver;
            int_t tree = myTreeIdxs[ilvl];
            if ((myGrid % (1 << ilvl)) == 0)
            {
                sender = myGrid;
                receiver = myGrid + (1 << (ilvl-1));
            }
            else
            {
                sender = myGrid - (1 << (ilvl-1));
                receiver = myGrid ;
            }
            int_t tr =  tree;
            for (int_t alvl = ilvl; alvl < maxLvl; alvl++)
            {
                // /* code */
                // printf("myGrid %5d tr %5d sender %5d receiver %5d\n",myGrid,tr, sender, receiver);
                // fflush(stdout);

                sp2pSolvedX3d(tr, sender, receiver, x, nrhs,  trf3Dpartition, LUstruct, grid3d, xtrsTimer);
                tr=(tr+1)/2-1;

            }
        }
	}

	return 0;
}




int_t sreduceSolvedX_newsolve(int_t treeId, int_t sender, int_t receiver, float* x, int nrhs,
                      strf3Dpartition_t*  trf3Dpartition, sLUstruct_t* LUstruct, gridinfo3d_t* grid3d, float* recvbuf, xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;
	sForest_t* sforest = sForests[treeId];
	if (!sforest) return 0;
	int_t nnodes = sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;

	gridinfo_t * grid = &(grid3d->grid2d);
	int_t myGrid = grid3d->zscp.Iam;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	sLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );
    float zero = 0.0;

	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow && mycol == kcol)
		{
			int_t lk = LBi(k, grid);
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize(k);
			if (myGrid == sender)
			{
				/* code */
				MPI_Send( &x[ii], knsupc * nrhs, MPI_FLOAT, receiver, k,  grid3d->zscp.comm);
                for(int_t i=0; i<knsupc * nrhs; i++){
                    x[ii+i]=zero;
                }
                xtrsTimer->trsDataSendZ += knsupc * nrhs;
            }
			else
			{
				MPI_Status status;
				MPI_Recv( recvbuf, knsupc * nrhs, MPI_FLOAT, sender, k, grid3d->zscp.comm, &status );
                for(int_t i=0; i<knsupc * nrhs; i++){
                    x[ii+i]+=recvbuf[i];
                }
                xtrsTimer->trsDataRecvZ += knsupc * nrhs;
			}
		}
	}

	return 0;
}




// Gather the solution vector from all grids to grid 0
int_t strs_X_gather3d(float* x, int nrhs, strf3Dpartition_t*  trf3Dpartition,
                     sLUstruct_t* LUstruct,
                     gridinfo3d_t* grid3d, xtrsTimer_t *xtrsTimer)

{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t myGrid = grid3d->zscp.Iam;
	int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;

	for (int_t ilvl = 0; ilvl < maxLvl - 1; ++ilvl)
	{
		int_t sender, receiver;
		if (!myZeroTrIdxs[ilvl])
		{
			if ((myGrid % (1 << (ilvl + 1))) == 0)
			{
				sender = myGrid + (1 << ilvl);
				receiver = myGrid;
			}
			else
			{
				sender = myGrid;
				receiver = myGrid - (1 << ilvl);
			}
			for (int_t alvl = 0; alvl <= ilvl; alvl++)
			{
				int_t diffLvl  = ilvl - alvl;
				int_t numTrees = 1 << diffLvl;
				int_t blvl = maxLvl - alvl - 1;
				int_t st = (1 << blvl) - 1 + (sender >> alvl);

				for (int_t tr = st; tr < st + numTrees; ++tr)
				{
					/* code */
					sp2pSolvedX3d(tr, sender, receiver, x, nrhs,  trf3Dpartition, LUstruct, grid3d, xtrsTimer);
				}
			}

		}
	}

	return 0;
}


int_t sp2pSolvedX3d(int_t treeId, int_t sender, int_t receiver, float* x, int nrhs,
                      strf3Dpartition_t*  trf3Dpartition, sLUstruct_t* LUstruct, gridinfo3d_t* grid3d, xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;
	sForest_t* sforest = sForests[treeId];
	if (!sforest) return 0;
	int_t nnodes = sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;

	gridinfo_t * grid = &(grid3d->grid2d);
	int_t myGrid = grid3d->zscp.Iam;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	sLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );


	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow && mycol == kcol)
		{
			int_t lk = LBi(k, grid);
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize(k);
			if (myGrid == sender)
			{
				/* code */
				MPI_Send( &x[ii], knsupc * nrhs, MPI_FLOAT, receiver, k,  grid3d->zscp.comm);
                xtrsTimer->trsDataSendZ += knsupc * nrhs;
            }
			else
			{
				MPI_Status status;
				MPI_Recv( &x[ii], knsupc * nrhs, MPI_FLOAT, sender, k, grid3d->zscp.comm, &status );
                xtrsTimer->trsDataRecvZ += knsupc * nrhs;
            }
		}
	}

	return 0;
}


int_t sfsolveReduceLsum3d(int_t treeId, int_t sender, int_t receiver, float* lsum, float* recvbuf, int nrhs,
                         strf3Dpartition_t*  trf3Dpartition, sLUstruct_t* LUstruct, gridinfo3d_t* grid3d ,
                         xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;
	sForest_t* sforest = sForests[treeId];
	if (!sforest) return 0;
	int_t nnodes = sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;

	gridinfo_t * grid = &(grid3d->grid2d);
	int_t myGrid = grid3d->zscp.Iam;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	sLocalLU_t *Llu = LUstruct->Llu;
	int_t* ilsum = Llu->ilsum;
	int_t* xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );


	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow )
		{
			int_t lk = LBi(k, grid);
			int_t knsupc = SuperSize(k);
			if (myGrid == sender)
			{
				/* code */
				int_t ii = LSUM_BLK (lk);
				float* lsum_k = &lsum[ii];
				superlu_scope_t *scp = &grid->rscp;
				MPI_Reduce( lsum_k, recvbuf, knsupc * nrhs,
				            MPI_FLOAT, MPI_SUM, kcol, scp->comm);
				xtrsTimer->trsDataSendXY += knsupc * nrhs;
				xtrsTimer->trsDataRecvXY += knsupc * nrhs;
				if (mycol == kcol)
				{
					MPI_Send( recvbuf, knsupc * nrhs, MPI_FLOAT, receiver, k,  grid3d->zscp.comm);
					xtrsTimer->trsDataSendZ += knsupc * nrhs;
				}
			}
			else
			{
				if (mycol == kcol)
				{
					MPI_Status status;
					MPI_Recv( recvbuf, knsupc * nrhs, MPI_FLOAT, sender, k, grid3d->zscp.comm, &status );
					xtrsTimer->trsDataRecvZ += knsupc * nrhs;
					int_t ii = LSUM_BLK (lk);
					float* dest = &lsum[ii];
					float* tempv = recvbuf;
					for (int_t j = 0; j < nrhs; ++j)
					{
						for (int_t i = 0; i < knsupc; ++i)
                            dest[i + j * knsupc] += tempv[i + j * knsupc];
                    }
				}

			}
		}
	}

	return 0;
}



int_t sbsolve_Xt_bcast(int_t ilvl, sxT_struct *xT_s, int nrhs, strf3Dpartition_t*  trf3Dpartition,
                     sLUstruct_t * LUstruct,gridinfo3d_t* grid3d , xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t *ilsum = Llu->ilsum;
    int_t* xsup = Glu_persist->xsup;

	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	gridinfo_t * grid = &(grid3d->grid2d);
	int_t myGrid = grid3d->zscp.Iam;
		int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t mycol = MYCOL( iam, grid );


	float *xT = xT_s->xT;
	int_t *ilsumT = xT_s->ilsumT;
	int_t ldaspaT = xT_s->ldaspaT;


	int_t sender, receiver;

	if ((myGrid % (1 << (ilvl + 1))) == 0)
	{
		receiver = myGrid + (1 << ilvl);
		sender = myGrid;
	}
	else
	{
		receiver = myGrid;
		sender = myGrid - (1 << ilvl);
	}

	for (int_t alvl = ilvl + 1; alvl < maxLvl; ++alvl)
	{
		/* code */

		int_t treeId = trf3Dpartition->myTreeIdxs[alvl];
		sForest_t* sforest = trf3Dpartition->sForests[treeId];
		if (sforest)
		{
			/* code */
			int_t nnodes = sforest->nNodes;
			int_t* nodeList = sforest->nodeList;
			for (int_t k0 = 0; k0 < nnodes ; ++k0)
			{
				/* code */
				int_t k = nodeList[k0];
				int_t krow = PROW (k, grid);
				int_t kcol = PCOL (k, grid);
				int_t knsupc = SuperSize (k);
				if (myGrid == sender)
				{
					/* code */
					if (mycol == kcol &&   myrow == krow)
					{

						int_t lk = LBj (k, grid);
						int_t ii = XT_BLK (lk);
						float* xk = &xT[ii];
						MPI_Send( xk, knsupc * nrhs, MPI_FLOAT, receiver, k,
						           grid3d->zscp.comm);
						xtrsTimer->trsDataSendZ += knsupc * nrhs;

					}
				}
				else
				{
					if (mycol == kcol)
					{
						/* code */
						if (myrow == krow )
						{
							/* code */
							int_t lk = LBj (k, grid);
							int_t ii = XT_BLK (lk);
							float* xk = &xT[ii];
							MPI_Status status;
							MPI_Recv( xk, knsupc * nrhs, MPI_FLOAT, sender,k,
							           grid3d->zscp.comm, &status);
							xtrsTimer->trsDataRecvZ += knsupc * nrhs;
						}
						sbCastXk2Pck( k,  xT_s,  nrhs, LUstruct, grid, xtrsTimer);
					}

				}

			}
		}
	}


	return 0;
}




int_t slsumForestFsolve(int_t k,
                       float *lsum, float *x, float* rtemp,  sxT_struct *xT_s, int    nrhs,
                       sLUstruct_t * LUstruct,
                       strf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat)
{
	gridinfo_t * grid = &(grid3d->grid2d);
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	sLocalLU_t *Llu = LUstruct->Llu;
	int_t* xsup = Glu_persist->xsup;

	int_t iam = grid->iam;
	int_t myrow = MYROW( iam, grid );
	int_t lk = LBj( k, grid ); /* Local block number, column-wise. */
	int_t *lsub = Llu->Lrowind_bc_ptr[lk];
	if (!lsub) return 0;
	float* lusup = Llu->Lnzval_bc_ptr[lk];
	int nsupr = lsub[1];
	int_t nlb = lsub[0];
	int_t lptr = BC_HEADER;
	int_t luptr = 0;
	int_t krow = PROW (k, grid);
	int knsupc = SuperSize(k);
	if (myrow == krow)
	{
		/* code */
		nlb = lsub[0] - 1;
		lptr +=  LB_DESCRIPTOR + knsupc;
		luptr += knsupc;
	}

	float *xT = xT_s->xT;
	int_t *ilsumT = xT_s->ilsumT;
	int_t ldaspaT = xT_s->ldaspaT;


	int_t *ilsum = Llu->ilsum;
	int_t ii = XT_BLK (lk);
	float* xk = &xT[ii];
	for (int_t lb = 0; lb < nlb; ++lb)
	{
		int_t ik = lsub[lptr]; /* Global block number, row-wise. */
		int nbrow = lsub[lptr + 1];
        float alpha = 1.0;
        float beta = 0.0;
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

		int_t lk = LBi( ik, grid ); /* Local block number, row-wise. */
		int_t iknsupc = SuperSize( ik );
		int_t il = LSUM_BLK( lk );
		float* dest = &lsum[il];
		lptr += LB_DESCRIPTOR;
		int_t rel = xsup[ik]; /* Global row index of block ik. */
		for (int_t i = 0; i < nbrow; ++i)
		{
			int_t irow = lsub[lptr++] - rel; /* Relative row. */
			for (int_t j = 0; j < nrhs; ++j)
                dest[irow + j * iknsupc] -= rtemp[i + j * nbrow];
		}
		luptr += nbrow;
	}

	return 0;
}



int_t snonLeafForestForwardSolve3d( int_t treeId,  sLUstruct_t * LUstruct,
                                   sScalePermstruct_t * ScalePermstruct,
                                   strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                                   float * x, float * lsum,
                                   sxT_struct *xT_s,
                                   float * recvbuf, float* rtemp,
                                   MPI_Request * send_req,
                                   int nrhs,
                                   sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{

	sForest_t** sForests = trf3Dpartition->sForests;

	sForest_t* sforest = sForests[treeId];
	if (!sforest)
	{
		/* code */
		return 0;
	}
	int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
	if (nnodes < 1) return 1;
	int_t *perm_c_supno = sforest->nodeList ;
	gridinfo_t * grid = &(grid3d->grid2d);

	sLocalLU_t *Llu = LUstruct->Llu;
	int_t *ilsum = Llu->ilsum;

	int_t* xsup =  LUstruct->Glu_persist->xsup;

	float *xT = xT_s->xT;
	int_t *ilsumT = xT_s->ilsumT;
	int_t ldaspaT = xT_s->ldaspaT;

	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);

	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = perm_c_supno[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);
		// printf("doing %d \n", k);
		/**
		 * Pkk(Yk) = sumOver_PrK (Yk)
		 */
		if (myrow == krow )
		{
			double tx = SuperLU_timer_();
			slsumReducePrK(k, x, lsum, recvbuf, nrhs, LUstruct, grid,xtrsTimer);
			// xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
			xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
		}

		if (mycol == kcol )
		{
			int_t lk = LBi (k, grid); /* Local block number, row-wise. */
			int_t ii = X_BLK (lk);
			if (myrow == krow )
			{
				/* Diagonal process. */
				double tx = SuperLU_timer_();
				slocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				int_t lkj = LBj (k, grid);
				int_t jj = XT_BLK (lkj);
				int_t knsupc = SuperSize(k);
				memcpy(&xT[jj], &x[ii], knsupc * nrhs * sizeof(float) );
				xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
			}                       /* if diagonal process ... */
			/*
			 * Send Xk to process column Pc[k].
			 */
			double tx = SuperLU_timer_();
			sbCastXk2Pck( k,  xT_s,  nrhs, LUstruct, grid, xtrsTimer);
			xtrsTimer->tfs_comm += SuperLU_timer_() - tx;

			/*
			 * Perform local block modifications: lsum[i] -= U_i,k * X[k]
			 * where i is in current sforest
			 */
			tx = SuperLU_timer_();
			slsumForestFsolve(k, lsum, x, rtemp, xT_s, nrhs,
			                 LUstruct, trf3Dpartition, grid3d, stat);
			xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
		}
	}                           /* for k ... */
	return 0;
}


int_t sleafForestForwardSolve3d(superlu_dist_options_t *options, int_t treeId, int_t n,  sLUstruct_t * LUstruct,
                               sScalePermstruct_t * ScalePermstruct,
                               strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                               float * x, float * lsum, float * recvbuf, float* rtemp,
                               MPI_Request * send_req,
                               int nrhs,
                               sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;

	sForest_t* sforest = sForests[treeId];
	if (!sforest) return 0;
	int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
	if (nnodes < 1)
	{
		return 1;
	}
	gridinfo_t * grid = &(grid3d->grid2d);
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	sLocalLU_t *Llu = LUstruct->Llu;
	int_t* xsup = Glu_persist->xsup;
	int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
	int_t nsupers = Glu_persist->supno[n - 1] + 1;
	int_t Pr = grid->nprow;
	int_t nlb = CEILING (nsupers, Pr);

	treeTopoInfo_t* treeTopoInfo = &sforest->topoInfo;
	int_t* eTreeTopLims = treeTopoInfo->eTreeTopLims;
	int_t *nodeList = sforest->nodeList ;


	int_t knsupc = sp_ienv_dist (3,options);
	int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);

	int **fsendx_plist = Llu->fsendx_plist;
	int_t* ilsum = Llu->ilsum;

	int* fmod = getfmodLeaf(nlb, LUstruct->Llu->fmod);
	int* frecv = getfrecvLeaf(sforest, nlb, fmod, LUstruct->Llu->mod_bit, grid);
	Llu->frecv = frecv;
	int  nfrecvx = getNfrecvxLeaf(sforest, LUstruct->Llu->Lrowind_bc_ptr, grid);
	int nleaf = 0;
	int_t nfrecvmod = getNfrecvmodLeaf(&nleaf, sforest, frecv, fmod,  grid);
    int_t myGrid = grid3d->zscp.Iam;
    // printf("igrid %5d, iam %5d, nfrecvx %5d, nfrecvmod %5d, nleaf %5d\n",myGrid,iam,nfrecvx,nfrecvmod,nleaf);


	/* factor the leaf to being the factorization*/
	for (int_t k0 = 0; k0 < nnodes && nleaf; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);
		if (myrow == krow && mycol == kcol)
		{
			/* Diagonal process */
			int_t knsupc = SuperSize (k);
			int_t lk = LBi (k, grid);
			if (frecv[lk] == 0 && fmod[lk] == 0)
			{
				double tx = SuperLU_timer_();
				fmod[lk] = -1;  /* Do not solve X[k] in the future. */
				int_t ii = X_BLK (lk);

				int_t lkj = LBj (k, grid); /* Local block number, column-wise. */
				int_t* lsub = Lrowind_bc_ptr[lkj];
				slocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				siBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, fsendx_plist, send_req, LUstruct, grid,xtrsTimer);
				nleaf--;
				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				int_t nb = lsub[0] - 1;
				int_t lptr = BC_HEADER + LB_DESCRIPTOR + knsupc;
				int_t luptr = knsupc; /* Skip diagonal block L(k,k). */

				slsum_fmod_leaf (treeId, trf3Dpartition, lsum, x, &x[ii], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
				xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
			}
		}                       /* if diagonal process ... */
	}


	while (nfrecvx || nfrecvmod)
	{
		/* While not finished. */
		/* Receive a message. */
		MPI_Status status;
		double tx = SuperLU_timer_();
		MPI_Recv (recvbuf, maxrecvsz, MPI_FLOAT,
		          MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status);
		xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
		int_t k = *recvbuf;
		xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
		tx = SuperLU_timer_();
		switch (status.MPI_TAG)
		{
		case Xk:
		{
			--nfrecvx;
			int_t lk = LBj (k, grid); /* Local block number, column-wise. */
			int_t *lsub = Lrowind_bc_ptr[lk];

			if (lsub)
			{
				int_t nb = lsub[0];
				int_t lptr = BC_HEADER;
				int_t luptr = 0;
				int_t knsupc = SuperSize (k);

				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				slsum_fmod_leaf (treeId, trf3Dpartition, lsum, x, &recvbuf[XK_H], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
			}                   /* if lsub */

			break;
		}

		case LSUM:             /* Receiver must be a diagonal process */
		{
			--nfrecvmod;
			int_t lk = LBi (k, grid); /* Local block number, row-wise. */
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize (k);
			float* tempv = &recvbuf[LSUM_H];
			for (int_t j = 0; j < nrhs; ++j)
			{
				for (int_t i = 0; i < knsupc; ++i)
					x[i + ii + j * knsupc] += tempv[i + j * knsupc];
			}

			if ((--frecv[lk]) == 0 && fmod[lk] == 0)
			{
				fmod[lk] = -1;  /* Do not solve X[k] in the future. */
				lk = LBj (k, grid); /* Local block number, column-wise. */
				int_t *lsub = Lrowind_bc_ptr[lk];
				slocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				/*
				  * Send Xk to process column Pc[k].
				  */
				siBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, fsendx_plist, send_req, LUstruct, grid, xtrsTimer);
				/*
				 * Perform local block modifications.
				 */
				int_t nb = lsub[0] - 1;
				int_t lptr = BC_HEADER + LB_DESCRIPTOR + knsupc;
				int_t luptr = knsupc; /* Skip diagonal block L(k,k). */

				slsum_fmod_leaf (treeId, trf3Dpartition, lsum, x, &x[ii], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
			}                   /* if */

			break;
		}

		default:
		{
			// printf ("(%2d) Recv'd wrong message tag %4d\n", status.MPI_TAG);
			break;
		}

		}                       /* switch */
		xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
	}                           /* while not finished ... */
	SUPERLU_FREE (fmod);
	SUPERLU_FREE (frecv);
	double tx = SuperLU_timer_();
	for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
	{
		MPI_Status status;
		MPI_Wait (&send_req[i], &status);
	}
	Llu->SolveMsgSent = 0;
	xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
	MPI_Barrier (grid->comm);
	return 0;
}




void slsum_fmod_leaf (
    int_t treeId,
    strf3Dpartition_t*  trf3Dpartition,
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
    SuperLUStat_t *stat,xtrsTimer_t *xtrsTimer)

{
    float alpha = 1.0;
    float beta = 0.0;
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

	for (lb = 0; lb < nlb; ++lb)
	{
		ik = lsub[lptr]; /* Global block number, row-wise. */
		nbrow = lsub[lptr + 1];
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
		for (i = 0; i < nbrow; ++i)
		{
			irow = lsub[lptr++] - rel; /* Relative row. */
			RHS_ITERATE(j)
            dest[irow + j * iknsupc] -= rtemp[i + j * nbrow];
		}
		luptr += nbrow;

#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat->utime[SOL_GEMM] += t2;
#endif


		if ( (--fmod[lk]) == 0  )   /* Local accumulation done. */
		{
			if (trf3Dpartition->supernode2treeMap[ik] == treeId)
			{
				ikcol = PCOL( ik, grid );
				p = PNUM( myrow, ikcol, grid );
				if ( iam != p )
				{
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
					xtrsTimer->trsDataSendXY += iknsupc * nrhs + LSUM_H;
				}
				else     /* Diagonal process: X[i] += lsum[i]. */
				{
					ii = X_BLK( lk );
					RHS_ITERATE(j)
					for (i = 0; i < iknsupc; ++i)
					    x[i + ii + j * iknsupc] += lsum[i + il + j * iknsupc];

					if ( frecv[lk] == 0 )   /* Becomes a leaf node. */
					{
						fmod[lk] = -1; /* Do not solve X[k] in the future. */


						lk = LBj( ik, grid );/* Local block number, column-wise. */
						lsub1 = Llu->Lrowind_bc_ptr[lk];
						lusup1 = Llu->Lnzval_bc_ptr[lk];
						nsupr1 = lsub1[1];
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


						stat->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;

						/*
						 * Send Xk to process column Pc[k].
						 */
						for (p = 0; p < grid->nprow; ++p)
						{
							if ( fsendx_plist[lk][p] != SLU_EMPTY )
							{
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

							}
						}
						xtrsTimer->trsDataSendXY += iknsupc * nrhs + XK_H;
						/*
						 * Perform local block modifications.
						 */
						nlb1 = lsub1[0] - 1;
						lptr1 = BC_HEADER + LB_DESCRIPTOR + iknsupc;
						luptr1 = iknsupc; /* Skip diagonal block L(I,I). */

						slsum_fmod_leaf(treeId, trf3Dpartition,
						                lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
						                fmod, nlb1, lptr1, luptr1, xsup,
						                grid, Llu, send_req, stat,xtrsTimer);
					} /* if frecv[lk] == 0 */
				} /* if iam == p */
			}
		}/* if fmod[lk] == 0 */

	} /* for lb ... */

} /* sLSUM_FMOD_LEAF */


int_t sleafForestForwardSolve3d_newsolve(superlu_dist_options_t *options, int_t n,  sLUstruct_t * LUstruct,
                               sScalePermstruct_t * ScalePermstruct,
                               strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                               float * x, float * lsum, float * recvbuf, float* rtemp,
                               MPI_Request * send_req,
                               int nrhs,
                               sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
	sForest_t** sForests = trf3Dpartition->sForests;

	// sForest_t* sforest = sForests[treeId];
	// if (!sforest) return 0;
	// int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
	// if (nnodes < 1)
	// {
	// 	return 1;
	// }
	gridinfo_t * grid = &(grid3d->grid2d);
	int_t iam = grid->iam;
    int_t myGrid = grid3d->zscp.Iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	sLocalLU_t *Llu = LUstruct->Llu;
	int_t* xsup = Glu_persist->xsup;
	int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
	int_t nsupers = Glu_persist->supno[n - 1] + 1;
	int_t Pr = grid->nprow;
	int_t nlb = CEILING (nsupers, Pr);
    int* supernodeMask = trf3Dpartition->supernodeMask;

	// treeTopoInfo_t* treeTopoInfo = &sforest->topoInfo;
	// int_t* eTreeTopLims = treeTopoInfo->eTreeTopLims;
	// int_t *nodeList = sforest->nodeList ;


	int_t knsupc = sp_ienv_dist (3,options);
	int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);

	int **fsendx_plist = Llu->fsendx_plist;
	int_t* ilsum = Llu->ilsum;

	int* fmod = getfmod_newsolve(nlb, nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);
	int* frecv = getfrecv_newsolve(nsupers, supernodeMask, nlb, fmod, LUstruct->Llu->mod_bit, grid);
	Llu->frecv = frecv;
	int  nfrecvx = getNfrecvx_newsolve(nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);
	int  nleaf = 0;
	int_t nfrecvmod = getNfrecvmod_newsolve(&nleaf, nsupers, supernodeMask, frecv, fmod,  grid);
    // printf("igrid %5d, iam %5d, nfrecvx %5d, nfrecvmod %5d, nleaf %5d\n",myGrid,iam,nfrecvx,nfrecvmod,nleaf);

	/* factor the leaf to being the factorization*/
	for (int_t k = 0; k < nsupers && nleaf; ++k)
	{
        if(supernodeMask[k]>0){
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);
		if (myrow == krow && mycol == kcol)
		{
			/* Diagonal process */
			int_t knsupc = SuperSize (k);
			int_t lk = LBi (k, grid);
			if (frecv[lk] == 0 && fmod[lk] == 0)
			{
				double tx = SuperLU_timer_();
				fmod[lk] = -1;  /* Do not solve X[k] in the future. */
				int_t ii = X_BLK (lk);

				int_t lkj = LBj (k, grid); /* Local block number, column-wise. */
				int_t* lsub = Lrowind_bc_ptr[lkj];
				slocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				siBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, fsendx_plist, send_req, LUstruct, grid,xtrsTimer);
				nleaf--;
				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				int_t nb = lsub[0] - 1;
				int_t lptr = BC_HEADER + LB_DESCRIPTOR + knsupc;
				int_t luptr = knsupc; /* Skip diagonal block L(k,k). */

				slsum_fmod_leaf_newsolve (trf3Dpartition, lsum, x, &x[ii], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
				xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
			}
		}                       /* if diagonal process ... */
        }
	}


	while (nfrecvx || nfrecvmod)
	{
		/* While not finished. */
		/* Receive a message. */
		MPI_Status status;
		double tx = SuperLU_timer_();
		MPI_Recv (recvbuf, maxrecvsz, MPI_FLOAT,
		          MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status);
		xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
		int_t k = *recvbuf;
		xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
		tx = SuperLU_timer_();
		switch (status.MPI_TAG)
		{
		case Xk:
		{
			--nfrecvx;
			int_t lk = LBj (k, grid); /* Local block number, column-wise. */
			int_t *lsub = Lrowind_bc_ptr[lk];

			if (lsub)
			{
				int_t nb = lsub[0];
				int_t lptr = BC_HEADER;
				int_t luptr = 0;
				int_t knsupc = SuperSize (k);

				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */
				slsum_fmod_leaf_newsolve (trf3Dpartition, lsum, x, &recvbuf[XK_H], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
			}                   /* if lsub */

			break;
		}

		case LSUM:             /* Receiver must be a diagonal process */
		{
			--nfrecvmod;
			int_t lk = LBi (k, grid); /* Local block number, row-wise. */
			int_t ii = X_BLK (lk);
			int_t knsupc = SuperSize (k);
			float* tempv = &recvbuf[LSUM_H];
			for (int_t j = 0; j < nrhs; ++j)
			{
				for (int_t i = 0; i < knsupc; ++i)
					x[i + ii + j * knsupc] += tempv[i + j * knsupc];
			}

			if ((--frecv[lk]) == 0 && fmod[lk] == 0)
			{
				fmod[lk] = -1;  /* Do not solve X[k] in the future. */
				lk = LBj (k, grid); /* Local block number, column-wise. */
				int_t *lsub = Lrowind_bc_ptr[lk];
				slocalSolveXkYk(  LOWER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
				/*
				  * Send Xk to process column Pc[k].
				  */
				siBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, fsendx_plist, send_req, LUstruct, grid, xtrsTimer);
				/*
				 * Perform local block modifications.
				 */
				int_t nb = lsub[0] - 1;
				int_t lptr = BC_HEADER + LB_DESCRIPTOR + knsupc;
				int_t luptr = knsupc; /* Skip diagonal block L(k,k). */

				slsum_fmod_leaf_newsolve (trf3Dpartition, lsum, x, &x[ii], rtemp, nrhs, knsupc, k,
				                 fmod, nb, lptr, luptr, xsup, grid, Llu,
				                 send_req, stat, xtrsTimer);
			}                   /* if */

			break;
		}

		default:
		{
			// printf ("(%2d) Recv'd wrong message tag %4d\n", status.MPI_TAG);
			break;
		}

		}                       /* switch */
		xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
	}                           /* while not finished ... */
	SUPERLU_FREE (fmod);
	SUPERLU_FREE (frecv);
	double tx = SuperLU_timer_();
	for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
	{
		MPI_Status status;
		MPI_Wait (&send_req[i], &status);
	}
	Llu->SolveMsgSent = 0;
	xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
	MPI_Barrier (grid->comm);
	return 0;
}





void sForwardSolve3d_newsolve_reusepdgstrs(superlu_dist_options_t *options, int_t n,  sLUstruct_t * LUstruct,
                               sScalePermstruct_t * ScalePermstruct,
                               int*  supernodeMask, gridinfo3d_t *grid3d,
                               float * x, float * lsum,
                               int nrhs,
                               sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    int_t myGrid = grid3d->zscp.Iam;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    float alpha = 1.0;
    float beta = 0.0;
    float zero = 0.0;
    float *lusup, *dest;
    float *recvbuf, *recvbuf_on, *tempv,
            *recvbufall, *recvbuf_BC_fwd, *recvbuf0, *xin, *recvbuf_BC_gpu,*recvbuf_RD_gpu;
    float *rtemp, *rtemp_loc; /* Result of full matrix-vector multiply. */
    float *Linv; /* Inverse of diagonal block */
    float *Uinv; /* Inverse of diagonal block */
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
    float **Lnzval_bc_ptr;
    float **Linv_bc_ptr;
    float **Uinv_bc_ptr;
    float sum;
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
    int fmod_tmp;
    int  **fsendx_plist = Llu->fsendx_plist;
    int  nfrecvx_buf=0;
    int *frecv;        /* Count of lsum[lk] contributions to be received
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
    double t1, t2, t3;
    float msg_vol = 0, msg_cnt = 0;
    // #endif

    int_t msgcnt[4]; /* Count the size of the message xfer'd in each buffer:
		      *     0 : transferred in Lsub_buf[]
		      *     1 : transferred in Lval_buf[]
		      *     2 : transferred in Usub_buf[]
		      *     3 : transferred in Uval_buf[]
		      */
    int iword = sizeof (int_t);
    int dword = sizeof (float);
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
    aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int num_thread = 1;
	int_t cnt1,cnt2;
    double tx;



#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK)

#if ( PRNTlevel>=1 )

    if (get_acc_solve()) /* GPU trisolve*/
    {
    iam = grid->iam;
	if ( !iam) printf(".. GPU trisolve\n");
	fflush(stdout);
    }
#endif

	const int nwrp_block = 1; /* number of warps in each block */
	const int warp_size = 32; /* number of threads per warp*/
	gpuStream_t sid=0;
	int gid=0;
	gridinfo_t *d_grid = NULL;
	float *d_x = NULL;
	float *d_lsum = NULL;
    int  *d_fmod = NULL;
#endif


// cudaProfilerStart();
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

    // MPI_Barrier( grid->comm );
    t1_sol = SuperLU_timer_();
    t = SuperLU_timer_();


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
    CHECK_MALLOC(iam, "Enter dForwardSolve3d_newsolve_reusepdgstrs()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    /* Save the count to be altered so it can be used by
       subsequent call to PSGSTRS. */

/* skip fmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
	fmod = getfmod_newsolve(nlb, nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);
}
	int  nfrecvx = getNfrecvx_newsolve(nsupers, supernodeMask, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Lindval_loc_bc_ptr, grid);

    if ( !(frecv = int32Calloc_dist(nlb)) )
	ABORT("Calloc fails for frecv[].");
    Llu->frecv = frecv;

    if ( !(leaf_send = intMalloc_dist((CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i)) )
	ABORT("Malloc fails for leaf_send[].");
    nleaf_send=0;


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
    maxrecvsz = knsupc * nrhs + SUPERLU_MAX( XK_H, LSUM_H );
    sizelsum = (((size_t)ldalsum)*nrhs + nlb*LSUM_H);
    sizelsum = ((sizelsum + (aln_d - 1)) / aln_d) * aln_d;



/* skip rtemp on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    sizertemp=ldalsum * nrhs;
    sizertemp = ((sizertemp + (aln_d - 1)) / aln_d) * aln_d;
    if ( !(rtemp = (float*)SUPERLU_MALLOC((sizertemp*num_thread + 1) * sizeof(float))) )
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
}


    if ( !(stat_loc = (SuperLUStat_t**) SUPERLU_MALLOC(num_thread*sizeof(SuperLUStat_t*))) )
	ABORT("Malloc fails for stat_loc[].");

    for ( i=0; i<num_thread; i++) {
	stat_loc[i] = (SuperLUStat_t*)SUPERLU_MALLOC(sizeof(SuperLUStat_t));
	PStatInit(stat_loc[i]);
    }

    // /* Set up the headers in lsum[]. */
    // for (k = 0; k < nsupers; ++k) {
	// krow = PROW( k, grid );
	// if ( myrow == krow ) {
	//     lk = LBi( k, grid );   /* Local block number. */
	//     il = LSUM_BLK( lk );
	    // lsum[il - LSUM_H] = k; /* Block number prepended in the header. */
	// }
    // }

	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */

	nbtree = 0;
	for (lk=0;lk<nsupers_j;++lk){
		if(LBtree_ptr[lk].empty_==NO){
			// printf("LBtree_ptr lk %5d\n",lk);
			if(C_BcTree_IsRoot(&LBtree_ptr[lk])==NO){
				nbtree++;
				if(LBtree_ptr[lk].destCnt_>0)nfrecvx_buf++;
			}
		}
	}



	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	leafsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for leafsups.");

	nrtree = 0;
	nleaf=0;
	nfrecvmod=0;


	for (lk=0;lk<nsupers_j;++lk){
		if(LBtree_ptr[lk].empty_==NO){
            xtrsTimer->trsDataSendXY  += LBtree_ptr[lk].msgSize_*nrhs+XK_H;
		}
    }
	for (lk=0;lk<nsupers_i;++lk){
		if(LRtree_ptr[lk].empty_==NO){
            xtrsTimer->trsDataSendXY  += LRtree_ptr[lk].msgSize_*nrhs+LSUM_H;
		}
    }


    /* skip fmod,leafsups,nleaf on CPU if using GPU solve*/
    if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
        if(procs==1){
            for (lk=0;lk<nsupers_i;++lk){
                gb = myrow+lk*grid->nprow;  /* not sure */
                if(gb<nsupers){
                        if (fmod[lk*aln_i]==0 && supernodeMask[gb]){
                                leafsups[nleaf]=gb;
                                ++nleaf;
                        }
                }
            }
        }else{
            for (lk=0;lk<nsupers_i;++lk){
                if(LRtree_ptr[lk].empty_==NO){
                        nrtree++;
                        // RdTree_allocateRequest(LRtree_ptr[lk],'s');
                        frecv[lk] = LRtree_ptr[lk].destCnt_;
                        nfrecvmod += frecv[lk];
                }else{
                        gb = myrow+lk*grid->nprow;  /* not sure */
                        if(gb<nsupers){
                                kcol = PCOL( gb, grid );
                                if(mycol==kcol) { /* Diagonal process */
                                    /* skip fmod,leafsups,nleaf on CPU if using GPU solve*/
                                    if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
                                        if (fmod[lk*aln_i]==0 && supernodeMask[gb]){
                                                leafsups[nleaf]=gb;
                                                ++nleaf;
                                        }
                                    }
                                }
                        }
                }
            }
        }
    }else{
        if(procs>1){
            for (lk=0;lk<nsupers_i;++lk){
                if(LRtree_ptr[lk].empty_==NO){
                    nrtree++;
                    // RdTree_allocateRequest(LRtree_ptr[lk],'s');
                    gb = myrow+lk*grid->nprow;  /* not sure */
                    if (supernodeMask[gb]==1){
                        frecv[lk] = LRtree_ptr[lk].destCnt_;
                        nfrecvmod += frecv[lk];
                    }
                }
            }
        }
    }

/* skip fmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
	for (i = 0; i < nlb; ++i) fmod[i*aln_i] += frecv[i];
}
	if ( !(recvbuf_BC_fwd = (float*)SUPERLU_MALLOC(maxrecvsz*(nfrecvx+1) * sizeof(float))) )  // this needs to be optimized for 1D row mapping
		ABORT("Malloc fails for recvbuf_BC_fwd[].");
	nfrecvx_buf=0;

	log_memory(nlb*aln_i*iword+nlb*iword+(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*2.0*iword+ nsupers_i*iword + sizelsum*num_thread * dword + (ldalsum * nrhs + nlb * XK_H) *dword + (sizertemp*num_thread + 1)*dword+maxrecvsz*(nfrecvx+1)*dword, stat);	//account for fmod, frecv, leaf_send, root_send, leafsups, recvbuf_BC_fwd	, lsum, x, rtemp


#if ( DEBUGlevel>=2 )
	printf("(%2d) nfrecvx %4d,  nfrecvmod %4d,  nleaf %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nfrecvx, nfrecvmod, nleaf, nbtree, nrtree);
	fflush(stdout);
#endif

// #if ( PRNTlevel>=1 )
#if 0
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Grid %3d: Setup L-solve time\t%8.4f\n", myGrid, t);
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

	// ii = X_BLK( 0 );
	// knsupc = SuperSize( 0 );
	// for (i=0 ; i<knsupc*nrhs ; i++){
	    // printf("x_l: %f\n",x[ii+i]);
	// fflush(stdout);
	// }


#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK)

    if (get_acc_solve()) /* GPU trisolve*/
    {
// #if 0 /* CPU trisolve*/

// #if HAVE_CUDA
// cudaProfilerStart();
// #elif defined(HAVE_HIP)
// roctracer_mark("before HIP LaunchKernel");
// roctxMark("before hipLaunchKernel");
// roctxRangePush("hipLaunchKernel");
// #endif

#if ( PROFlevel>=1 )
    t = SuperLU_timer_();
#endif

    d_fmod=SOLVEstruct->d_fmod;
    d_lsum=SOLVEstruct->d_lsum;
	d_x=SOLVEstruct->d_x;
	d_grid=Llu->d_grid;

	checkGPU(gpuMemcpy(d_fmod, SOLVEstruct->d_fmod_save, nlb * sizeof(int), gpuMemcpyDeviceToDevice));
    checkGPU(gpuMemcpy(d_lsum, SOLVEstruct->d_lsum_save, sizelsum * sizeof(float), gpuMemcpyDeviceToDevice));
	checkGPU(gpuMemcpy(d_x, x, (ldalsum * nrhs + nlb * XK_H) * sizeof(float), gpuMemcpyHostToDevice));

	k = CEILING( nsupers, grid->npcol);/* Number of local block columns divided by #warps per block used as number of thread blocks*/
	knsupc = sp_ienv_dist(3, options);

    if(procs>1){ /* only nvshmem needs the following*/
    #ifdef HAVE_NVSHMEM
    checkGPU(gpuMemcpy(d_status, mystatus, k * sizeof(int), gpuMemcpyHostToDevice));
	checkGPU(gpuMemcpy(d_statusmod, mystatusmod, 2* nlb * sizeof(int), gpuMemcpyHostToDevice));
	//for(int i=0;i<2*nlb;i++) printf("(%d),mystatusmod[%d]=%d\n",iam,i,mystatusmod[i]);
	checkGPU(gpuMemset(flag_rd_q, 0, RDMA_FLAG_SIZE * nlb * 2 * sizeof(int)));
    checkGPU(gpuMemset(flag_bc_q, 0, RDMA_FLAG_SIZE * (k+1)  * sizeof(int)));
	checkGPU(gpuMemset(sready_x, 0, maxrecvsz*CEILING( nsupers, grid->npcol) * sizeof(float)));
    checkGPU(gpuMemset(sready_lsum, 0, 2*maxrecvsz*CEILING( nsupers, grid->nprow) * sizeof(float)));
    checkGPU(gpuMemset(d_msgnum, 0, h_nfrecv[1] * sizeof(int)));
	//printf("2-(%d) maxrecvsz=%d,sready_x=%d, sready_lsum=%d,RDMA_FLAG_SIZE=%d,k=%d,nlb=%d\n",iam,maxrecvsz,maxrecvsz*CEILING( nsupers, grid->npcol),2*maxrecvsz*CEILING( nsupers, grid->nprow),RDMA_FLAG_SIZE,k,nlb);
	//fflush(stdout);
    // MUST have this barrier, otherwise the code hang.
	MPI_Barrier( grid->comm );
    #endif
    }

	// k -> Llu->nbcol_masked ???
    int nblock_loc;
    if(procs==1){
        nblock_loc=Llu->nbcol_masked;
    }else{
        nblock_loc=k;
    }
	slsum_fmod_inv_gpu_wrap(nblock_loc,nlb,DIM_X,DIM_Y,d_lsum,d_x,nrhs,knsupc,nsupers,d_fmod,Llu->d_LBtree_ptr,Llu->d_LRtree_ptr,Llu->d_ilsum,Llu->d_Lrowind_bc_dat, Llu->d_Lrowind_bc_offset, Llu->d_Lnzval_bc_dat, Llu->d_Lnzval_bc_offset, Llu->d_Linv_bc_dat, Llu->d_Linv_bc_offset, Llu->d_Lindval_loc_bc_dat, Llu->d_Lindval_loc_bc_offset,Llu->d_xsup,Llu->d_bcols_masked, d_grid,
                         maxrecvsz,
	                        flag_bc_q, flag_rd_q, sready_x, sready_lsum, my_flag_bc, my_flag_rd, d_nfrecv, h_nfrecv,
	                        d_status,d_colnum,d_mynum, d_mymaskstart,d_mymasklength,
	                        d_nfrecvmod,d_statusmod,d_colnummod,d_mynummod,d_mymaskstartmod,d_mymasklengthmod,d_recv_cnt,d_msgnum,d_flag_mod,procs);
	checkGPU(gpuMemcpy(x, d_x, (ldalsum * nrhs + nlb * XK_H) * sizeof(float), gpuMemcpyDeviceToHost));


#if ( PROFlevel>=1 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Grid %3d: around L kernel time\t%8.4f\n", myGrid, t);
#endif

	stat_loc[0]->ops[SOLVE]+=Llu->Lnzval_bc_cnt*nrhs*2; // YL: this is a rough estimate

    } else
    
#endif /* match #if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK) */
    { /* CPU trisolve */

tx = SuperLU_timer_();

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
					SGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
					sgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
					sgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
							&alpha, Linv, &knsupc, &x[ii],
							&knsupc, &beta, rtemp_loc, &knsupc );
#endif

					for (i=0 ; i<knsupc*nrhs ; i++){
				        x[ii+i] = rtemp_loc[i];
					}
							// printf("\n");
							// printf("k: %5d\n",k);
					// for (i=0 ; i<knsupc*nrhs ; i++){
                        // printf("x_l: %f\n",x[ii+i]);
					// fflush(stdout);
					// }


#if ( PROFlevel>=1 )
					TOC(t2, t1);
					stat_loc[thread_id]->utime[SOL_TRSM] += t2;

#endif

			        stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc - 1) * nrhs;



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
						// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],'s');
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
   		    STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
				lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
		    strsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
				lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
 		    strsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
					lusup, &nsupr, &x[ii], &knsupc);
#endif

#if ( PROFlevel>=1 )
		    TOC(t2, t1);
		    stat_loc[thread_id]->utime[SOL_TRSM] += t2;

#endif

            stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc - 1) * nrhs;

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
	    } /* end else ... diagonal is not invedted */
	  }
	} /* end omp parallel */

	jj=0;

#if ( DEBUGlevel>=2 )
	printf("(%2d) end solving nleaf %4d\n", iam, nleaf);
	fflush(stdout);
#endif

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
#else
	{
#endif

#ifdef _OPENMP
#pragma omp master
#endif
		    {

#ifdef _OPENMP
#if defined __GNUC__  && !defined __NVCOMPILER
#pragma	omp taskloop private (k,ii,lk,thread_id) num_tasks(num_thread*8) nogroup
#endif
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
				slsum_fmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, fmod, xsup, grid, Llu, stat_loc, leaf_send, &nleaf_send,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);
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
					// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'s')*nrhs+XK_H,'s');
					C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);

				}else{ // this is a reduce forwarding
					lk = -lk - 1;
					il = LSUM_BLK( lk );
					// RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(LRtree_ptr[lk],'s')*nrhs+LSUM_H,'s');
					C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
				}
			}
        xtrsTimer->tfs_compute += SuperLU_timer_() - tx;


#ifdef USE_VTUNE
	__itt_pause(); // stop VTune
	__SSC_MARK(0x222); // stop SDE tracing
#endif

			/* -----------------------------------------------------------
			   Compute the internal nodes asynchronously by all processes.
			   ----------------------------------------------------------- */

#ifdef _OPENMP
#pragma omp parallel default (shared)
			{
	int thread_id = omp_get_thread_num();
#else
	{
	thread_id=0;
#endif

#ifdef _OPENMP
#pragma omp master
#endif
				{
					for ( nfrecv =0; nfrecv<nfrecvx+nfrecvmod;nfrecv++) { /* While not finished. */
						thread_id = 0;
#if ( PROFlevel>=1 )
						TIC(t1);
						// msgcnt[1] = maxrecvsz;
#endif

						recvbuf0 = &recvbuf_BC_fwd[nfrecvx_buf*maxrecvsz];
                        double tx = SuperLU_timer_();
						/* Receive a message. */
						MPI_Recv( recvbuf0, maxrecvsz, MPI_FLOAT,
								MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status );
                        xtrsTimer->tfs_comm += SuperLU_timer_() - tx;

						// MPI_Irecv(recvbuf0,maxrecvsz,MPI_FLOAT,MPI_ANY_SOURCE,MPI_ANY_TAG,grid->comm,&req);
						// ready=0;
						// while(ready==0){
						// MPI_Test(&req,&ready,&status);
						// #pragma omp taskyield
						// }

#if ( PROFlevel>=1 )
						TOC(t2, t1);
						stat_loc[thread_id]->utime[SOL_COMM] += t2;

						msg_cnt += 1;
						msg_vol += maxrecvsz * dword;
#endif

						{
                            double tx = SuperLU_timer_();
			                k = *recvbuf0;

#if ( DEBUGlevel>=2 )
							printf("(%2d) Recv'd block %d, tag %2d\n", iam, k, status.MPI_TAG);
#endif

							if(status.MPI_TAG==BC_L){
                                xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
								// --nfrecvx;
								nfrecvx_buf++;
								{
									lk = LBj( k, grid );    /* local block number */

									if(LBtree_ptr[lk].destCnt_>0){

										// BcTree_forwardMessageSimple(LBtree_ptr[lk],recvbuf0,BcTree_GetMsgSize(LBtree_ptr[lk],'s')*nrhs+XK_H,'s');
										C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], recvbuf0, LBtree_ptr[lk].msgSize_*nrhs+XK_H);
										// nfrecvx_buf++;
									}

									/*
									 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
									 */

									lk = LBj( k, grid ); /* Local block number, column-wise. */
									lsub = Lrowind_bc_ptr[lk];
									lusup = Lnzval_bc_ptr[lk];
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

										slsum_fmod_inv_master(lsum, x, xin, rtemp, nrhs, knsupc, k,
												fmod, nb, xsup, grid, Llu,
												stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);

									} /* if lsub */
								}

							}else if(status.MPI_TAG==RD_L){
                                xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + LSUM_H;
								// --nfrecvmod;
								lk = LBi( k, grid ); /* Local block number, row-wise. */

								knsupc = SuperSize( k );
								tempv = &recvbuf0[LSUM_H];
								il = LSUM_BLK( lk );
								RHS_ITERATE(j) {
									for (i = 0; i < knsupc; ++i)
					                    lsum[i + il + j*knsupc + thread_id*sizelsum] += tempv[i + j*knsupc];

								}

								// #ifdef _OPENMP
								// #pragma omp atomic capture
								// #endif
								fmod_tmp=--fmod[lk*aln_i];
								{
									thread_id = 0;
									rtemp_loc = &rtemp[sizertemp* thread_id];
									if ( fmod_tmp==0 ) {
										if(C_RdTree_IsRoot(&LRtree_ptr[lk])==YES){
											// ii = X_BLK( lk );
											knsupc = SuperSize( k );
											for (ii=1;ii<num_thread;ii++)
												for (jj=0;jj<knsupc*nrhs;jj++)
						                            lsum[il + jj ] += lsum[il + jj + ii*sizelsum];


											ii = X_BLK( lk );
											RHS_ITERATE(j)
												for (i = 0; i < knsupc; ++i)
					                                x[i + ii + j*knsupc] += lsum[i + il + j*knsupc];

											// fmod[lk] = -1; /* Do not solve X[k] in the future. */
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
												SGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
												sgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
												sgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
														&alpha, Linv, &knsupc, &x[ii],
														&knsupc, &beta, rtemp_loc, &knsupc );
#endif
												for (i=0 ; i<knsupc*nrhs ; i++){
				                                    x[ii+i] = rtemp_loc[i];
												}
											}
											else{
#ifdef _CRAY
												STRSM(ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
												strsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
												strsm_("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
														lusup, &nsupr, &x[ii], &knsupc);
#endif
											}

#if ( PROFlevel>=1 )
											TOC(t2, t1);
											stat_loc[thread_id]->utime[SOL_TRSM] += t2;
#endif

			                                stat_loc[thread_id]->ops[SOLVE] += knsupc * (knsupc - 1) * nrhs;

#if ( DEBUGlevel>=2 )
											printf("(%2d) Solve X[%2d]\n", iam, k);
#endif

											/*
											 * Send Xk to process column Pc[k].
											 */
											if(LBtree_ptr[lk].empty_==NO){
												// BcTree_forwardMessageSimple(LBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(LBtree_ptr[lk],'s')*nrhs+XK_H,'s');
												C_BcTree_forwardMessageSimple(&LBtree_ptr[lk], &x[ii - XK_H], LBtree_ptr[lk].msgSize_*nrhs+XK_H);
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
												slsum_fmod_inv_master(lsum, x, xin, rtemp, nrhs, knsupc, k,
														fmod, nb, xsup, grid, Llu,
														stat_loc,sizelsum,sizertemp,0,maxsuper,thread_id,num_thread);
											} /* if lsub */
											// }

									}else{

										il = LSUM_BLK( lk );
										knsupc = SuperSize( k );

										for (ii=1;ii<num_thread;ii++)
											for (jj=0;jj<knsupc*nrhs;jj++)
                                                lsum[il + jj ] += lsum[il + jj + ii*sizelsum];
										// RdTree_forwardMessageSimple(LRtree_ptr[lk],&lsum[il-LSUM_H],RdTree_GetMsgSize(LRtree_ptr[lk],'s')*nrhs+LSUM_H,'s');
										C_RdTree_forwardMessageSimple(&LRtree_ptr[lk],&lsum[il - LSUM_H ],LRtree_ptr[lk].msgSize_*nrhs+LSUM_H);
									}

								}

							}
						} /* check Tag */
					    xtrsTimer->tfs_compute += SuperLU_timer_() - tx;
                    }

				} /* while not finished ... */

			}
		} // end of parallel
	}  /* end CPU trisolve */


// #if ( PRNTlevel>=1 )
#if 0
		t = SuperLU_timer_() - t;
		stat->utime[SOL_TOT] += t;
		// if ( !iam ) {
		// 	printf(".. L-solve time\t%8.4f\n", t);
		// 	fflush(stdout);
		// }


		MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE,
				MPI_MAX, 0, grid->comm);
		if ( !iam ) {
			printf(".. Grid %3d: L-solve time (MAX) \t%8.4f\n", myGrid, tmax);
			fflush(stdout);
		}


		t = SuperLU_timer_();
#endif


// stat->utime[SOLVE] = SuperLU_timer_() - t1_sol;

#if ( DEBUGlevel==2 )
		{
		  printf("(%d) .. After L-solve: y =\n", iam); fflush(stdout);
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
/* skip fmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
		SUPERLU_FREE(fmod);
}
		SUPERLU_FREE(frecv);
		SUPERLU_FREE(leaf_send);
		SUPERLU_FREE(leafsups);
		SUPERLU_FREE(recvbuf_BC_fwd);
		log_memory(-nlb*aln_i*iword-nlb*iword-(CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*iword- nsupers_i*iword -maxrecvsz*(nfrecvx+1)*dword, stat);	//account for fmod, frecv, leaf_send, leafsups, recvbuf_BC_fwd

		for (lk=0;lk<nsupers_j;++lk){
			if(LBtree_ptr[lk].empty_==NO){
				// if(BcTree_IsRoot(LBtree_ptr[lk],'s')==YES){
				// BcTree_waitSendRequest(LBtree_ptr[lk],'s');
				C_BcTree_waitSendRequest(&LBtree_ptr[lk]);
				// }
				// deallocate requests here
			}
		}

		for (lk=0;lk<nsupers_i;++lk){
			if(LRtree_ptr[lk].empty_==NO){
				C_RdTree_waitSendRequest(&LRtree_ptr[lk]);
				// deallocate requests here
			}
		}
		// MPI_Barrier( grid->comm );

#if ( VAMPIR>=1 )
		VT_traceoff();
		VT_finalize();
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

/* skip rtemp on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
		SUPERLU_FREE(rtemp);
}
		// SUPERLU_FREE(lsum);
		// SUPERLU_FREE(x);

		// MPI_Barrier( grid->comm );


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
				printf ("\tPSGSTRS comm stat:"
						"\tAvg\tMax\t\tAvg\tMax\n"
						"\t\t\tCount:\t%.0f\t%.0f\tVol(MB)\t%.2f\t%.2f\n",
						msg_cnt_sum / Pr / Pc, msg_cnt_max,
						msg_vol_sum / Pr / Pc * 1e-6, msg_vol_max * 1e-6);
			}
		}
#endif

    stat->utime[SOLVE] = SuperLU_timer_() - t1_sol;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit dForwardSolve3d_newsolve_reusepdgstrs()");
#endif


#if ( PRNTlevel>=2 )
	    float for_lu, total, max, avg, temp;
		superlu_dist_mem_usage_t num_mem_usage;

	    sQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);
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

// cudaProfilerStop();

    return;
} /* sForwardSolve3d_newsolve_reusepdgstrs */







void slsum_fmod_leaf_newsolve (
    strf3Dpartition_t*  trf3Dpartition,
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
    SuperLUStat_t *stat,xtrsTimer_t *xtrsTimer)

{
    float alpha = 1.0;
    float beta = 0.0;
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

	for (lb = 0; lb < nlb; ++lb)
	{
		ik = lsub[lptr]; /* Global block number, row-wise. */

        if (trf3Dpartition->supernodeMask[ik])
        {

		nbrow = lsub[lptr + 1];
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
		for (i = 0; i < nbrow; ++i)
		{
			irow = lsub[lptr++] - rel; /* Relative row. */
			RHS_ITERATE(j)
                dest[irow + j * iknsupc] -= rtemp[i + j * nbrow];
		}
		luptr += nbrow;

#if ( PROFlevel>=1 )
		TOC(t2, t1);
		stat->utime[SOL_GEMM] += t2;
#endif


		if ( (--fmod[lk]) == 0  )   /* Local accumulation done. */
		{
            ikcol = PCOL( ik, grid );
            p = PNUM( myrow, ikcol, grid );
            if ( iam != p )
            {
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
                xtrsTimer->trsDataSendXY += iknsupc * nrhs + LSUM_H;
            }
            else     /* Diagonal process: X[i] += lsum[i]. */
            {
                ii = X_BLK( lk );
                RHS_ITERATE(j)
                for (i = 0; i < iknsupc; ++i)
                    x[i + ii + j * iknsupc] += lsum[i + il + j * iknsupc];
                if ( frecv[lk] == 0 )   /* Becomes a leaf node. */
                {
                    fmod[lk] = -1; /* Do not solve X[k] in the future. */


                    lk = LBj( ik, grid );/* Local block number, column-wise. */
                    lsub1 = Llu->Lrowind_bc_ptr[lk];
                    lusup1 = Llu->Lnzval_bc_ptr[lk];
                    nsupr1 = lsub1[1];
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


                    stat->ops[SOLVE] += iknsupc * (iknsupc - 1) * nrhs;

                    /*
                        * Send Xk to process column Pc[k].
                        */
                    for (p = 0; p < grid->nprow; ++p)
                    {
                        if ( fsendx_plist[lk][p] != SLU_EMPTY )
                        {
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

                        }
                    }
                    xtrsTimer->trsDataSendXY += iknsupc * nrhs + XK_H;
                    /*
                        * Perform local block modifications.
                        */
                    nlb1 = lsub1[0] - 1;
                    lptr1 = BC_HEADER + LB_DESCRIPTOR + iknsupc;
                    luptr1 = iknsupc; /* Skip diagonal block L(I,I). */

                    slsum_fmod_leaf_newsolve(trf3Dpartition,
                                    lsum, x, &x[ii], rtemp, nrhs, iknsupc, ik,
                                    fmod, nlb1, lptr1, luptr1, xsup,
                                    grid, Llu, send_req, stat,xtrsTimer);
                } /* if frecv[lk] == 0 */
            } /* if iam == p */
		}/* if fmod[lk] == 0 */

	    }
    }/* for lb ... */
} /* sLSUM_FMOD_LEAF */




int_t slasum_bmod_Tree(int_t  pTree, int_t cTree, float *lsum, float *x,
                       sxT_struct *xT_s,
                       int    nrhs, slsumBmod_buff_t* lbmod_buf,
                       sLUstruct_t * LUstruct,
                       strf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    sForest_t* pforest = trf3Dpartition->sForests[pTree];
    sForest_t* cforest = trf3Dpartition->sForests[cTree];
    if (!pforest || !cforest) return 0;

    int_t nnodes = pforest->nNodes;
    if (nnodes < 1) return 0;
    int_t* nodeList =  pforest->nodeList;
    int_t iam = grid->iam;
    int_t mycol = MYCOL( iam, grid );
    for (int_t k0 = 0; k0 < nnodes; ++k0)
    {
        /* code */
        int_t k = nodeList[k0];
        int_t kcol = PCOL (k, grid);
        if (mycol == kcol)
        {
            /* code */
            slsumForestBsolve(k, cTree, lsum, x, xT_s, nrhs, lbmod_buf,
                             LUstruct, trf3Dpartition, grid3d, stat);
        }
    }
    return 0;
}


int_t sinitLsumBmod_buff(int_t ns, int nrhs, slsumBmod_buff_t* lbmod_buf)
{
    lbmod_buf->tX = SUPERLU_MALLOC(ns * nrhs * sizeof(float));
    lbmod_buf->tU = SUPERLU_MALLOC(ns * ns * sizeof(float));
    lbmod_buf->indCols = SUPERLU_MALLOC(ns * sizeof(int_t));
    return 0;
}

int_t sfreeLsumBmod_buff(slsumBmod_buff_t* lbmod_buf)
{
    SUPERLU_FREE(lbmod_buf->tX);
    SUPERLU_FREE(lbmod_buf->tU);
    SUPERLU_FREE(lbmod_buf->indCols);
    return 0;
}


int spackUblock(int ldu, int_t* indCols,
                 int_t knsupc, int_t iklrow,  int_t* usub,
                 float* tempu, float* uval )
{
    float zero = 0.0;
    int ncols = 0;
    for (int_t jj = 0; jj < knsupc; ++jj)
    {

        int_t segsize = iklrow - usub[jj];
        if ( segsize )
        {
            int_t lead_zero = ldu - segsize;
            for (int_t i = 0; i < lead_zero; ++i) tempu[i] = zero;
            tempu += lead_zero;
            for (int_t i = 0; i < segsize; ++i)
            {
                tempu[i] = uval[i];
            }

            uval += segsize;
            tempu += segsize;
            indCols[ncols] = jj;
            ncols++;
        }

    } /* for jj ... */

    return ncols;
}


int_t spackXbmod( int_t knsupc, int_t ncols, int_t nrhs, int_t* indCols, float* xk, float* tempx)
{

    for (int_t j = 0; j < nrhs; ++j)
    {
        float* dest = &tempx[j * ncols];
        float* y = &xk[j * knsupc];

        for (int_t jj = 0; jj < ncols; ++jj)
        {
            dest[jj] = y[indCols[jj]];
        } /* for jj ... */
    }

    return 0;
}

int_t slsumBmod(int_t gik, int_t gjk, int nrhs, slsumBmod_buff_t* lbmod_buf,
               int_t* usub,  float* uval,
               float* xk, float* lsum, int_t* xsup, SuperLUStat_t * stat)
{

    int_t* indCols = lbmod_buf->indCols;
    float* tempu = lbmod_buf->tU;
    float* tempx = lbmod_buf->tX;
    int iknsupc = (int)SuperSize( gik );
    int_t knsupc = SuperSize( gjk );
    int_t iklrow = FstBlockC( gik + 1 );
    int ldu = getldu(knsupc, iklrow,
                       usub // use &usub[i]
                      );

    int ncols = spackUblock(ldu, indCols, knsupc, iklrow, usub,
                             tempu, uval );

    float alpha = -1.0;
    float beta = 1.0;
    float* X;

    if (ncols < knsupc)
    {
        /* code */
        spackXbmod(knsupc, ncols, nrhs, indCols, xk, tempx);
        X = tempx;
    }
    else
    {
        X = xk;
    }

    float* V = &lsum[iknsupc - ldu];


#if defined (USE_VENDOR_BLAS)
	sgemm_("N", "N", &ldu, &nrhs, &ncols, &alpha,
	tempu, &ldu,
	X, &ncols, &beta, V, &iknsupc, 1, 1);
#else
	sgemm_("N", "N", &ldu, &nrhs, &ncols, &alpha,
	tempu, &ldu,
	X, &ncols, &beta, V, &iknsupc);
#endif





    stat->ops[SOLVE] += 2 * ldu * nrhs * ncols;
    return 0;
}

int_t slsumForestBsolve(int_t k, int_t treeId,
                       float *lsum, float *x,  sxT_struct *xT_s, int    nrhs, slsumBmod_buff_t* lbmod_buf,
                       sLUstruct_t * LUstruct,
                       strf3Dpartition_t*  trf3Dpartition,
                       gridinfo3d_t* grid3d, SuperLUStat_t * stat)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t iam = grid->iam;
    int_t myrow = MYROW( iam, grid );
    int_t knsupc = SuperSize( k );
    float *xT = xT_s->xT;
    int_t *ilsumT = xT_s->ilsumT;
    int_t ldaspaT = xT_s->ldaspaT;

    int_t lk = LBj( k, grid ); /* Local block number, column-wise. */
    int_t nub = Urbs[lk];      /* Number of U blocks in block column lk */
    int_t *ilsum = Llu->ilsum;
    int_t ii = XT_BLK (lk);
    float* xk = &xT[ii];
    for (int_t ub = 0; ub < nub; ++ub)
    {
        int_t ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
        int_t gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */

        if (trf3Dpartition->supernode2treeMap[gik] == treeId)
        {
            int_t* usub = Llu->Ufstnz_br_ptr[ik];
            float* uval = Llu->Unzval_br_ptr[ik];
            int_t i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
            i += UB_DESCRIPTOR;
            int_t il = LSUM_BLK( ik );
#if 1
            slsumBmod(gik, k, nrhs, lbmod_buf,
                     &usub[i], &uval[Ucb_valptr[lk][ub]], xk,
                     &lsum[il], xsup, stat);
#else
            int_t iknsupc = SuperSize( gik );
            int_t ikfrow = FstBlockC( gik );
            int_t iklrow = FstBlockC( gik + 1 );

            for (int_t j = 0; j < nrhs; ++j)
            {
                float* dest = &lsum[il + j * iknsupc];
                float* y = &xk[j * knsupc];
                int_t uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
                for (int_t jj = 0; jj < knsupc; ++jj)
                {
                    int_t fnz = usub[i + jj];
                    if ( fnz < iklrow )
                    {
                        /* Nonzero segment. */
                        /* AXPY */
                        for (int_t irow = fnz; irow < iklrow; ++irow)
                            dest[irow - ikfrow] -= uval[uptr++] * y[jj];
                        stat->ops[SOLVE] += 2 * (iklrow - fnz);
                    }
                } /* for jj ... */
            } /*for (int_t j = 0;*/
#endif

        }

    }
    return 0;
}


int_t  sbCastXk2Pck  (int_t k, sxT_struct *xT_s, int nrhs,
                     sLUstruct_t * LUstruct, gridinfo_t * grid, xtrsTimer_t *xtrsTimer)
{
    /*
     * Send Xk to process column Pc[k].
     */

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t *ilsum = Llu->ilsum;
    int_t* xsup = Glu_persist->xsup;

    float *xT = xT_s->xT;
    int_t *ilsumT = xT_s->ilsumT;
    int_t ldaspaT = xT_s->ldaspaT;

    int_t lk = LBj (k, grid);
    int_t ii = XT_BLK (lk);
    float* xk = &xT[ii];
    superlu_scope_t *scp = &grid->cscp;
    int_t knsupc = SuperSize (k);
    int_t krow = PROW (k, grid);
    MPI_Bcast( xk, knsupc * nrhs, MPI_FLOAT, krow,
               scp->comm);

    xtrsTimer->trsDataRecvXY  += knsupc * nrhs;
    xtrsTimer->trsDataSendXY  += knsupc * nrhs;
    return 0;
}

int_t  slsumReducePrK (int_t k, float*x, float* lsum, float* recvbuf, int nrhs,
                      sLUstruct_t * LUstruct, gridinfo_t * grid, xtrsTimer_t *xtrsTimer)
{
    /*
     * Send Xk to process column Pc[k].
     */

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t *ilsum = Llu->ilsum;
    int_t* xsup = Glu_persist->xsup;

    int_t knsupc = SuperSize (k);
    int_t lk = LBi (k, grid);
    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);

    int_t ii = LSUM_BLK (lk);
    float* lsum_k = &lsum[ii];
    superlu_scope_t *scp = &grid->rscp;
    MPI_Reduce( lsum_k, recvbuf, knsupc * nrhs,
                MPI_FLOAT, MPI_SUM, kcol, scp->comm);

    xtrsTimer->trsDataRecvXY  += knsupc * nrhs;
    xtrsTimer->trsDataSendXY  += knsupc * nrhs;

    if (mycol == kcol)
    {
        int_t ii = X_BLK( lk );
        float* dest = &x[ii];
        float* tempv = recvbuf;
        for (int_t j = 0; j < nrhs; ++j)
        {
            for (int_t i = 0; i < knsupc; ++i)
                x[i + ii + j * knsupc] += tempv[i + j * knsupc];
        }
    }

    return 0;
}

int_t snonLeafForestBackSolve3d( int_t treeId,  sLUstruct_t * LUstruct,
                                sScalePermstruct_t * ScalePermstruct,
                                strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                                float * x, float * lsum,
                                sxT_struct *xT_s,
                                float * recvbuf,
                                MPI_Request * send_req,
                                int nrhs, slsumBmod_buff_t* lbmod_buf,
                                sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    sForest_t** sForests = trf3Dpartition->sForests;

    sForest_t* sforest = sForests[treeId];
    if (!sforest)
    {
        /* code */
        return 0;
    }
    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1) return 1;
    int_t *perm_c_supno = sforest->nodeList ;
    gridinfo_t * grid = &(grid3d->grid2d);

    sLocalLU_t *Llu = LUstruct->Llu;
    int_t *ilsum = Llu->ilsum;

    int_t* xsup =  LUstruct->Glu_persist->xsup;

    float *xT = xT_s->xT;
    int_t *ilsumT = xT_s->ilsumT;
    int_t ldaspaT = xT_s->ldaspaT;

    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);

    for (int_t k0 = nnodes - 1; k0 >= 0; --k0)
    {
        int_t k = perm_c_supno[k0];
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
        // printf("doing %d \n", k);
        /**
         * Pkk(Yk) = sumOver_PrK (Yk)
         */
        if (myrow == krow )
        {
            double tx = SuperLU_timer_();
            slsumReducePrK(k, x, lsum, recvbuf, nrhs, LUstruct, grid, xtrsTimer);
            xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
        }

        if (mycol == kcol )
        {
            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            int_t ii = X_BLK (lk);
            if (myrow == krow )
            {
                double tx = SuperLU_timer_();
                /* Diagonal process. */
                slocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                int_t lkj = LBj (k, grid);
                int_t jj = XT_BLK (lkj);
                int_t knsupc = SuperSize(k);
                memcpy(&xT[jj], &x[ii], knsupc * nrhs * sizeof(float) );
                xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
            }                       /* if diagonal process ... */

            /*
             * Send Xk to process column Pc[k].
             */
            double tx = SuperLU_timer_();
            sbCastXk2Pck( k,  xT_s,  nrhs, LUstruct, grid,xtrsTimer);
            xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
            /*
             * Perform local block modifications: lsum[i] -= U_i,k * X[k]
             * where i is in current sforest
             */
            tx = SuperLU_timer_();
            slsumForestBsolve(k, treeId, lsum, x, xT_s, nrhs, lbmod_buf,
                             LUstruct, trf3Dpartition, grid3d, stat);
            xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
        }
    }                           /* for k ... */
    return 0;
}



int_t sleafForestBackSolve3d(superlu_dist_options_t *options, int_t treeId, int_t n,  sLUstruct_t * LUstruct,
                            sScalePermstruct_t * ScalePermstruct,
                            strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                            float * x, float * lsum, float * recvbuf,
                            MPI_Request * send_req,
                            int nrhs, slsumBmod_buff_t* lbmod_buf,
                            sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{

    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    sForest_t* sforest = trf3Dpartition->sForests[treeId];
    if (!sforest) return 0;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    // float** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t *ilsum = Llu->ilsum;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t myGrid = grid3d->zscp.Iam;
    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t knsupc = sp_ienv_dist (3,options);
    int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);

    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1) return 1;
    int_t *perm_c_supno = sforest->nodeList ;

    int **bsendx_plist = Llu->bsendx_plist;
    int_t Pr = grid->nprow;
    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t nlb = CEILING (nsupers, Pr);
    int* bmod =  getBmod3d(treeId, nlb, sforest, xsup, Llu->Ufstnz_br_ptr, trf3Dpartition->supernode2treeMap, grid);
    // for (int_t l=0;l<nsupers;l++)
        // printf("iam %5d lk %5d bmod %5d \n",grid->iam,l,bmod[l]);
    int* brecv = getBrecvTree(nlb, sforest, bmod, grid);
    Llu->brecv = brecv;

    int_t nbrecvmod = 0;
    int nroot = getNrootUsolveTree(&nbrecvmod, sforest, brecv, bmod, grid);
    int nbrecvx = getNbrecvX(sforest, Urbs, grid);
    // printf("igrid %5d, iam %5d, nbrecvx %5d, nbrecvmod %5d, nroot %5d\n",myGrid,iam,nbrecvx,nbrecvmod,nroot);

    /*before starting the solve; intialize the 3d lsum*/

    for (int_t k0 = nnodes - 1; k0 >= 0 ; --k0)
    {
        int_t k = perm_c_supno[k0];
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
        if (myrow == krow)
        {
            /* Diagonal process. */

            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            if (bmod[lk] == 0)
            {
                /* code */
                int_t il = LSUM_BLK( lk );
                int_t knsupc = SuperSize(k);
                if (mycol != kcol)
                {
                    /* code */
                    int_t p = PNUM( myrow, kcol, grid );
                    MPI_Isend( &lsum[il - LSUM_H], knsupc * nrhs + LSUM_H,
                               MPI_FLOAT, p, LSUM, grid->comm,
                               &send_req[Llu->SolveMsgSent++] );
                    xtrsTimer->trsDataSendXY += knsupc * nrhs + LSUM_H;
                }
                else
                {
                    int_t ii = X_BLK( lk );
                    float* dest = &x[ii];
                    for (int_t j = 0; j < nrhs; ++j)
                        for (int_t i = 0; i < knsupc; ++i)
                            dest[i + j * knsupc] += lsum[i + il + j * knsupc];

                    if (brecv[lk] == 0 )
                    {
                        double tx = SuperLU_timer_();
                        bmod[lk] = -1;  /* Do not solve X[k] in the future. */

                        int_t ii = X_BLK (lk);
                        int_t lkj = LBj (k, grid); /* Local block number, column-wise */

                        slocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                        --nroot;
                        /*
                         * Send Xk to process column Pc[k].
                         */
                        siBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, bsendx_plist, send_req, LUstruct, grid,xtrsTimer);
                        /*
                         * Perform local block modifications: lsum[i] -= U_i,k * X[k]
                         */
                        if (Urbs[lkj])
                            slsum_bmod_GG (lsum, x, &x[ii], nrhs, lbmod_buf,  k, bmod, Urbs,
                                           Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                           send_req, stat,xtrsTimer);
                        xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
                    }                   /* if root ... */
                }
            }

        }                       /* if diagonal process ... */
    }                           /* for k ... */
    while (nbrecvx || nbrecvmod)
    {
        /* While not finished. */

        /* Receive a message. */
        MPI_Status status;
        double tx = SuperLU_timer_();
        MPI_Recv (recvbuf, maxrecvsz, MPI_FLOAT,
                  MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status);
        xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
		int_t k = *recvbuf;

        tx = SuperLU_timer_();
        switch (status.MPI_TAG)
        {
        case Xk:
        {
            --nbrecvx;
            xtrsTimer->trsDataRecvXY += SuperSize(k)*nrhs + XK_H;
            /*
             * Perform local block modifications:
             *         lsum[i] -= U_i,k * X[k]
             */
            slsum_bmod_GG (lsum, x, &recvbuf[XK_H], nrhs, lbmod_buf, k, bmod, Urbs,
                           Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                           send_req, stat,xtrsTimer);
            break;
        }
        case LSUM:             /* Receiver must be a diagonal process */
        {
            --nbrecvmod;
            xtrsTimer->trsDataRecvXY += SuperSize(k)*nrhs + LSUM_H;
            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            int_t ii = X_BLK (lk);
            int_t knsupc = SuperSize (k);
            float* tempv = &recvbuf[LSUM_H];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
					x[i + ii + j * knsupc] += tempv[i + j * knsupc];
            }

            if ((--brecv[lk]) == 0 && bmod[lk] == 0)
            {
                bmod[lk] = -1;  /* Do not solve X[k] in the future. */
                int_t lk = LBj (k, grid); /* Local block number, column-wise. */
                // int_t* lsub = Lrowind_bc_ptr[lk];
                slocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                siBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, bsendx_plist, send_req, LUstruct, grid,xtrsTimer);
                if (Urbs[lk])
                    slsum_bmod_GG (lsum, x, &x[ii], nrhs, lbmod_buf, k, bmod, Urbs,
                                   Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                   send_req, stat,xtrsTimer);
            }                   /* if becomes solvable */

            break;
        }
        }                       /* switch */
        xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
    }                           /* while not finished ... */

    double tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    SUPERLU_FREE(bmod);
    SUPERLU_FREE(brecv);
    Llu->SolveMsgSent = 0;
    xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
    return 0;

}



int_t sleafForestBackSolve3d_newsolve(superlu_dist_options_t *options, int_t n,  sLUstruct_t * LUstruct,
                            strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                            float * x, float * lsum, float * recvbuf,
                            MPI_Request * send_req,
                            int nrhs, slsumBmod_buff_t* lbmod_buf,
                            sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{

    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    // sForest_t* sforest = trf3Dpartition->sForests[treeId];
    // if (!sforest) return 0;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    // float** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t *ilsum = Llu->ilsum;
    int_t iam = grid->iam;
    int_t myGrid = grid3d->zscp.Iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);

    int_t  *Urbs = Llu->Urbs; /* Number of row blocks in each block column of U. */
    Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
    int_t  **Ucb_valptr = Llu->Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
    int_t knsupc = sp_ienv_dist (3,options);
    int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);

    // int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    // if (nnodes < 1) return 1;
    // int_t *perm_c_supno = sforest->nodeList ;

    int **bsendx_plist = Llu->bsendx_plist;
    int_t Pr = grid->nprow;
    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t nlb = CEILING (nsupers, Pr);
    int* supernodeMask = trf3Dpartition->supernodeMask;


    int* bmod=  getBmod3d_newsolve(nlb, nsupers, supernodeMask, xsup, Llu->Ufstnz_br_ptr, grid);
    // for (int_t l=0;l<nsupers;l++)
    //     printf("iam %5d lk %5d bmod %5d \n",grid->iam,l,bmod[l]);
    int* brecv = getBrecvTree_newsolve(nlb, nsupers, supernodeMask, bmod, grid);
    Llu->brecv = brecv;

    int_t nbrecvmod = 0;
    int nroot= getNrootUsolveTree_newsolve(&nbrecvmod, nsupers, supernodeMask, brecv, bmod, grid);
    int nbrecvx= getNbrecvX_newsolve(nsupers, supernodeMask, Urbs, Ucb_indptr, grid);

    // printf("igrid %5d, iam %5d, nbrecvx %5d, nbrecvmod %5d, nroot %5d\n",myGrid,iam,nbrecvx,nbrecvmod,nroot);

    /*before starting the solve; intialize the 3d lsum*/

    for (int_t k = nsupers - 1; k >= 0 && nroot; --k)
    {
        if(supernodeMask[k]>0){
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
        if (myrow == krow && mycol == kcol)
        {
            /* Diagonal process. */

            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            if (bmod[lk] == 0)
            {
                /* code */
                int_t il = LSUM_BLK( lk );
                int_t knsupc = SuperSize(k);
                {
                int_t ii = X_BLK( lk );
                if (brecv[lk] == 0 )
                {
                    double tx = SuperLU_timer_();
                    bmod[lk] = -1;  /* Do not solve X[k] in the future. */

                    int_t ii = X_BLK (lk);
                    int_t lkj = LBj (k, grid); /* Local block number, column-wise */


                    // if(4327==k)
                    // for(int_t i=0;i<knsupc;i++)
                    // printf("before xk root: lk %5d, k %5d, x[ii] %15.6f iam %5d knsupc %5d tree %5d \n",lk,k,x[ii+i],grid3d->iam,knsupc,trf3Dpartition->supernode2treeMap[k]);


                    slocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                    --nroot;

                    int_t knsupc = SuperSize(k);

                    // // // for(int_t i=0;i<knsupc;i++)
                    // if(4327==k)
                    // for(int_t i=0;i<knsupc;i++)
                    // printf("check xk root: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+i],grid3d->iam);


                    /*
                        * Send Xk to process column Pc[k].
                        */
                    siBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, bsendx_plist, send_req, LUstruct, grid,xtrsTimer);
                    /*
                        * Perform local block modifications: lsum[i] -= U_i,k * X[k]
                        */
                    if (Urbs[lkj])
                        slsum_bmod_GG_newsolve (trf3Dpartition, lsum, x, &x[ii], nrhs, lbmod_buf,  k, bmod, Urbs,
                                        Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                        send_req, stat,xtrsTimer);
                    xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
                }                   /* if root ... */
                }
            }
        }                       /* if diagonal process ... */
        }                       /* if(supernodeMask[k]) */
    }                           /* for k ... */
    while (nbrecvx || nbrecvmod)
    {
        /* While not finished. */

        /* Receive a message. */
        MPI_Status status;
        double tx = SuperLU_timer_();
        MPI_Recv (recvbuf, maxrecvsz, MPI_FLOAT,
                  MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status);
        xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
		int_t k = *recvbuf;

        tx = SuperLU_timer_();
        switch (status.MPI_TAG)
        {
        case Xk:
        {
            --nbrecvx;
            xtrsTimer->trsDataRecvXY += SuperSize(k)*nrhs + XK_H;
            /*
             * Perform local block modifications:
             *         lsum[i] -= U_i,k * X[k]
             */
            slsum_bmod_GG_newsolve (trf3Dpartition, lsum, x, &recvbuf[XK_H], nrhs, lbmod_buf, k, bmod, Urbs,
                           Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                           send_req, stat,xtrsTimer);
            break;
        }
        case LSUM:             /* Receiver must be a diagonal process */
        {
            --nbrecvmod;
            xtrsTimer->trsDataRecvXY += SuperSize(k)*nrhs + LSUM_H;
            int_t lk = LBi (k, grid); /* Local block number, row-wise. */
            int_t ii = X_BLK (lk);
            int_t knsupc = SuperSize (k);
            float* tempv = &recvbuf[LSUM_H];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
					x[i + ii + j * knsupc] += tempv[i + j * knsupc];
            }

            if ((--brecv[lk]) == 0 && bmod[lk] == 0)
            {
                bmod[lk] = -1;  /* Do not solve X[k] in the future. */
                int_t lk = LBj (k, grid); /* Local block number, column-wise. */
                // int_t* lsub = Lrowind_bc_ptr[lk];
                slocalSolveXkYk(  UPPER_TRI,  k,  &x[ii],  nrhs, LUstruct,   grid, stat);
                siBcastXk2Pck( k,  &x[ii - XK_H],  nrhs, bsendx_plist, send_req, LUstruct, grid,xtrsTimer);
                if (Urbs[lk])
                    slsum_bmod_GG_newsolve (trf3Dpartition, lsum, x, &x[ii], nrhs, lbmod_buf, k, bmod, Urbs,
                                   Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                   send_req, stat,xtrsTimer);
            }                   /* if becomes solvable */

            break;
        }
        }                       /* switch */
        xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
    }                           /* while not finished ... */

    double tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    Llu->SolveMsgSent = 0;
    xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
    SUPERLU_FREE(bmod);
    SUPERLU_FREE(brecv);
    return 0;

}





void sBackSolve3d_newsolve_reusepdgstrs(superlu_dist_options_t *options, int_t n,  sLUstruct_t * LUstruct,
                               int*  supernodeMask, gridinfo3d_t *grid3d,
                               float * x, float * lsum,
                               int nrhs,
                               sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    int_t myGrid = grid3d->zscp.Iam;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    float alpha = 1.0;
    float beta = 0.0;
    float zero = 0.0;
    float *lusup, *dest;
    float *recvbuf, *recvbuf_on, *tempv,
            *recvbufall, *recvbuf_BC_fwd, *recvbuf0, *xin, *recvbuf_BC_gpu,*recvbuf_RD_gpu;
    float *rtemp, *rtemp_loc; /* Result of full matrix-vector multiply. */
    float *Linv; /* Inverse of diagonal block */
    float *Uinv; /* Inverse of diagonal block */
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
    float **Lnzval_bc_ptr;
    float **Linv_bc_ptr;
    float **Uinv_bc_ptr;
    float sum;
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
    int fmod_tmp;
    int  **fsendx_plist = Llu->fsendx_plist;
    int  nfrecvx_buf=0;
    int *frecv;        /* Count of lsum[lk] contributions to be received
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
    double t1, t2, t3;
    float msg_vol = 0, msg_cnt = 0;
    // #endif

    int_t msgcnt[4]; /* Count the size of the message xfer'd in each buffer:
		      *     0 : transferred in Lsub_buf[]
		      *     1 : transferred in Lval_buf[]
		      *     2 : transferred in Usub_buf[]
		      *     3 : transferred in Uval_buf[]
		      */
    int iword = sizeof (int_t);
    int dword = sizeof (float);
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
    aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    int num_thread = 1;
	int_t cnt1,cnt2;
    double tx;

#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK)

	const int nwrp_block = 1; /* number of warps in each block */
	const int warp_size = 32; /* number of threads per warp*/
	gpuStream_t sid=0;
	int gid=0;
	gridinfo_t *d_grid = NULL;
	float *d_x = NULL;
	float *d_lsum = NULL;
    int  *d_bmod = NULL;
#endif


// cudaProfilerStart();
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

    // MPI_Barrier( grid->comm );
    t1_sol = SuperLU_timer_();
    t = SuperLU_timer_();


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

    // stat->utime[SOL_COMM] = 0.0;
    // stat->utime[SOL_GEMM] = 0.0;
    // stat->utime[SOL_TRSM] = 0.0;
    // stat->utime[SOL_TOT] = 0.0;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter dBackSolve3d_newsolve_reusepdgstrs()");
#endif

    // stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    /* Save the count to be altered so it can be used by
       subsequent call to PSGSTRS. */

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
    maxrecvsz = knsupc * nrhs + SUPERLU_MAX( XK_H, LSUM_H );
    sizelsum = (((size_t)ldalsum)*nrhs + nlb*LSUM_H);
    sizelsum = ((sizelsum + (aln_d - 1)) / aln_d) * aln_d;

/* skip rtemp on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    sizertemp=ldalsum * nrhs;
    sizertemp = ((sizertemp + (aln_d - 1)) / aln_d) * aln_d;
    if ( !(rtemp = (float*)SUPERLU_MALLOC((sizertemp*num_thread + 1) * sizeof(float))) )
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
}

    if ( !(stat_loc = (SuperLUStat_t**) SUPERLU_MALLOC(num_thread*sizeof(SuperLUStat_t*))) )
	ABORT("Malloc fails for stat_loc[].");

    for ( i=0; i<num_thread; i++) {
	stat_loc[i] = (SuperLUStat_t*)SUPERLU_MALLOC(sizeof(SuperLUStat_t));
	PStatInit(stat_loc[i]);
    }


	/* ---------------------------------------------------------
	   Initialize the async Bcast trees on all processes.
	   --------------------------------------------------------- */
	nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */


/* skip bmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    bmod=  getBmod3d_newsolve(nlb, nsupers, supernodeMask, xsup, Llu->Ufstnz_br_ptr, grid);
}
    nbrecvx= getNbrecvX_newsolve(nsupers, supernodeMask, Urbs, Ucb_indptr, grid);


		/* Save the count to be altered so it can be used by
		   subsequent call to PSGSTRS. */
		if ( !(brecv = int32Calloc_dist(nlb)) )
			ABORT("Calloc fails for brecv[].");
		Llu->brecv = brecv;

		/* Re-initialize lsum to zero. Each block header is already in place. */


/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
#ifdef _OPENMP
#pragma omp parallel default(shared) private(ii)
	{
		int thread_id = omp_get_thread_num();
		for(ii=0;ii<sizelsum;ii++)
			lsum[thread_id*sizelsum+ii]=zero;
	}
    /* Set up the headers in lsum[]. */
    for (k = 0; k < nsupers; ++k) {
	krow = PROW( k, grid );
	if ( myrow == krow ) {
	    lk = LBi( k, grid );   /* Local block number. */
	    il = LSUM_BLK( lk );
	    lsum[il - LSUM_H] = k; /* Block number prepended in the header. */
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
}


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
		if(UBtree_ptr[lk].empty_==NO){
			// printf("UBtree_ptr lk %5d\n",lk);
			if(C_BcTree_IsRoot(&UBtree_ptr[lk])==NO){
				nbtree++;
				if(UBtree_ptr[lk].destCnt_>0)nbrecvx_buf++;
			}
			// BcTree_allocateRequest(UBtree_ptr[lk],'s');
		}
	}

	nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	if ( !(	rootsups = (int_t*)intCalloc_dist(nsupers_i)) )
		ABORT("Calloc fails for rootsups.");


	for (lk=0;lk<nsupers_j;++lk){
		if(UBtree_ptr[lk].empty_==NO){
            xtrsTimer->trsDataSendXY  += UBtree_ptr[lk].msgSize_*nrhs+XK_H;
		}
    }
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
            xtrsTimer->trsDataSendXY  += URtree_ptr[lk].msgSize_*nrhs+LSUM_H;
		}
    }

/* skip bmod/rootsups/nroot on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
	nrtree = 0;
	nroot=0;
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
			// printf("here lk %5d myid %5d\n",lk,iam);
			// fflush(stdout);
			nrtree++;
			// RdTree_allocateRequest(URtree_ptr[lk],'s');
			brecv[lk] = URtree_ptr[lk].destCnt_;
			nbrecvmod += brecv[lk];
		}else{
			gb = myrow+lk*grid->nprow;  /* not sure */
			if(gb<nsupers){
				kcol = PCOL( gb, grid );
				if(mycol==kcol) { /* Diagonal process */
					if (bmod[lk*aln_i]==0 && supernodeMask[gb]>0){
						rootsups[nroot]=gb;
						++nroot;
					}
                }
            }
        }
    }
}else{
	nrtree = 0;
	for (lk=0;lk<nsupers_i;++lk){
		if(URtree_ptr[lk].empty_==NO){
			// printf("here lk %5d myid %5d\n",lk,iam);
			// fflush(stdout);
			nrtree++;
			// RdTree_allocateRequest(URtree_ptr[lk],'s');
			brecv[lk] = URtree_ptr[lk].destCnt_;
			nbrecvmod += brecv[lk];
		}
    }
}

/* skip bmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
	for (i = 0; i < nlb; ++i) bmod[i*aln_i] += brecv[i];
	// for (i = 0; i < nlb; ++i)printf("bmod[i]: %5d\n",bmod[i]);
}

	if ( !(recvbuf_BC_fwd = (float*)SUPERLU_MALLOC(maxrecvsz*(nbrecvx+1) * sizeof(float))) )  // this needs to be optimized for 1D row mapping
		ABORT("Malloc fails for recvbuf_BC_fwd[].");
	nbrecvx_buf=0;

	log_memory(nlb*aln_i*iword+nlb*iword + nsupers_i*iword + maxrecvsz*(nbrecvx+1)*dword, stat);	//account for bmod, brecv, rootsups, recvbuf_BC_fwd

#if ( DEBUGlevel>=2 )
	printf("(%2d) nbrecvx %4d,  nbrecvmod %4d,  nroot %4d\n,  nbtree %4d\n,  nrtree %4d\n",
			iam, nbrecvx, nbrecvmod, nroot, nbtree, nrtree);
	fflush(stdout);
#endif

// #if ( PRNTlevel>=1 )
#if 0
	t = SuperLU_timer_() - t;
    if ( !iam) printf(".. Grid %3d: Setup U-solve time\t%8.4f\n", myGrid, t);
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





if (get_acc_solve()){  /* GPU trisolve*/
#if defined(GPU_ACC) && defined(SLU_HAVE_LAPACK)
// #if 0 /* CPU trisolve*/


#if ( PROFlevel>=1 )
    t = SuperLU_timer_();
#endif

    d_bmod=SOLVEstruct->d_bmod;
    d_lsum=SOLVEstruct->d_lsum;
	d_x=SOLVEstruct->d_x;
	d_grid=Llu->d_grid;

	checkGPU(gpuMemcpy(d_bmod, SOLVEstruct->d_bmod_save, nlb * sizeof(int), gpuMemcpyDeviceToDevice));
    checkGPU(gpuMemcpy(d_lsum, SOLVEstruct->d_lsum_save, sizelsum * sizeof(float), gpuMemcpyDeviceToDevice));
    checkGPU(gpuMemcpy(d_x, x, (ldalsum * nrhs + nlb * XK_H) * sizeof(float), gpuMemcpyHostToDevice));

	k = CEILING( nsupers, grid->npcol);/* Number of local block columns divided by #warps per block used as number of thread blocks*/
	knsupc = sp_ienv_dist(3, options);

    if(procs>1){ /* only nvshmem needs the following*/
    #ifdef HAVE_NVSHMEM
    checkGPU(gpuMemcpy(d_status, mystatus_u, k * sizeof(int), gpuMemcpyHostToDevice));
    checkGPU(gpuMemcpy(d_statusmod, mystatusmod_u, 2* nlb * sizeof(int), gpuMemcpyHostToDevice));
    //for(int i=0;i<2*nlb;i++) printf("(%d),mystatusmod[%d]=%d\n",iam,i,mystatusmod[i]);
    checkGPU(gpuMemset(flag_rd_q, 0, RDMA_FLAG_SIZE * nlb * 2 * sizeof(int)));
    checkGPU(gpuMemset(flag_bc_q, 0, RDMA_FLAG_SIZE * (k+1)  * sizeof(int)));
    checkGPU(gpuMemset(sready_x, 0, maxrecvsz*CEILING( nsupers, grid->npcol) * sizeof(float)));
    checkGPU(gpuMemset(sready_lsum, 0, 2*maxrecvsz*CEILING( nsupers, grid->nprow) * sizeof(float)));
    checkGPU(gpuMemset(d_msgnum, 0, h_nfrecv_u[1] * sizeof(int)));
    // MUST have this barrier, otherwise the code hang.
	MPI_Barrier( grid->comm );
    #endif
    }

    slsum_bmod_inv_gpu_wrap(options, k,nlb,DIM_X,DIM_Y,d_lsum,d_x,nrhs,knsupc,nsupers,d_bmod,
                        Llu->d_UBtree_ptr,Llu->d_URtree_ptr,
                        Llu->d_ilsum,Llu->d_Ucolind_bc_dat,Llu->d_Ucolind_bc_offset,Llu->d_Ucolind_br_dat,Llu->d_Ucolind_br_offset,
                        Llu->d_Uind_br_dat,Llu->d_Uind_br_offset,
                        Llu->d_Unzval_bc_dat,Llu->d_Unzval_bc_offset,Llu->d_Unzval_br_new_dat,Llu->d_Unzval_br_new_offset,
                        Llu->d_Uinv_bc_dat,Llu->d_Uinv_bc_offset,
                        Llu->d_Uindval_loc_bc_dat,Llu->d_Uindval_loc_bc_offset,
                        Llu->d_xsup,d_grid,
                        maxrecvsz, flag_bc_q, flag_rd_q, sready_x, sready_lsum,
                        my_flag_bc, my_flag_rd,
                        d_nfrecv_u, h_nfrecv_u, d_status, d_colnum_u, d_mynum_u,
                        d_mymaskstart_u,d_mymasklength_u,
                        d_nfrecvmod_u, d_statusmod, d_colnummod_u, d_mynummod_u,
                        d_mymaskstartmod_u, d_mymasklengthmod_u,
                        d_recv_cnt_u, d_msgnum, d_flag_mod_u, procs);


    checkGPU(gpuMemcpy(x, d_x, (ldalsum * nrhs + nlb * XK_H) * sizeof(float), gpuMemcpyDeviceToHost));


#if ( PROFlevel>=1 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Grid %3d: around U kernel time\t%8.4f\n", myGrid, t);
#endif

	stat_loc[0]->ops[SOLVE]+=Llu->Unzval_br_cnt*nrhs*2; // YL: this is a rough estimate
#endif
}else{  /* CPU trisolve*/


tx = SuperLU_timer_();

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
#else
	{
#endif
#ifdef _OPENMP
#pragma omp master
#endif
		{
#ifdef _OPENMP
#if defined __GNUC__  && !defined __NVCOMPILER
#pragma	omp	taskloop firstprivate (nrhs,beta,alpha,x,rtemp,ldalsum) private (ii,jj,k,knsupc,lk,luptr,lsub,nsupr,lusup,t1,t2,Uinv,i,lib,rtemp_loc,nroot_send_tmp,thread_id) nogroup
#endif
#endif
		for (jj=0;jj<nroot;jj++){
			k=rootsups[jj];

#if ( PROFlevel>=1 )
			TIC(t1);
#endif
#ifdef _OPENMP
			thread_id=omp_get_thread_num();
#else
			thread_id=0;
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
				SGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
				sgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
				sgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
						&alpha, Uinv, &knsupc, &x[ii],
						&knsupc, &beta, rtemp_loc, &knsupc );
#endif
				for (i=0 ; i<knsupc*nrhs ; i++){
				    x[ii+i] = rtemp_loc[i];
				}
			}else{
#ifdef _CRAY
				STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
				strsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
						lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
				strsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
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
		} /* for k ... */
	}
}


#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
#else
	{
#endif
#ifdef _OPENMP
#pragma omp master
#endif
		{
#ifdef _OPENMP
#if defined __GNUC__  && !defined __NVCOMPILER
#pragma	omp	taskloop private (ii,jj,k,lk,thread_id) nogroup
#endif
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
				slsum_bmod_inv(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
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
		// BcTree_forwardMessageSimple(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'s')*nrhs+XK_H,'s');
		C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H);
	}else{ // this is a reduce forwarding
		lk = -lk - 1;
		il = LSUM_BLK( lk );
		// RdTree_forwardMessageSimple(URtree_ptr[lk],&lsum[il - LSUM_H ],RdTree_GetMsgSize(URtree_ptr[lk],'s')*nrhs+LSUM_H,'s');
		C_RdTree_forwardMessageSimple(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H);
	}
}
xtrsTimer->tbs_compute += SuperLU_timer_() - tx;

		/*
		 * Compute the internal nodes asychronously by all processes.
		 */

#ifdef _OPENMP
#pragma omp parallel default (shared)
	{
	int thread_id=omp_get_thread_num();
#else
	{
	thread_id=0;
#endif
#ifdef _OPENMP
#pragma omp master
#endif
		for ( nbrecv =0; nbrecv<nbrecvx+nbrecvmod;nbrecv++) { /* While not finished. */

			// printf("iam %4d nbrecv %4d nbrecvx %4d nbrecvmod %4d\n", iam, nbrecv, nbrecvxnbrecvmod);
			// fflush(stdout);



			thread_id = 0;
#if ( PROFlevel>=1 )
			TIC(t1);
#endif

			recvbuf0 = &recvbuf_BC_fwd[nbrecvx_buf*maxrecvsz];
            double tx = SuperLU_timer_();
			/* Receive a message. */
			MPI_Recv( recvbuf0, maxrecvsz, MPI_FLOAT,
					MPI_ANY_SOURCE, MPI_ANY_TAG, grid->comm, &status );
            xtrsTimer->tbs_comm += SuperLU_timer_() - tx;

#if ( PROFlevel>=1 )
			TOC(t2, t1);
			stat_loc[thread_id]->utime[SOL_COMM] += t2;

			msg_cnt += 1;
			msg_vol += maxrecvsz * dword;
#endif

			k = *recvbuf0;
#if ( DEBUGlevel>=2 )
			printf("(%2d) Recv'd block %d, tag %2d\n", iam, k, status.MPI_TAG);
			fflush(stdout);
#endif
            tx = SuperLU_timer_();
			if(status.MPI_TAG==BC_U){
                xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + XK_H;
				// --nfrecvx;
				nbrecvx_buf++;

				lk = LBj( k, grid );    /* local block number */

				if(UBtree_ptr[lk].destCnt_>0){

					// BcTree_forwardMessageSimple(UBtree_ptr[lk],recvbuf0,BcTree_GetMsgSize(UBtree_ptr[lk],'s')*nrhs+XK_H,'s');
					C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], recvbuf0, UBtree_ptr[lk].msgSize_*nrhs+XK_H);
					// nfrecvx_buf++;
				}

				/*
				 * Perform local block modifications: lsum[i] -= L_i,k * X[k]
				 */

				lk = LBj( k, grid ); /* Local block number, column-wise. */
				slsum_bmod_inv_master(lsum, x, &recvbuf0[XK_H], rtemp, nrhs, k, bmod, Urbs,
						Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
						stat_loc, sizelsum,sizertemp,thread_id,num_thread);
			}else if(status.MPI_TAG==RD_U){
                xtrsTimer->trsDataRecvXY  += SuperSize (k)*nrhs + LSUM_H;
				lk = LBi( k, grid ); /* Local block number, row-wise. */

				knsupc = SuperSize( k );
				tempv = &recvbuf0[LSUM_H];
				il = LSUM_BLK( lk );
				RHS_ITERATE(j) {
					for (i = 0; i < knsupc; ++i)
					    lsum[i + il + j*knsupc + thread_id*sizelsum] += tempv[i + j*knsupc];

				}
			// #ifdef _OPENMP
			// #pragma omp atomic capture
			// #endif
				bmod_tmp=--bmod[lk*aln_i];
				thread_id = 0;
				rtemp_loc = &rtemp[sizertemp* thread_id];
				if ( bmod_tmp==0 ) {
					if(C_RdTree_IsRoot(&URtree_ptr[lk])==YES){

						knsupc = SuperSize( k );
						for (ii=1;ii<num_thread;ii++)
							for (jj=0;jj<knsupc*nrhs;jj++)
					            lsum[il+ jj ] += lsum[il + jj + ii*sizelsum];


						ii = X_BLK( lk );
						RHS_ITERATE(j)
							for (i = 0; i < knsupc; ++i)
					            x[i + ii + j*knsupc] += lsum[i + il + j*knsupc];

						lk = LBj( k, grid ); /* Local block number, column-wise. */
						lsub = Lrowind_bc_ptr[lk];
						lusup = Lnzval_bc_ptr[lk];
						nsupr = lsub[1];

						if(Llu->inv == 1){

							Uinv = Uinv_bc_ptr[lk];

#ifdef _CRAY
							SGEMM( ftcs2, ftcs2, &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc );
#elif defined (USE_VENDOR_BLAS)
							sgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc, 1, 1 );
#else
							sgemm_( "N", "N", &knsupc, &nrhs, &knsupc,
									&alpha, Uinv, &knsupc, &x[ii],
									&knsupc, &beta, rtemp_loc, &knsupc );
#endif

							for (i=0 ; i<knsupc*nrhs ; i++){
				                x[ii+i] = rtemp_loc[i];
							}
						}else{
#ifdef _CRAY
							STRSM(ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &knsupc);
#elif defined (USE_VENDOR_BLAS)
							strsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
									lusup, &nsupr, &x[ii], &knsupc, 1, 1, 1, 1);
#else
							strsm_("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
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

						/*
						 * Send Xk to process column Pc[k].
						 */
						if(UBtree_ptr[lk].empty_==NO){
							// BcTree_forwardMessageSimple(UBtree_ptr[lk],&x[ii - XK_H],BcTree_GetMsgSize(UBtree_ptr[lk],'s')*nrhs+XK_H,'s');
							C_BcTree_forwardMessageSimple(&UBtree_ptr[lk], &x[ii - XK_H], UBtree_ptr[lk].msgSize_*nrhs+XK_H);
						}


						/*
						 * Perform local block modifications:
						 *         lsum[i] -= U_i,k * X[k]
						 */
						if ( Urbs[lk] )
							slsum_bmod_inv_master(lsum, x, &x[ii], rtemp, nrhs, k, bmod, Urbs,
									Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
									stat_loc, sizelsum,sizertemp,thread_id,num_thread);

					}else{
						il = LSUM_BLK( lk );
						knsupc = SuperSize( k );

						for (ii=1;ii<num_thread;ii++)
							for (jj=0;jj<knsupc*nrhs;jj++)
					            lsum[il+ jj ] += lsum[il + jj + ii*sizelsum];

						// RdTree_forwardMessageSimple(URtree_ptr[lk],&lsum[il-LSUM_H],RdTree_GetMsgSize(URtree_ptr[lk],'s')*nrhs+LSUM_H,'s');
						C_RdTree_forwardMessageSimple(&URtree_ptr[lk],&lsum[il - LSUM_H ],URtree_ptr[lk].msgSize_*nrhs+LSUM_H);
					}

				}
			}
            xtrsTimer->tbs_compute += SuperLU_timer_() - tx;
		} /* while not finished ... */
	}

    }

// #if ( PRNTlevel>=1 )
#if 0
		t = SuperLU_timer_() - t;
		stat->utime[SOL_TOT] += t;
		// if ( !iam ) printf(".. U-solve time\t%8.4f\n", t);
		MPI_Reduce (&t, &tmax, 1, MPI_DOUBLE,
				MPI_MAX, 0, grid->comm);
		if ( !iam ) {
			printf(".. Grid %3d: U-solve time (MAX) \t%8.4f\n", myGrid, tmax);
			fflush(stdout);
		}
		t = SuperLU_timer_();
#endif


#if ( DEBUGlevel>=2 )
		{
			float *x_col;
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
/* skip rtemp on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
		SUPERLU_FREE(rtemp);
}

/* skip bmod on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
		SUPERLU_FREE(bmod);
}
		SUPERLU_FREE(brecv);
		SUPERLU_FREE(root_send);

		SUPERLU_FREE(rootsups);
		SUPERLU_FREE(recvbuf_BC_fwd);

		log_memory(-nlb*aln_i*iword-nlb*iword - nsupers_i*iword - (CEILING( nsupers, Pr )+CEILING( nsupers, Pc ))*aln_i*iword - maxrecvsz*(nbrecvx+1)*dword - sizelsum*num_thread * dword - (ldalsum * nrhs + nlb * XK_H) *dword - (sizertemp*num_thread + 1)*dword, stat);	//account for bmod, brecv, root_send, rootsups, recvbuf_BC_fwd,rtemp,lsum,x

		for (lk=0;lk<nsupers_j;++lk){
			if(UBtree_ptr[lk].empty_==NO){
				// if(BcTree_IsRoot(LBtree_ptr[lk],'s')==YES){
				C_BcTree_waitSendRequest(&UBtree_ptr[lk]);
				// }
				// deallocate requests here
			}
		}

		for (lk=0;lk<nsupers_i;++lk){
			if(URtree_ptr[lk].empty_==NO){
				C_RdTree_waitSendRequest(&URtree_ptr[lk]);
				// deallocate requests here
			}
		}
		// MPI_Barrier( grid->comm );


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
				printf ("\tPSGSTRS comm stat:"
						"\tAvg\tMax\t\tAvg\tMax\n"
						"\t\t\tCount:\t%.0f\t%.0f\tVol(MB)\t%.2f\t%.2f\n",
						msg_cnt_sum / Pr / Pc, msg_cnt_max,
						msg_vol_sum / Pr / Pc * 1e-6, msg_vol_max * 1e-6);
			}
		}
#endif

    stat->utime[SOLVE] = SuperLU_timer_() - t1_sol;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit dBackSolve3d_newsolve_reusepdgstrs()");
#endif


#if ( PRNTlevel>=2 )
	    float for_lu, total, max, avg, temp;
		superlu_dist_mem_usage_t num_mem_usage;

	    sQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);
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

// cudaProfilerStop();

    return;
} /* sBackSolve3d_newsolve_reusepdgstrs */

/************************************************************************/

/************************************************************************/
void slsum_bmod_GG (
    float *lsum,        /* Sum of local modifications.                    */
    float *x,           /* X array (local).                               */
    float *xk,          /* X[k].                                          */
    int    nrhs,          /* Number of right-hand sides.                    */
    slsumBmod_buff_t* lbmod_buf,
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
    , xtrsTimer_t *xtrsTimer)
{
    // printf("bmodding %d\n", k);
    /*
     * Purpose
     * =======
     *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
     */
    float alpha = 1.0;
    float beta = 0.0;
    int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
    int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
           j, jj, lk, lk1, nub, ub, uptr;
    int_t  *usub;
    float *uval, *dest, *y;
    int_t  *lsub;
    float *lusup;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int  *brecv = Llu->brecv;
    int  **bsendx_plist = Llu->bsendx_plist;
    MPI_Status status;
    int test_flag;

    iam = grid->iam;
    myrow = MYROW( iam, grid );
    knsupc = SuperSize( k );
    lk = LBj( k, grid ); /* Local block number, column-wise. */
    nub = Urbs[lk];      /* Number of U blocks in block column lk */

    for (ub = 0; ub < nub; ++ub)
    {
        ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
        usub = Llu->Ufstnz_br_ptr[ik];
        uval = Llu->Unzval_br_ptr[ik];
        i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
        i += UB_DESCRIPTOR;
        il = LSUM_BLK( ik );
        gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
        iknsupc = SuperSize( gik );
#if 1
        slsumBmod(gik, k, nrhs, lbmod_buf,
                 &usub[i], &uval[Ucb_valptr[lk][ub]], xk,
                 &lsum[il], xsup, stat);
#else

        ikfrow = FstBlockC( gik );
        iklrow = FstBlockC( gik + 1 );

        for (int_t j = 0; j < nrhs; ++j)
        {
            dest = &lsum[il + j * iknsupc];
            y = &xk[j * knsupc];
            uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
            for (jj = 0; jj < knsupc; ++jj)
            {
                fnz = usub[i + jj];
                if ( fnz < iklrow )   /* Nonzero segment. */
                {
                    /* AXPY */
                    for (irow = fnz; irow < iklrow; ++irow)
                        dest[irow - ikfrow] -= uval[uptr++] * y[jj];
                    stat->ops[SOLVE] += 2 * (iklrow - fnz);
                }
            } /* for jj ... */
        } /*for (int_t j = 0;*/
#endif
        // printf(" updating %d  %d  \n",ik, bmod[ik] );
        if ( (--bmod[ik]) == 0 )   /* Local accumulation done. */
        {
            // printf("Local accumulation done %d  %d, brecv[ik]=%d  ",ik, bmod[ik],brecv[ik] );
            gikcol = PCOL( gik, grid );
            p = PNUM( myrow, gikcol, grid );
            if ( iam != p )
            {
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
                       iam, lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H, p);
#endif
                xtrsTimer->trsDataSendXY += iknsupc * nrhs + LSUM_H;
            }
            else     /* Diagonal process: X[i] += lsum[i]. */
            {
                ii = X_BLK( ik );
                dest = &x[ii];
                for (int_t j = 0; j < nrhs; ++j)
                    for (i = 0; i < iknsupc; ++i)
                        dest[i + j * iknsupc] += lsum[i + il + j * iknsupc];
                if ( !brecv[ik] )   /* Becomes a leaf node. */
                {
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
                    for (p = 0; p < grid->nprow; ++p)
                    {
                        if ( bsendx_plist[lk1][p] != SLU_EMPTY )
                        {
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
                                   iam, x[ii - XK_H], pi);
#endif
                        }
                    }
                    xtrsTimer->trsDataSendXY += iknsupc * nrhs + XK_H;
                    /*
                     * Perform local block modifications.
                     */
                    if ( Urbs[lk1] )
                        slsum_bmod_GG(lsum, x, &x[ii], nrhs, lbmod_buf, gik, bmod, Urbs,
                                      Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                      send_req, stat,xtrsTimer);
                } /* if brecv[ik] == 0 */
            }
        } /* if bmod[ik] == 0 */

    } /* for ub ... */

} /* slSUM_BMOD */




/************************************************************************/
void slsum_bmod_GG_newsolve (
    strf3Dpartition_t*  trf3Dpartition,
    float *lsum,        /* Sum of local modifications.                    */
    float *x,           /* X array (local).                               */
    float *xk,          /* X[k].                                          */
    int    nrhs,          /* Number of right-hand sides.                    */
    slsumBmod_buff_t* lbmod_buf,
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
    , xtrsTimer_t *xtrsTimer)
{
    // printf("bmodding %d\n", k);
    /*
     * Purpose
     * =======
     *   Perform local block modifications: lsum[i] -= U_i,k * X[k].
     */
    float alpha = 1.0;
    float beta = 0.0;
    int    iam, iknsupc, knsupc, myrow, nsupr, p, pi;
    int_t  fnz, gik, gikcol, i, ii, ik, ikfrow, iklrow, il, irow,
           j, jj, lk, lk1, nub, ub, uptr;
    int_t  *usub;
    float *uval, *dest, *y;
    int_t  *lsub;
    float *lusup;
    int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
    int  *brecv = Llu->brecv;
    int  **bsendx_plist = Llu->bsendx_plist;
    MPI_Status status;
    int test_flag;

    iam = grid->iam;
    myrow = MYROW( iam, grid );
    knsupc = SuperSize( k );
    lk = LBj( k, grid ); /* Local block number, column-wise. */
    nub = Urbs[lk];      /* Number of U blocks in block column lk */

    for (ub = 0; ub < nub; ++ub)
    {
        ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
        gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
        if (trf3Dpartition->supernodeMask[gik]>0)
        {
        usub = Llu->Ufstnz_br_ptr[ik];
        uval = Llu->Unzval_br_ptr[ik];
        i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
        i += UB_DESCRIPTOR;
        il = LSUM_BLK( ik );
        iknsupc = SuperSize( gik );
#if 1
        slsumBmod(gik, k, nrhs, lbmod_buf,
                 &usub[i], &uval[Ucb_valptr[lk][ub]], xk,
                 &lsum[il], xsup, stat);
#else

        ikfrow = FstBlockC( gik );
        iklrow = FstBlockC( gik + 1 );

        for (int_t j = 0; j < nrhs; ++j)
        {
            dest = &lsum[il + j * iknsupc];
            y = &xk[j * knsupc];
            uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */
            for (jj = 0; jj < knsupc; ++jj)
            {
                fnz = usub[i + jj];
                if ( fnz < iklrow )   /* Nonzero segment. */
                {
                    /* AXPY */
                    for (irow = fnz; irow < iklrow; ++irow)
                        dest[irow - ikfrow] -= uval[uptr++] * y[jj];
                    stat->ops[SOLVE] += 2 * (iklrow - fnz);
                }
            } /* for jj ... */
        } /*for (int_t j = 0;*/
#endif
        // printf(" updating %d  %d  \n",ik, bmod[ik] );
        if ( (--bmod[ik]) == 0 )   /* Local accumulation done. */
        {
            // printf("Local accumulation done %d  %d, brecv[ik]=%d  ",ik, bmod[ik],brecv[ik] );
            gikcol = PCOL( gik, grid );
            p = PNUM( myrow, gikcol, grid );
            if ( iam != p )
            {
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
                       iam, lsum[il - LSUM_H], iknsupc * nrhs + LSUM_H, p);
#endif
                xtrsTimer->trsDataSendXY += iknsupc * nrhs + LSUM_H;
            }
            else     /* Diagonal process: X[i] += lsum[i]. */
            {
                ii = X_BLK( ik );
                dest = &x[ii];
                for (int_t j = 0; j < nrhs; ++j)
                    for (i = 0; i < iknsupc; ++i)
                    dest[i + j * iknsupc] += lsum[i + il + j * iknsupc];

                if ( !brecv[ik] )   /* Becomes a leaf node. */
                {
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
                    for (p = 0; p < grid->nprow; ++p)
                    {
                        if ( bsendx_plist[lk1][p] != SLU_EMPTY )
                        {
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
                                   iam, x[ii - XK_H], pi);
#endif
                        }
                    }
                    xtrsTimer->trsDataSendXY += iknsupc * nrhs + XK_H;
                    /*
                     * Perform local block modifications.
                     */
                    if ( Urbs[lk1] )
                        slsum_bmod_GG_newsolve(trf3Dpartition, lsum, x, &x[ii], nrhs, lbmod_buf, gik, bmod, Urbs,
                                      Ucb_indptr, Ucb_valptr, xsup, grid, Llu,
                                      send_req, stat,xtrsTimer);
                } /* if brecv[ik] == 0 */
            }
        } /* if bmod[ik] == 0 */
        } /* if (trf3Dpartition->supernodeMask[gik]>0) */
    } /* for ub ... */

} /* slsum_bmod_GG_newsolve */



/*
 * Sketch of the algorithm for L-solve:
 * =======================
 *
 * Self-scheduling loop:
 *
 *   while ( not finished ) { .. use message counter to control
 *
 *      reveive a message;
 *
 * 	if ( message is Xk ) {
 * 	    perform local block modifications into lsum[];
 *                 lsum[i] -= L_i,k * X[k]
 *          if all local updates done, Isend lsum[] to diagonal process;
 *
 *      } else if ( message is LSUM ) { .. this must be a diagonal process
 *          accumulate LSUM;
 *          if ( all LSUM are received ) {
 *              perform triangular solve for Xi;
 *              Isend Xi down to the current process column;
 *              perform local block modifications into lsum[];
 *          }
 *      }
 *   }
 *
 *
 * Auxiliary data structures: lsum[] / ilsum (pointer to lsum array)
 * =======================
 *
 * lsum[] array (local)
 *   + lsum has "nrhs" columns, row-wise is partitioned by supernodes
 *   + stored by row blocks, column wise storage within a row block
 *   + prepend a header recording the global block number.
 *
 *         lsum[]                        ilsum[nsupers + 1]
 *
 *         -----
 *         | | |  <- header of size 2     ---
 *         --------- <--------------------| |
 *         | | | | |			  ---
 * 	   | | | | |	      |-----------| |
 *         | | | | | 	      |           ---
 *	   ---------          |   |-------| |
 *         | | |  <- header   |   |       ---
 *         --------- <--------|   |  |----| |
 *         | | | | |		  |  |    ---
 * 	   | | | | |              |  |
 *         | | | | |              |  |
 *	   ---------              |  |
 *         | | |  <- header       |  |
 *         --------- <------------|  |
 *         | | | | |                 |
 * 	   | | | | |                 |
 *         | | | | |                 |
 *	   --------- <---------------|
 */

/*#define ISEND_IRECV*/

/*
 * Function prototypes
 */
#ifdef _CRAY
fortran void STRSM(_fcd, _fcd, _fcd, _fcd, int*, int*, float*,
		   float*, int*, float*, int*);
_fcd ftcs1;
_fcd ftcs2;
_fcd ftcs3;
#endif





int_t slocalSolveXkYk( trtype_t trtype, int_t k, float* x, int nrhs,
                      sLUstruct_t * LUstruct, gridinfo_t * grid,
                      SuperLUStat_t * stat)
{
    // printf("Solving %d \n",k );
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    float alpha = 1.0;
    int_t* xsup = Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    float** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int knsupc = (int)SuperSize (k);
    int_t lk = LBj (k, grid); /* Local block number, column-wise */
    int_t *lsub = Lrowind_bc_ptr[lk];
    float* lusup = Lnzval_bc_ptr[lk];
    int nsupr = (int) lsub[1];

    if (trtype == UPPER_TRI)
    {
        /* upper triangular matrix */
#ifdef _CRAY
        STRSM (ftcs1, ftcs3, ftcs2, ftcs2, &knsupc, &nrhs, &alpha,
               lusup, &nsupr, x, &knsupc);
#elif defined (USE_VENDOR_BLAS)
        strsm_ ("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
                lusup, &nsupr, x, &knsupc, 1, 1, 1, 1);
#else
        strsm_ ("L", "U", "N", "N", &knsupc, &nrhs, &alpha,
                lusup, &nsupr, x, &knsupc);
#endif
    }
    else
    {
        /* lower triangular matrix */
#ifdef _CRAY
        STRSM (ftcs1, ftcs1, ftcs2, ftcs3, &knsupc, &nrhs, &alpha,
               lusup, &nsupr, x, &knsupc);
#elif defined (USE_VENDOR_BLAS)
        strsm_ ("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
                lusup, &nsupr, x, &knsupc, 1, 1, 1, 1);
#else
        strsm_ ("L", "L", "N", "U", &knsupc, &nrhs, &alpha,
                lusup, &nsupr, x, &knsupc);
#endif
    }
    stat->ops[SOLVE] += knsupc * (knsupc + 1) * nrhs;
    return 0;
}

int_t siBcastXk2Pck(int_t k, float* x, int nrhs,
                   int** sendList, MPI_Request *send_req,
                   sLUstruct_t * LUstruct, gridinfo_t * grid,xtrsTimer_t *xtrsTimer)
{
    /*
     * Send Xk to process column Pc[k].
     */
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t Pr = grid->nprow;
    int_t knsupc = SuperSize (k);
    int_t lk = LBj (k, grid);
    int_t kcol = PCOL (k, grid);
    for (int_t p = 0; p < Pr; ++p)
    {
        if (sendList[lk][p] != SLU_EMPTY)
        {
            int_t pi = PNUM (p, kcol, grid);

            MPI_Isend (x, knsupc * nrhs + XK_H,
                       MPI_FLOAT, pi, Xk, grid->comm,
                       &send_req[Llu->SolveMsgSent++]);

        }
    }

    xtrsTimer->trsDataSendXY += (double) SuperSize(k)*nrhs + XK_H;
    // printf("Data sent so far =%g and in this round= %g \n",xtrsTimer->trsDataSendXY, (double) SuperSize(k)*nrhs + XK_H );

    return 0;
}

/*! \brief
 *
 * <pre>
 * Purpose
 *
 *   Re-distribute B on the diagonal processes of the 2D process mesh (only on grid 0).
 *
 * Note
 *
 *   This routine can only be called after the routine pxgstrs_init(),
 *   in which the structures of the send and receive buffers are set up.
 *
 * Arguments
 *
  *
 * B      (input) float*
 *        The distributed right-hand side matrix of the possibly
 *        equilibrated system.
 *
 * m_loc  (input) int (local)
 *        The local row dimension of matrix B.
 *
 * nrhs   (input) int (global)
 *        Number of right-hand sides.
 *
 * ldb    (input) int (local)
 *        Leading dimension of matrix B.
 *
 * fst_row (input) int (global)
 *        The row number of B's first row in the global matrix.
 *
 * ilsum  (input) int* (global)
 *        Starting position of each supernode in a full array.
 *
 * x      (output) float*
 *        The solution vector. It is valid only on the diagonal processes.
 *
 * ScalePermstruct (input) dScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * SOLVEstruct (input) dSOLVEstruct_t*
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * Return value

 * </pre>
 */

int_t
psReDistribute3d_B_to_X (float *B, int_t m_loc, int nrhs, int_t ldb,
                         int_t fst_row, int_t * ilsum, float *x,
                         sScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist,
                         gridinfo3d_t * grid3d, sSOLVEstruct_t * SOLVEstruct)
{
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *sdispls_nrhs, *rdispls, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *perm_r, *perm_c;     /* row and column permutation vectors */
    int_t *send_ibuf, *recv_ibuf;
    float *send_dbuf, *recv_dbuf;
    int_t *xsup, *supno;
    int_t i, ii, irow, gbi, jj, k, knsupc, l, lk;
    int p, procs;
    gridinfo_t * grid = &(grid3d->grid2d);
    if (!grid3d->zscp.Iam)
    {
        pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

#if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter psReDistribute3d_B_to_X()");
#endif

        /* ------------------------------------------------------------
           INITIALIZATION.
           ------------------------------------------------------------ */
        perm_r = ScalePermstruct->perm_r;
        perm_c = ScalePermstruct->perm_c;
        procs = grid->nprow * grid->npcol;
        xsup = Glu_persist->xsup;
        supno = Glu_persist->supno;
        SendCnt = gstrs_comm->B_to_X_SendCnt;
        SendCnt_nrhs = gstrs_comm->B_to_X_SendCnt + procs;
        RecvCnt = gstrs_comm->B_to_X_SendCnt + 2 * procs;
        RecvCnt_nrhs = gstrs_comm->B_to_X_SendCnt + 3 * procs;
        sdispls = gstrs_comm->B_to_X_SendCnt + 4 * procs;
        sdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 5 * procs;
        rdispls = gstrs_comm->B_to_X_SendCnt + 6 * procs;
        rdispls_nrhs = gstrs_comm->B_to_X_SendCnt + 7 * procs;
        ptr_to_ibuf = gstrs_comm->ptr_to_ibuf;
        ptr_to_dbuf = gstrs_comm->ptr_to_dbuf;

        /* ------------------------------------------------------------
           NOW COMMUNICATE THE ACTUAL DATA.
           ------------------------------------------------------------ */
        k = sdispls[procs - 1] + SendCnt[procs - 1];    /* Total number of sends */
        l = rdispls[procs - 1] + RecvCnt[procs - 1];    /* Total number of receives */
        if (!(send_ibuf = intMalloc_dist (k + l)))
            ABORT ("Malloc fails for send_ibuf[].");
        recv_ibuf = send_ibuf + k;
        if (!(send_dbuf = floatMalloc_dist ((k + l) * (size_t) nrhs)))
            ABORT ("Malloc fails for send_dbuf[].");
        recv_dbuf = send_dbuf + k * nrhs;

        for (p = 0; p < procs; ++p)
        {
            ptr_to_ibuf[p] = sdispls[p];
            ptr_to_dbuf[p] = sdispls[p] * nrhs;
        }

        /* Copy the row indices and values to the send buffer. */
        for (i = 0, l = fst_row; i < m_loc; ++i, ++l)
        {
            irow = perm_c[perm_r[l]];   /* Row number in Pc*Pr*B */
            gbi = BlockNum (irow);
            p = PNUM (PROW (gbi, grid), PCOL (gbi, grid), grid);    /* Diagonal process */
            k = ptr_to_ibuf[p];
            send_ibuf[k] = irow;
            k = ptr_to_dbuf[p];
            for (int_t j = 0; j < nrhs; ++j)
            {
                /* RHS is stored in row major in the buffer. */
                send_dbuf[k++] = B[i + j * ldb];
            }
            ++ptr_to_ibuf[p];
            ptr_to_dbuf[p] += nrhs;
        }

        /* Communicate the (permuted) row indices. */
        MPI_Alltoallv (send_ibuf, SendCnt, sdispls, mpi_int_t,
                       recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid->comm);

        /* Communicate the numerical values. */
        MPI_Alltoallv (send_dbuf, SendCnt_nrhs, sdispls_nrhs, MPI_FLOAT,
                       recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, MPI_FLOAT,
                       grid->comm);

        /* ------------------------------------------------------------
           Copy buffer into X on the diagonal processes.
           ------------------------------------------------------------ */
        ii = 0;
        for (p = 0; p < procs; ++p)
        {
            jj = rdispls_nrhs[p];
            for (int_t i = 0; i < RecvCnt[p]; ++i)
            {
                /* Only the diagonal processes do this; the off-diagonal processes
                   have 0 RecvCnt. */
                irow = recv_ibuf[ii];   /* The permuted row index. */
                k = BlockNum (irow);
                knsupc = SuperSize (k);
                lk = LBi (k, grid); /* Local block number. */
                l = X_BLK (lk);
			    x[l - XK_H] = k;      /* Block number prepended in the header. */
                irow = irow - FstBlockC (k);    /* Relative row number in X-block */
                for (int_t j = 0; j < nrhs; ++j)
                {
                    x[l + irow + j * knsupc] = recv_dbuf[jj++];
                }
                ++ii;
            }
        }

        SUPERLU_FREE (send_ibuf);
        SUPERLU_FREE (send_dbuf);
    }
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid->iam, "Exit psReDistribute3d_B_to_X()");
#endif
    return 0;
}                               /* psReDistribute3d_B_to_X */

/*! \brief
 *
 * <pre>
 * Purpose
 *
 *   Re-distribute X on the diagonal processes to B distributed on all
 *   the processes (only on grid 0)
 *
 * Note
 *
 *   This routine can only be called after the routine pxgstrs_init(),
 *   in which the structures of the send and receive buffers are set up.
 * </pre>
 */

int_t
psReDistribute3d_X_to_B (int_t n, float *B, int_t m_loc, int_t ldb,
                         int_t fst_row, int nrhs, float *x, int_t * ilsum,
                         sScalePermstruct_t * ScalePermstruct,
                         Glu_persist_t * Glu_persist, gridinfo3d_t * grid3d,
                         sSOLVEstruct_t * SOLVEstruct)
{
    int_t i, ii, irow,  jj, k, knsupc, nsupers, l, lk;
    int_t *xsup, *supno;
    int *SendCnt, *SendCnt_nrhs, *RecvCnt, *RecvCnt_nrhs;
    int *sdispls, *rdispls, *sdispls_nrhs, *rdispls_nrhs;
    int *ptr_to_ibuf, *ptr_to_dbuf;
    int_t *send_ibuf, *recv_ibuf;
    float *send_dbuf, *recv_dbuf;
    int iam, p, q, pkk, procs;
    int_t num_diag_procs, *diag_procs;
    gridinfo_t * grid = &(grid3d->grid2d);
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid->iam, "Enter pdReDistribute_X_to_B()");
#endif

    /* ------------------------------------------------------------
       INITIALIZATION.
       ------------------------------------------------------------ */
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = Glu_persist->supno[n - 1] + 1;
    iam = grid->iam;
    procs = grid->nprow * grid->npcol;
    if (!grid3d->zscp.Iam)
    {
        int_t *row_to_proc = SOLVEstruct->row_to_proc;  /* row-process mapping */
        pxgstrs_comm_t *gstrs_comm = SOLVEstruct->gstrs_comm;

        SendCnt = gstrs_comm->X_to_B_SendCnt;
        SendCnt_nrhs = gstrs_comm->X_to_B_SendCnt + procs;
        RecvCnt = gstrs_comm->X_to_B_SendCnt + 2 * procs;
        RecvCnt_nrhs = gstrs_comm->X_to_B_SendCnt + 3 * procs;
        sdispls = gstrs_comm->X_to_B_SendCnt + 4 * procs;
        sdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 5 * procs;
        rdispls = gstrs_comm->X_to_B_SendCnt + 6 * procs;
        rdispls_nrhs = gstrs_comm->X_to_B_SendCnt + 7 * procs;
        ptr_to_ibuf = gstrs_comm->ptr_to_ibuf;
        ptr_to_dbuf = gstrs_comm->ptr_to_dbuf;

        k = sdispls[procs - 1] + SendCnt[procs - 1];    /* Total number of sends */
        l = rdispls[procs - 1] + RecvCnt[procs - 1];    /* Total number of receives */
        if (!(send_ibuf = intMalloc_dist (k + l)))
            ABORT ("Malloc fails for send_ibuf[].");
        recv_ibuf = send_ibuf + k;
        if (!(send_dbuf = floatMalloc_dist ((k + l) * nrhs)))
            ABORT ("Malloc fails for send_dbuf[].");
        recv_dbuf = send_dbuf + k * nrhs;
        for (p = 0; p < procs; ++p)
        {
            ptr_to_ibuf[p] = sdispls[p];
            ptr_to_dbuf[p] = sdispls_nrhs[p];
        }
        num_diag_procs = SOLVEstruct->num_diag_procs;
        diag_procs = SOLVEstruct->diag_procs;

        for (p = 0; p < num_diag_procs; ++p)
        {
            /* For all diagonal processes. */
            pkk = diag_procs[p];
            if (iam == pkk)
            {
                for (k = p; k < nsupers; k += num_diag_procs)
                {
                    knsupc = SuperSize (k);
                    lk = LBi (k, grid); /* Local block number */
                    irow = FstBlockC (k);
                    l = X_BLK (lk);
                    for (i = 0; i < knsupc; ++i)
                    {

                        ii = irow;

                        q = row_to_proc[ii];
                        jj = ptr_to_ibuf[q];
                        send_ibuf[jj] = ii;
                        jj = ptr_to_dbuf[q];
                        for (int_t j = 0; j < nrhs; ++j)
                        {
                            /* RHS stored in row major in buffer. */
                            send_dbuf[jj++] = x[l + i + j * knsupc];
                        }
                        ++ptr_to_ibuf[q];
                        ptr_to_dbuf[q] += nrhs;
                        ++irow;
                    }
                }
            }
        }

        /* ------------------------------------------------------------
           COMMUNICATE THE (PERMUTED) ROW INDICES AND NUMERICAL VALUES.
           ------------------------------------------------------------ */
        MPI_Alltoallv (send_ibuf, SendCnt, sdispls, mpi_int_t,
                       recv_ibuf, RecvCnt, rdispls, mpi_int_t, grid->comm);
        MPI_Alltoallv (send_dbuf, SendCnt_nrhs, sdispls_nrhs, MPI_FLOAT,
                       recv_dbuf, RecvCnt_nrhs, rdispls_nrhs, MPI_FLOAT,
                       grid->comm);

        /* ------------------------------------------------------------
           COPY THE BUFFER INTO B.
           ------------------------------------------------------------ */
        for (i = 0, k = 0; i < m_loc; ++i)
        {
            irow = recv_ibuf[i];
            irow -= fst_row;        /* Relative row number */
            for (int_t j = 0; j < nrhs; ++j)
            {
                /* RHS is stored in row major in the buffer. */
                B[irow + j * ldb] = recv_dbuf[k++];
            }
        }

        SUPERLU_FREE (send_ibuf);
        SUPERLU_FREE (send_dbuf);
    }
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid->iam, "Exit psReDistribute_X_to_B()");
#endif
    return 0;

}                               /* psReDistribute_X_to_B */


/*! \brief
 *
 * <pre>
 * Purpose
 *
 *
 * PSGSTRS solves a system of distributed linear equations
 * A*X = B with a general N-by-N matrix A using the LU factorization
 * computed by PSGSTRF.
 * If the equilibration, and row and column permutations were performed,
 * the LU factorization was performed for A1 where
 *     A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 * and the linear system solved is
 *     A1 * Y = Pc*Pr*B1, where B was overwritten by B1 = diag(R)*B, and
 * the permutation to B1 by Pc*Pr is applied internally in this routine.
 *
 * Arguments
 *
 *
 * n      (input) int (global)
 *        The order of the system of linear equations.
 *
 * LUstruct (input) sLUstruct_t*
 *        The distributed data structures storing L and U factors.
 *        The L and U factors are obtained from PSGSTRF for
 *        the possibly scaled and permuted matrix A.
 *        See superlu_ddefs.h for the definition of 'sLUstruct_t'.
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
 * B      (input/output) float*
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
 * SOLVEstruct (input) sSOLVEstruct_t* (global)
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the triangular solves.
 *        See util.h for the definition of 'SuperLUStat_t'.
 *
 * info   (output) int*
 *     = 0: successful exit
 *     < 0: if info = -i, the i-th argument had an illegal value
 * </pre>
 */

void
psgstrs3d (superlu_dist_options_t *options, int_t n, sLUstruct_t * LUstruct,
           sScalePermstruct_t * ScalePermstruct,
           strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, float *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info)
{
    // printf("Using pdgstr3d ..\n");
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;

    float *lsum;               /* Local running sum of the updates to B-components */
    float *x;                  /* X component at step k. */
    /* NOTE: x and lsum are of same size. */

    float *recvbuf;


    int_t iam,  mycol, myrow;
    int_t i, k;
    int_t  nlb, nsupers;
    int_t *xsup, *supno;
    int_t *ilsum;               /* Starting position of each supernode in lsum (LOCAL) */
    int_t Pc, Pr;
    int knsupc;
    int ldalsum;                /* Number of lsum entries locally owned. */
    int maxrecvsz;
    int_t **Lrowind_bc_ptr;
    float **Lnzval_bc_ptr;
    MPI_Status status;
    MPI_Request *send_req;


    double t;
#if ( DEBUGlevel>=2 )
    int_t Ublocks = 0;
#endif



    t = SuperLU_timer_ ();

    /* Test input parameters. */
    *info = 0;
    if ( n < 0 ) *info = -1;
    else if ( nrhs < 0 ) *info = -9;
    if ( *info ) {
	pxerr_dist("PSGSTRS", grid, -*info);
	return;
    }
#ifdef _CRAY
    ftcs1 = _cptofcd ("L", strlen ("L"));
    ftcs2 = _cptofcd ("N", strlen ("N"));
    ftcs3 = _cptofcd ("U", strlen ("U"));
#endif

    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW (iam, grid);
    mycol = MYCOL (iam, grid);
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n - 1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    nlb = CEILING (nsupers, Pr);    /* Number of local block rows. */
    int_t nub = CEILING (nsupers, Pc);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter pdgstrs3d()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;


    k = SUPERLU_MAX (Llu->nfsendx, Llu->nbsendx) + nlb;
    if (!
            (send_req =
                 (MPI_Request *) SUPERLU_MALLOC (k * sizeof (MPI_Request))))
        ABORT ("Malloc fails for send_req[].");




    /* Obtain ilsum[] and ldalsum for process column 0. */


    ilsum = Llu->ilsum;
    ldalsum = Llu->ldalsum;

    /* Allocate working storage. */
    knsupc = sp_ienv_dist (3,options);
    maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);
    if (!
            (lsum = floatCalloc_dist (((size_t) ldalsum) * nrhs + nlb * LSUM_H)))
        ABORT ("Calloc fails for lsum[].");
    if (!(x = floatMalloc_dist (ldalsum * nrhs + nlb * XK_H)))
        ABORT ("Malloc fails for x[].");
    if (!(recvbuf = floatMalloc_dist (maxrecvsz)))
        ABORT ("Malloc fails for recvbuf[].");

    /**
     * Initializing xT
     */

    int_t* ilsumT = SUPERLU_MALLOC (sizeof(int_t) * (nub + 1));
    int_t ldaspaT = 0;
    ilsumT[0] = 0;
    for (int_t jb = 0; jb < nsupers; ++jb)
    {
        if ( mycol == PCOL( jb, grid ) )
        {
            int_t i = SuperSize( jb );
            ldaspaT += i;
            int_t ljb = LBj( jb, grid );
            ilsumT[ljb + 1] = ilsumT[ljb] + i;
        }
    }
    float* xT;
    if (!(xT = floatMalloc_dist (ldaspaT * nrhs + nub * XK_H)))
        ABORT ("Malloc fails for xT[].");
    /**
     * Setup the headers for xT
     */
    for (int_t jb = 0; jb < nsupers; ++jb)
    {
        if ( mycol == PCOL( jb, grid ) )
        {
            int_t ljb = LBj( jb, grid );
            int_t jj = XT_BLK (ljb);

	        xT[jj] = jb;

        }
    }

    sxT_struct xT_s;
    xT_s.xT = xT;
    xT_s.ilsumT = ilsumT;
    xT_s.ldaspaT = ldaspaT;

    xtrsTimer_t xtrsTimer;

    initTRStimer(&xtrsTimer, grid);
    double tx = SuperLU_timer_();
    /* Redistribute B into X on the diagonal processes. */
    psReDistribute3d_B_to_X(B, m_loc, nrhs, ldb, fst_row, ilsum, x,
                            ScalePermstruct, Glu_persist, grid3d, SOLVEstruct);

    xtrsTimer.t_pxReDistribute_B_to_X = SuperLU_timer_() - tx;

    /*---------------------------------------------------
     * Forward solve Ly = b.
     *---------------------------------------------------*/

    strs_B_init3d(nsupers, x, nrhs, LUstruct, grid3d);

    MPI_Barrier (grid3d->comm);
    tx = SuperLU_timer_();
    stat->utime[SOLVE] = 0.0;
    double tx_st= SuperLU_timer_();


    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 printf("before psgsTrForwardSolve3d: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }

    psgsTrForwardSolve3d(options, n,  LUstruct, ScalePermstruct, trf3Dpartition, grid3d, x,  lsum, &xT_s,
                          recvbuf, send_req,  nrhs, SOLVEstruct,  stat, &xtrsTimer);
    // psgsTrForwardSolve3d_2d( n,  LUstruct, ScalePermstruct, trf3Dpartition, grid3d, x,  lsum, &xT_s,
    //                          recvbuf, send_req,  nrhs, SOLVEstruct,  stat, info);
    xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx;



    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 // for(int_t i=0;i<knsupc;i++)
    //                 printf("check x after L solve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }

    /*---------------------------------------------------
     * Back solve Ux = y.
     *
     * The Y components from the forward solve is already
     * on the diagonal processes.
     *---------------------------------------------------*/
    tx = SuperLU_timer_();
    psgsTrBackSolve3d(options, n,  LUstruct, ScalePermstruct, trf3Dpartition, grid3d, x,  lsum, &xT_s,
                       recvbuf, send_req,  nrhs, SOLVEstruct,  stat, &xtrsTimer);

    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 // for(int_t i=0;i<knsupc;i++)
    //                 printf("check x after U solve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }

    xtrsTimer.t_backwardSolve = SuperLU_timer_() - tx;
    MPI_Barrier (grid3d->comm);
    stat->utime[SOLVE] = SuperLU_timer_ () - tx_st;
    strs_X_gather3d(x, nrhs, trf3Dpartition, LUstruct, grid3d, &xtrsTimer);
    tx = SuperLU_timer_();
    psReDistribute3d_X_to_B(n, B, m_loc, ldb, fst_row, nrhs, x, ilsum,
                            ScalePermstruct, Glu_persist, grid3d, SOLVEstruct);

    xtrsTimer.t_pxReDistribute_X_to_B = SuperLU_timer_() - tx;

    /**
     * Reduce the Solve flops from all the grids to grid zero
     */
    reduceStat(SOLVE, stat, grid3d);
    /* Deallocate storage. */
    SUPERLU_FREE (lsum);
    SUPERLU_FREE (x);
    SUPERLU_FREE (recvbuf);
    SUPERLU_FREE (ilsumT);
    SUPERLU_FREE (xT);


    /*for (i = 0; i < Llu->SolveMsgSent; ++i) MPI_Request_free(&send_req[i]); */

    for (i = 0; i < Llu->SolveMsgSent; ++i)
        MPI_Wait (&send_req[i], &status);
    SUPERLU_FREE (send_req);

    MPI_Barrier (grid->comm);

#if ( PRNTlevel >= 1)
    printTRStimer(&xtrsTimer, grid3d);
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit psgstrs3d()");
#endif

    return;
}                               /* psgstrs3d */


void
psgstrs3d_newsolve (superlu_dist_options_t *options, int_t n, sLUstruct_t * LUstruct,
           sScalePermstruct_t * ScalePermstruct,
           strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d, float *B,
           int_t m_loc, int_t fst_row, int_t ldb, int nrhs,
           sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, int *info)
{
    // printf("Using pdgstr3d ..\n");
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;

    float *lsum;               /* Local running sum of the updates to B-components */
    float *x;                  /* X component at step k. */
    /* NOTE: x and lsum are of same size. */

    float *recvbuf;
    float zero = 0.0;


    int_t iam,  mycol, myrow;
    int_t i, k, ii;
    int_t  nlb, nsupers;
    int_t *xsup, *supno;
    int_t *ilsum;               /* Starting position of each supernode in lsum (LOCAL) */
    int_t Pc, Pr;
    int knsupc;
    int ldalsum;                /* Number of lsum entries locally owned. */
    int maxrecvsz;
    int_t **Lrowind_bc_ptr;
    float **Lnzval_bc_ptr;
    MPI_Status status;
    MPI_Request *send_req;


    double t;
#if ( DEBUGlevel>=2 )
    int_t Ublocks = 0;
#endif

    t = SuperLU_timer_ ();

    /* Test input parameters. */
    *info = 0;
    if ( n < 0 ) *info = -1;
    else if ( nrhs < 0 ) *info = -9;
    if ( *info ) {
	pxerr_dist("PSGSTRS", grid, -*info);
	return;
    }
#ifdef _CRAY
    ftcs1 = _cptofcd ("L", strlen ("L"));
    ftcs2 = _cptofcd ("N", strlen ("N"));
    ftcs3 = _cptofcd ("U", strlen ("U"));
#endif

    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW (iam, grid);
    mycol = MYCOL (iam, grid);
    xsup = Glu_persist->xsup;
    supno = Glu_persist->supno;
    nsupers = supno[n - 1] + 1;
    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    nlb = CEILING (nsupers, Pr);    /* Number of local block rows. */
    int_t nub = CEILING (nsupers, Pc);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter pdgstrs3d_newsolve()");
#endif

    stat->ops[SOLVE] = 0.0;
    Llu->SolveMsgSent = 0;

    k = SUPERLU_MAX (Llu->nfsendx, Llu->nbsendx) + nlb;
 /* skip send_req on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    if (!(send_req =
                 (MPI_Request *) SUPERLU_MALLOC (k * sizeof (MPI_Request))))
        ABORT ("Malloc fails for send_req[].");
}

    /* Obtain ilsum[] and ldalsum for process column 0. */


    ilsum = Llu->ilsum;
    ldalsum = Llu->ldalsum;

    /* Allocate working storage. */
    knsupc = sp_ienv_dist (3,options);
    maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);


    int_t sizelsum,sizertemp,aln_d,aln_i;
    aln_d = 1;//ceil(CACHELINE/(double)dword);
    aln_i = 1;//ceil(CACHELINE/(double)iword);
    sizelsum = (((size_t)ldalsum)*nrhs + nlb*LSUM_H);
    sizelsum = ((sizelsum + (aln_d - 1)) / aln_d) * aln_d;

    int num_thread = 1;
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
    if( grid3d->iam==0 ) {
	printf("num_thread: %5d\n", num_thread);
	fflush(stdout);
    }
#endif



/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
#ifdef _OPENMP
    if ( !(lsum = (float*)SUPERLU_MALLOC(sizelsum*num_thread * sizeof(float))))
	ABORT("Malloc fails for lsum[].");
#pragma omp parallel default(shared) private(ii)
    {
	int thread_id = omp_get_thread_num(); //mjc
	for (ii=0; ii<sizelsum; ii++)
    	    lsum[thread_id*sizelsum+ii]=zero;
    }
#else
    if ( !(lsum = (float*)SUPERLU_MALLOC(sizelsum*num_thread * sizeof(float))))
  	    ABORT("Malloc fails for lsum[].");
    for ( ii=0; ii < sizelsum*num_thread; ii++ )
	lsum[ii]=zero;
#endif
}

    /* intermediate solution x[] vector has same structure as lsum[], see leading comment */
    if ( !(x = floatCalloc_dist(ldalsum * nrhs + nlb * XK_H)) )
	ABORT("Calloc fails for x[].");
    if (!(recvbuf = floatMalloc_dist (maxrecvsz)))
        ABORT ("Malloc fails for recvbuf[].");

    xtrsTimer_t xtrsTimer;

    initTRStimer(&xtrsTimer, grid);
    double tx = SuperLU_timer_();
    /* Redistribute B into X on the diagonal processes. */
    psReDistribute3d_B_to_X(B, m_loc, nrhs, ldb, fst_row, ilsum, x,
                            ScalePermstruct, Glu_persist, grid3d, SOLVEstruct);

    xtrsTimer.t_pxReDistribute_B_to_X = SuperLU_timer_() - tx;

    /*---------------------------------------------------
     * Forward solve Ly = b.
     *---------------------------------------------------*/

    strs_B_init3d_newsolve(nsupers, x, nrhs, LUstruct, grid3d, trf3Dpartition);

    MPI_Barrier (grid3d->comm);
    tx = SuperLU_timer_();
    stat->utime[SOLVE] = 0.0;
    double tx_st= SuperLU_timer_();


    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 printf("before psgsTrForwardSolve3d_newsolve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }



    psgsTrForwardSolve3d_newsolve(options, n,  LUstruct, ScalePermstruct, trf3Dpartition, grid3d, x,  lsum,
                          recvbuf, send_req,  nrhs, SOLVEstruct,  stat, &xtrsTimer);
    xtrsTimer.t_forwardSolve = SuperLU_timer_() - tx;

    // printf("Llu->SolveMsgSent %10d size %10d\n",Llu->SolveMsgSent,SUPERLU_MAX (Llu->nfsendx, Llu->nbsendx) + nlb);
    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 printf("before strs_x_reduction_newsolve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }

    tx = SuperLU_timer_();
    strs_x_reduction_newsolve(nsupers, x, nrhs, LUstruct, grid3d, trf3Dpartition, recvbuf, &xtrsTimer);
    strs_x_broadcast_newsolve(nsupers, x, nrhs, LUstruct, grid3d, trf3Dpartition, recvbuf, &xtrsTimer);
    xtrsTimer.trs_comm_z += SuperLU_timer_() - tx;

    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    ////                 if(grid3d->iam==7)
    //                 printf("check x after L solve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }

    /*---------------------------------------------------
     * Back solve Ux = y.
     *
     * The Y components from the forward solve is already
     * on the diagonal processes.
     *---------------------------------------------------*/
    tx = SuperLU_timer_();
    psgsTrBackSolve3d_newsolve(options, n,  LUstruct, trf3Dpartition, grid3d, x,  lsum,
                       recvbuf, send_req,  nrhs, SOLVEstruct,  stat, &xtrsTimer);

    // printf("psgsTrBackSolve3d_newsolve Llu->SolveMsgSent %10d size %10d\n",Llu->SolveMsgSent,SUPERLU_MAX (Llu->nfsendx, Llu->nbsendx) + nlb);
    // {
    // int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	// for (int_t ilvl = 0; ilvl < maxLvl ; ++ilvl)
	// {
    //     int_t tree = trf3Dpartition->myTreeIdxs[ilvl];
    //     sForest_t** sForests = trf3Dpartition->sForests;
    //     sForest_t* sforest = sForests[tree];
	// 	if (sforest)
	// 	{
    //         int_t nnodes = sforest->nNodes ;
	//         int_t *nodeList = sforest->nodeList ;


    //         for (int_t k0 = 0; k0 < nnodes; ++k0)
    //         {
    //             int_t k = nodeList[k0];
    //             int_t krow = PROW (k, grid);
    //             int_t kcol = PCOL (k, grid);

    //             if (myrow == krow && mycol == kcol)
    //             {
    //                 int_t lk = LBi(k, grid);
    //                 int_t ii = X_BLK (lk);
    //                 int_t knsupc = SuperSize(k);

    //                 // for(int_t i=0;i<knsupc;i++)
    ////                 if(grid3d->iam==7)
    //                 printf("check x after U solve: lk %5d, k %5d, x[ii] %15.6f iam %5d \n",lk,k,x[ii+knsupc-1],grid3d->iam);

    //             }
    //         }
    //     }
    // }
    // }


    xtrsTimer.t_backwardSolve = SuperLU_timer_() - tx;
    MPI_Barrier (grid3d->comm);
    stat->utime[SOLVE] = SuperLU_timer_ () - tx_st;
    strs_X_gather3d(x, nrhs, trf3Dpartition, LUstruct, grid3d, &xtrsTimer);
    tx = SuperLU_timer_();
    psReDistribute3d_X_to_B(n, B, m_loc, ldb, fst_row, nrhs, x, ilsum,
                            ScalePermstruct, Glu_persist, grid3d, SOLVEstruct);

    xtrsTimer.t_pxReDistribute_X_to_B = SuperLU_timer_() - tx;

    /**
     * Reduce the Solve flops from all the grids to grid zero
     */
    reduceStat(SOLVE, stat, grid3d);
    /* Deallocate storage. */

/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    SUPERLU_FREE (lsum);
}
    SUPERLU_FREE (x);
    SUPERLU_FREE (recvbuf);


/* skip send_req on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    /*for (i = 0; i < Llu->SolveMsgSent; ++i) MPI_Request_free(&send_req[i]); */
    for (i = 0; i < Llu->SolveMsgSent; ++i)
        MPI_Wait (&send_req[i], &status);
    SUPERLU_FREE (send_req);
}
    // MPI_Barrier (grid->comm);

#if ( PRNTlevel >= 1 )
    printTRStimer(&xtrsTimer, grid3d);
#endif    
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit pdgstrs3d_newsolve()");
#endif

    return;
}                               /* psgstrs3d_newsolve */




int_t psgsTrForwardSolve3d(superlu_dist_options_t *options, int_t n, sLUstruct_t * LUstruct,
                           sScalePermstruct_t * ScalePermstruct,
                           strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                           float *x3d, float *lsum3d,
                           sxT_struct *xT_s,
                           float * recvbuf,
                           MPI_Request * send_req, int nrhs,
                           sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    float zero = 0.0;
    int_t* xsup = Glu_persist->xsup;

    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t *ilsum = Llu->ilsum;

    int_t knsupc = sp_ienv_dist (3,options);
    int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);
    float* rtemp;
    if (!(rtemp = floatCalloc_dist (maxrecvsz)))
        ABORT ("Malloc fails for rtemp[].");

    /**
     *  Loop over all the levels from root to leaf
     */
    int_t ii = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t knsupc = SuperSize (k);
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* Local block number. */
            int_t il = LSUM_BLK (lk);
	        lsum3d[il - LSUM_H] = k; /* Block number prepended in the header. */

        }
        ii += knsupc;
    }

    /*initilize lsum to zero*/
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t knsupc = SuperSize (k);
            int_t lk = LBi (k, grid);
            int_t il = LSUM_BLK (lk);
            float* dest = &lsum3d[il];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
                    dest[i + j * knsupc] = zero;
            }
        }
    }


    Llu->SolveMsgSent = 0;
    for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        double tx = SuperLU_timer_();
        /* if I participate in this level */
        if (!myZeroTrIdxs[ilvl])
        {
            int_t tree = myTreeIdxs[ilvl];

            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

            /*main loop over all the super nodes*/
            if (sforest)
            {
                if (ilvl == 0)
                    sleafForestForwardSolve3d(options, tree, n, LUstruct,
                                              ScalePermstruct, trf3Dpartition, grid3d,
                                              x3d,  lsum3d, recvbuf, rtemp,
                                              send_req,  nrhs, SOLVEstruct,  stat, xtrsTimer);
                else
                    snonLeafForestForwardSolve3d(tree, LUstruct,
                                                ScalePermstruct, trf3Dpartition, grid3d,  x3d,  lsum3d, xT_s, recvbuf, rtemp,
                                                send_req, nrhs, SOLVEstruct,  stat, xtrsTimer);

            }
            if (ilvl != maxLvl - 1)
            {
                /* code */
                int_t myGrid = grid3d->zscp.Iam;


                int_t sender, receiver;
                if ((myGrid % (1 << (ilvl + 1))) == 0)
                {
                    sender = myGrid + (1 << ilvl);
                    receiver = myGrid;
                }
                else
                {

                    sender = myGrid;
                    receiver = myGrid - (1 << ilvl);
                }
                double tx = SuperLU_timer_();
                for (int_t alvl = ilvl + 1; alvl <  maxLvl; ++alvl)
                {
                    /* code */
                    int_t treeId = myTreeIdxs[alvl];
                    sfsolveReduceLsum3d(treeId, sender, receiver, lsum3d, recvbuf, nrhs,
                                       trf3Dpartition, LUstruct, grid3d,xtrsTimer );
                }
                xtrsTimer->trs_comm_z += SuperLU_timer_() - tx;
            }
        }
        xtrsTimer->tfs_tree[ilvl] = SuperLU_timer_() - tx;
    }

    double tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    Llu->SolveMsgSent = 0;
    xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
    SUPERLU_FREE(rtemp);


    return 0;
}



int_t psgsTrForwardSolve3d_newsolve(superlu_dist_options_t *options, int_t n, sLUstruct_t * LUstruct,
                           sScalePermstruct_t * ScalePermstruct,
                           strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                           float *x3d, float *lsum3d,
                           float * recvbuf,
                           MPI_Request * send_req, int nrhs,
                           sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    float zero = 0.0;
    int_t* xsup = Glu_persist->xsup;

    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t *ilsum = Llu->ilsum;

    int_t knsupc = sp_ienv_dist (3,options);
    int_t maxrecvsz = knsupc * nrhs + SUPERLU_MAX (XK_H, LSUM_H);
    float* rtemp;
    if (!(rtemp = floatCalloc_dist (maxrecvsz)))
        ABORT ("Malloc fails for rtemp[].");

/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    int_t ii = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t knsupc = SuperSize (k);
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* Local block number. */
            int_t il = LSUM_BLK (lk);
	        lsum3d[il - LSUM_H] = k; /* Block number prepended in the header. */
        }
        ii += knsupc;
    }

    /*initilize lsum to zero*/
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t knsupc = SuperSize (k);
            int_t lk = LBi (k, grid);
            int_t il = LSUM_BLK (lk);
            float* dest = &lsum3d[il];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
                    dest[i + j * knsupc] = zero;
            }
        }
    }
}

    Llu->SolveMsgSent = 0;

    double tx = SuperLU_timer_();

if (get_new3dsolvetreecomm()){
    sForwardSolve3d_newsolve_reusepdgstrs(options, n, LUstruct,
                                ScalePermstruct, trf3Dpartition->supernodeMask, grid3d,
                                x3d, lsum3d, nrhs, SOLVEstruct, stat, xtrsTimer);
}else{
    sleafForestForwardSolve3d_newsolve(options, n, LUstruct,
                                ScalePermstruct, trf3Dpartition, grid3d,
                                x3d,  lsum3d, recvbuf, rtemp,
                                send_req,  nrhs, SOLVEstruct,  stat, xtrsTimer);
}


    xtrsTimer->tfs_tree[0] = SuperLU_timer_() - tx;
    tx = SuperLU_timer_();
/* skip send_req on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
}
    Llu->SolveMsgSent = 0;
    xtrsTimer->tfs_comm += SuperLU_timer_() - tx;
    SUPERLU_FREE(rtemp);
    return 0;
}





int_t psgsTrBackSolve3d(superlu_dist_options_t *options, int_t n, sLUstruct_t * LUstruct,
                        sScalePermstruct_t * ScalePermstruct,
                        strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                        float *x3d, float *lsum3d,
                        sxT_struct *xT_s,
                        float * recvbuf,
                        MPI_Request * send_req, int nrhs,
                        sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{
    // printf("Using psgsTrBackSolve3d_2d \n");

    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    float zero = 0.0;
    int_t* xsup = Glu_persist->xsup;

    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t *ilsum = Llu->ilsum;

    /**
     *  Loop over all the levels from root to leaf
     */

    /*initilize lsum to zero*/
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t knsupc = SuperSize (k);
            int_t lk = LBi (k, grid);
            int_t il = LSUM_BLK (lk);
            float* dest = &lsum3d[il];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
                    dest[i + j * knsupc] = zero;
            }
        }
    }

    /**
     * Adding dlsumBmod_buff_t* lbmod_buf
     */

    slsumBmod_buff_t lbmod_buf;
    int_t nsupc = sp_ienv_dist (3,options);
    sinitLsumBmod_buff(nsupc, nrhs, &lbmod_buf);

    int_t numTrees = 2 * grid3d->zscp.Np - 1;
    int_t nLeafTrees = grid3d->zscp.Np;
    Llu->SolveMsgSent = 0;
    for (int_t ilvl = maxLvl - 1; ilvl >= 0  ; --ilvl)
    {
        /* code */
        double tx = SuperLU_timer_();
        if (!myZeroTrIdxs[ilvl])
        {
            double tx = SuperLU_timer_();
            sbsolve_Xt_bcast(ilvl, xT_s, nrhs, trf3Dpartition,
                            LUstruct, grid3d,xtrsTimer );
            xtrsTimer->trs_comm_z += SuperLU_timer_() - tx;


            int_t tree = myTreeIdxs[ilvl];

            int_t trParent = (tree + 1) / 2  - 1;
            tx = SuperLU_timer_();
            while (trParent > -1 )
            {
                slasum_bmod_Tree(trParent, tree, lsum3d, x3d,  xT_s, nrhs, &lbmod_buf,
                                 LUstruct, trf3Dpartition, grid3d, stat);
                trParent = (trParent + 1) / 2 - 1;

            }
            xtrsTimer->tbs_compute += SuperLU_timer_() - tx;


            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

            /*main loop over all the super nodes*/
            if (sforest)
            {
                if (ilvl == 0)
                    sleafForestBackSolve3d(options, tree, n, LUstruct,
                                           ScalePermstruct, trf3Dpartition, grid3d, x3d,  lsum3d, recvbuf,
                                           send_req,  nrhs, &lbmod_buf,
                                            SOLVEstruct,  stat, xtrsTimer);
                else
                    snonLeafForestBackSolve3d(tree, LUstruct,
                                             ScalePermstruct, trf3Dpartition, grid3d,  x3d,  lsum3d, xT_s, recvbuf,
                                             send_req, nrhs, &lbmod_buf,
                                              SOLVEstruct,  stat, xtrsTimer);


            }
        }
        xtrsTimer->tbs_tree[ilvl] = SuperLU_timer_() - tx;
    }
    double tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
    Llu->SolveMsgSent = 0;
    sfreeLsumBmod_buff(&lbmod_buf);

    return 0;
}




int_t psgsTrBackSolve3d_newsolve(superlu_dist_options_t *options, int_t n, sLUstruct_t * LUstruct,
                        strf3Dpartition_t*  trf3Dpartition, gridinfo3d_t *grid3d,
                        float *x3d, float *lsum3d,
                        float * recvbuf,
                        MPI_Request * send_req, int nrhs,
                        sSOLVEstruct_t * SOLVEstruct, SuperLUStat_t * stat, xtrsTimer_t *xtrsTimer)
{

    gridinfo_t * grid = &(grid3d->grid2d);
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    sLocalLU_t *Llu = LUstruct->Llu;
    float zero = 0.0;
    int_t* xsup = Glu_persist->xsup;

    int_t nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    int_t *ilsum = Llu->ilsum;


/* skip lsum on CPU if using GPU solve*/
if ( !(get_new3dsolvetreecomm() && get_acc_solve())){
    /*initilize lsum to zero*/
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t knsupc = SuperSize (k);
            int_t lk = LBi (k, grid);
            int_t il = LSUM_BLK (lk);
            float* dest = &lsum3d[il];
            for (int_t j = 0; j < nrhs; ++j)
            {
                for (int_t i = 0; i < knsupc; ++i)
                    dest[i + j * knsupc] = zero;
            }
        }
    }
}

    /**
     * Adding dlsumBmod_buff_t* lbmod_buf
     */

    slsumBmod_buff_t lbmod_buf;
    int_t nsupc = sp_ienv_dist (3,options);
    sinitLsumBmod_buff(nsupc, nrhs, &lbmod_buf);

    Llu->SolveMsgSent = 0;
    double tx = SuperLU_timer_();

if (get_new3dsolvetreecomm()){
    sBackSolve3d_newsolve_reusepdgstrs(options, n, LUstruct,
                                trf3Dpartition->supernodeMask, grid3d,
                                x3d, lsum3d, nrhs, SOLVEstruct, stat, xtrsTimer);
}else{
    sleafForestBackSolve3d_newsolve(options, n, LUstruct, trf3Dpartition, grid3d, x3d,  lsum3d, recvbuf,
                            send_req,  nrhs, &lbmod_buf, SOLVEstruct,  stat, xtrsTimer);
}

    xtrsTimer->tbs_tree[0] = SuperLU_timer_() - tx;

    tx = SuperLU_timer_();
    for (int_t i = 0; i < Llu->SolveMsgSent; ++i)
    {
        MPI_Status status;
        MPI_Wait (&send_req[i], &status);
    }
    xtrsTimer->tbs_comm += SuperLU_timer_() - tx;
    Llu->SolveMsgSent = 0;

    sfreeLsumBmod_buff(&lbmod_buf);

    return 0;
}
