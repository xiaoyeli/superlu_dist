/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Auxiliary routine for 3D factorization.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 * May 12, 2021
 */

#include "superlu_ddefs.h"

#if 0
#include "pdgstrf3d.h"
#include "trfAux.h"
#endif

/* Inititalize the data structure to assist HALO offload of Schur-complement. */
void dInit_HyP(HyP_t* HyP, dLocalLU_t *Llu, int_t mcb, int_t mrb )
{
    HyP->last_offload = -1;
#if 0
    HyP->lookAhead_info = (Remain_info_t *) _mm_malloc((mrb) * sizeof(Remain_info_t), 64);

    HyP->lookAhead_L_buff = (double *) _mm_malloc( sizeof(double) * (Llu->bufmax[1]), 64);

    HyP->Remain_L_buff = (double *) _mm_malloc( sizeof(double) * (Llu->bufmax[1]), 64);
    HyP->Remain_info = (Remain_info_t *) _mm_malloc(mrb * sizeof(Remain_info_t), 64);
    HyP->Ublock_info_Phi = (Ublock_info_t *) _mm_malloc(mcb * sizeof(Ublock_info_t), 64);
    HyP->Ublock_info = (Ublock_info_t *) _mm_malloc(mcb * sizeof(Ublock_info_t), 64);
    HyP->Lblock_dirty_bit = (int_t *) _mm_malloc(mcb * sizeof(int_t), 64);
    HyP->Ublock_dirty_bit = (int_t *) _mm_malloc(mrb * sizeof(int_t), 64);
#else
    HyP->lookAhead_info = (Remain_info_t *) SUPERLU_MALLOC((mrb) * sizeof(Remain_info_t));
    HyP->lookAhead_L_buff = (double *) doubleMalloc_dist((Llu->bufmax[1]));
    HyP->Remain_L_buff = (double *) doubleMalloc_dist((Llu->bufmax[1]));
    HyP->Remain_info = (Remain_info_t *) SUPERLU_MALLOC(mrb * sizeof(Remain_info_t));
    HyP->Ublock_info_Phi = (Ublock_info_t *) SUPERLU_MALLOC(mcb * sizeof(Ublock_info_t));
    HyP->Ublock_info = (Ublock_info_t *) SUPERLU_MALLOC(mcb * sizeof(Ublock_info_t));
    HyP->Lblock_dirty_bit = (int_t *) intMalloc_dist(mcb);
    HyP->Ublock_dirty_bit = (int_t *) intMalloc_dist(mrb);
#endif

    for (int_t i = 0; i < mcb; ++i)
    {
        HyP->Lblock_dirty_bit[i] = -1;
    }

    for (int_t i = 0; i < mrb; ++i)
    {
        HyP->Ublock_dirty_bit[i] = -1;
    }

    HyP->last_offload = -1;
    HyP->superlu_acc_offload = get_acc_offload ();

    HyP->nGPUStreams =0;
} /* dInit_HyP */

/*init3DLUstruct with forest interface */
void dinit3DLUstructForest( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
                           sForest_t**  sForests, dLUstruct_t* LUstruct,
                           gridinfo3d_t* grid3d)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t numForests = (1 << maxLvl) - 1;
    int_t* gNodeCount = INT_T_ALLOC (numForests);
    int_t** gNodeLists =  (int_t**) SUPERLU_MALLOC(numForests * sizeof(int_t*));

    for (int i = 0; i < numForests; ++i)
	{
	    gNodeCount[i] = 0;
	    gNodeLists[i] = NULL;
	    /* code */
	    if (sForests[i])
		{	
                    gNodeCount[i] = sForests[i]->nNodes;
		    gNodeLists[i] = sForests[i]->nodeList;
		}
	}
    
    /*call the old forest*/
    dinit3DLUstruct( myTreeIdxs, myZeroTrIdxs,
		     gNodeCount, gNodeLists, LUstruct, grid3d);

    SUPERLU_FREE(gNodeCount);  // sherry added
    SUPERLU_FREE(gNodeLists);
}

int_t dSchurComplementSetup(
    int_t k,
    int *msgcnt,
    Ublock_info_t*  Ublock_info,
    Remain_info_t*  Remain_info,
    uPanelInfo_t *uPanelInfo,
    lPanelInfo_t *lPanelInfo,
    int_t* iperm_c_supno,
    int_t * iperm_u,
    int_t * perm_u,
    double *bigU,
    int_t* Lsub_buf,
    double *Lval_buf,
    int_t* Usub_buf,
    double *Uval_buf,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct
)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;

    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);

    int_t krow = PROW (k, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;

    int_t *usub;
    double* uval;
    int_t* lsub;
    double* lusup;

    if (mycol == kcol)
    {
        /*send the L panel to myrow*/
        int_t  lk = LBj (k, grid);     /* Local block number. */
        lsub = Lrowind_bc_ptr[lk];
        lPanelInfo->lsub = Lrowind_bc_ptr[lk];
        lusup = Lnzval_bc_ptr[lk];
        lPanelInfo->lusup = Lnzval_bc_ptr[lk];
    }
    else
    {
        lsub = Lsub_buf;
        lPanelInfo->lsub = Lsub_buf;
        lusup = Lval_buf;
        lPanelInfo->lusup = Lval_buf;
    }

    if (myrow == krow)
    {
        int_t  lk = LBi (k, grid);
        usub = Ufstnz_br_ptr[lk];
        uval = Unzval_br_ptr[lk];
        uPanelInfo->usub = usub;
    }
    else
    {
        if (ToRecv[k] == 2)
        {
            usub = Usub_buf;
            uval = Uval_buf;
            uPanelInfo->usub = usub;
        }
    }

    /*now each procs does the schurcomplement update*/
    int_t msg0 = msgcnt[0];
    int_t msg2 = msgcnt[2];
    int_t knsupc = SuperSize (k);

    int_t lptr0, luptr0;
    int_t LU_nonempty = msg0 && msg2;
    if (LU_nonempty == 0) return 0;
    if (msg0 && msg2)       /* L(:,k) and U(k,:) are not empty. */
    {
        lPanelInfo->nsupr = lsub[1];
        int_t nlb;
        if (myrow == krow)  /* Skip diagonal block L(k,k). */
        {
            lptr0 = BC_HEADER + LB_DESCRIPTOR + lsub[BC_HEADER + 1];
            luptr0 = knsupc;
            nlb = lsub[0] - 1;
            lPanelInfo->nlb = nlb;
        }
        else
        {
            lptr0 = BC_HEADER;
            luptr0 = 0;
            nlb = lsub[0];
            lPanelInfo->nlb = nlb;
        }
        int_t iukp = BR_HEADER;   /* Skip header; Pointer to index[] of U(k,:) */
        int_t rukp = 0;           /* Pointer to nzval[] of U(k,:) */
        int_t nub = usub[0];      /* Number of blocks in the block row U(k,:) */
        int_t klst = FstBlockC (k + 1);
        uPanelInfo->klst = klst;

        /* --------------------------------------------------------------
           Update the look-ahead block columns A(:,k+1:k+num_look_ahead).
           -------------------------------------------------------------- */
        int_t iukp0 = iukp;
        int_t rukp0 = rukp;

        /* reorder the remaining columns in bottom-up */
        for (int_t jj = 0; jj < nub; jj++)
        {
#ifdef ISORT
            iperm_u[jj] = iperm_c_supno[usub[iukp]];    /* Global block number of block U(k,j). */
            perm_u[jj] = jj;
#else
            perm_u[2 * jj] = iperm_c_supno[usub[iukp]]; /* Global block number of block U(k,j). */
            perm_u[2 * jj + 1] = jj;
#endif
            int_t jb = usub[iukp];    /* Global block number of block U(k,j). */
            int_t nsupc = SuperSize (jb);
            iukp += UB_DESCRIPTOR;  /* Start fstnz of block U(k,j). */
            iukp += nsupc;
        }
        iukp = iukp0;
#ifdef ISORT
        isort (nub, iperm_u, perm_u);
#else
        qsort (perm_u, (size_t) nub, 2 * sizeof (int_t),
               &superlu_sort_perm);
#endif
        // j = jj0 = 0;

        int_t ldu   = 0;
        int_t full  = 1;
        int_t num_u_blks = 0;

        for (int_t j = 0; j < nub ; ++j)
        {
            int_t iukp, temp_ncols;

            temp_ncols = 0;
            int_t  rukp, jb, ljb, nsupc, segsize;
            arrive_at_ublock(
                j, &iukp, &rukp, &jb, &ljb, &nsupc,
                iukp0, rukp0, usub, perm_u, xsup, grid
            );

            int_t jj = iukp;
            for (; jj < iukp + nsupc; ++jj)
            {
                segsize = klst - usub[jj];
                if ( segsize ) ++temp_ncols;
            }
            Ublock_info[num_u_blks].iukp = iukp;
            Ublock_info[num_u_blks].rukp = rukp;
            Ublock_info[num_u_blks].jb = jb;
            Ublock_info[num_u_blks].eo = iperm_c_supno[jb];
            /* Prepare to call DGEMM. */
            jj = iukp;

            for (; jj < iukp + nsupc; ++jj)
            {
                segsize = klst - usub[jj];
                if ( segsize )
                {
                    if ( segsize != ldu ) full = 0;
                    if ( segsize > ldu ) ldu = segsize;
                }
            }

            Ublock_info[num_u_blks].ncols = temp_ncols;
            // ncols += temp_ncols;
            num_u_blks++;

        }

        uPanelInfo->ldu = ldu;
        uPanelInfo->nub = num_u_blks;

        Ublock_info[0].full_u_cols = Ublock_info[0 ].ncols;
        Ublock_info[0].StCol = 0;
        for ( int_t j = 1; j < num_u_blks; ++j)
        {
            Ublock_info[j].full_u_cols = Ublock_info[j ].ncols + Ublock_info[j - 1].full_u_cols;
            Ublock_info[j].StCol = Ublock_info[j - 1].StCol + Ublock_info[j - 1].ncols;
        }

        dgather_u(num_u_blks, Ublock_info, usub,  uval,  bigU,  ldu, xsup, klst );

        sort_U_info_elm(Ublock_info, num_u_blks );

        int_t cum_nrow = 0;
        int_t RemainBlk = 0;

        int_t lptr = lptr0;
        int_t luptr = luptr0;
        for (int_t i = 0; i < nlb; ++i)
        {
            int_t ib = lsub[lptr];        /* Row block L(i,k). */
            int_t temp_nbrow = lsub[lptr + 1]; /* Number of full rows. */

            Remain_info[RemainBlk].nrows = temp_nbrow;
            Remain_info[RemainBlk].StRow = cum_nrow;
            Remain_info[RemainBlk].FullRow = cum_nrow;
            Remain_info[RemainBlk].lptr = lptr;
            Remain_info[RemainBlk].ib = ib;
            Remain_info[RemainBlk].eo = iperm_c_supno[ib];
            RemainBlk++;

            cum_nrow += temp_nbrow;
            lptr += LB_DESCRIPTOR;  /* Skip descriptor. */
            lptr += temp_nbrow;
            luptr += temp_nbrow;
        }

        lptr = lptr0;
        luptr = luptr0;
        sort_R_info_elm( Remain_info, lPanelInfo->nlb );
        lPanelInfo->luptr0 = luptr0;
    }
    return LU_nonempty;
} /* dSchurComplementSetup */

/* 
 * Gather L and U panels into respective buffers, to prepare for GEMM call.
 * Divide Schur complement update into two parts: CPU vs. GPU.
 */
int_t dSchurComplementSetupGPU(
    int_t k, msgs_t* msgs,
    packLUInfo_t* packLUInfo,
    int_t* myIperm, 
    int_t* iperm_c_supno, int_t*perm_c_supno,
    gEtreeInfo_t*   gEtreeInfo, factNodelists_t* fNlists,
    dscuBufs_t* scuBufs, dLUValSubBuf_t* LUvsb,
    gridinfo_t *grid, dLUstruct_t *LUstruct,
    HyP_t* HyP)
{
    int_t * Lsub_buf  = LUvsb->Lsub_buf;
    double * Lval_buf  = LUvsb->Lval_buf;
    int_t * Usub_buf  = LUvsb->Usub_buf;
    double * Uval_buf  = LUvsb->Uval_buf;
    uPanelInfo_t* uPanelInfo = packLUInfo->uPanelInfo;
    lPanelInfo_t* lPanelInfo = packLUInfo->lPanelInfo;
    int* msgcnt  = msgs->msgcnt;
    int_t* iperm_u  = fNlists->iperm_u;
    int_t* perm_u  = fNlists->perm_u;
    double* bigU = scuBufs->bigU;

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;

    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);

    int_t krow = PROW (k, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;

    int_t *usub;
    double* uval;
    int_t* lsub;
    double* lusup;

    HyP->lookAheadBlk = 0, HyP->RemainBlk = 0;
    HyP->Lnbrow =0, HyP->Rnbrow=0;
    HyP->num_u_blks_Phi=0;
    HyP->num_u_blks=0;

    if (mycol == kcol)
    {
        /*send the L panel to myrow*/
        int_t  lk = LBj (k, grid);     /* Local block number. */
        lsub = Lrowind_bc_ptr[lk];
        lPanelInfo->lsub = Lrowind_bc_ptr[lk];
        lusup = Lnzval_bc_ptr[lk];
        lPanelInfo->lusup = Lnzval_bc_ptr[lk];
    }
    else
    {
        lsub = Lsub_buf;
        lPanelInfo->lsub = Lsub_buf;
        lusup = Lval_buf;
        lPanelInfo->lusup = Lval_buf;
    }
    if (myrow == krow)
    {
        int_t  lk = LBi (k, grid);
        usub = Ufstnz_br_ptr[lk];
        uval = Unzval_br_ptr[lk];
        uPanelInfo->usub = usub;
    }
    else
    {
        if (ToRecv[k] == 2)
        {
            usub = Usub_buf;
            uval = Uval_buf;
            uPanelInfo->usub = usub;
        }
    }

    /*now each procs does the schurcomplement update*/
    int_t msg0 = msgcnt[0];
    int_t msg2 = msgcnt[2];
    int_t knsupc = SuperSize (k);

    int_t lptr0, luptr0;
    int_t LU_nonempty = msg0 && msg2;
    if (LU_nonempty == 0) return 0;
    if (msg0 && msg2)       /* L(:,k) and U(k,:) are not empty. */
    {
        lPanelInfo->nsupr = lsub[1];
        int_t nlb;
        if (myrow == krow)  /* Skip diagonal block L(k,k). */
        {
            lptr0 = BC_HEADER + LB_DESCRIPTOR + lsub[BC_HEADER + 1];
            luptr0 = knsupc;
            nlb = lsub[0] - 1;
            lPanelInfo->nlb = nlb;
        }
        else
        {
            lptr0 = BC_HEADER;
            luptr0 = 0;
            nlb = lsub[0];
            lPanelInfo->nlb = nlb;
        }
        int_t iukp = BR_HEADER;   /* Skip header; Pointer to index[] of U(k,:) */

        int_t nub = usub[0];      /* Number of blocks in the block row U(k,:) */
        int_t klst = FstBlockC (k + 1);
        uPanelInfo->klst = klst;

        /* --------------------------------------------------------------
           Update the look-ahead block columns A(:,k+1:k+num_look_ahead).
           -------------------------------------------------------------- */
        int_t iukp0 = iukp;

        /* reorder the remaining columns in bottom-up */
        for (int_t jj = 0; jj < nub; jj++)
        {
#ifdef ISORT
            iperm_u[jj] = iperm_c_supno[usub[iukp]];    /* Global block number of block U(k,j). */
            perm_u[jj] = jj;
#else
            perm_u[2 * jj] = iperm_c_supno[usub[iukp]]; /* Global block number of block U(k,j). */
            perm_u[2 * jj + 1] = jj;
#endif
            int_t jb = usub[iukp];    /* Global block number of block U(k,j). */
            int_t nsupc = SuperSize (jb);
            iukp += UB_DESCRIPTOR;  /* Start fstnz of block U(k,j). */
            iukp += nsupc;
        }
        iukp = iukp0;
#ifdef ISORT
        isort (nub, iperm_u, perm_u);
#else
        qsort (perm_u, (size_t) nub, 2 * sizeof (int_t),
               &superlu_sort_perm);
#endif
        HyP->Lnbrow = 0;
        HyP->Rnbrow = 0;
        HyP->num_u_blks_Phi=0;
	HyP->num_u_blks=0;

        dRgather_L(k, lsub, lusup,  gEtreeInfo, Glu_persist, grid, HyP, myIperm, iperm_c_supno);
        if (HyP->Lnbrow + HyP->Rnbrow > 0)
        {
            dRgather_U( k, 0, usub, uval, bigU,  gEtreeInfo, Glu_persist, grid, HyP, myIperm, iperm_c_supno, perm_u);
        }/*if(nbrow>0) */

    }

    return LU_nonempty;
} /* dSchurComplementSetupGPU */


double* dgetBigV(int_t ldt, int_t num_threads)
{
    double *bigV;
    if (!(bigV = doubleMalloc_dist (8 * ldt * ldt * num_threads)))
        ABORT ("Malloc failed for dgemm buffV");
    return bigV;
}

double* dgetBigU(superlu_dist_options_t *options,
	 int_t nsupers, gridinfo_t *grid, dLUstruct_t *LUstruct)
{
    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);

    /* Following circuit is for finding maximum block size */
    int local_max_row_size = 0;
    int max_row_size;

    for (int_t i = 0; i < nsupers; ++i)
    {
        int_t tpc = PCOL (i, grid);
        if (mycol == tpc)
        {
            int_t lk = LBj (i, grid);
            int_t* lsub = LUstruct->Llu->Lrowind_bc_ptr[lk];
            if (lsub != NULL)
            {
                local_max_row_size = SUPERLU_MAX (local_max_row_size, lsub[1]);
            }
        }

    }

    /* Max row size is global reduction of within A row */
    MPI_Allreduce (&local_max_row_size, &max_row_size, 1, MPI_INT, MPI_MAX,
                   (grid->rscp.comm));

    // int_t Threads_per_process = get_thread_per_process ();

    /*Buffer size is max of of look ahead window*/

    int_t bigu_size =
	8 * sp_ienv_dist(3, options) * (max_row_size) * SUPERLU_MAX(Pr / Pc, 1);
	//Sherry: 8 * sp_ienv_dist (3) * (max_row_size) * MY_MAX(Pr / Pc, 1);

    // printf("Size of big U is %d\n",bigu_size );
    double* bigU = doubleMalloc_dist(bigu_size);

    return bigU;
} /* dgetBigU */


dtrf3Dpartition_t* dinitTrf3Dpartition(int_t nsupers,
				      superlu_dist_options_t *options,
				      dLUstruct_t *LUstruct, gridinfo3d_t * grid3d
				      )
{
    gridinfo_t* grid = &(grid3d->grid2d);

#if ( DEBUGlevel>=1 )
    int iam = grid3d->iam;
    CHECK_MALLOC (iam, "Enter dinitTrf3Dpartition()");
#endif
    int_t* perm_c_supno = getPerm_c_supno(nsupers, options,
                                         LUstruct->etree,
    	   		                 LUstruct->Glu_persist,
		                         LUstruct->Llu->Lrowind_bc_ptr,
					 LUstruct->Llu->Ufstnz_br_ptr, grid);
    int_t* iperm_c_supno = getFactIperm(perm_c_supno, nsupers);

    // calculating tree factorization
    int_t *setree = supernodal_etree(nsupers, LUstruct->etree, LUstruct->Glu_persist->supno, LUstruct->Glu_persist->xsup);
    treeList_t* treeList = setree2list(nsupers, setree );

    /*update treelist with weight and depth*/
    getSCUweight(nsupers, treeList, LUstruct->Glu_persist->xsup,
		  LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Ufstnz_br_ptr,
		  grid3d);

    calcTreeWeight(nsupers, setree, treeList, LUstruct->Glu_persist->xsup);

    gEtreeInfo_t gEtreeInfo;
    gEtreeInfo.setree = setree;
    gEtreeInfo.numChildLeft = (int_t* ) SUPERLU_MALLOC(sizeof(int_t) * nsupers);
    for (int_t i = 0; i < nsupers; ++i)
    {
        /* code */
        gEtreeInfo.numChildLeft[i] = treeList[i].numChild;
    }

    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    sForest_t**  sForests = getForests( maxLvl, nsupers, setree, treeList);
    /*indexes of trees for my process grid in gNodeList size(maxLvl)*/
    int_t* myTreeIdxs = getGridTrees(grid3d);
    int_t* myZeroTrIdxs = getReplicatedTrees(grid3d);
    int_t*  gNodeCount = getNodeCountsFr(maxLvl, sForests);
    int_t** gNodeLists = getNodeListFr(maxLvl, sForests); // reuse NodeLists stored in sForests[]

    dinit3DLUstructForest(myTreeIdxs, myZeroTrIdxs,
                         sForests, LUstruct, grid3d);
    int_t* myNodeCount = getMyNodeCountsFr(maxLvl, myTreeIdxs, sForests);
    int_t** treePerm = getTreePermFr( myTreeIdxs, sForests, grid3d);

    dLUValSubBuf_t *LUvsb = SUPERLU_MALLOC(sizeof(dLUValSubBuf_t));
    dLluBufInit(LUvsb, LUstruct);

    int_t* supernode2treeMap = SUPERLU_MALLOC(nsupers*sizeof(int_t));
    int_t numForests = (1 << maxLvl) - 1;
    for (int_t Fr = 0; Fr < numForests; ++Fr)
    {
        /* code */
        for (int_t nd = 0; nd < gNodeCount[Fr]; ++nd)
        {
            /* code */
            supernode2treeMap[gNodeLists[Fr][nd]]=Fr;
        }
    }

    dtrf3Dpartition_t*  trf3Dpartition = SUPERLU_MALLOC(sizeof(dtrf3Dpartition_t));

    trf3Dpartition->gEtreeInfo = gEtreeInfo;
    trf3Dpartition->iperm_c_supno = iperm_c_supno;
    trf3Dpartition->myNodeCount = myNodeCount;
    trf3Dpartition->myTreeIdxs = myTreeIdxs;
    trf3Dpartition->myZeroTrIdxs = myZeroTrIdxs;
    trf3Dpartition->sForests = sForests;
    trf3Dpartition->treePerm = treePerm;
    trf3Dpartition->LUvsb = LUvsb;
    trf3Dpartition->supernode2treeMap = supernode2treeMap;

    // Sherry added
    // Deallocate storage
    SUPERLU_FREE(gNodeCount); 
    SUPERLU_FREE(gNodeLists); 
    SUPERLU_FREE(perm_c_supno);
    free_treelist(nsupers, treeList);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit dinitTrf3Dpartition()");
#endif
    return trf3Dpartition;
} /* dinitTrf3Dpartition */

/* Free memory allocated for trf3Dpartition structure. Sherry added this routine */
void dDestroy_trf3Dpartition(dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d)
{
    int i;
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Enter dDestroy_trf3Dpartition()");
#endif
    SUPERLU_FREE(trf3Dpartition->gEtreeInfo.setree);
    SUPERLU_FREE(trf3Dpartition->gEtreeInfo.numChildLeft);
    SUPERLU_FREE(trf3Dpartition->iperm_c_supno);
    SUPERLU_FREE(trf3Dpartition->myNodeCount);
    SUPERLU_FREE(trf3Dpartition->myTreeIdxs);
    SUPERLU_FREE(trf3Dpartition->myZeroTrIdxs);
    SUPERLU_FREE(trf3Dpartition->treePerm); // double pointer pointing to sForests->nodeList

    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t numForests = (1 << maxLvl) - 1;
    sForest_t** sForests = trf3Dpartition->sForests;
    for (i = 0; i < numForests; ++i) {
	if ( sForests[i] ) {
	    SUPERLU_FREE(sForests[i]->nodeList);
	    SUPERLU_FREE((sForests[i]->topoInfo).eTreeTopLims);
	    SUPERLU_FREE((sForests[i]->topoInfo).myIperm);
	    SUPERLU_FREE(sForests[i]); // Sherry added
	}
    }
    SUPERLU_FREE(trf3Dpartition->sForests); // double pointer 
    SUPERLU_FREE(trf3Dpartition->supernode2treeMap);

    SUPERLU_FREE((trf3Dpartition->LUvsb)->Lsub_buf);
    SUPERLU_FREE((trf3Dpartition->LUvsb)->Lval_buf);
    SUPERLU_FREE((trf3Dpartition->LUvsb)->Usub_buf);
    SUPERLU_FREE((trf3Dpartition->LUvsb)->Uval_buf);
    SUPERLU_FREE(trf3Dpartition->LUvsb); // Sherry: check this ...

    SUPERLU_FREE(trf3Dpartition);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Exit dDestroy_trf3Dpartition()");
#endif
}


#if 0  //**** Sherry: following two routines are old, the new ones are in util.c
int_t num_full_cols_U(int_t kk,  int_t **Ufstnz_br_ptr, int_t *xsup,
                      gridinfo_t *grid, int_t *perm_u)
{
    int_t lk = LBi (kk, grid);
    int_t *usub = Ufstnz_br_ptr[lk];

    if (usub == NULL)
    {
        /* code */
        return 0;
    }
    int_t iukp = BR_HEADER;   /* Skip header; Pointer to index[] of U(k,:) */
    int_t rukp = 0;           /* Pointer to nzval[] of U(k,:) */
    int_t nub = usub[0];      /* Number of blocks in the block row U(k,:) */

    int_t klst = FstBlockC (kk + 1);
    int_t iukp0 = iukp;
    int_t rukp0 = rukp;
    int_t jb, ljb;
    int_t nsupc;
    int_t temp_ncols = 0;
    int_t segsize;

    temp_ncols = 0;

    for (int_t j = 0; j < nub; ++j)
    {
        arrive_at_ublock(
            j, &iukp, &rukp, &jb, &ljb, &nsupc,
            iukp0, rukp0, usub, perm_u, xsup, grid
        );

        for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
        {
            segsize = klst - usub[jj];
            if ( segsize ) ++temp_ncols;
        }
    }
    return temp_ncols;
}

// Sherry: this is old; new version is in util.c 
int_t estimate_bigu_size( int_t nsupers, int_t ldt, int_t**Ufstnz_br_ptr,
                          Glu_persist_t *Glu_persist,  gridinfo_t* grid, int_t* perm_u)
{

    int_t iam = grid->iam;

    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);

    int_t* xsup = Glu_persist->xsup;

    int ncols = 0;
    int_t ldu = 0;

    /*initilize perm_u*/
    for (int i = 0; i < nsupers; ++i)
    {
        perm_u[i] = i;
    }

    for (int lk = myrow; lk < nsupers; lk += Pr )
    {
        ncols = SUPERLU_MAX(ncols, num_full_cols_U(lk, Ufstnz_br_ptr,
						   xsup, grid, perm_u, &ldu));
    }

    int_t max_ncols = 0;

    MPI_Allreduce(&ncols, &max_ncols, 1, mpi_int_t, MPI_MAX, grid->cscp.comm);

    printf("max_ncols =%d, bigu_size=%ld\n", (int) max_ncols, (long long) ldt * max_ncols);
    return ldt * max_ncols;
} /* old estimate_bigu_size. New one is in util.c */
#endif /**** end old ones ****/


