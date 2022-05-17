/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Factorization routines for the subtree using 2D process grid.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.2) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 * May 12, 2021
 *
 * Last update: December 12, 2021  v7.2.0
 */
 
#include "superlu_zdefs.h"
#if 0
#include "treeFactorization.h"
#include "trfCommWrapper.h"
#endif

int_t zLluBufInit(zLUValSubBuf_t* LUvsb, zLUstruct_t *LUstruct)
{
    zLocalLU_t *Llu = LUstruct->Llu;
    LUvsb->Lsub_buf = intMalloc_dist(Llu->bufmax[0]); //INT_T_ALLOC(Llu->bufmax[0]);
    LUvsb->Lval_buf = doublecomplexMalloc_dist(Llu->bufmax[1]); //DOUBLE_ALLOC(Llu->bufmax[1]);
    LUvsb->Usub_buf = intMalloc_dist(Llu->bufmax[2]); //INT_T_ALLOC(Llu->bufmax[2]);
    LUvsb->Uval_buf = doublecomplexMalloc_dist(Llu->bufmax[3]); //DOUBLE_ALLOC(Llu->bufmax[3]);
    return 0;
}

zdiagFactBufs_t** zinitDiagFactBufsArr(int_t mxLeafNode, int_t ldt, gridinfo_t* grid)
{
    zdiagFactBufs_t** dFBufs;

    /* Sherry fix:
     * mxLeafNode can be 0 for the replicated layers of the processes ?? */
    if ( mxLeafNode ) dFBufs = (zdiagFactBufs_t** )
                          SUPERLU_MALLOC(mxLeafNode * sizeof(zdiagFactBufs_t*));

    for (int i = 0; i < mxLeafNode; ++i)
    {
        /* code */
        dFBufs[i] = (zdiagFactBufs_t* ) SUPERLU_MALLOC(sizeof(zdiagFactBufs_t));
        assert(dFBufs[i]);
        zinitDiagFactBufs(ldt, dFBufs[i]);

    }/*Minor for loop -2 for (int i = 0; i < mxLeafNode; ++i)*/

    return dFBufs;
}

// sherry added
int zfreeDiagFactBufsArr(int_t mxLeafNode, zdiagFactBufs_t** dFBufs)
{
    for (int i = 0; i < mxLeafNode; ++i) {
	SUPERLU_FREE(dFBufs[i]->BlockUFactor);
	SUPERLU_FREE(dFBufs[i]->BlockLFactor);
	SUPERLU_FREE(dFBufs[i]);
    }

    /* Sherry fix:
     * mxLeafNode can be 0 for the replicated layers of the processes ?? */
    if ( mxLeafNode ) SUPERLU_FREE(dFBufs);

    return 0;
}

zLUValSubBuf_t** zLluBufInitArr(int_t numLA, zLUstruct_t *LUstruct)
{
    zLUValSubBuf_t** LUvsbs = (zLUValSubBuf_t**) SUPERLU_MALLOC(numLA * sizeof(zLUValSubBuf_t*));
    for (int_t i = 0; i < numLA; ++i)
    {
        /* code */
        LUvsbs[i] = (zLUValSubBuf_t*) SUPERLU_MALLOC(sizeof(zLUValSubBuf_t));
        zLluBufInit(LUvsbs[i], LUstruct);
    } /*minor for loop-3 for (int_t i = 0; i < numLA; ++i)*/

    return LUvsbs;
}

// sherry added
int zLluBufFreeArr(int_t numLA, zLUValSubBuf_t **LUvsbs)
{
    for (int_t i = 0; i < numLA; ++i) {
	SUPERLU_FREE(LUvsbs[i]->Lsub_buf);
	SUPERLU_FREE(LUvsbs[i]->Lval_buf);
	SUPERLU_FREE(LUvsbs[i]->Usub_buf);
	SUPERLU_FREE(LUvsbs[i]->Uval_buf);
	SUPERLU_FREE(LUvsbs[i]);
    }
    SUPERLU_FREE(LUvsbs);
    return 0;
}


int_t zinitScuBufs(superlu_dist_options_t *options,
                  int_t ldt, int_t num_threads, int_t nsupers,
                  zscuBufs_t* scuBufs,
                  zLUstruct_t* LUstruct,
                  gridinfo_t * grid)
{
    scuBufs->bigV = zgetBigV(ldt, num_threads);
    scuBufs->bigU = zgetBigU(options, nsupers, grid, LUstruct);
    return 0;
}

// sherry added
int zfreeScuBufs(zscuBufs_t* scuBufs)
{
    SUPERLU_FREE(scuBufs->bigV);
    SUPERLU_FREE(scuBufs->bigU);
    return 0;
}

int_t zinitDiagFactBufs(int_t ldt, zdiagFactBufs_t* dFBuf)
{
    dFBuf->BlockUFactor = doublecomplexMalloc_dist(ldt * ldt); //DOUBLE_ALLOC( ldt * ldt);
    dFBuf->BlockLFactor = doublecomplexMalloc_dist(ldt * ldt); //DOUBLE_ALLOC( ldt * ldt);
    return 0;
}

int_t zdenseTreeFactor(
    int_t nnodes,          // number of nodes in the tree
    int_t *perm_c_supno,    // list of nodes in the order of factorization
    commRequests_t *comReqs,    // lists of communication requests
    zscuBufs_t *scuBufs,   // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t*msgs,
    zLUValSubBuf_t* LUvsb,
    zdiagFactBufs_t *dFBuf,
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
)
{
    gridinfo_t* grid = &(grid3d->grid2d);
    zLocalLU_t *Llu = LUstruct->Llu;

    /*main loop over all the super nodes*/
    for (int_t k0 = 0; k0 < nnodes   ; ++k0)
    {
        int_t k = perm_c_supno[k0];   // direct computation no perm_c_supno

        /* diagonal factorization */
#if 0
        sDiagFactIBCast(k,  dFBuf, factStat, comReqs, grid,
                        options, thresh, LUstruct, stat, info, SCT, tag_ub);
#else
	zDiagFactIBCast(k, k, dFBuf->BlockUFactor, dFBuf->BlockLFactor,
			factStat->IrecvPlcd_D,
			comReqs->U_diag_blk_recv_req, 
			comReqs->L_diag_blk_recv_req,
			comReqs->U_diag_blk_send_req, 
			comReqs->L_diag_blk_send_req,
			grid, options, thresh, LUstruct, stat, info, SCT, tag_ub);
#endif

#if 0
        /*L update */
        sLPanelUpdate(k,  dFBuf, factStat, comReqs, grid, LUstruct, SCT);
        /*L Ibcast*/
        sIBcastRecvLPanel( k, comReqs,  LUvsb,  msgs, factStat, grid, LUstruct, SCT, tag_ub );
        /*U update*/
        sUPanelUpdate(k, ldt, dFBuf, factStat, comReqs, scuBufs,
                      packLUInfo, grid, LUstruct, stat, SCT);
        /*U bcast*/
        sIBcastRecvUPanel( k, comReqs,  LUvsb,  msgs, factStat, grid, LUstruct, SCT, tag_ub );
        /*Wait for L panel*/
        sWaitL(k, comReqs, msgs, grid, LUstruct, SCT);
        /*Wait for U panel*/
        sWaitU(k, comReqs, msgs, grid, LUstruct, SCT);
#else
        /*L update */
	zLPanelUpdate(k, factStat->IrecvPlcd_D, factStat->factored_L,
		      comReqs->U_diag_blk_recv_req, dFBuf->BlockUFactor, grid, LUstruct, SCT);
        /*L Ibcast*/
	zIBcastRecvLPanel(k, k, msgs->msgcnt, comReqs->send_req, comReqs->recv_req,
			  LUvsb->Lsub_buf, LUvsb->Lval_buf, factStat->factored, 
			  grid, LUstruct, SCT, tag_ub);
        /*U update*/
	zUPanelUpdate(k, factStat->factored_U, comReqs->L_diag_blk_recv_req,
		      dFBuf->BlockLFactor, scuBufs->bigV, ldt,
		      packLUInfo->Ublock_info, grid, LUstruct, stat, SCT);
        /*U bcast*/
	zIBcastRecvUPanel(k, k, msgs->msgcnt, comReqs->send_requ, comReqs->recv_requ,
			  LUvsb->Usub_buf, LUvsb->Uval_buf, 
			  grid, LUstruct, SCT, tag_ub);
	zWaitL(k, msgs->msgcnt, msgs->msgcntU, comReqs->send_req, comReqs->recv_req,
	       grid, LUstruct, SCT);
	zWaitU(k, msgs->msgcnt, comReqs->send_requ, comReqs->recv_requ, grid, LUstruct, SCT);
#endif
        double tsch = SuperLU_timer_();
#if 0
        int_t LU_nonempty = sSchurComplementSetup(k,
                            msgs, packLUInfo, gIperm_c_supno, perm_c_supno,
                            fNlists, scuBufs,  LUvsb, grid, LUstruct);
#else
	int_t LU_nonempty= zSchurComplementSetup(k, msgs->msgcnt,
				 packLUInfo->Ublock_info, packLUInfo->Remain_info,
				 packLUInfo->uPanelInfo, packLUInfo->lPanelInfo,
				 gIperm_c_supno, fNlists->iperm_u, fNlists->perm_u,
				 scuBufs->bigU, LUvsb->Lsub_buf, LUvsb->Lval_buf,
				 LUvsb->Usub_buf, LUvsb->Uval_buf,
				 grid, LUstruct);
#endif
        if (LU_nonempty)
        {
            Ublock_info_t* Ublock_info = packLUInfo->Ublock_info;
            Remain_info_t*  Remain_info = packLUInfo->Remain_info;
            uPanelInfo_t* uPanelInfo = packLUInfo->uPanelInfo;
            lPanelInfo_t* lPanelInfo = packLUInfo->lPanelInfo;
            int* indirect  = fNlists->indirect;
            int* indirect2  = fNlists->indirect2;
            /*Schurcomplement Update*/
            int_t nub = uPanelInfo->nub;
            int_t nlb = lPanelInfo->nlb;
            doublecomplex* bigV = scuBufs->bigV;
            doublecomplex* bigU = scuBufs->bigU;

#ifdef _OPENMP    
#pragma omp parallel for schedule(dynamic)
#endif
            for (int_t ij = 0; ij < nub * nlb; ++ij)
            {
                /* code */
                int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
                doublecomplex** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
                int_t** Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
                doublecomplex** Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;
                int_t* xsup = LUstruct->Glu_persist->xsup;
                int_t ub = ij / nlb;
                int_t lb
                    = ij % nlb;
                doublecomplex *L_mat = lPanelInfo->lusup;
                int_t ldl = lPanelInfo->nsupr;
                int_t luptr0 = lPanelInfo->luptr0;
                doublecomplex *U_mat = bigU;
                int_t ldu = uPanelInfo->ldu;
                int_t knsupc = SuperSize(k);
                int_t klst = FstBlockC (k + 1);
                int_t *lsub = lPanelInfo->lsub;
                int_t *usub = uPanelInfo->usub;
#ifdef _OPENMP		
                int thread_id = omp_get_thread_num();
#else		
                int thread_id = 0;
#endif		
                zblock_gemm_scatter( lb, ub,
                                    Ublock_info,
                                    Remain_info,
                                    &L_mat[luptr0], ldl,
                                    U_mat, ldu,
                                    bigV,
                                    knsupc, klst,
                                    lsub, usub, ldt,
                                    thread_id, indirect, indirect2,
                                    Lrowind_bc_ptr, Lnzval_bc_ptr,
                                    Ufstnz_br_ptr, Unzval_br_ptr,
                                    xsup, grid, stat
#ifdef SCATTER_PROFILE
                                    , Host_TheadScatterMOP, Host_TheadScatterTimer
#endif
                                  );
            } /*for (int_t ij = 0; ij < nub * nlb;*/
        } /*if (LU_nonempty)*/
        SCT->NetSchurUpTimer += SuperLU_timer_() - tsch;
#if 0
        sWait_LUDiagSend(k,  comReqs, grid, SCT);
#else
	Wait_LUDiagSend(k, comReqs->U_diag_blk_send_req, comReqs->L_diag_blk_send_req, 
			grid, SCT);
#endif
    }/*for main loop (int_t k0 = 0; k0 < gNodeCount[tree]; ++k0)*/

    return 0;
} /* zdenseTreeFactor */

/*
 * 2D factorization at individual subtree. -- CPU only
 */
int_t zsparseTreeFactor_ASYNC(
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    zscuBufs_t *scuBufs,       // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    zLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    zdiagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    zLUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
)
{
    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1)
    {
        return 1;
    }

    /* Test the input parameters. */
    *info = 0;
    
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Enter zsparseTreeFactor_ASYNC()");
#endif

    int_t *perm_c_supno = sforest->nodeList ;  // list of nodes in the order of factorization
    treeTopoInfo_t* treeTopoInfo = &sforest->topoInfo;
    int_t* myIperm = treeTopoInfo->myIperm;

    gridinfo_t* grid = &(grid3d->grid2d);
    /*main loop over all the levels*/

    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t* eTreeTopLims = treeTopoInfo->eTreeTopLims;
    int_t * IrecvPlcd_D = factStat->IrecvPlcd_D;
    int_t* factored_D = factStat->factored_D;
    int_t * factored_L = factStat->factored_L;
    int_t * factored_U = factStat->factored_U;
    int_t* IbcastPanel_L = factStat->IbcastPanel_L;
    int_t* IbcastPanel_U = factStat->IbcastPanel_U;
    int_t* xsup = LUstruct->Glu_persist->xsup;

    int_t numLAMax = getNumLookAhead(options);
    int_t numLA = numLAMax;

    for (int_t k0 = 0; k0 < eTreeTopLims[1]; ++k0)
    {
        int_t k = perm_c_supno[k0];   // direct computation no perm_c_supno
        int_t offset = k0;
        /* k-th diagonal factorization */
        /*Now factor and broadcast diagonal block*/
#if 0
        sDiagFactIBCast(k,  dFBufs[offset], factStat, comReqss[offset], grid,
                        options, thresh, LUstruct, stat, info, SCT, tag_ub);
#else
	zDiagFactIBCast(k, k, dFBufs[offset]->BlockUFactor, dFBufs[offset]->BlockLFactor,
			factStat->IrecvPlcd_D,
			comReqss[offset]->U_diag_blk_recv_req, 
			comReqss[offset]->L_diag_blk_recv_req,
			comReqss[offset]->U_diag_blk_send_req, 
			comReqss[offset]->L_diag_blk_send_req,
			grid, options, thresh, LUstruct, stat, info, SCT, tag_ub);
#endif
        factored_D[k] = 1;
    }

    for (int_t topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)
    {
        /* code */
        int_t k_st = eTreeTopLims[topoLvl];
        int_t k_end = eTreeTopLims[topoLvl + 1];
        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0];   // direct computation no perm_c_supno
            int_t offset = k0 - k_st;
            /* diagonal factorization */
            if (!factored_D[k] )
            {
                /*If LU panels from GPU are not reduced then reduce
                them before diagonal factorization*/
#if 0
                sDiagFactIBCast(k, dFBufs[offset], factStat, comReqss[offset], grid,
                                options, thresh, LUstruct, stat, info, SCT, tag_ub);
#else
		zDiagFactIBCast(k, k, dFBufs[offset]->BlockUFactor,
				dFBufs[offset]->BlockLFactor, factStat->IrecvPlcd_D,
				comReqss[offset]->U_diag_blk_recv_req, 
				comReqss[offset]->L_diag_blk_recv_req,
				comReqss[offset]->U_diag_blk_send_req, 
				comReqss[offset]->L_diag_blk_send_req,
				grid, options, thresh, LUstruct, stat, info, SCT, tag_ub);
#endif
            }
        }
        double t_apt = SuperLU_timer_();

        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0];   // direct computation no perm_c_supno
            int_t offset = k0 - k_st;

            /*L update */
            if (factored_L[k] == 0)
            {  
#if 0
		sLPanelUpdate(k, dFBufs[offset], factStat, comReqss[offset],
			      grid, LUstruct, SCT);
#else
		zLPanelUpdate(k, factStat->IrecvPlcd_D, factStat->factored_L,
			      comReqss[offset]->U_diag_blk_recv_req, 
			      dFBufs[offset]->BlockUFactor, grid, LUstruct, SCT);
#endif
                factored_L[k] = 1;
            }
            /*U update*/
            if (factored_U[k] == 0)
            {
#if 0
		sUPanelUpdate(k, ldt, dFBufs[offset], factStat, comReqss[offset],
			      scuBufs, packLUInfo, grid, LUstruct, stat, SCT);
#else
		zUPanelUpdate(k, factStat->factored_U, comReqss[offset]->L_diag_blk_recv_req,
			      dFBufs[offset]->BlockLFactor, scuBufs->bigV, ldt,
			      packLUInfo->Ublock_info, grid, LUstruct, stat, SCT);
#endif
                factored_U[k] = 1;
            }
        }

        for (int_t k0 = k_st; k0 < SUPERLU_MIN(k_end, k_st + numLA); ++k0)
        {
            int_t k = perm_c_supno[k0];   // direct computation no perm_c_supno
            int_t offset = k0 % numLA;
            /* diagonal factorization */

            /*L Ibcast*/
            if (IbcastPanel_L[k] == 0)
	    {
#if 0
                sIBcastRecvLPanel( k, comReqss[offset],  LUvsbs[offset],
                                   msgss[offset], factStat, grid, LUstruct, SCT, tag_ub );
#else
		zIBcastRecvLPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_req,
				  comReqss[offset]->recv_req, LUvsbs[offset]->Lsub_buf,
				  LUvsbs[offset]->Lval_buf, factStat->factored, 
				  grid, LUstruct, SCT, tag_ub);
#endif
                IbcastPanel_L[k] = 1; /*for consistancy; unused later*/
            }

            /*U Ibcast*/
            if (IbcastPanel_U[k] == 0)
            {
#if 0
                sIBcastRecvUPanel( k, comReqss[offset],  LUvsbs[offset],
                                   msgss[offset], factStat, grid, LUstruct, SCT, tag_ub );
#else
		zIBcastRecvUPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_requ,
				  comReqss[offset]->recv_requ, LUvsbs[offset]->Usub_buf,
				  LUvsbs[offset]->Uval_buf, grid, LUstruct, SCT, tag_ub);
#endif
                IbcastPanel_U[k] = 1;
            }
        }

        // if (topoLvl) SCT->tAsyncPipeTail += SuperLU_timer_() - t_apt;
        SCT->tAsyncPipeTail += SuperLU_timer_() - t_apt;

        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0];   // direct computation no perm_c_supno
            int_t offset = k0 % numLA;

#if 0
            sWaitL(k, comReqss[offset], msgss[offset], grid, LUstruct, SCT);
            /*Wait for U panel*/
            sWaitU(k, comReqss[offset], msgss[offset], grid, LUstruct, SCT);
#else
	    zWaitL(k, msgss[offset]->msgcnt, msgss[offset]->msgcntU, 
		   comReqss[offset]->send_req, comReqss[offset]->recv_req,
		   grid, LUstruct, SCT);
	    zWaitU(k, msgss[offset]->msgcnt, comReqss[offset]->send_requ, 
		   comReqss[offset]->recv_requ, grid, LUstruct, SCT);
#endif
            double tsch = SuperLU_timer_();
            int_t LU_nonempty = zSchurComplementSetupGPU(k,
							 msgss[offset], packLUInfo,
							 myIperm, gIperm_c_supno, 
							 perm_c_supno, gEtreeInfo,
							 fNlists, scuBufs,
							 LUvsbs[offset],
							 grid, LUstruct, HyP);
            // initializing D2H data transfer
            int_t jj_cpu = 0;

            scuStatUpdate( SuperSize(k), HyP,  SCT, stat);
            uPanelInfo_t* uPanelInfo = packLUInfo->uPanelInfo;
            lPanelInfo_t* lPanelInfo = packLUInfo->lPanelInfo;
            int_t *lsub = lPanelInfo->lsub;
            int_t *usub = uPanelInfo->usub;
            int* indirect  = fNlists->indirect;
            int* indirect2  = fNlists->indirect2;

            /*Schurcomplement Update*/

            int_t knsupc = SuperSize(k);
            int_t klst = FstBlockC (k + 1);

            doublecomplex* bigV = scuBufs->bigV;
	    
#ifdef _OPENMP    
#pragma omp parallel
#endif
            {
#ifdef _OPENMP    
#pragma omp for schedule(dynamic,2) nowait
#endif
		/* Each thread is assigned one loop index ij, responsible for
		   block update L(lb,k) * U(k,j) -> tempv[]. */
                for (int_t ij = 0; ij < HyP->lookAheadBlk * HyP->num_u_blks; ++ij)
                {
		    /* Get the entire area of L (look-ahead) X U (all-blocks). */
		    /* for each j-block in U, go through all L-blocks in the
		       look-ahead window. */
                    int_t j   = ij / HyP->lookAheadBlk; 
							   
                    int_t lb  = ij % HyP->lookAheadBlk;
                    zblock_gemm_scatterTopLeft( lb,  j, bigV, knsupc, klst, lsub,
					       usub, ldt,  indirect, indirect2, HyP,
					       LUstruct, grid, SCT, stat );
                }

#ifdef _OPENMP    
#pragma omp for schedule(dynamic,2) nowait
#endif
                for (int_t ij = 0; ij < HyP->lookAheadBlk * HyP->num_u_blks_Phi; ++ij)
                {
                    int_t j   = ij / HyP->lookAheadBlk ;
                    int_t lb  = ij % HyP->lookAheadBlk;
                    zblock_gemm_scatterTopRight( lb,  j, bigV, knsupc, klst, lsub,
                                                usub, ldt,  indirect, indirect2, HyP,
						LUstruct, grid, SCT, stat);
                }

#ifdef _OPENMP    
#pragma omp for schedule(dynamic,2) nowait
#endif
                for (int_t ij = 0; ij < HyP->RemainBlk * HyP->num_u_blks; ++ij) //
                {
                    int_t j   = ij / HyP->RemainBlk;
                    int_t lb  = ij % HyP->RemainBlk;
                    zblock_gemm_scatterBottomLeft( lb,  j, bigV, knsupc, klst, lsub,
                                                  usub, ldt,  indirect, indirect2,
						  HyP, LUstruct, grid, SCT, stat);
                } /*for (int_t ij =*/
            }

            if (topoLvl < maxTopoLevel - 1)
            {
                int_t k_parent = gEtreeInfo->setree[k];
                gEtreeInfo->numChildLeft[k_parent]--;
                if (gEtreeInfo->numChildLeft[k_parent] == 0)
                {
                    int_t k0_parent =  myIperm[k_parent];
                    if (k0_parent > 0)
                    {
                        /* code */
                        assert(k0_parent < nnodes);
                        int_t offset = k0_parent - k_end;
#if 0
                        sDiagFactIBCast(k_parent,  dFBufs[offset], factStat,
					comReqss[offset], grid, options, thresh,
					LUstruct, stat, info, SCT, tag_ub);
#else
			zDiagFactIBCast(k_parent, k_parent, dFBufs[offset]->BlockUFactor,
					dFBufs[offset]->BlockLFactor, factStat->IrecvPlcd_D,
					comReqss[offset]->U_diag_blk_recv_req, 
					comReqss[offset]->L_diag_blk_recv_req,
					comReqss[offset]->U_diag_blk_send_req, 
					comReqss[offset]->L_diag_blk_send_req,
					grid, options, thresh, LUstruct, stat, info, SCT, tag_ub);
#endif
                        factored_D[k_parent] = 1;
                    }

                }
            }

#ifdef _OPENMP    
#pragma omp parallel
#endif
            {
#ifdef _OPENMP    
#pragma omp for schedule(dynamic,2) nowait
#endif
                for (int_t ij = 0; ij < HyP->RemainBlk * (HyP->num_u_blks_Phi - jj_cpu) ; ++ij)
                {
                    int_t j   = ij / HyP->RemainBlk + jj_cpu;
                    int_t lb  = ij % HyP->RemainBlk;
                    zblock_gemm_scatterBottomRight( lb,  j, bigV, knsupc, klst, lsub,
                                                   usub, ldt,  indirect, indirect2,
						   HyP, LUstruct, grid, SCT, stat);
                } /*for (int_t ij =*/

            }

            SCT->NetSchurUpTimer += SuperLU_timer_() - tsch;
            // finish waiting for diag block send
            int_t abs_offset = k0 - k_st;
#if 0
            sWait_LUDiagSend(k,  comReqss[abs_offset], grid, SCT);
#else
	    Wait_LUDiagSend(k, comReqss[abs_offset]->U_diag_blk_send_req, 
			    comReqss[abs_offset]->L_diag_blk_send_req, 
			    grid, SCT);
#endif
            /*Schedule next I bcasts*/
            for (int_t next_k0 = k0 + 1; next_k0 < SUPERLU_MIN( k0 + 1 + numLA, nnodes); ++next_k0)
            {
                /* code */
                int_t next_k = perm_c_supno[next_k0];
                int_t offset = next_k0 % numLA;

                /*L Ibcast*/
                if (IbcastPanel_L[next_k] == 0 && factored_L[next_k])
                {
#if 0
                    sIBcastRecvLPanel( next_k, comReqss[offset], 
				       LUvsbs[offset], msgss[offset], factStat,
				       grid, LUstruct, SCT, tag_ub );
#else
		    zIBcastRecvLPanel(next_k, next_k, msgss[offset]->msgcnt, 
				      comReqss[offset]->send_req, comReqss[offset]->recv_req,
				      LUvsbs[offset]->Lsub_buf, LUvsbs[offset]->Lval_buf,
				      factStat->factored, grid, LUstruct, SCT, tag_ub);
#endif
                    IbcastPanel_L[next_k] = 1; /*will be used later*/
                }
                /*U Ibcast*/
                if (IbcastPanel_U[next_k] == 0 && factored_U[next_k])
                {
#if 0
                    sIBcastRecvUPanel( next_k, comReqss[offset],
				       LUvsbs[offset], msgss[offset], factStat,
				       grid, LUstruct, SCT, tag_ub );
#else
		    zIBcastRecvUPanel(next_k, next_k, msgss[offset]->msgcnt, 
				      comReqss[offset]->send_requ, comReqss[offset]->recv_requ,
				      LUvsbs[offset]->Usub_buf, LUvsbs[offset]->Uval_buf, 
				      grid, LUstruct, SCT, tag_ub);
#endif
                    IbcastPanel_U[next_k] = 1;
                }
            }

            if (topoLvl < maxTopoLevel - 1)
            {

                /*look ahead LU factorization*/
                int_t kx_st = eTreeTopLims[topoLvl + 1];
                int_t kx_end = eTreeTopLims[topoLvl + 2];
                for (int_t k0x = kx_st; k0x < kx_end; k0x++)
                {
                    /* code */
                    int_t kx = perm_c_supno[k0x];
                    int_t offset = k0x - kx_st;
                    if (IrecvPlcd_D[kx] && !factored_L[kx])
                    {
                        /*check if received*/
                        int_t recvUDiag = checkRecvUDiag(kx, comReqss[offset],
                                                         grid, SCT);
                        if (recvUDiag)
                        {
#if 0
                            sLPanelTrSolve( kx,  dFBufs[offset],
                                            factStat, comReqss[offset],
                                            grid, LUstruct, SCT);
#else
			    zLPanelTrSolve( kx, factStat->factored_L, 
					    dFBufs[offset]->BlockUFactor, grid, LUstruct);
#endif

                            factored_L[kx] = 1;

                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_L[kx] == 0 &&
                                    k0x - k0 < numLA + 1  && // is within lookahead window
                                    factored_L[kx])
                            {
                                int_t offset1 = k0x % numLA;
#if 0
                                sIBcastRecvLPanel( kx, comReqss[offset1], LUvsbs[offset1],
                                                   msgss[offset1], factStat,
						   grid, LUstruct, SCT, tag_ub);
#else
				zIBcastRecvLPanel(kx, kx, msgss[offset1]->msgcnt, 
						  comReqss[offset1]->send_req,
						  comReqss[offset1]->recv_req,
						  LUvsbs[offset1]->Lsub_buf,
						  LUvsbs[offset1]->Lval_buf, 
						  factStat->factored, 
						  grid, LUstruct, SCT, tag_ub);
#endif
                                IbcastPanel_L[kx] = 1; /*will be used later*/
                            }

                        }
                    }

                    if (IrecvPlcd_D[kx] && !factored_U[kx])
                    {
                        /*check if received*/
                        int_t recvLDiag = checkRecvLDiag( kx, comReqss[offset],
                                                          grid, SCT);
                        if (recvLDiag)
                        {
#if 0
                            sUPanelTrSolve( kx, ldt, dFBufs[offset], scuBufs, packLUInfo,
                                            grid, LUstruct, stat, SCT);
#else
			    zUPanelTrSolve( kx, dFBufs[offset]->BlockLFactor,
                                            scuBufs->bigV,
					    ldt, packLUInfo->Ublock_info, 
					    grid, LUstruct, stat, SCT);
#endif
                            factored_U[kx] = 1;
                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_U[kx] == 0 &&
                                    k0x - k0 < numLA + 1  && // is within lookahead window
                                    factored_U[kx])
                            {
                                int_t offset = k0x % numLA;
#if 0
                                sIBcastRecvUPanel( kx, comReqss[offset],
						   LUvsbs[offset],
						   msgss[offset], factStat,
						   grid, LUstruct, SCT, tag_ub);
#else
				zIBcastRecvUPanel(kx, kx, msgss[offset]->msgcnt, 
						  comReqss[offset]->send_requ,
						  comReqss[offset]->recv_requ,
						  LUvsbs[offset]->Usub_buf,
						  LUvsbs[offset]->Uval_buf, 
						  grid, LUstruct, SCT, tag_ub);
#endif
                                IbcastPanel_U[kx] = 1; /*will be used later*/
                            }
                        }
                    }
                }

            }
        }/*for main loop (int_t k0 = 0; k0 < gNodeCount[tree]; ++k0)*/

    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Exit zsparseTreeFactor_ASYNC()");
#endif

    return 0;
} /* zsparseTreeFactor_ASYNC */
