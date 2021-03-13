#include "superlu_ddefs.h"

int_t dsparseTreeFactor_v100(
    sForest_t* sforest,
    commRequests_t **comReqss,    // lists of communication requests // size maxEtree level
    scuBufs_t *scuBufs,          // contains buffers for schur complement update
    packLUInfo_t*packLUInfo,
    msgs_t**msgss,                  // size=num Look ahead
    dLUValSubBuf_t** LUvsbs,          // size=num Look ahead
    diagFactBufs_t **dFBufs,         // size maxEtree level
    factStat_t *factStat,
    factNodelists_t  *fNlists,
    gEtreeInfo_t*   gEtreeInfo,        // global etree info
    superlu_dist_options_t *options,
    int_t * gIperm_c_supno,
    int_t ldt,
    HyP_t* HyP,
    LUstruct_t *LUstruct, gridinfo3d_t * grid3d, SuperLUStat_t *stat,
    double thresh,  SCT_t *SCT, int tag_ub,
    int *info
)
{
    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1)
    {
        return 1;
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Enter dsparseTreeFactor_ASYNC()");
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
	dDiagFactIBCast(k, k, dFBufs[offset]->BlockUFactor, dFBufs[offset]->BlockLFactor,
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
		dDiagFactIBCast(k, k, dFBufs[offset]->BlockUFactor,
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
		dLPanelUpdate(k, factStat->IrecvPlcd_D, factStat->factored_L,
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
		dUPanelUpdate(k, factStat->factored_U, comReqss[offset]->L_diag_blk_recv_req,
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
		dIBcastRecvLPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_req,
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
		dIBcastRecvUPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_requ,
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
	    dWaitL(k, msgss[offset]->msgcnt, msgss[offset]->msgcntU, 
		   comReqss[offset]->send_req, comReqss[offset]->recv_req,
		   grid, LUstruct, SCT);
	    dWaitU(k, msgss[offset]->msgcnt, comReqss[offset]->send_requ, 
		   comReqss[offset]->recv_requ, grid, LUstruct, SCT);
#endif
            double tsch = SuperLU_timer_();
            int_t LU_nonempty = dSchurComplementSetupGPU(k,
							 msgss[offset], packLUInfo,
							 myIperm, gIperm_c_supno, 
							 perm_c_supno, gEtreeInfo,
							 fNlists, scuBufs,
							 LUvsbs[offset],
							 grid, LUstruct, HyP);
            // initializing D2H data transfer
            int_t jj_cpu = 0;

            // scuStatUpdate( SuperSize(k), HyP,  SCT, stat);
            uPanelInfo_t* uPanelInfo = packLUInfo->uPanelInfo;
            lPanelInfo_t* lPanelInfo = packLUInfo->lPanelInfo;
            int_t *lsub = lPanelInfo->lsub;
            int_t *usub = uPanelInfo->usub;
            int* indirect  = fNlists->indirect;
            int* indirect2  = fNlists->indirect2;

            /*Schurcomplement Update*/

            int_t knsupc = SuperSize(k);
            int_t klst = FstBlockC (k + 1);

            double* bigV = scuBufs->bigV;

#pragma omp parallel
            {
#pragma omp for schedule(dynamic,2) nowait
		/* Each thread is assigned one loop index ij, responsible for
		   block update L(lb,k) * U(k,j) -> tempv[]. */
                for (int_t ij = 0; ij < HyP->lookAheadBlk * HyP->num_u_blks; ++ij)
                {
		    /* Get the entire area of L (look-ahead) X U (all-blocks). */
		    /* for each j-block in U, go through all L-blocks in the
		       look-ahead window. */
                    int_t j   = ij / HyP->lookAheadBlk; 
							   
                    int_t lb  = ij % HyP->lookAheadBlk;
                    dblock_gemm_scatterTopLeft( lb,  j, bigV, knsupc, klst, lsub,
					       usub, ldt,  indirect, indirect2, HyP,
					       LUstruct, grid, SCT, stat );
                }

#pragma omp for schedule(dynamic,2) nowait
                for (int_t ij = 0; ij < HyP->lookAheadBlk * HyP->num_u_blks_Phi; ++ij)
                {
                    int_t j   = ij / HyP->lookAheadBlk ;
                    int_t lb  = ij % HyP->lookAheadBlk;
                    dblock_gemm_scatterTopRight( lb,  j, bigV, knsupc, klst, lsub,
                                                usub, ldt,  indirect, indirect2, HyP,
						LUstruct, grid, SCT, stat);
                }

#pragma omp for schedule(dynamic,2) nowait
                for (int_t ij = 0; ij < HyP->RemainBlk * HyP->num_u_blks; ++ij) //
                {
                    int_t j   = ij / HyP->RemainBlk;
                    int_t lb  = ij % HyP->RemainBlk;
                    dblock_gemm_scatterBottomLeft( lb,  j, bigV, knsupc, klst, lsub,
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
			dDiagFactIBCast(k_parent, k_parent, dFBufs[offset]->BlockUFactor,
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

#pragma omp parallel
            {
#pragma omp for schedule(dynamic,2) nowait
                for (int_t ij = 0; ij < HyP->RemainBlk * (HyP->num_u_blks_Phi - jj_cpu) ; ++ij)
                {
                    int_t j   = ij / HyP->RemainBlk + jj_cpu;
                    int_t lb  = ij % HyP->RemainBlk;
                    dblock_gemm_scatterBottomRight( lb,  j, bigV, knsupc, klst, lsub,
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
		    dIBcastRecvLPanel(next_k, next_k, msgss[offset]->msgcnt, 
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
		    dIBcastRecvUPanel(next_k, next_k, msgss[offset]->msgcnt, 
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
			    dLPanelTrSolve( kx, factStat->factored_L, 
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
				dIBcastRecvLPanel(kx, kx, msgss[offset1]->msgcnt, 
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
			    dUPanelTrSolve( kx, dFBufs[offset]->BlockLFactor,
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
				dIBcastRecvUPanel(kx, kx, msgss[offset]->msgcnt, 
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
    CHECK_MALLOC (grid3d->iam, "Exit dsparseTreeFactor_ASYNC()");
#endif

    return 0;
} /* dsparseTreeFactor_ASYNC */
