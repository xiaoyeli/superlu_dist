// #include "treeFactorization.h"
// #include "trfCommWrapper.h"
#include "dlustruct_gpu.h"

/* 
/-- num_u_blks--\ /-- num_u_blks_Phi --\
----------------------------------------
|  host_cols    ||    GPU   |   host   |
----------------------------------------
                  ^          ^
                  0          jj_cpu
*/
static int_t getAccUPartition(HyP_t *HyP)
{
    /* Sherry: what if num_u_blks_phi == 0 ? Need to fix the bug */
    int_t total_cols_1 = HyP->Ublock_info_Phi[HyP->num_u_blks_Phi - 1].full_u_cols;

    int_t host_cols = HyP->Ublock_info[HyP->num_u_blks - 1].full_u_cols;
    double cpu_time_0 = estimate_cpu_time(HyP->Lnbrow, total_cols_1, HyP->ldu_Phi) +
                        estimate_cpu_time(HyP->Rnbrow, host_cols, HyP->ldu) + estimate_cpu_time(HyP->Lnbrow, host_cols, HyP->ldu);

    int jj_cpu;

#if 0 /* Ignoe those estimates */
    jj_cpu = tuned_partition(HyP->num_u_blks_Phi, HyP->Ublock_info_Phi,
                                   HyP->Remain_info, HyP->RemainBlk, cpu_time_0, HyP->Rnbrow, HyP->ldu_Phi );
#else /* Sherry: new */
    jj_cpu = HyP->num_u_blks_Phi;
#endif

    if (jj_cpu != 0 && HyP->Rnbrow > 0) // ###
    {
        HyP->offloadCondition = 1;
    }
    else
    {
        HyP->offloadCondition = 0;
        jj_cpu = 0; // ###
    }

    return jj_cpu;
}

int dsparseTreeFactor_ASYNC_GPU(
    sForest_t *sforest,
    commRequests_t **comReqss, // lists of communication requests,
                               // size = maxEtree level
    scuBufs_t *scuBufs,        // contains buffers for schur complement update
    packLUInfo_t *packLUInfo,
    msgs_t **msgss,          // size = num Look ahead
    dLUValSubBuf_t **LUvsbs, // size = num Look ahead
    diagFactBufs_t **dFBufs, // size = maxEtree level
    factStat_t *factStat,
    factNodelists_t *fNlists,
    gEtreeInfo_t *gEtreeInfo, // global etree info
    superlu_dist_options_t *options,
    int_t *gIperm_c_supno,
    int_t ldt,
    sluGPU_t *sluGPU,
    d2Hreduce_t *d2Hred,
    HyP_t *HyP,
    dLUstruct_t *LUstruct, gridinfo3d_t *grid3d, SuperLUStat_t *stat,
    double thresh, SCT_t *SCT, int tag_ub,
    int *info)
{
    // sforest.nNodes, sforest.nodeList,
    // &sforest.topoInfo,
    int_t nnodes = sforest->nNodes; // number of nodes in supernodal etree
    if (nnodes < 1)
    {
        return 1;
    }

    int_t *perm_c_supno = sforest->nodeList; // list of nodes in the order of factorization
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t *myIperm = treeTopoInfo->myIperm;

    gridinfo_t *grid = &(grid3d->grid2d);
    /*main loop over all the levels*/

    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;
    int_t *IrecvPlcd_D = factStat->IrecvPlcd_D;
    int_t *factored_D = factStat->factored_D;
    int_t *factored_L = factStat->factored_L;
    int_t *factored_U = factStat->factored_U;
    int_t *IbcastPanel_L = factStat->IbcastPanel_L;
    int_t *IbcastPanel_U = factStat->IbcastPanel_U;
    int_t *gpuLUreduced = factStat->gpuLUreduced;
    int_t *xsup = LUstruct->Glu_persist->xsup;

    // int_t numLAMax = getNumLookAhead();
    int_t numLAMax = getNumLookAhead(options);
    int_t numLA = numLAMax; // number of look-ahead panels
    int_t superlu_acc_offload = HyP->superlu_acc_offload;
    int_t last_flag = 1;                       /* for updating nsuper-1 only once */
    int_t nGPUStreams = sluGPU->nGPUStreams; // number of gpu streams

    if (superlu_acc_offload)
        syncAllfunCallStreams(sluGPU, SCT);

    /* Go through each leaf node */
    for (int_t k0 = 0; k0 < eTreeTopLims[1]; ++k0)
    {
        int_t k = perm_c_supno[k0]; // direct computation no perm_c_supno
        int_t offset = k0;
        /* k-th diagonal factorization */

        /* If LU panels from GPU are not reduced, then reduce
	   them before diagonal factorization */
        if (!gpuLUreduced[k] && superlu_acc_offload)
        {
            double tt_start1 = SuperLU_timer_();

            initD2Hreduce(k, d2Hred, last_flag,
                          HyP, sluGPU, grid, LUstruct, SCT);
            int_t copyL_kljb = d2Hred->copyL_kljb;
            int_t copyU_kljb = d2Hred->copyU_kljb;

            if (copyL_kljb || copyU_kljb)
                SCT->PhiMemCpyCounter++;
            sendLUpanelGPU2HOST(k, d2Hred, sluGPU);

            reduceGPUlu(last_flag, d2Hred,
                        sluGPU, SCT, grid, LUstruct);

            gpuLUreduced[k] = 1;
            SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
        }

        double t1 = SuperLU_timer_();

        /*Now factor and broadcast diagonal block*/
        // sDiagFactIBCast(k, dFBufs[offset], factStat, comReqss[offset], grid,
        //                 options, thresh, LUstruct, stat, info, SCT);

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

        SCT->pdgstrf2_timer += (SuperLU_timer_() - t1);
    } /* for all leaves ... */

    //printf(".. SparseFactor_GPU: after leaves\n"); fflush(stdout);

    /* Process supernodal etree level by level */
    for (int_t topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)
    // for (int_t topoLvl = 0; topoLvl < 1; ++topoLvl)
    {
        //      printf("(%d) factor level %d, maxTopoLevel %d\n",grid3d->iam,topoLvl,maxTopoLevel); fflush(stdout);
        /* code */
        int_t k_st = eTreeTopLims[topoLvl];
        int_t k_end = eTreeTopLims[topoLvl + 1];

        /* Process all the nodes in 'topoLvl': diagonal factorization */
        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int_t offset = k0 - k_st;

            if (!factored_D[k])
            {
                /*If LU panels from GPU are not reduced then reduce
		  them before diagonal factorization*/
                if (!gpuLUreduced[k] && superlu_acc_offload)
                {
                    double tt_start1 = SuperLU_timer_();
                    initD2Hreduce(k, d2Hred, last_flag,
                                  HyP, sluGPU, grid, LUstruct, SCT);
                    int_t copyL_kljb = d2Hred->copyL_kljb;
                    int_t copyU_kljb = d2Hred->copyU_kljb;

                    if (copyL_kljb || copyU_kljb)
                        SCT->PhiMemCpyCounter++;
                    sendLUpanelGPU2HOST(k, d2Hred, sluGPU);
                    /*
                        Reduce the LU panels from GPU
                    */
                    reduceGPUlu(last_flag, d2Hred,
                                sluGPU, SCT, grid, LUstruct);

                    gpuLUreduced[k] = 1;
                    SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
                }

                double t1 = SuperLU_timer_();
                /* Factor diagonal block on CPU */
                // sDiagFactIBCast(k, dFBufs[offset], factStat, comReqss[offset], grid,
                //                 options, thresh, LUstruct, stat, info, SCT);
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
                SCT->pdgstrf2_timer += (SuperLU_timer_() - t1);
            }
        } /* for all nodes in this level */

        //printf(".. SparseFactor_GPU: after diag factorization\n"); fflush(stdout);

        double t_apt = SuperLU_timer_(); /* Async Pipe Timer */

        /* Process all the nodes in 'topoLvl': panel updates on CPU */
        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0]; // direct computation no perm_c_supno
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
        } /* end panel update */

        //printf(".. after CPU panel updates. numLA %d\n", numLA); fflush(stdout);

        /* Process all the panels in look-ahead window: 
	   broadcast L and U panels. */
        for (int_t k0 = k_st; k0 < SUPERLU_MIN(k_end, k_st + numLA); ++k0)
        {
            int_t k = perm_c_supno[k0]; // direct computation no perm_c_supno
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
        } /* end for panels in look-ahead window */

        //printf(".. after CPU look-ahead updates\n"); fflush(stdout);

        // if (topoLvl) SCT->tAsyncPipeTail += SuperLU_timer_() - t_apt;
        SCT->tAsyncPipeTail += (SuperLU_timer_() - t_apt);

        /* Process all the nodes in level 'topoLvl': Schur complement update
	   (no MPI communication)  */
        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int_t offset = k0 % numLA;

            double tsch = SuperLU_timer_();

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

            int_t LU_nonempty = dSchurComplementSetupGPU(k,
                                                         msgss[offset], packLUInfo,
                                                         myIperm, gIperm_c_supno, perm_c_supno,
                                                         gEtreeInfo, fNlists, scuBufs,
                                                         LUvsbs[offset], grid, LUstruct, HyP);
            // initializing D2H data transfer. D2H = Device To Host.
            int_t jj_cpu; /* limit between CPU and GPU */

#if 1
            if (superlu_acc_offload)
            {
                jj_cpu = HyP->num_u_blks_Phi; // -1 ??
                HyP->offloadCondition = 1;
            }
            else
            {
                /* code */
                HyP->offloadCondition = 0;
                jj_cpu = 0;
            }

#else
            if (superlu_acc_offload)
            {
                jj_cpu = getAccUPartition(HyP);

                if (jj_cpu > 0)
                    jj_cpu = HyP->num_u_blks_Phi;

                /* Sherry force this --> */
                jj_cpu = HyP->num_u_blks_Phi; // -1 ??
                HyP->offloadCondition = 1;
            }
            else
            {
                jj_cpu = 0;
            }
#endif

            // int_t jj_cpu = HyP->num_u_blks_Phi-1;
            // if (HyP->Rnbrow > 0 && jj_cpu>=0)
            //     HyP->offloadCondition = 1;
            // else
            //     HyP->offloadCondition = 0;
            //     jj_cpu=0;
#if 0
	    if ( HyP->offloadCondition ) {
	    printf("(%d) k=%d, nub=%d, nub_host=%d, nub_phi=%d, jj_cpu %d, offloadCondition %d\n",
		   grid3d->iam, k, HyP->num_u_blks+HyP->num_u_blks_Phi ,
		   HyP->num_u_blks, HyP->num_u_blks_Phi,
		   jj_cpu, HyP->offloadCondition);
	    fflush(stdout);
	    }
#endif
            scuStatUpdate(SuperSize(k), HyP, SCT, stat);

            int_t offload_condition = HyP->offloadCondition;
            uPanelInfo_t *uPanelInfo = packLUInfo->uPanelInfo;
            lPanelInfo_t *lPanelInfo = packLUInfo->lPanelInfo;
            int_t *lsub = lPanelInfo->lsub;
            int_t *usub = uPanelInfo->usub;
            int_t *indirect = fNlists->indirect;
            int_t *indirect2 = fNlists->indirect2;

            /* Schur Complement Update */

            int_t knsupc = SuperSize(k);
            int_t klst = FstBlockC(k + 1);

            double *bigV = scuBufs->bigV;
            double *bigU = scuBufs->bigU;

            double t1 = SuperLU_timer_();

#pragma omp parallel /* Look-ahead update on CPU */
            {
                int_t thread_id = omp_get_thread_num();

#pragma omp for
                for (int_t ij = 0; ij < HyP->lookAheadBlk * HyP->num_u_blks; ++ij)
                {
                    int_t j = ij / HyP->lookAheadBlk;
                    int_t lb = ij % HyP->lookAheadBlk;
                    dblock_gemm_scatterTopLeft(lb, j, bigV, knsupc, klst, lsub,
                                               usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
                }

#pragma omp for
                for (int_t ij = 0; ij < HyP->lookAheadBlk * HyP->num_u_blks_Phi; ++ij)
                {
                    int_t j = ij / HyP->lookAheadBlk;
                    int_t lb = ij % HyP->lookAheadBlk;
                    dblock_gemm_scatterTopRight(lb, j, bigV, knsupc, klst, lsub,
                                                usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
                }

#pragma omp for
                for (int_t ij = 0; ij < HyP->RemainBlk * HyP->num_u_blks; ++ij)
                {
                    int_t j = ij / HyP->RemainBlk;
                    int_t lb = ij % HyP->RemainBlk;
                    dblock_gemm_scatterBottomLeft(lb, j, bigV, knsupc, klst, lsub,
                                                  usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
                } /* for int_t ij = ... */
            }     /* end parallel region ... end look-ahead update */

            SCT->lookaheadupdatetimer += (SuperLU_timer_() - t1);

            //printf("... after look-ahead update, topoLvl %d\t maxTopoLevel %d\n", topoLvl, maxTopoLevel); fflush(stdout);

            /* Reduce the L & U panels from GPU to CPU.       */
            if (topoLvl < maxTopoLevel - 1)
            { /* Not the root */
                int_t k_parent = gEtreeInfo->setree[k];
                gEtreeInfo->numChildLeft[k_parent]--;
                if (gEtreeInfo->numChildLeft[k_parent] == 0 && k_parent < nnodes)
                { /* if k is the last child in this level */
                    int_t k0_parent = myIperm[k_parent];
                    if (k0_parent > 0)
                    {
                        /* code */
                        //      printf("Before assert: iam %d, k %d, k_parent %d, k0_parent %d, nnodes %d\n", grid3d->iam, k, k_parent, k0_parent, nnodes); fflush(stdout);
                        //	      exit(-1);
                        assert(k0_parent < nnodes);
                        int_t offset = k0_parent - k_end;
                        if (!gpuLUreduced[k_parent] && superlu_acc_offload)
                        {
                            double tt_start1 = SuperLU_timer_();

                            initD2Hreduce(k_parent, d2Hred, last_flag,
                                          HyP, sluGPU, grid, LUstruct, SCT);
                            int_t copyL_kljb = d2Hred->copyL_kljb;
                            int_t copyU_kljb = d2Hred->copyU_kljb;

                            if (copyL_kljb || copyU_kljb)
                                SCT->PhiMemCpyCounter++;
                            sendLUpanelGPU2HOST(k_parent, d2Hred, sluGPU);

                            /* Reduce the LU panels from GPU */
                            reduceGPUlu(last_flag, d2Hred,
                                        sluGPU, SCT, grid, LUstruct);

                            gpuLUreduced[k_parent] = 1;
                            SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
                        }

                        /* Factorize diagonal block on CPU */
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
                    } /* end if k0_parent > 0 */

                } /* end if all children are done */
            }     /* end if non-root */

#pragma omp parallel
            {
                /* Master thread performs Schur complement update on GPU. */
#pragma omp master
                {
                    if (superlu_acc_offload)
                    {
                        int_t thread_id = omp_get_thread_num();
                        double t1 = SuperLU_timer_();

                        if (offload_condition)
                        {
                            SCT->datatransfer_count++;
                            int_t streamId = k0 % nGPUStreams;

                            /*wait for previous offload to get finished*/
                            if (sluGPU->lastOffloadStream[streamId] != -1)
                            {
                                waitGPUscu(streamId, sluGPU, SCT);
                                sluGPU->lastOffloadStream[streamId] = -1;
                            }

                            int_t Remain_lbuf_send_size = knsupc * HyP->Rnbrow;
                            int_t bigu_send_size = jj_cpu < 1 ? 0 : HyP->ldu_Phi * HyP->Ublock_info_Phi[jj_cpu - 1].full_u_cols;
                            assert(bigu_send_size < HyP->bigu_size);

                            /* !! Sherry add the test to avoid seg_fault inside sendSCUdataHost2GPU */
                            if (bigu_send_size > 0)
                            {
                                sendSCUdataHost2GPU(streamId, lsub, usub, bigU, bigu_send_size,
                                                    Remain_lbuf_send_size, sluGPU, HyP);

                                sluGPU->lastOffloadStream[streamId] = k0;
                                int_t usub_len = usub[2];
                                int_t lsub_len = lsub[1] + BC_HEADER + lsub[0] * LB_DESCRIPTOR;
                                //{printf("... before SchurCompUpdate_GPU, bigu_send_size %d\n", bigu_send_size); fflush(stdout);}

                                SchurCompUpdate_GPU(
                                    streamId, 0, jj_cpu, klst, knsupc, HyP->Rnbrow, HyP->RemainBlk,
                                    Remain_lbuf_send_size, bigu_send_size, HyP->ldu_Phi, HyP->num_u_blks_Phi,
                                    HyP->buffer_size, lsub_len, usub_len, ldt, k0, sluGPU, grid);
                            } /* endif bigu_send_size > 0 */

                            // sendLUpanelGPU2HOST( k0, d2Hred, sluGPU);

                            SCT->schurPhiCallCount++;
                            HyP->jj_cpu = jj_cpu;
                            updateDirtyBit(k0, HyP, grid);
                        } /* endif (offload_condition) */

                        double t2 = SuperLU_timer_();
                        SCT->SchurCompUdtThreadTime[thread_id * CACHE_LINE_SIZE] += (double)(t2 - t1); /* not used */
                        SCT->CPUOffloadTimer += (double)(t2 - t1);                                     // Sherry added

                    } /* endif (superlu_acc_offload) */

                } /* end omp master thread */

#pragma omp for
                /* The following update is on CPU. Should not be necessary now,
		   because we set jj_cpu equal to num_u_blks_Phi.      		*/
                for (int_t ij = 0; ij < HyP->RemainBlk * (HyP->num_u_blks_Phi - jj_cpu); ++ij)
                {
                    //printf(".. WARNING: should NOT get here\n");
                    int_t j = ij / HyP->RemainBlk + jj_cpu;
                    int_t lb = ij % HyP->RemainBlk;
                    dblock_gemm_scatterBottomRight(lb, j, bigV, knsupc, klst, lsub,
                                                   usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
                } /* for int_t ij = ... */

            } /* end omp parallel region */

            //SCT->NetSchurUpTimer += SuperLU_timer_() - tsch;

            // finish waiting for diag block send
            int_t abs_offset = k0 - k_st;
#if 0
            sWait_LUDiagSend(k,  comReqss[abs_offset], grid, SCT);
#else
            Wait_LUDiagSend(k, comReqss[abs_offset]->U_diag_blk_send_req,
                            comReqss[abs_offset]->L_diag_blk_send_req,
                            grid, SCT);
#endif

            /*Schedule next I bcasts within look-ahead window */
            for (int_t next_k0 = k0 + 1; next_k0 < SUPERLU_MIN(k0 + 1 + numLA, nnodes); ++next_k0)
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
            } /* end for look-ahead window */

            if (topoLvl < maxTopoLevel - 1) /* not root */
            {
                /*look-ahead LU factorization*/
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
                            dLPanelTrSolve(kx, factStat->factored_L,
                                           dFBufs[offset]->BlockUFactor, grid, LUstruct);
#endif

                            factored_L[kx] = 1;

                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_L[kx] == 0 &&
                                k0x - k0 < numLA + 1 && // is within look-ahead window
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
                        int_t recvLDiag = checkRecvLDiag(kx, comReqss[offset],
                                                         grid, SCT);
                        if (recvLDiag)
                        {
#if 0
                            sUPanelTrSolve( kx, ldt, dFBufs[offset], scuBufs, packLUInfo,
                                            grid, LUstruct, stat, SCT);
#else
                            dUPanelTrSolve(kx, dFBufs[offset]->BlockLFactor,
                                           scuBufs->bigV,
                                           ldt, packLUInfo->Ublock_info,
                                           grid, LUstruct, stat, SCT);
#endif
                            factored_U[kx] = 1;
                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_U[kx] == 0 &&
                                k0x - k0 < numLA + 1 && // is within lookahead window
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
                } /* end look-ahead */

            } /* end if non-root level */

            /* end Schur complement update */
            SCT->NetSchurUpTimer += SuperLU_timer_() - tsch;

        } /* end Schur update for all the nodes in level 'topoLvl' */

    } /* end for all levels of the tree */

    return 0;
} /* end sparseTreeFactor_ASYNC_GPU */
