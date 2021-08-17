#include "superlu_ddefs.h"

#ifdef MAP_PROFILE
#include "mapsampler_api.h"
#endif

#ifdef GPU_ACC
#include "dlustruct_gpu.h"
#include "acc_aux.c"
#endif

#include "lupanels.hpp"
#include "superlu_summit.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int_t pdgstrf3d_summit(superlu_dist_options_t *options, int m, int n, double anorm,
                           trf3Dpartition_t *trf3Dpartition, SCT_t *SCT,
                           dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                           SuperLUStat_t *stat, int *info)
    {
        gridinfo_t *grid = &(grid3d->grid2d);
        dLocalLU_t *Llu = LUstruct->Llu;

        // problem specific contants
        int_t ldt = sp_ienv_dist(3); /* Size of maximum supernode */
        //    double s_eps = slamch_ ("Epsilon");  -Sherry
        double s_eps = smach_dist("Epsilon");
        double thresh = s_eps * anorm;

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Enter pdgstrf3d()");
#endif

        // Initilize stat
        stat->ops[FACT] = 0;
        stat->current_buffer = 0.0;
        stat->peak_buffer = 0.0;
        stat->gpu_buffer = 0.0;
        //if (!grid3d->zscp.Iam && !grid3d->iam) printf("Using NSUP=%d\n", (int) ldt);

        //getting Nsupers
        int_t nsupers = getNsupers(n, LUstruct->Glu_persist);

        // Grid related Variables
        int_t iam = grid->iam; // in 2D grid
        int num_threads = getNumThreads(grid3d->iam);

        factStat_t factStat;
        initFactStat(nsupers, &factStat);

        SCT->tStartup = SuperLU_timer_();
        packLUInfo_t packLUInfo;
        initPackLUInfo(nsupers, &packLUInfo);

        dscuBufs_t scuBufs;
        dinitScuBufs(ldt, num_threads, nsupers, &scuBufs, LUstruct, grid);

        factNodelists_t fNlists;
        initFactNodelists(ldt, num_threads, nsupers, &fNlists);

        // tag_ub initialization
        int tag_ub = set_tag_ub();
        int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

#if (PRNTlevel >= 1)
        if (!iam)
        {
            printf("MPI tag upper bound = %d\n", tag_ub);
            fflush(stdout);
        }
#endif

        // trf3Dpartition_t*  trf3Dpartition = initTrf3Dpartition(nsupers, options, LUstruct, grid3d);
        gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
        int_t *iperm_c_supno = trf3Dpartition->iperm_c_supno;
        int_t *myNodeCount = trf3Dpartition->myNodeCount;
        int_t *myTreeIdxs = trf3Dpartition->myTreeIdxs;
        int_t *myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
        sForest_t **sForests = trf3Dpartition->sForests;
        int_t **treePerm = trf3Dpartition->treePerm;
        dLUValSubBuf_t *LUvsb = trf3Dpartition->LUvsb;

        /* Initializing factorization specific buffers */

        int_t numLA = getNumLookAhead(options);
        dLUValSubBuf_t **LUvsbs = dLluBufInitArr(SUPERLU_MAX(numLA, grid3d->zscp.Np), LUstruct);
        msgs_t **msgss = initMsgsArr(numLA);
        int_t mxLeafNode = 0;
        for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
        {
            if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode)
                mxLeafNode = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
        }
        ddiagFactBufs_t **dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);
        commRequests_t **comReqss = initCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), ldt, grid);

        // TODO: remove the following for time being
        int_t first_l_block_acc = 0;
        int_t first_u_block_acc = 0;
        int_t Pc = grid->npcol;
        int_t Pr = grid->nprow;
        int_t mrb = (nsupers + Pr - 1) / Pr; // Sherry check ... use ceiling
        int_t mcb = (nsupers + Pc - 1) / Pc;
        HyP_t *HyP = (HyP_t *)SUPERLU_MALLOC(sizeof(HyP_t));
        dInit_HyP(HyP, Llu, mcb, mrb);
        HyP->first_l_block_acc = first_l_block_acc;
        HyP->first_u_block_acc = first_u_block_acc;

        /*******************************************
    *
    *   New code starts 
    * ******************************************/
        // Create the new LU structure
        int_t *isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, myNodeCount, treePerm);
        int superlu_acc_offload = get_acc_offload();
        LUstruct_v100 LU_packed(nsupers, ldt, isNodeInMyGrid, superlu_acc_offload, LUstruct, grid3d,
                                SCT, options, stat);
        if(superlu_acc_offload)
            LU_packed.setLUstruct_GPU();

        /*====  starting main factorization loop =====*/
        MPI_Barrier(grid3d->comm);
        SCT->tStartup = SuperLU_timer_() - SCT->tStartup;

        SCT->pdgstrfTimer = SuperLU_timer_();

        for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
        {
            /* if I participate in this level */
            if (!myZeroTrIdxs[ilvl])
            {

                sForest_t *sforest = sForests[myTreeIdxs[ilvl]];

                /* main loop over all the supernodes */
                if (sforest) /* 2D factorization at individual subtree */
                {
                    double tilvl = SuperLU_timer_();

                    if (superlu_acc_offload)
                        LU_packed.dsparseTreeFactorGPU(sforest, comReqss, &scuBufs, &packLUInfo,
                                                       msgss, LUvsbs, dFBufs,
                                                       &gEtreeInfo, iperm_c_supno,
                                                       thresh, tag_ub, info);
                    else
                        LU_packed.dsparseTreeFactor(sforest, comReqss, &scuBufs, &packLUInfo,
                                                    msgss, LUvsbs, dFBufs,
                                                    &gEtreeInfo, iperm_c_supno,
                                                    thresh, tag_ub, info);

                    /*now reduce the updates*/
                    SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                    sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
                }

                if (ilvl < maxLvl - 1) /*then reduce before factorization*/
                {
                    if (superlu_acc_offload)
                        LU_packed.ancestorReduction3dGPU(ilvl, myNodeCount, treePerm);
                    else
                        LU_packed.ancestorReduction3d(ilvl, myNodeCount, treePerm);
                }
            } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/

            SCT->tSchCompUdt3d[ilvl] = ilvl == 0 ? SCT->NetSchurUpTimer
                                                 : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];
        } /*for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)*/

        MPI_Barrier(grid3d->comm);
        SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;

        if (superlu_acc_offload)
            LU_packed.copyLUGPUtoHost();
        LU_packed.packedU2skyline(LUstruct);

        if (!grid3d->zscp.Iam)
        {
            SCT_printSummary(grid, SCT);
            // if (superlu_acc_offload )
            //     printGPUStats(sluGPU->A_gpu, grid);
        }

#ifdef ITAC_PROF
        VT_traceoff();
#endif

#ifdef MAP_PROFILE
        allinea_stop_sampling();
#endif

        reduceStat(FACT, stat, grid3d);

        // sherry added
        /* Deallocate factorization specific buffers */
        freePackLUInfo(&packLUInfo);
        dfreeScuBufs(&scuBufs);
        freeFactStat(&factStat);
        freeFactNodelists(&fNlists);
        freeMsgsArr(numLA, msgss);
        freeCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), comReqss);
        dLluBufFreeArr(numLA, LUvsbs);
        dfreeDiagFactBufsArr(mxLeafNode, dFBufs);
        Free_HyP(HyP);
        // if (superlu_acc_offload )
        //     freeSluGPU(sluGPU);

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit pdgstrf3d()");
#endif
        return 0;

    } /* pdgstrf3d */

#ifdef __cplusplus
}
#endif
//UrowindPtr_host