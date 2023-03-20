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
                           dtrf3Dpartition_t *trf3Dpartition, SCT_t *SCT,
                           dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                           SuperLUStat_t *stat, int *info)
    {
        gridinfo_t *grid = &(grid3d->grid2d);
        dLocalLU_t *Llu = LUstruct->Llu;

        // problem specific contants
        int_t ldt = sp_ienv_dist(3, options); /* Size of maximum supernode */
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
        // if (!grid3d->zscp.Iam && !grid3d->iam) printf("Using NSUP=%d\n", (int) ldt);

        // getting Nsupers
        int_t nsupers = getNsupers(n, LUstruct->Glu_persist);

        // Grid related Variables
        int_t iam = grid->iam; // in 2D grid
        int num_threads = getNumThreads(grid3d->iam);

        SCT->tStartup = SuperLU_timer_();

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

        int_t mxLeafNode = 0;
        for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
        {
            if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode)
                mxLeafNode = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
        }
        ddiagFactBufs_t **dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);

        /*******************************************
         *
         *   New code starts
         * ******************************************/
        // Create the new LU structure
        int *isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, myNodeCount, treePerm);
        int superlu_acc_offload = get_acc_offload();
        double tConst = SuperLU_timer_();
        LUstruct_v100 LU_packed(nsupers, ldt, trf3Dpartition, LUstruct, grid3d,
                                SCT, options, stat, thresh, info);

        tConst = SuperLU_timer_() - tConst;
        printf("Time to intialize New DS= %g\n", tConst);

        /*====  starting main factorization loop =====*/
        MPI_Barrier(grid3d->comm);
        SCT->tStartup = SuperLU_timer_() - SCT->tStartup;
#if 1
        LU_packed.pdgstrf3d();
#else
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
                    LU_packed.dsparseTreeFactorGPU(sforest, dFBufs,
                                                   &gEtreeInfo,
                                                   tag_ub);
                else
                    LU_packed.dsparseTreeFactor(sforest, dFBufs,
                                                &gEtreeInfo,
                                                tag_ub);

                /*now reduce the updates*/
                SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
            }

            if (ilvl < maxLvl - 1) /*then reduce before factorization*/
            {
                if (superlu_acc_offload)
                {
#define NDEBUG
#ifndef NDEBUG
                    LU_packed.checkGPU();
                    LU_packed.ancestorReduction3d(ilvl, myNodeCount, treePerm);
#endif
                    LU_packed.ancestorReduction3dGPU(ilvl, myNodeCount, treePerm);
#ifndef NDEBUG
                    LU_packed.checkGPU();
#endif
                }

                else
                    LU_packed.ancestorReduction3d(ilvl, myNodeCount, treePerm);
            }
        } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/

        SCT->tSchCompUdt3d[ilvl] = ilvl == 0 ? SCT->NetSchurUpTimer
                                             : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];
    } /*for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)*/

    MPI_Barrier(grid3d->comm);
    SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;
#endif
        double tXferGpu2Host = SuperLU_timer_();
        if (superlu_acc_offload)
        {
            cudaStreamSynchronize(LU_packed.A_gpu.cuStreams[0]); // in theory I don't need it
            LU_packed.copyLUGPUtoHost();
        }

        LU_packed.packedU2skyline(LUstruct);
        tXferGpu2Host = SuperLU_timer_() - tXferGpu2Host;
        printf("Time to send data back= %g\n", tXferGpu2Host);

        if (!grid3d->zscp.Iam)
        {
            // SCT_printSummary(grid, SCT);
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

        dfreeDiagFactBufsArr(mxLeafNode, dFBufs);

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit pdgstrf3d()");
#endif
        return 0;

    } /* pdgstrf3d */

    int_t LUstruct_v100::pdgstrf3d()
    {
        int tag_ub = set_tag_ub();
        gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
        int_t *iperm_c_supno = trf3Dpartition->iperm_c_supno;
        int_t *myNodeCount = trf3Dpartition->myNodeCount;
        int_t *myTreeIdxs = trf3Dpartition->myTreeIdxs;
        int_t *myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
        sForest_t **sForests = trf3Dpartition->sForests;
        int_t **treePerm = trf3Dpartition->treePerm;

        SCT->pdgstrfTimer = SuperLU_timer_();

        for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
        {

            sForest_t *sforest = sForests[myTreeIdxs[ilvl]];
            if (sforest) /* 2D factorization at individual subtree */
            {
                double tilvl = SuperLU_timer_();
                if (ilvl == 0)
                    dsparseTreeFactor(sforest, dFBufs,
                                      &gEtreeInfo,
                                      tag_ub);
                else
                    dAncestorFactorBaseline(ilvl, sforest, dFBufs,
                                            &gEtreeInfo,
                                            tag_ub);

                /*now reduce the updates*/
                SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
            }

            SCT->tSchCompUdt3d[ilvl] = ilvl == 0 ? SCT->NetSchurUpTimer
                                                 : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];
        } /*for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)*/

        MPI_Barrier(grid3d->comm);

        SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;
        if (superlu_acc_offload)
        {
            copyLUHosttoGPU();
            
            // LU check passed
        }
        return 0;
    }

#ifdef __cplusplus
}
#endif
// UrowindPtr_host