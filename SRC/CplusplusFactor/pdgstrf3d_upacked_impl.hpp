#pragma once
#include "superlu_ddefs.h"

#ifdef MAP_PROFILE
#include "mapsampler_api.h"
#endif

#ifdef HAVE_CUDA
#include "dlustruct_gpu.h"
// #include "acc_aux.c"
#endif


#include "lupanels.hpp"
#include "superlu_upacked.h"
#include "luAuxStructTemplated.hpp"
#include "anc25d-GPU_impl.hpp"
#include "dAncestorFactor_impl.hpp"
#include "anc25d_impl.hpp"
#include "dsparseTreeFactorGPU_impl.hpp"  //needed???
#include "dsparseTreeFactor_upacked_impl.hpp"
#include "schurCompUpdate_impl.cuh"
#include "l_panels_impl.hpp"
#include "u_panels_impl.hpp"
#include "lupanels_impl.hpp"
#include "lupanels_GPU_impl.hpp"
#include "lupanelsComm3dGPU_impl.hpp"
#include "lupanels_comm3d_impl.hpp"
// #include "sparseTreeFactor_impl.hpp"
// pxgstrf3d<double>
template <typename Ftype>
int_t pdgstrf3d_upacked(superlu_dist_options_t *options, int m, int n, AnormType<Ftype> anorm,
		       trf3dpartitionType<Ftype> *trf3Dpartition, SCT_t *SCT,
		       LUStruct_type<Ftype> *LUstruct, gridinfo3d_t *grid3d,
		       SuperLUStat_t *stat, int *info)
{
        gridinfo_t *grid = &(grid3d->grid2d);
        LocalLU_type<Ftype> *Llu = LUstruct->Llu;

        // problem specific contants
        int_t ldt = sp_ienv_dist(3, options); /* Size of maximum supernode */
        //    double s_eps = slamch_ ("Epsilon");  -Sherry
        AnormType<Ftype> s_eps = smach_dist("Epsilon");
        AnormType<Ftype> thresh = s_eps * anorm;

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Enter pdgstrf3d_upacked()");
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
        LUValSubBuf_type<Ftype> *LUvsb = trf3Dpartition->LUvsb;

        /* Initializing factorization specific buffers */

        int_t numLA = getNumLookAhead(options);

        int_t mxLeafNode = 0;
        for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
        {
            if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode)
                mxLeafNode = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
        }
        diagFactBufs_type<Ftype> **dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);

        /*******************************************
         *
         *   New code starts
         * ******************************************/
        // Create the new LU structure
        int *isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, myNodeCount, treePerm);
        int superlu_acc_offload = sp_ienv_dist(10, options); //get_acc_offload();
        double tConst = SuperLU_timer_();
        xLUstruct_t<Ftype> LU_packed(nsupers, ldt, trf3Dpartition, LUstruct, grid3d,
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
                #ifdef HAVE_CUDA
                    LU_packed.dsparseTreeFactorGPU(sforest, dFBufs,
                                                   &gEtreeInfo,
                                                   tag_ub);
                #endif
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
#ifdef HAVE_CUDA
		  //#define NDEBUG
#ifndef NDEBUG
                    LU_packed.checkGPU();
                    LU_packed.ancestorReduction3d(ilvl, myNodeCount, treePerm);
#endif
                    LU_packed.ancestorReduction3dGPU(ilvl, myNodeCount, treePerm);
#ifndef NDEBUG
                    LU_packed.checkGPU();
#endif
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
    
#endif /* match line 106 */
    
        double tXferGpu2Host = SuperLU_timer_();
        if (superlu_acc_offload)
        {
        #ifdef HAVE_CUDA
            cudaStreamSynchronize(LU_packed.A_gpu.cuStreams[0]);    // in theory I don't need it
            LU_packed.copyLUGPUtoHost();
        #endif
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

	/* free L panels and U panels. called in LUstruct_v100 destructor */
        // dfreeDiagFactBufsArr(mxLeafNode, dFBufs);

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit pdgstrf3d_upacked()");
#endif
        return 0;

} /* pdgstrf3d_upacked */


/* This can be accessed from the C handle  */
template <typename Ftype>
int_t xLUstruct_t<Ftype>::pdgstrf3d()
{
        int tag_ub = set_tag_ub();
        gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
        // int_t *iperm_c_supno = trf3Dpartition->iperm_c_supno;
        int_t *myNodeCount = trf3Dpartition->myNodeCount;
        int_t *myTreeIdxs = trf3Dpartition->myTreeIdxs;
        int_t *myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
        sForest_t **sForests = trf3Dpartition->sForests;
        int_t **treePerm = trf3Dpartition->treePerm;

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Enter pdgstrf3d()");
	printf(".maxLvl %d\n", maxLvl);
#endif
	
        SCT->pdgstrfTimer = SuperLU_timer_();
        // get environment variables ANC25D
        int useAnc25D = 0;
        if (getenv("ANC25D"))
            useAnc25D = atoi(getenv("ANC25D"));
        if (useAnc25D)
            printf("-- Using ANC25D; ONLY CPU supported \n");

        for (int ilvl = 0; ilvl < maxLvl; ++ilvl) /* maxLvel is the tree level
						     along Z-dim process grid */
        {
            if (useAnc25D)
            {
                sForest_t *sforest = sForests[myTreeIdxs[ilvl]];
                if (sforest) /* 2D factorization at individual subtree */
                {
                    double tilvl = SuperLU_timer_();
                    if (superlu_acc_offload)
                    {
                        printf("-- ANC25D on GPU is not working yet!!!!! \n");    
                        if (ilvl == 0)
                            dsparseTreeFactorGPU(sforest, dFBufs,
                                                 &gEtreeInfo,
                                                 tag_ub);
                        else
                            dAncestorFactorBaselineGPU(ilvl, sforest, dFBufs,
                                                    &gEtreeInfo,
                                                    tag_ub);
                    }
                    else
                    {
                        if (ilvl == 0)
                            dsparseTreeFactor(sforest, dFBufs,
                                              &gEtreeInfo,
                                              tag_ub);
                        else
                            dAncestorFactorBaseline(ilvl, sforest, dFBufs,
                                                    &gEtreeInfo,
                                                    tag_ub);
                    }

                    /*now reduce the updates*/
                    SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                    sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
                }
            }
            else
            {
                /* if I participate in this level */
                if (!myZeroTrIdxs[ilvl])
                {

                    sForest_t *sforest = sForests[myTreeIdxs[ilvl]];

                    /* main loop over all the supernodes */
                    if (sforest) /* 2D factorization at individual subtree */
                    {
                        double tilvl = SuperLU_timer_();

                        if ( superlu_acc_offload ) {
                            if ( options->batchCount==0 )
                                dsparseTreeFactorGPU(sforest, dFBufs, &gEtreeInfo, tag_ub);
                            else {
				printf("Batch ERROR: should not get to this branch!\n");
				// Sherry commented out the following
                                //dsparseTreeFactorBatchGPU(sforest, dFBufs, &gEtreeInfo, tag_ub);
			    }
                        } else {
                            dsparseTreeFactor(sforest, dFBufs,
                                            &gEtreeInfo,
                                            tag_ub);

                        }

                        /*now reduce the updates*/
                        SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                        sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
                    }

                    if (ilvl < maxLvl - 1) /*then reduce before factorization*/
                    {
                        if (superlu_acc_offload)
                        {
                //#define NDEBUG
    // #ifndef NDEBUG
    //                         checkGPU();
    //                         ancestorReduction3d(ilvl, myNodeCount, treePerm);
    // #endif
                
                            ancestorReduction3dGPU(ilvl, myNodeCount, treePerm);
                
    // #ifndef NDEBUG
    //                         checkGPU();
    // #endif
                        }

                        else
                            this->ancestorReduction3d(ilvl, myNodeCount, treePerm);
                    }
                } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/
            }

            SCT->tSchCompUdt3d[ilvl] = ilvl == 0 ? SCT->NetSchurUpTimer
                                                 : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];
        } /*for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)*/

        MPI_Barrier(grid3d->comm);
        SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit pdgstrf3d()");
#endif

	// dfreeDiagFactBufsArr(maxLeafNodes, dFBufs); // called in LUstruct_v100 destructor

        return 0;
} /* pdgstrf3d */



// UrowindPtr_host
