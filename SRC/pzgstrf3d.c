/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Performs LU factorization in 3D process grid.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.2) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 * May 12, 2021
 * Last update: December 12, 2021  v7.2.0
 */

#include "superlu_zdefs.h"
#if 0
#include "pdgstrf3d.h"
#include "trfCommWrapper.h"
#include "trfAux.h"
//#include "load-balance/supernodal_etree.h"
//#include "load-balance/supernodalForest.h"
#include "supernodal_etree.h"
#include "supernodalForest.h"
#include "p3dcomm.h"
#include "treeFactorization.h"
#include "ancFactorization.h"
#include "xtrf3Dpartition.h"
#endif

#ifdef MAP_PROFILE
#include  "mapsampler_api.h"
#endif

#ifdef GPU_ACC
#include "zlustruct_gpu.h"
//#include "acc_aux.c"  //no need anymore
#endif


/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * PZGSTRF3D performs the LU factorization in parallel using 3D process grid,
 * which is a communication-avoiding algorithm compared to the 2D algorithm.
 *
 * Arguments
 * =========
 *
 * options (input) superlu_dist_options_t*
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *         The following field should be defined:
 *         o ReplaceTinyPivot (yes_no_t)
 *           Specifies whether to replace the tiny diagonals by
 *           sqrt(epsilon)*norm(A) during LU factorization.
 *
 * m      (input) int
 *        Number of rows in the matrix.
 *
 * n      (input) int
 *        Number of columns in the matrix.
 *
 * anorm  (input) double
 *        The norm of the original matrix A, or the scaled A if
 *        equilibration was done.
 *
 * trf3Dpartition (input) trf3Dpartition*
 *        Matrix partitioning information in 3D process grid.
 *
 * SCT    (input/output) SCT_t*
 *        Various statistics of 3D factorization.
 *
 * LUstruct (input/output) zLUstruct_t*
 *         The data structures to store the distributed L and U factors.
 *         The following fields should be defined:
 *
 *         o Glu_persist (input) Glu_persist_t*
 *           Global data structure (xsup, supno) replicated on all processes,
 *           describing the supernode partition in the factored matrices
 *           L and U:
 *         xsup[s] is the leading column of the s-th supernode,
 *             supno[i] is the supernode number to which column i belongs.
 *
 *         o Llu (input/output) zLocalLU_t*
 *           The distributed data structures to store L and U factors.
 *           See superlu_zdefs.h for the definition of 'zLocalLU_t'.
 *
 * grid3d (input) gridinfo3d_t*
 *        The 3D process mesh. It contains the MPI communicator, the number
 *        of process rows (NPROW), the number of process columns (NPCOL),
 *        and replication factor in Z-dimension. It is an input argument to all
 *        the 3D parallel routines.
 *        Grid3d can be initialized by subroutine SUPERLU_GRIDINIT3D.
 *        See superlu_defs.h for the definition of 'gridinfo3d_t'.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics on runtime and floating-point operation count.
 *        See util_dist.h for the definition of 'SuperLUStat_t'.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 * </pre>
 */
int_t pzgstrf3d(superlu_dist_options_t *options, int m, int n, double anorm,
		ztrf3Dpartition_t*  trf3Dpartition, SCT_t *SCT,
		zLUstruct_t *LUstruct, gridinfo3d_t * grid3d,
		SuperLUStat_t *stat, int *info)
{
    gridinfo_t* grid = &(grid3d->grid2d);
    zLocalLU_t *Llu = LUstruct->Llu;

    // problem specific contants
    int_t ldt = sp_ienv_dist(3, options);     /* Size of maximum supernode */
    //    double s_eps = slamch_ ("Epsilon");  -Sherry
    double s_eps = smach_dist("Epsilon");
    double thresh = s_eps * anorm;

    /* Test the input parameters. */
    *info = 0;
    
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Enter pzgstrf3d()");
#endif

    // Initilize stat
    stat->ops[FACT] = 0;
    stat->current_buffer = 0.0;
    stat->peak_buffer    = 0.0;
    stat->gpu_buffer     = 0.0;
    //if (!grid3d->zscp.Iam && !grid3d->iam) printf("Using NSUP=%d\n", (int) ldt);

    //getting Nsupers
    int_t nsupers = getNsupers(n, LUstruct->Glu_persist);

    // Grid related Variables
    int_t iam = grid->iam; // in 2D grid
    int num_threads = getNumThreads(grid3d->iam);

    factStat_t factStat;
    initFactStat(nsupers, &factStat);

#if 0  // sherry: not used
    zdiagFactBufs_t dFBuf;
    zinitDiagFactBufs(ldt, &dFBuf);

    commRequests_t comReqs;   
    initCommRequests(&comReqs, grid);

    msgs_t msgs;
    initMsgs(&msgs);
#endif

    SCT->tStartup = SuperLU_timer_();
    packLUInfo_t packLUInfo;
    initPackLUInfo(nsupers, &packLUInfo);

    zscuBufs_t scuBufs;
    zinitScuBufs(options, ldt, num_threads, nsupers, &scuBufs, LUstruct, grid);

    factNodelists_t  fNlists;
    initFactNodelists( ldt, num_threads, nsupers, &fNlists);

    // tag_ub initialization
    int tag_ub = set_tag_ub();
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

#if ( PRNTlevel>=1 )
    if (grid3d->iam == 0) {
        printf ("MPI tag upper bound = %d\n", tag_ub); fflush(stdout);
    }
#endif

    // ztrf3Dpartition_t*  trf3Dpartition = initTrf3Dpartition(nsupers, options, LUstruct, grid3d);
    gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
    int_t* iperm_c_supno = trf3Dpartition->iperm_c_supno;
    int_t* myNodeCount = trf3Dpartition->myNodeCount;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t** treePerm = trf3Dpartition->treePerm ;
    zLUValSubBuf_t *LUvsb = trf3Dpartition->LUvsb;

    /* Initializing factorization specific buffers */

    int_t numLA = getNumLookAhead(options);
    zLUValSubBuf_t** LUvsbs = zLluBufInitArr( SUPERLU_MAX( numLA, grid3d->zscp.Np ), LUstruct);
    msgs_t**msgss = initMsgsArr(numLA);
    int_t mxLeafNode    = 0;
    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
        if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode )
            mxLeafNode    = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
    }
    zdiagFactBufs_t** dFBufs = zinitDiagFactBufsArr(mxLeafNode, ldt, grid);
    commRequests_t** comReqss = initCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), ldt, grid);

    /* Setting up GPU related data structures */

    int_t first_l_block_acc = 0;
    int_t first_u_block_acc = 0;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t mrb =    (nsupers + Pr - 1) / Pr;
    int_t mcb =    (nsupers + Pc - 1) / Pc;
    HyP_t *HyP = (HyP_t *) SUPERLU_MALLOC(sizeof(HyP_t));

    zInit_HyP(HyP, Llu, mcb, mrb);
    HyP->first_l_block_acc = first_l_block_acc;
    HyP->first_u_block_acc = first_u_block_acc;

    int superlu_acc_offload = HyP->superlu_acc_offload;

    //int_t bigu_size = getBigUSize(nsupers, grid, LUstruct);
    int_t bigu_size = getBigUSize(options, nsupers, grid,
    	  	                  LUstruct->Llu->Lrowind_bc_ptr);
    HyP->bigu_size = bigu_size;
    int_t buffer_size = sp_ienv_dist(8, options); // get_max_buffer_size ();
    HyP->buffer_size = buffer_size;
    HyP->nsupers = nsupers;

#ifdef GPU_ACC

    /*Now initialize the GPU data structure*/
    zLUstruct_gpu_t *A_gpu, *dA_gpu;

    d2Hreduce_t d2HredObj;
    d2Hreduce_t* d2Hred = &d2HredObj;
    zsluGPU_t sluGPUobj;
    zsluGPU_t *sluGPU = &sluGPUobj;
    sluGPU->isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, myNodeCount, treePerm);
    if (superlu_acc_offload)
    {
#if 0 	/* Sherry: For GPU code on titan, we do not need performance 
	   lookup tables since due to difference in CPU-GPU performance,
	   it didn't make much sense to do any Schur-complement update
	   on CPU, except for the lookahead-update on CPU. Same should
	   hold for summit as well. (from Piyush)   */

        /*Initilize the lookup tables */
        LookUpTableInit(iam);
        acc_async_cost = get_acc_async_cost();
#endif

	//OLD: int_t* perm_c_supno = getPerm_c_supno(nsupers, options, LUstruct, grid);
	int_t* perm_c_supno = getPerm_c_supno(nsupers, options,
					      LUstruct->etree,
					      LUstruct->Glu_persist,
					      LUstruct->Llu->Lrowind_bc_ptr,
					      LUstruct->Llu->Ufstnz_br_ptr,
					      grid);

	/* Initialize GPU data structures */
        zinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                        n, buffer_size, bigu_size, ldt, stat);

        HyP->first_u_block_acc = sluGPU->A_gpu->first_u_block_gpu;
        HyP->first_l_block_acc = sluGPU->A_gpu->first_l_block_gpu;
        HyP->nGPUStreams = sluGPU->nGPUStreams;
    }

#endif  // end GPU_ACC

    /*====  starting main factorization loop =====*/
    MPI_Barrier( grid3d->comm);
    SCT->tStartup = SuperLU_timer_() - SCT->tStartup;
    // int_t myGrid = grid3d->zscp.Iam;

#ifdef ITAC_PROF
    VT_traceon();
#endif
#ifdef MAP_PROFILE
    allinea_start_sampling();
#endif
    SCT->pdgstrfTimer = SuperLU_timer_();

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        /* if I participate in this level */
        if (!myZeroTrIdxs[ilvl])
        {
            //int_t tree = myTreeIdxs[ilvl];

            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

            /* main loop over all the supernodes */
            if (sforest) /* 2D factorization at individual subtree */
            {
                double tilvl = SuperLU_timer_();
#ifdef GPU_ACC
                zsparseTreeFactor_ASYNC_GPU(
                    sforest,
                    comReqss, &scuBufs,  &packLUInfo,
                    msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                    &gEtreeInfo, options,  iperm_c_supno, ldt,
                    sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                    thresh,  SCT, tag_ub, info);
#else
                zsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
					msgss, LUvsbs, dFBufs, &factStat, &fNlists,
					&gEtreeInfo, options, iperm_c_supno, ldt,
					HyP, LUstruct, grid3d, stat,
					thresh,  SCT, tag_ub, info );
#endif

                /*now reduce the updates*/
                SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
            }

            if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
            {
#ifdef GPU_ACC
                zreduceAllAncestors3d_GPU(
                    ilvl, myNodeCount, treePerm, LUvsb,
                    LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                    SCT, stat );
#else

                zreduceAllAncestors3d( ilvl, myNodeCount, treePerm,
                                      LUvsb, LUstruct, grid3d, SCT );
#endif

            }
        } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/

        SCT->tSchCompUdt3d[ilvl] = ilvl == 0 ? SCT->NetSchurUpTimer
	    : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];
    } /* end for (int ilvl = 0; ilvl < maxLvl; ++ilvl) */

    /* Prepare error message - find the smallesr index i that U(i,i)==0 */
    int iinfo;
    if ( *info == 0 ) *info = n + 1;
    MPI_Allreduce (info, &iinfo, 1, MPI_INT, MPI_MIN, grid3d->comm);
    if ( iinfo == (n + 1) ) *info = 0;
    else *info = iinfo;
    //printf("After factorization: INFO = %d\n", *info); fflush(stdout);

    SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;

#ifdef ITAC_PROF
    VT_traceoff();
#endif

#ifdef MAP_PROFILE
    allinea_stop_sampling();
#endif

#ifdef GPU_ACC
    /* This frees the GPU storage allocateed in initSluGPU3D_t() */
    if (superlu_acc_offload) {
        if ( options->PrintStat ) {
	    printGPUStats(nsupers, stat, grid3d);
	}
        zfree_LUstruct_gpu (sluGPU, stat);
    }
#endif
    reduceStat(FACT, stat, grid3d);

    // sherry added
    /* Deallocate factorization specific buffers */
    freePackLUInfo(&packLUInfo);
    zfreeScuBufs(&scuBufs);
    freeFactStat(&factStat);
    freeFactNodelists(&fNlists);
    freeMsgsArr(numLA, msgss);
    freeCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), comReqss);
    zLluBufFreeArr(numLA, LUvsbs);
    zfreeDiagFactBufsArr(mxLeafNode, dFBufs);
    Free_HyP(HyP);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Exit pzgstrf3d()");
#endif
    return 0;

} /* pzgstrf3d */
