#include <cstdio>
#include "superlu_ddefs.h"
#include "lupanels.hpp"
#include "lupanels_GPU.cuh"

int_t LUstruct_v100::dAncestorFactorBaselineGPU(
    int_t alvl,
    sForest_t *sforest,
    ddiagFactBufs_t **dFBufs, // size maxEtree level
    gEtreeInfo_t *gEtreeInfo, // global etree info
    int tag_ub)
{
    int_t nnodes = sforest->nNodes; // number of nodes in the tree
    if (nnodes < 1)
    {
        return 1;
    }

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Enter dAncestorFactor_ASYNC()");
#endif

    int_t *perm_c_supno = sforest->nodeList; // list of nodes in the order of factorization
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t *myIperm = treeTopoInfo->myIperm;
    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;

    /*main loop over all the levels*/
    int_t numLA = getNumLookAhead(options);

    for (int_t topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)
    {
        /* code */
        int_t k_st = eTreeTopLims[topoLvl];
        int_t k_end = eTreeTopLims[topoLvl + 1];
        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0];
            int kRoot = anc25d.rootRank(k0, alvl);
            // reduce the l and u panels to the root with MPI_Comm = anc25d.getComm(alvl);
            if (mycol == kcol(k))
            {
                void* sendBuf = (void*) lPanelVec[g2lCol(k)].gpuPanel.val;
                if (anc25d.rankHasGrid(k0, alvl))
                    sendBuf = MPI_IN_PLACE;

                MPI_Reduce(sendBuf, lPanelVec[g2lCol(k)].gpuPanel.val, 
                           lPanelVec[g2lCol(k)].nzvalSize(), MPI_DOUBLE, MPI_SUM, kRoot, anc25d.getComm(alvl));

            }
                
            if (myrow == krow(k))
            {
                void* sendBuf =  (void*) uPanelVec[g2lRow(k)].gpuPanel.val;
                if (anc25d.rankHasGrid(k0, alvl))
                    sendBuf = MPI_IN_PLACE;
                MPI_Reduce(sendBuf, uPanelVec[g2lRow(k)].gpuPanel.val, 
                           uPanelVec[g2lRow(k)].nzvalSize(), MPI_DOUBLE, MPI_SUM, kRoot, anc25d.getComm(alvl));
            }
                

            if (anc25d.rankHasGrid(k0, alvl))
            {

                int_t offset = k0 - k_st;
                int_t ksupc = SuperSize(k);
                dDFactPSolveGPU(k, offset, dFBufs);

                #if 0
                /*=======   Diagonal Factorization      ======*/
                if (iam == procIJ(k, k))
                {
                    lPanelVec[g2lCol(k)].diagFactor(k, dFBufs[offset]->BlockUFactor, ksupc,
                                                    thresh, xsup, options, stat, info);
                    lPanelVec[g2lCol(k)].packDiagBlock(dFBufs[offset]->BlockLFactor, ksupc);
                }

                /*=======   Diagonal Broadcast          ======*/
                if (myrow == krow(k))
                    MPI_Bcast((void *)dFBufs[offset]->BlockLFactor, ksupc * ksupc,
                              MPI_DOUBLE, kcol(k), (grid->rscp).comm);
                if (mycol == kcol(k))
                    MPI_Bcast((void *)dFBufs[offset]->BlockUFactor, ksupc * ksupc,
                              MPI_DOUBLE, krow(k), (grid->cscp).comm);

                /*=======   Panel Update                ======*/
                if (myrow == krow(k))
                    uPanelVec[g2lRow(k)].panelSolve(ksupc, dFBufs[offset]->BlockLFactor, ksupc);

                if (mycol == kcol(k))
                    lPanelVec[g2lCol(k)].panelSolve(ksupc, dFBufs[offset]->BlockUFactor, ksupc);
                #endif 
                /*=======   Panel Broadcast             ======*/
                // upanel_t k_upanel(UidxRecvBufs[0], UvalRecvBufs[0]);
                // lpanel_t k_lpanel(LidxRecvBufs[0], LvalRecvBufs[0]);
                            /*=======   Panel Broadcast             ======*/
                upanel_t k_upanel(UidxRecvBufs[0], UvalRecvBufs[0],
                                A_gpu.UidxRecvBufs[0], A_gpu.UvalRecvBufs[0]);
                lpanel_t k_lpanel(LidxRecvBufs[0], LvalRecvBufs[0],
                                A_gpu.LidxRecvBufs[0], A_gpu.LvalRecvBufs[0]);
                if (myrow == krow(k))
                {
                    k_upanel = uPanelVec[g2lRow(k)];
                }
                if (mycol == kcol(k))
                    k_lpanel = lPanelVec[g2lCol(k)];

                if (UidxSendCounts[k] > 0)
                {
                    MPI_Bcast(k_upanel.gpuPanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
                    MPI_Bcast(k_upanel.gpuPanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
                }

                if (LidxSendCounts[k] > 0)
                {
                    MPI_Bcast(k_lpanel.gpuPanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
                    MPI_Bcast(k_lpanel.gpuPanel.val, LvalSendCounts[k], MPI_DOUBLE, kcol(k), grid3d->rscp.comm);
                }

/*=======   Schurcomplement Update      ======*/
#warning single node only
                // dSchurComplementUpdate(k, lPanelVec[g2lCol(k)], uPanelVec[g2lRow(k)]);
                // dSchurComplementUpdate(k, lPanelVec[g2lCol(k)], k_upanel);
                if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
                {
                    k_upanel.checkCorrectness();
                    // dSchurComplementUpdate(k, k_lpanel, k_upanel);
                    int streamId = 0;
                    dSchurComplementUpdateGPU(
                    streamId, 
                    k, k_lpanel, k_upanel);
                }
            } /** if (anc25d.rankHasGrid(k0, alvl)) */
            
            // Brodcast the l and u panels to the root with MPI_Comm = anc25d.getComm(alvl);
            if (mycol == kcol(k))
                MPI_Bcast(lPanelVec[g2lCol(k)].gpuPanel.val, 
                          lPanelVec[g2lCol(k)].nzvalSize(), MPI_DOUBLE, kRoot, anc25d.getComm(alvl));
                           
            if (myrow == krow(k))
                MPI_Bcast(uPanelVec[g2lRow(k)].gpuPanel.val, 
                           uPanelVec[g2lRow(k)].nzvalSize(), MPI_DOUBLE, kRoot, anc25d.getComm(alvl));
            // MPI_Barrier(grid3d->comm);

        } /*for k0= k_st:k_end */

    } /*for topoLvl = 0:maxTopoLevel*/

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Exit dAncestorFactor_ASYNC()");
#endif

    return 0;
} /* dAncestorFactor_ASYNC */
