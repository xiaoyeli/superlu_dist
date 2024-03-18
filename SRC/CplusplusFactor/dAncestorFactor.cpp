#include "superlu_ddefs.h"
#include "lupanels.hpp"

int_t LUstruct_v100::dAncestorFactor(
    int_t ilvl,
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

    int_t *donePanelBcast = intMalloc_dist(nnodes);
    int_t *donePanelSolve = intMalloc_dist(nnodes);
    int_t *localNumChildrenLeft = intMalloc_dist(nnodes);

    // TODO: not needed, remove after testing
    for (int_t i = 0; i < nnodes; i++)
    {
        donePanelBcast[i] = 0;
        donePanelSolve[i] = 0;
        localNumChildrenLeft[i] = 0;
    }

    for (int_t k0 = 0; k0 < nnodes; k0++)
    {
        int_t k = perm_c_supno[k0];
        int_t k_parent = gEtreeInfo->setree[k];
        int_t ik = myIperm[k_parent];
        if (ik > -1 && ik < nnodes)
            localNumChildrenLeft[ik]++;
    }

    // start the pipeline
    int_t topoLvl = 0;
    int_t k_st = eTreeTopLims[topoLvl];
    int_t k_end = eTreeTopLims[topoLvl + 1];
    // TODO: make this asynchronous
    for (int_t k0 = k_st; k0 < k_end; k0++)
    {
        int_t k = perm_c_supno[k0];
        int_t offset = 0;
        dDiagFactorPanelSolve(k, offset, dFBufs);
        donePanelSolve[k0] = 1;
    }

    // TODO: its really the panels that needs to be doubled
    //  everything else can remain as it is
    int_t winSize = SUPERLU_MIN(numLA / 2, eTreeTopLims[1]);
    for (int k0 = k_st; k0 < winSize; ++k0)
    {
        int_t k = perm_c_supno[k0];
        int_t offset = k0 % numLA;
        if (!donePanelBcast[k0])
        {
            dPanelBcast(k, offset);
            donePanelBcast[k0] = 1;
        }
    } /*for (int k0 = k_st; k0 < SUPERLU_MIN(k_end, k_st + numLA); ++k0)*/

    int_t k1 = 0;
    int_t winParity = 0;
    int_t halfWin = numLA / 2;
    while (k1 < nnodes)
    {
        for (int_t k0 = k1; k0 < SUPERLU_MIN(nnodes, k1 + winSize); ++k0)
        {
            int_t k = perm_c_supno[k0];
            int_t offset = (k0 - k1) % winSize;
            if (winParity % 2)
                offset += halfWin; //

            /*=======   SchurComplement Update ======*/
            upanel_t k_upanel(UidxRecvBufs[offset], UvalRecvBufs[offset]);
            lpanel_t k_lpanel(LidxRecvBufs[offset], LvalRecvBufs[offset]);
            if (myrow == krow(k))
                k_upanel = uPanelVec[g2lRow(k)];
            if (mycol == kcol(k))
                k_lpanel = lPanelVec[g2lCol(k)];

            int_t k_parent = gEtreeInfo->setree[k];
            /* Look Ahead Panel Update */
            if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
                lookAheadUpdate(k, k_parent, k_lpanel, k_upanel);

            /* Look Ahead Panel Solve */
            if (k_parent < nsupers)
            {
                int_t k0_parent = myIperm[k_parent];
                if (k0_parent > 0 && k0_parent < nnodes)
                {
                    localNumChildrenLeft[k0_parent]--;
                    if (topoLvl < maxTopoLevel - 1 && !localNumChildrenLeft[k0_parent])
                    {
                        int_t dOffset = 0;
                        dDiagFactorPanelSolve(k_parent, dOffset, dFBufs);
                        donePanelSolve[k0_parent] = 1;
                    }
                }
            }

            /*proceed with remaining SchurComplement update */
            if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
                dSchurCompUpdateExcludeOne(k, k_parent, k_lpanel, k_upanel);
        }

        k1 = k1 + winSize;
        for (int_t k0_next = k1; k0_next < SUPERLU_MIN(nnodes, k1 + winSize); ++k0_next)
        {
            int k_next = perm_c_supno[k0_next];
            if (!localNumChildrenLeft[k0_next])
            {
                int offset_next = (k0_next - k1) % winSize;
                if (!(winParity % 2))
                    offset_next += halfWin;
                dPanelBcast(k_next, offset_next);
                donePanelBcast[k0_next] = 1;
            }
            else
            {
                winSize = k0_next - k1;
                break;
            }
        }

        winParity++;
    }

    SUPERLU_FREE(donePanelBcast);
    SUPERLU_FREE(donePanelSolve);
    SUPERLU_FREE(localNumChildrenLeft);
#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Exit dAncestorFactor_ASYNC()");
#endif
}

