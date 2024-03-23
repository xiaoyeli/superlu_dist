#include <algorithm>
#include <iostream>
#include <cassert>
#include "superlu_defs.h"

#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#endif
#include "lupanels.hpp"

// #define cudaCheckError()                                                              
//     {                                                                        \
//         cudaError_t e = cudaGetLastError();                                   \
//         if (e != cudaSuccess)                                                \
//         {                                                                    \
//             printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
//             exit(0);                                                         \
//         }                                                                    \
//     }

#ifdef HAVE_CUDA
upanel_t LUstruct_v100::getKUpanel(int_t k, int_t offset)
{
    return (
        myrow == krow(k) ? 
        uPanelVec[g2lRow(k)] : 
        upanel_t(UidxRecvBufs[offset], UvalRecvBufs[offset],
            A_gpu.UidxRecvBufs[offset], A_gpu.UvalRecvBufs[offset])
    );
}

lpanel_t LUstruct_v100::getKLpanel(int_t k, int_t offset)
{ 
    return (
        mycol == kcol(k) ? 
        lPanelVec[g2lCol(k)] : 
        lpanel_t(LidxRecvBufs[offset], LvalRecvBufs[offset],
            A_gpu.LidxRecvBufs[offset], A_gpu.LvalRecvBufs[offset])
    );
}
#endif

/* Constructor */
LUstruct_v100::LUstruct_v100(int_t nsupers_, int_t ldt_,
                             dtrf3Dpartition_t *trf3Dpartition_, 
                             dLUstruct_t *LUstruct,
                             gridinfo3d_t *grid3d_in,
                             SCT_t *SCT_, superlu_dist_options_t *options_,
                             SuperLUStat_t *stat_, double thresh_, int *info_) :
                             nsupers(nsupers_), trf3Dpartition(trf3Dpartition_),
                             ldt(ldt_), /* maximum supernode size */
			     grid3d(grid3d_in), SCT(SCT_),
			     options(options_), stat(stat_),
			     thresh(thresh_), info(info_), anc25d(grid3d_in)
{
    maxLvl = log2i(grid3d->zscp.Np) + 1;
    isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, trf3Dpartition->myNodeCount, trf3Dpartition->treePerm);
    superlu_acc_offload = sp_ienv_dist(10, options); //get_acc_offload();

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d_in->iam, "Enter LUstruct_v100 constructor");
#endif
    grid = &(grid3d->grid2d);
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW(iam, grid);
    mycol = MYCOL(iam, grid);
    xsup = LUstruct->Glu_persist->xsup;
    int_t **Lrowind_bc_ptr = LUstruct->Llu->Lrowind_bc_ptr;
    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    double **Lnzval_bc_ptr = LUstruct->Llu->Lnzval_bc_ptr;
    double **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    lPanelVec = new lpanel_t[CEILING(nsupers, Pc)];
    uPanelVec = new upanel_t[CEILING(nsupers, Pr)];
    // create the lvectors
    maxLvalCount = 0;
    maxLidxCount = 0;
    maxUvalCount = 0;
    maxUidxCount = 0;

    std::vector<int_t> localLvalSendCounts(CEILING(nsupers, Pc), 0);
    std::vector<int_t> localUvalSendCounts(CEILING(nsupers, Pr), 0);
    std::vector<int_t> localLidxSendCounts(CEILING(nsupers, Pc), 0);
    std::vector<int_t> localUidxSendCounts(CEILING(nsupers, Pr), 0);

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        int_t k0 = i * Pc + mycol;
        if (Lrowind_bc_ptr[i] != NULL && isNodeInMyGrid[k0] == 1)
        {
            int_t isDiagIncluded = 0;

            if (myrow == krow(k0))
                isDiagIncluded = 1;
            lpanel_t lpanel(k0, Lrowind_bc_ptr[i], Lnzval_bc_ptr[i], xsup, isDiagIncluded);
            lPanelVec[i] = lpanel;
            maxLvalCount = std::max(lPanelVec[i].nzvalSize(), maxLvalCount);
            maxLidxCount = std::max(lPanelVec[i].indexSize(), maxLidxCount);
            localLvalSendCounts[i] = lPanelVec[i].nzvalSize();
            localLidxSendCounts[i] = lPanelVec[i].indexSize();
        }
    }

    // create the vectors
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            int_t globalId = i * Pr + myrow;
            upanel_t upanel(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
            uPanelVec[i] = upanel;
            maxUvalCount = std::max(uPanelVec[i].nzvalSize(), maxUvalCount);
            maxUidxCount = std::max(uPanelVec[i].indexSize(), maxUidxCount);
            localUvalSendCounts[i] = uPanelVec[i].nzvalSize();
            localUidxSendCounts[i] = uPanelVec[i].indexSize();
        }
    }

    // compute the send sizes
    // send and recv count for 2d comm
    LvalSendCounts.resize(nsupers);
    UvalSendCounts.resize(nsupers);
    LidxSendCounts.resize(nsupers);
    UidxSendCounts.resize(nsupers);

    std::vector<int_t> recvBuf(std::max(CEILING(nsupers, Pr), CEILING(nsupers, Pc)), 0);

    for (int pr = 0; pr < Pr; pr++)
    {
        int npr = CEILING(nsupers, Pr);
        std::copy(localUvalSendCounts.begin(), localUvalSendCounts.end(), recvBuf.begin());
        // Send the value counts ;
        MPI_Bcast((void *)recvBuf.data(), npr, mpi_int_t, pr, grid3d->cscp.comm);
        for (int i = 0; i * Pr + pr < nsupers; i++)
        {
            UvalSendCounts[i * Pr + pr] = recvBuf[i];
        }

        std::copy(localUidxSendCounts.begin(), localUidxSendCounts.end(), recvBuf.begin());
        // send the index count
        MPI_Bcast((void *)recvBuf.data(), npr, mpi_int_t, pr, grid3d->cscp.comm);
        for (int i = 0; i * Pr + pr < nsupers; i++)
        {
            UidxSendCounts[i * Pr + pr] = recvBuf[i];
        }
    }

    for (int pc = 0; pc < Pc; pc++)
    {
        int npc = CEILING(nsupers, Pc);
        std::copy(localLvalSendCounts.begin(), localLvalSendCounts.end(), recvBuf.begin());
        // Send the value counts ;
        MPI_Bcast((void *)recvBuf.data(), npc, mpi_int_t, pc, grid3d->rscp.comm);
        for (int i = 0; i * Pc + pc < nsupers; i++)
        {
            LvalSendCounts[i * Pc + pc] = recvBuf[i];
        }

        std::copy(localLidxSendCounts.begin(), localLidxSendCounts.end(), recvBuf.begin());
        // send the index count
        MPI_Bcast((void *)recvBuf.data(), npc, mpi_int_t, pc, grid3d->rscp.comm);
        for (int i = 0; i * Pc + pc < nsupers; i++)
        {
            LidxSendCounts[i * Pc + pc] = recvBuf[i];
        }
    }

    maxUvalCount = *std::max_element(UvalSendCounts.begin(), UvalSendCounts.end());
    maxUidxCount = *std::max_element(UidxSendCounts.begin(), UidxSendCounts.end());
    maxLvalCount = *std::max_element(LvalSendCounts.begin(), LvalSendCounts.end());
    maxLidxCount = *std::max_element(LidxSendCounts.begin(), LidxSendCounts.end());

    // Allocate bigV, indirect
    nThreads = getNumThreads(iam);
    bigV = dgetBigV(ldt, nThreads);
    indirect = (int_t *)SUPERLU_MALLOC(nThreads * ldt * sizeof(int_t));
    indirectRow = (int_t *)SUPERLU_MALLOC(nThreads * ldt * sizeof(int_t));
    indirectCol = (int_t *)SUPERLU_MALLOC(nThreads * ldt * sizeof(int_t));

    // allocating communication buffers
    LvalRecvBufs.resize(options->num_lookaheads);
    UvalRecvBufs.resize(options->num_lookaheads);
    LidxRecvBufs.resize(options->num_lookaheads);
    UidxRecvBufs.resize(options->num_lookaheads);
    bcastLval.resize(options->num_lookaheads);
    bcastUval.resize(options->num_lookaheads);
    bcastLidx.resize(options->num_lookaheads);
    bcastUidx.resize(options->num_lookaheads);

    for (int i = 0; i < options->num_lookaheads; i++)
    {
        LvalRecvBufs[i] = (double *)SUPERLU_MALLOC(sizeof(double) * maxLvalCount);
        UvalRecvBufs[i] = (double *)SUPERLU_MALLOC(sizeof(double) * maxUvalCount);
        LidxRecvBufs[i] = (int_t *)SUPERLU_MALLOC(sizeof(int_t) * maxLidxCount);
        UidxRecvBufs[i] = (int_t *)SUPERLU_MALLOC(sizeof(int_t) * maxUidxCount);

        //TODO: check if setup correctly
        bcastStruct bcLval(grid3d->rscp.comm, MPI_DOUBLE, SYNC);
        bcastLval[i] = bcLval;
        bcastStruct bcUval(grid3d->cscp.comm, MPI_DOUBLE, SYNC);
        bcastUval[i] = bcUval;
        bcastStruct bcLidx(grid3d->rscp.comm, mpi_int_t, SYNC);
        bcastLidx[i] = bcLidx;
        bcastStruct bcUidx(grid3d->cscp.comm, mpi_int_t, SYNC);
        bcastUidx[i] = bcUidx;
    }

    diagFactBufs.resize(numDiagBufs);  /* Sherry?? numDiagBufs == 32 hard-coded */
    bcastDiagRow.resize(numDiagBufs);
    bcastDiagCol.resize(numDiagBufs);

    for (int i = 0; i < numDiagBufs; i++) /* Sherry?? these strcutures not used */
    {
        diagFactBufs[i] = (double *)SUPERLU_MALLOC(sizeof(double) * ldt * ldt);
        bcastStruct bcDiagRow(grid3d->rscp.comm, MPI_DOUBLE, SYNC);
        bcastDiagRow[i] = bcDiagRow;
        bcastStruct bcDiagCol(grid3d->cscp.comm, MPI_DOUBLE, SYNC);
        bcastDiagCol[i] = bcDiagCol;
    }

    int mxLeafNode = 0;
    int_t *myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t *myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t **sForests = trf3Dpartition->sForests;
    for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode)
            mxLeafNode = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
    }
    //Yang: how is dFBufs being used in the c++ factorization code? Shall we call dinitDiagFactBufsArrMod instead to save memory? 
    dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);
    maxLeafNodes = mxLeafNode;

    
    double tGPU = SuperLU_timer_();
    if(superlu_acc_offload)
    {
    #ifdef HAVE_CUDA
        setLUstruct_GPU();  /* Set up LU structure and buffers on GPU */
	
        // TODO: remove it, checking is very slow 
        if(0)
            checkGPU();     
    #endif
    }
        
    tGPU = SuperLU_timer_() -tGPU;
    printf("Time to intialize GPU DS= %g\n",tGPU );
    //

    // if (superluAccOffload)

    // for(int pc=0;pc<Pc; pc++)
    // {
    //     MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    //     ...
    // }

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d_in->iam, "Exit LUstruct_v100 constructor");
#endif
    
} /* constructor LUstruct_v100 */

int_t LUstruct_v100::dSchurComplementUpdate(
    int_t k, lpanel_t &lpanel, upanel_t &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;
        blockUpdate(k, ii, jj, lpanel, upanel);
    }

    return 0;
}

// should be called from an openMP region
int_t *LUstruct_v100::computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                                         int_t dstLen, int_t *dstVec)
{
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        return srcVec;
    }
    int_t thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif
    int_t *dstIdx = indirect + thread_id * ldt;
    for (int_t i = 0; i < dstLen; i++)
    {
        // if(thread_id < dstLen)
        dstIdx[dstVec[i]] = i;
    }

    int_t *RCmap = (direction == ROW_MAP) ? indirectRow : indirectCol;
    RCmap += thread_id * ldt;

    for (int_t i = 0; i < srcLen; i++)
    {
        RCmap[i] = dstIdx[srcVec[i]];
    }

    return RCmap;
}

int_t LUstruct_v100::dScatter(int_t m, int_t n,
                              int_t gi, int_t gj,
                              double *Src, int_t ldsrc,
                              int_t *srcRowList, int_t *srcColList)
{

    double *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    if (gj > gi) // its in upanel
    {
        int li = g2lRow(gi);
        int lj = uPanelVec[li].find(gj);
        Dst = uPanelVec[li].blkPtr(lj);
        lddst = supersize(gi);
        dstRowLen = supersize(gi);
        dstRowList = NULL;
        dstColLen = uPanelVec[li].nbcol(lj);
        dstColList = uPanelVec[li].colList(lj);
        // std::cout<<li<<" "<<lj<<" Dst[0] is"<<Dst[0] << "\n";
    }
    else
    {
        int lj = g2lCol(gj);
        int li = lPanelVec[lj].find(gi);
        Dst = lPanelVec[lj].blkPtr(li);
        lddst = lPanelVec[lj].LDA();
        dstRowLen = lPanelVec[lj].nbrow(li);
        dstRowList = lPanelVec[lj].rowList(li);
        dstColLen = supersize(gj);
        dstColList = NULL;
    }

    // compute source row to dest row mapping
    int_t *rowS2D = computeIndirectMap(ROW_MAP, m, srcRowList,
                                       dstRowLen, dstRowList);

    // compute source col to dest col mapping
    int_t *colS2D = computeIndirectMap(COL_MAP, n, srcColList,
                                       dstColLen, dstColList);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            // if( i+j<5 && gi==521 && gj==521)
            //     printf(" (%d %d, %lf %lf)\n", i, j, Dst[rowS2D[i] + lddst * colS2D[j]], Src[i + ldsrc * j]);
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
        }
    }

    return 0;
}

int_t LUstruct_v100::packedU2skyline(dLUstruct_t *LUstruct)
{

    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    double **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            int_t globalId = i * Pr + myrow;
            uPanelVec[i].packed2skyline(globalId, Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
        }
    }

    return 0;
}

int numProcsPerNode(MPI_Comm baseCommunicator)
{
    MPI_Comm sharedComm;
    MPI_Comm_split_type(baseCommunicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedComm);
    int count = 0;
    MPI_Comm_size(sharedComm, &count);
    return count;
}


int_t LUstruct_v100::lookAheadUpdate(
    int_t k, int_t laIdx, lpanel_t &lpanel, upanel_t &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t laILoc = lpanel.find(laIdx);
    int_t nub = upanel.nblocks();
    int_t laJLoc = upanel.find(laIdx);

#pragma omp parallel
    {
        /*Next lpanelUpdate*/
#pragma omp for nowait
        for (size_t ii = st_lb; ii < nlb; ii++)
        {
            int_t jj = laJLoc;
            if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
                blockUpdate(k, ii, jj, lpanel, upanel);
        }

        /*Next upanelUpdate*/
#pragma omp for nowait
        for (size_t jj = 0; jj < nub; jj++)
        {
            int_t ii = laILoc;
            if (laILoc != GLOBAL_BLOCK_NOT_FOUND && jj != laJLoc)
                blockUpdate(k, ii, jj, lpanel, upanel);
        }
    }

    return 0;
}

int_t LUstruct_v100::blockUpdate(int_t k,
                                 int_t ii, int_t jj, lpanel_t &lpanel, upanel_t &upanel)
{
    int thread_id;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif

    double *V = bigV + thread_id * ldt * ldt;

    double alpha = 1.0;
    double beta = 0.0;
    superlu_dgemm("N", "N",
                  lpanel.nbrow(ii), upanel.nbcol(jj), supersize(k), alpha,
                  lpanel.blkPtr(ii), lpanel.LDA(),
                  upanel.blkPtr(jj), upanel.LDA(), beta,
                  V, lpanel.nbrow(ii));

    // now do the scatter
    int_t ib = lpanel.gid(ii);
    int_t jb = upanel.gid(jj);

    dScatter(lpanel.nbrow(ii), upanel.nbcol(jj),
             ib, jb, V, lpanel.nbrow(ii),
             lpanel.rowList(ii), upanel.colList(jj));
    return 0;
}

int_t LUstruct_v100::dSchurCompUpdateExcludeOne(
    int_t k, int_t ex, // suypernodes to be excluded
    lpanel_t &lpanel, upanel_t &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = upanel.find(ex);

#pragma omp parallel for
    for (size_t ij = 0; ij < (nlb - st_lb) * nub; ij++)
    {
        int_t ii = ij / nub + st_lb;
        int_t jj = ij % nub;

        if (ii != exILoc && jj != exJLoc)
            blockUpdate(k, ii, jj, lpanel, upanel);
    }
    return 0;
}

int_t LUstruct_v100::dDiagFactorPanelSolve(int_t k, int_t offset, ddiagFactBufs_t **dFBufs)
{

    int_t ksupc = SuperSize(k);
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

    return 0;
}

int_t LUstruct_v100::dPanelBcast(int_t k, int_t offset)
{
    /*=======   Panel Broadcast             ======*/
    upanel_t k_upanel(UidxRecvBufs[offset], UvalRecvBufs[offset]);
    lpanel_t k_lpanel(LidxRecvBufs[offset], LvalRecvBufs[offset]);
    if (myrow == krow(k))
        k_upanel = uPanelVec[g2lRow(k)];

    if (mycol == kcol(k))
        k_lpanel = lPanelVec[g2lCol(k)];

    if (UidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_upanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
        MPI_Bcast(k_upanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
    }

    if (LidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_lpanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
        MPI_Bcast(k_lpanel.val, LvalSendCounts[k], MPI_DOUBLE, kcol(k), grid3d->rscp.comm);
    }
    return 0;
}

// double *gpuLvalBasePtr, *gpuUvalBasePtr;
//     int_t *gpuLidxBasePtr, *gpuUidxBasePtr;
//     size_t gpuLvalSize, gpuUvalSize, gpuLidxSize, gpuUidxSize;
//     lpanelGPU_t* copyLpanelsToGPU();
//     upanelGPU_t* copyUpanelsToGPU();

/*Adding new function lupanel initialization

we have the following variable and functions 
double *gpuLvalBasePtr, *gpuUvalBasePtr;
    int_t *gpuLidxBasePtr, *gpuUidxBasePtr;
    size_t gpuLvalSize, gpuUvalSize, gpuLidxSize, gpuUidxSize;
    
    upanelGPU_t* copyUpanelsToGPU();
*/


