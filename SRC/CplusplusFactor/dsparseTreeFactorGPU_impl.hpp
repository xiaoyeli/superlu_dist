#pragma once
#include <cstdio>
#include <cinttypes>
#include "superlu_ddefs.h"
#include "lupanels.hpp"
#include "xlupanels.hpp"

#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#include "xlupanels_GPU.cuh"
#include "batch_block_copy.h"


int getBufferOffset(int k0, int k1, int winSize, int winParity, int halfWin)
{
    int offset = (k0 - k1) % winSize;
    if (winParity % 2)
         offset += halfWin;

    return offset;
}

#if 0 //////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Ftype>
xLUMarshallData<Ftype>::xLUMarshallData()
{
    dev_diag_ptrs = dev_panel_ptrs = NULL;
    dev_diag_ld_array = dev_diag_dim_array = dev_info_array = NULL;
    dev_panel_ld_array = dev_panel_dim_array = NULL;
}

template <typename Ftype>
xLUMarshallData<Ftype>::~xLUMarshallData()
{
    gpuErrchk(cudaFree(dev_diag_ptrs));
    gpuErrchk(cudaFree(dev_panel_ptrs));
    gpuErrchk(cudaFree(dev_diag_ld_array));
    gpuErrchk(cudaFree(dev_diag_dim_array));
    gpuErrchk(cudaFree(dev_info_array));
    gpuErrchk(cudaFree(dev_panel_ld_array));
    gpuErrchk(cudaFree(dev_panel_dim_array));
}

template <typename Ftype>
void xLUMarshallData<Ftype>::setBatchSize(int batch_size)
{
    gpuErrchk(cudaMalloc(&dev_diag_ptrs, batch_size * sizeof(Ftype *)));
    gpuErrchk(cudaMalloc(&dev_panel_ptrs, batch_size * sizeof(Ftype *)));

    gpuErrchk(cudaMalloc(&dev_diag_ld_array, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_diag_dim_array, (batch_size + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_info_array, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_panel_ld_array, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_panel_dim_array, (batch_size + 1) * sizeof(int)));

    host_diag_ptrs.resize(batch_size);
    host_diag_ld_array.resize(batch_size);
    host_diag_dim_array.resize(batch_size);
    host_panel_ptrs.resize(batch_size);
    host_panel_ld_array.resize(batch_size);
    host_panel_dim_array.resize(batch_size);
}

template <typename Ftype>
void xLUMarshallData<Ftype>::setMaxDiag()
{
    max_diag = 0;
    for (int i = 0; i < batchsize; i++)
        max_diag = SUPERLU_MAX(max_diag, host_diag_dim_array[i]);
}

template <typename Ftype>
void xLUMarshallData<Ftype>::setMaxPanel()
{
    max_panel = 0;
    for (int i = 0; i < batchsize; i++)
        max_panel = SUPERLU_MAX(max_panel, host_panel_dim_array[i]);
}

template <typename Ftype>
xSCUMarshallData<Ftype>::xSCUMarshallData()
{
    dev_A_ptrs = dev_B_ptrs = dev_C_ptrs = NULL;
    dev_lda_array = dev_ldb_array = dev_ldc_array = NULL;
    dev_m_array = dev_n_array = dev_k_array = NULL;
    dev_gpu_lpanels = NULL;
    dev_gpu_upanels = NULL;
    dev_ist = dev_iend = dev_jst = dev_jend = NULL;
    dev_maxGemmCols = dev_maxGemmRows = NULL;
}

template <typename Ftype>
xSCUMarshallData<Ftype>::~xSCUMarshallData()
{
    gpuErrchk(cudaFree(dev_A_ptrs));
    gpuErrchk(cudaFree(dev_B_ptrs));
    gpuErrchk(cudaFree(dev_C_ptrs));
    gpuErrchk(cudaFree(dev_lda_array));
    gpuErrchk(cudaFree(dev_ldb_array));
    gpuErrchk(cudaFree(dev_ldc_array));
    gpuErrchk(cudaFree(dev_m_array));
    gpuErrchk(cudaFree(dev_n_array));
    gpuErrchk(cudaFree(dev_k_array));
    gpuErrchk(cudaFree(dev_gpu_lpanels));
    gpuErrchk(cudaFree(dev_gpu_upanels));
    gpuErrchk(cudaFree(dev_ist));
    gpuErrchk(cudaFree(dev_iend));
    gpuErrchk(cudaFree(dev_jst));
    gpuErrchk(cudaFree(dev_jend));
    gpuErrchk(cudaFree(dev_maxGemmCols));
    gpuErrchk(cudaFree(dev_maxGemmRows));
}

template <typename Ftype>
void xSCUMarshallData<Ftype>::setBatchSize(int batch_size)
{
    gpuErrchk(cudaMalloc(&dev_A_ptrs, batch_size * sizeof(Ftype *)));
    gpuErrchk(cudaMalloc(&dev_B_ptrs, batch_size * sizeof(Ftype *)));
    gpuErrchk(cudaMalloc(&dev_C_ptrs, batch_size * sizeof(Ftype *)));

    gpuErrchk(cudaMalloc(&dev_lda_array, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_ldb_array, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_ldc_array, batch_size * sizeof(int)));

    gpuErrchk(cudaMalloc(&dev_m_array, (batch_size + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_n_array, (batch_size + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_k_array, (batch_size + 1) * sizeof(int)));

    gpuErrchk(cudaMalloc(&dev_ist, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_iend, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_jst, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_jend, batch_size * sizeof(int)));

    gpuErrchk(cudaMalloc(&dev_maxGemmCols, batch_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&dev_maxGemmRows, batch_size * sizeof(int)));

    gpuErrchk(cudaMalloc(&dev_gpu_lpanels, batch_size * sizeof(lpanelGPU_t)));
    gpuErrchk(cudaMalloc(&dev_gpu_upanels, batch_size * sizeof(upanelGPU_t)));

    host_A_ptrs.resize(batch_size);
    host_B_ptrs.resize(batch_size);
    host_C_ptrs.resize(batch_size);
    host_lda_array.resize(batch_size);
    host_ldb_array.resize(batch_size);
    host_ldc_array.resize(batch_size);
    host_m_array.resize(batch_size);
    host_n_array.resize(batch_size);
    host_k_array.resize(batch_size);
    upanels.resize(batch_size);
    lpanels.resize(batch_size);
    host_gpu_upanels.resize(batch_size);
    host_gpu_lpanels.resize(batch_size);
    ist.resize(batch_size);
    iend.resize(batch_size);
    jst.resize(batch_size);
    jend.resize(batch_size);
    maxGemmRows.resize(batch_size);
    maxGemmCols.resize(batch_size);
}

template <typename Ftype>
void xSCUMarshallData<Ftype>::setMaxDims()
{
    max_n = max_k = max_m = max_ilen = max_jlen = 0;
    for (int i = 0; i < batchsize; i++)
    {
        max_m = SUPERLU_MAX(max_m, host_m_array[i]);
        max_n = SUPERLU_MAX(max_n, host_n_array[i]);
        max_k = SUPERLU_MAX(max_k, host_k_array[i]);
        max_ilen = SUPERLU_MAX(max_ilen, iend[i] - ist[i]);
        max_jlen = SUPERLU_MAX(max_jlen, jend[i] - jst[i]);
    }
}

template <typename Ftype>
void xSCUMarshallData<Ftype>::copyToGPU()
{
    gpuErrchk(cudaMemcpy(dev_A_ptrs, host_A_ptrs.data(), batchsize * sizeof(Ftype *), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_B_ptrs, host_B_ptrs.data(), batchsize * sizeof(Ftype *), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_C_ptrs, host_C_ptrs.data(), batchsize * sizeof(Ftype *), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dev_lda_array, host_lda_array.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_ldb_array, host_ldb_array.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_ldc_array, host_ldc_array.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dev_m_array, host_m_array.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_n_array, host_n_array.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_k_array, host_k_array.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dev_ist, ist.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_iend, iend.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_jst, jst.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_jend, jend.data(), batchsize * sizeof(int), cudaMemcpyHostToDevice));
}

template <typename Ftype>
void xSCUMarshallData<Ftype>::copyPanelDataToGPU()
{
    gpuErrchk(cudaMemcpy(dev_gpu_lpanels, host_gpu_lpanels.data(), batchsize * sizeof(lpanelGPU_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_gpu_upanels, host_gpu_upanels.data(), batchsize * sizeof(upanelGPU_t), cudaMemcpyHostToDevice));
}

#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dDFactPSolveGPU(int_t k, int_t offset, diagFactBufs_type<Ftype> **dFBufs)
{
    // this is new version with diagonal factor being performed on GPU
    // different from dDiagFactorPanelSolveGPU (it performs diag factor in CPU)

    /* Sherry: argument dFBufs[] is on CPU, not used in this routine */

    double t0 = SuperLU_timer_();
    int ksupc = SuperSize(k);
    cublasHandle_t cubHandle = A_gpu.cuHandles[offset];
    cusolverDnHandle_t cusolverH = A_gpu.cuSolveHandles[offset];
    cudaStream_t cuStream = A_gpu.cuStreams[offset];

    /*======= Diagonal Factorization ======*/
    if (iam == procIJ(k, k))
    {
        lPanelVec[g2lCol(k)].diagFactorCuSolver(k,
                                                cusolverH, cuStream,
                                                A_gpu.diagFactWork[offset], A_gpu.diagFactInfo[offset], // CPU pointers
                                                A_gpu.dFBufs[offset], ksupc,                            // CPU pointers
                                                thresh, xsup, options, stat, info);
    }

    // CHECK_MALLOC(iam, "after diagFactorCuSolver()");

    // TODO: need to synchronize the cuda stream
    /*======= Diagonal Broadcast ======*/
    if (myrow == krow(k))
        MPI_Bcast((void *)A_gpu.dFBufs[offset], ksupc * ksupc,
                  get_mpi_type<Ftype>(), kcol(k), (grid->rscp).comm);

    // CHECK_MALLOC(iam, "after row Bcast");

    if (mycol == kcol(k))
        MPI_Bcast((void *)A_gpu.dFBufs[offset], ksupc * ksupc,
                  get_mpi_type<Ftype>(), krow(k), (grid->cscp).comm);

    // do the panels solver
    if (myrow == krow(k))
    {
        uPanelVec[g2lRow(k)].panelSolveGPU(
            cubHandle, cuStream,
            ksupc, A_gpu.dFBufs[offset], ksupc);
        cudaStreamSynchronize(cuStream); // synchronize befpre broadcast
    }

    if (mycol == kcol(k))
    {
        lPanelVec[g2lCol(k)].panelSolveGPU(
            cubHandle, cuStream,
            ksupc, A_gpu.dFBufs[offset], ksupc);
        cudaStreamSynchronize(cuStream);
    }
    SCT->tDiagFactorPanelSolve += (SuperLU_timer_() - t0);

    return 0;
} /* dDFactPSolveGPU */

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dDFactPSolveGPU(int_t k, int_t handle_offset, int buffer_offset, diagFactBufs_type<Ftype> **dFBufs)
{
    // this is new version with diagonal factor being performed on GPU
    // different from dDiagFactorPanelSolveGPU (it performs diag factor in CPU)

    /* Sherry: argument dFBufs[] is on CPU, not used in this routine */

    double t0 = SuperLU_timer_();
    int ksupc = SuperSize(k);
    cublasHandle_t cubHandle = A_gpu.cuHandles[handle_offset];
    cusolverDnHandle_t cusolverH = A_gpu.cuSolveHandles[handle_offset];
    cudaStream_t cuStream = A_gpu.cuStreams[handle_offset];

    /*======= Diagonal Factorization ======*/
    if (iam == procIJ(k, k))
    {
        lPanelVec[g2lCol(k)].diagFactorCuSolver(k,
                                                cusolverH, cuStream,
                                                A_gpu.diagFactWork[handle_offset], A_gpu.diagFactInfo[handle_offset], // CPU pointers
                                                A_gpu.dFBufs[buffer_offset], ksupc,                                   // CPU pointers
                                                thresh, xsup, options, stat, info);
    }

    // CHECK_MALLOC(iam, "after diagFactorCuSolver()");

    // TODO: need to synchronize the cuda stream
    /*======= Diagonal Broadcast ======*/
    if (myrow == krow(k))
        MPI_Bcast((void *)A_gpu.dFBufs[buffer_offset], ksupc * ksupc,
                  get_mpi_type<Ftype>(), kcol(k), (grid->rscp).comm);

    // CHECK_MALLOC(iam, "after row Bcast");

    if (mycol == kcol(k))
        MPI_Bcast((void *)A_gpu.dFBufs[buffer_offset], ksupc * ksupc,
                  get_mpi_type<Ftype>(), krow(k), (grid->cscp).comm);

    // do the panels solver
    if (myrow == krow(k))
    {
        uPanelVec[g2lRow(k)].panelSolveGPU(
            cubHandle, cuStream,
            ksupc, A_gpu.dFBufs[buffer_offset], ksupc);
        cudaStreamSynchronize(cuStream); // synchronize befpre broadcast
    }

    if (mycol == kcol(k))
    {
        lPanelVec[g2lCol(k)].panelSolveGPU(
            cubHandle, cuStream,
            ksupc, A_gpu.dFBufs[buffer_offset], ksupc);
        cudaStreamSynchronize(cuStream);
    }
    SCT->tDiagFactorPanelSolve += (SuperLU_timer_() - t0);

    return 0;
} /* dDFactPSolveGPU */

/* This performs diag factor on CPU */
template <typename Ftype>
int_t xLUstruct_t<Ftype>::dDiagFactorPanelSolveGPU(int_t k, int_t offset, diagFactBufs_type<Ftype> **dFBufs)
{
    double t0 = SuperLU_timer_();
    int_t ksupc = SuperSize(k);
    cublasHandle_t cubHandle = A_gpu.cuHandles[offset];
    cudaStream_t cuStream = A_gpu.cuStreams[offset];
    if (iam == procIJ(k, k))
    {

        lPanelVec[g2lCol(k)].diagFactorPackDiagBlockGPU(k,
                                                        dFBufs[offset]->BlockUFactor, ksupc, // CPU pointers
                                                        dFBufs[offset]->BlockLFactor, ksupc, // CPU pointers
                                                        thresh, xsup, options, stat, info);
    }

    /*=======   Diagonal Broadcast          ======*/
    if (myrow == krow(k))
        MPI_Bcast((void *)dFBufs[offset]->BlockLFactor, ksupc * ksupc,
                  get_mpi_type<Ftype>(), kcol(k), (grid->rscp).comm);
    if (mycol == kcol(k))
        MPI_Bcast((void *)dFBufs[offset]->BlockUFactor, ksupc * ksupc,
                  get_mpi_type<Ftype>(), krow(k), (grid->cscp).comm);

    /*=======   Panel Update                ======*/
    if (myrow == krow(k))
    {

        cudaMemcpy(A_gpu.dFBufs[offset], dFBufs[offset]->BlockLFactor,
                   ksupc * ksupc * sizeof(Ftype), cudaMemcpyHostToDevice);
        uPanelVec[g2lRow(k)].panelSolveGPU(
            cubHandle, cuStream,
            ksupc, A_gpu.dFBufs[offset], ksupc);
        cudaStreamSynchronize(cuStream); // synchronize befpre broadcast
    }

    if (mycol == kcol(k))
    {
        cudaMemcpy(A_gpu.dFBufs[offset], dFBufs[offset]->BlockUFactor,
                   ksupc * ksupc * sizeof(Ftype), cudaMemcpyHostToDevice);
        lPanelVec[g2lCol(k)].panelSolveGPU(
            cubHandle, cuStream,
            ksupc, A_gpu.dFBufs[offset], ksupc);
        cudaStreamSynchronize(cuStream);
    }
    SCT->tDiagFactorPanelSolve += (SuperLU_timer_() - t0);

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dPanelBcastGPU(int_t k, int_t offset)
{
    double t0 = SuperLU_timer_();
    /*=======   Panel Broadcast             ======*/
    // upanel_t k_upanel(UidxRecvBufs[offset], UvalRecvBufs[offset],
    //                   A_gpu.UidxRecvBufs[offset], A_gpu.UvalRecvBufs[offset]);
    // lpanel_t k_lpanel(LidxRecvBufs[offset], LvalRecvBufs[offset],
    //                   A_gpu.LidxRecvBufs[offset], A_gpu.LvalRecvBufs[offset]);
    // if (myrow == krow(k))
    // {
    //     k_upanel = uPanelVec[g2lRow(k)];
    // }
    // if (mycol == kcol(k))
    //     k_lpanel = lPanelVec[g2lCol(k)];
    xupanel_t<Ftype> k_upanel = getKUpanel(k, offset);
    xlpanel_t<Ftype> k_lpanel = getKLpanel(k, offset);

    if (UidxSendCounts[k] > 0)
    {
        // assuming GPU direct is available
        MPI_Bcast(k_upanel.gpuPanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
        MPI_Bcast(k_upanel.gpuPanel.val, UvalSendCounts[k], get_mpi_type<Ftype>(), krow(k), grid3d->cscp.comm);
        // copy the index to cpu
        gpuErrchk(cudaMemcpy(k_upanel.index, k_upanel.gpuPanel.index,
                             sizeof(int_t) * UidxSendCounts[k], cudaMemcpyDeviceToHost));
    }

    if (LidxSendCounts[k] > 0)
    {
        MPI_Bcast(k_lpanel.gpuPanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
        MPI_Bcast(k_lpanel.gpuPanel.val, LvalSendCounts[k], get_mpi_type<Ftype>(), kcol(k), grid3d->rscp.comm);
        gpuErrchk(cudaMemcpy(k_lpanel.index, k_lpanel.gpuPanel.index,
                             sizeof(int_t) * LidxSendCounts[k], cudaMemcpyDeviceToHost));
    }
    SCT->tPanelBcast += (SuperLU_timer_() - t0);
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dsparseTreeFactorGPU(
    sForest_t *sforest,
    diagFactBufs_type<Ftype> **dFBufs, // size maxEtree level
    gEtreeInfo_t *gEtreeInfo, // global etree info

    int tag_ub)
{
    int_t nnodes = sforest->nNodes; // number of nodes in the tree
    if (nnodes < 1)
    {
        return 1;
    }

    int_t *perm_c_supno = sforest->nodeList; // list of nodes in the order of factorization
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t *myIperm = treeTopoInfo->myIperm;
    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;

    /*main loop over all the levels*/
    int_t numLA = SUPERLU_MIN(A_gpu.numCudaStreams, getNumLookAhead(options));

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Enter dsparseTreeFactorGPU()");

    printf("Using New code V100 with GPU acceleration\n");
    fflush(stdout);
    printf(". lookahead numLA %d\n", numLA);
    fflush(stdout);
#endif
    // start the pipeline.  Sherry: need to free these 3 arrays
    int *donePanelBcast = int32Malloc_dist(nnodes);
    int *donePanelSolve = int32Malloc_dist(nnodes);
    int *localNumChildrenLeft = int32Malloc_dist(nnodes);

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
        // dDiagFactorPanelSolveGPU(k, offset, dFBufs);
        dDFactPSolveGPU(k, offset, dFBufs);
        donePanelSolve[k0] = 1;
    }

    // TODO: its really the panels that needs to be doubled
    //  everything else can remain as it is
    int_t winSize = SUPERLU_MIN(numLA / 2, eTreeTopLims[1]);

    // printf(". lookahead winSize %d\n", winSize);
#if ( PRNTlevel >= 1 )    
    printf(". lookahead winSize %" PRIdMAX "\n", static_cast<intmax_t>(winSize));
    fflush(stdout);
#endif
    
    for (int k0 = k_st; k0 < winSize; ++k0)
    {
        int_t k = perm_c_supno[k0];
        int_t offset = k0 % numLA;
        if (!donePanelBcast[k0])
        {
            dPanelBcastGPU(k, offset);
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
            int_t offset = getBufferOffset(k0, k1, winSize, winParity, halfWin);
            xupanel_t<Ftype> k_upanel = getKUpanel(k, offset);
            xlpanel_t<Ftype> k_lpanel = getKLpanel(k, offset);
            int_t k_parent = gEtreeInfo->setree[k];
            /* L o o k   A h e a d   P a n e l   U p d a t e */
            if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
                lookAheadUpdateGPU(offset, k, k_parent, k_lpanel, k_upanel);
        }

        for (int_t k0 = k1; k0 < SUPERLU_MIN(nnodes, k1 + winSize); ++k0)
        {
            // int_t k = perm_c_supno[k0];
            int_t offset = getBufferOffset(k0, k1, winSize, winParity, halfWin);
            SyncLookAheadUpdate(offset);
        }

        for (int_t k0 = k1; k0 < SUPERLU_MIN(nnodes, k1 + winSize); ++k0)
        {
            int_t k = perm_c_supno[k0];
            int_t offset = getBufferOffset(k0, k1, winSize, winParity, halfWin);
            xupanel_t<Ftype> k_upanel = getKUpanel(k, offset);
            xlpanel_t<Ftype> k_lpanel = getKLpanel(k, offset);
            int_t k_parent = gEtreeInfo->setree[k];
            /* Look Ahead Panel Solve */
            if (k_parent < nsupers)
            {
                int_t k0_parent = myIperm[k_parent];
                if (k0_parent > 0 && k0_parent < nnodes)
                {
                    localNumChildrenLeft[k0_parent]--;
                    if (topoLvl < maxTopoLevel - 1 && !localNumChildrenLeft[k0_parent])
                    {
#if (PRNTlevel >= 2)
                        printf("parent %d of node %d during second phase\n", k0_parent, k0);
#endif
                        int_t dOffset = 0; // this is wrong
                        // dDiagFactorPanelSolveGPU(k_parent, dOffset,dFBufs);
                        dDFactPSolveGPU(k_parent, dOffset, dFBufs);
                        donePanelSolve[k0_parent] = 1;
                    }
                }
            }

            /*proceed with remaining SchurComplement update */
            if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
                dSchurCompUpdateExcludeOneGPU(offset, k, k_parent, k_lpanel, k_upanel);
        }

        int_t k1_next = k1 + winSize;
        int_t oldWinSize = winSize;
        for (int_t k0_next = k1_next; k0_next < SUPERLU_MIN(nnodes, k1_next + winSize); ++k0_next)
        {
            int k_next = perm_c_supno[k0_next];
            if (!localNumChildrenLeft[k0_next])
            {
                // int offset_next = (k0_next-k1_next)%winSize;
                // if(!(winParity%2))
                //     offset_next += halfWin;

                int_t offset_next = getBufferOffset(k0_next, k1_next, winSize, winParity + 1, halfWin);
                dPanelBcastGPU(k_next, offset_next);
                donePanelBcast[k0_next] = 1;
                // printf("Trying  %d on offset %d\n", k0_next, offset_next);
            }
            else
            {
                winSize = k0_next - k1_next;
                break;
            }
        }

        for (int_t k0 = k1; k0 < SUPERLU_MIN(nnodes, k1 + oldWinSize); ++k0)
        {
            int_t k = perm_c_supno[k0];
            // int_t offset = (k0-k1)%oldWinSize;
            // if(winParity%2)
            //     offset+= halfWin;
            int_t offset = getBufferOffset(k0, k1, oldWinSize, winParity, halfWin);
            // printf("Syncing stream %d on offset %d\n", k0, offset);
            if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
                gpuErrchk(cudaStreamSynchronize(A_gpu.cuStreams[offset]));
        }

        k1 = k1_next;
        winParity++;
    }

#if 0
    
    for (int_t topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)
    {
        /* code */
        int_t k_st = eTreeTopLims[topoLvl];
        int_t k_end = eTreeTopLims[topoLvl + 1];
        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0];

            int_t ksupc = SuperSize(k);
            cublasHandle_t cubHandle = A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            dDiagFactorPanelSolveGPU(k, 0, dFBufs);
            /*=======   Panel Broadcast             ======*/
            // panelBcastGPU(k, 0);
            int_t offset = k0%numLA;
            dPanelBcastGPU(k, offset);

            /*=======   Schurcomplement Update      ======*/
            upanel_t k_upanel(UidxRecvBufs[offset], UvalRecvBufs[offset],
                              A_gpu.UidxRecvBufs[offset], A_gpu.UvalRecvBufs[offset]);
            lpanel_t k_lpanel(LidxRecvBufs[offset], LvalRecvBufs[offset],
                              A_gpu.LidxRecvBufs[offset], A_gpu.LvalRecvBufs[offset]);
            if (myrow == krow(k))
            {
                k_upanel = uPanelVec[g2lRow(k)];
            }
            if (mycol == kcol(k))
                k_lpanel = lPanelVec[g2lCol(k)];

            if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
            {
                int streamId = 0;

#if 0

                dSchurComplementUpdateGPU(
                    streamId,
                    k, k_lpanel, k_upanel);
#else
                int_t k_parent = gEtreeInfo->setree[k];
                lookAheadUpdateGPU(
                    streamId,
                    k, k_parent, k_lpanel, k_upanel);
                dSchurCompUpdateExcludeOneGPU(
                    streamId,
                    k, k_parent, k_lpanel, k_upanel);

#endif

            }
        } /*for k0= k_st:k_end */
    } /*for topoLvl = 0:maxTopoLevel*/

#endif /* match  #if 0 at line 562 before "for (int_t topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)" */

    /* Sherry added 2/1/23 */
    SUPERLU_FREE(donePanelBcast);
    SUPERLU_FREE(donePanelSolve);
    SUPERLU_FREE(localNumChildrenLeft);

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Exit dsparseTreeFactorGPU()");
#endif

    return 0;
} /* dsparseTreeFactorGPU */


#if 0 //////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Ftype>
void xLUstruct_t<Ftype>::marshallBatchedBufferCopyData(int k_st, int k_end, int_t *perm_c_supno)
{
    // First gather up all the pointer and meta data on the host
    LUMarshallData &mdata = A_gpu.marshall_data;
    Ftype **panel_ptrs = mdata.host_panel_ptrs.data();
    int *panel_ld_batch = mdata.host_panel_ld_array.data();
    int *panel_dim_batch = mdata.host_panel_dim_array.data();
    Ftype **diag_ptrs = mdata.host_diag_ptrs.data();
    int *diag_ld_batch = mdata.host_diag_ld_array.data();
    int *diag_dim_batch = mdata.host_diag_dim_array.data();

    mdata.batchsize = 0;

    for (int_t k0 = k_st; k0 < k_end; k0++)
    {
        int_t k = perm_c_supno[k0];
        int_t buffer_offset = k0 - k_st;
        int ksupc = SuperSize(k);

        if (iam == procIJ(k, k))
        {
            lpanel_t &lpanel = lPanelVec[g2lCol(k)];

            assert(mdata.batchsize < mdata.host_diag_ptrs.size());

            panel_ptrs[mdata.batchsize] = lpanel.blkPtrGPU(0);
            panel_ld_batch[mdata.batchsize] = lpanel.LDA();
            panel_dim_batch[mdata.batchsize] = ksupc;

            diag_ptrs[mdata.batchsize] = A_gpu.dFBufs[buffer_offset];
            diag_ld_batch[mdata.batchsize] = ksupc;
            diag_dim_batch[mdata.batchsize] = ksupc;

            mdata.batchsize++;
        }
    }

    mdata.setMaxDiag();
    mdata.setMaxPanel();

    // Then copy the marshalled data over to the GPU
    gpuErrchk(cudaMemcpy(mdata.dev_diag_ptrs, diag_ptrs, mdata.batchsize * sizeof(Ftype *), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mdata.dev_diag_ld_array, diag_ld_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mdata.dev_diag_dim_array, diag_dim_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mdata.dev_panel_ptrs, panel_ptrs, mdata.batchsize * sizeof(Ftype *), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mdata.dev_panel_ld_array, panel_ld_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mdata.dev_panel_dim_array, panel_dim_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
}

#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Ftype>
void xLUstruct_t<Ftype>::dFactBatchSolve(int k_st, int k_end, int_t *perm_c_supno)
{
#if 0//def HAVE_MAGMA
    //#ifdef HAVE_MAGMA
    // Marshall the data for the leaf level batched LU decomposition
    LUMarshallData &mdata = A_gpu.marshall_data;
    cudaStream_t stream = A_gpu.cuStreams[0];
    marshallBatchedLUData(k_st, k_end, perm_c_supno);

    int info = magma_dgetrf_vbatched(
        mdata.dev_diag_dim_array, mdata.dev_diag_dim_array,
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array,
        NULL, mdata.dev_info_array, mdata.batchsize,
        A_gpu.magma_queue);

    // Hackathon change: assuming individual matrices in a local batch are not distributed
    // so we don't do the buffer copies anymore
    // // Copy the diagonal block data over to the bcast buffers
    // marshallBatchedBufferCopyData(k_st, k_end, perm_c_supno);

    // copyBlock_vbatch(
    //     stream, mdata.dev_diag_dim_array, mdata.dev_panel_dim_array, mdata.max_diag, mdata.max_panel,
    //     mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_panel_ptrs, mdata.dev_panel_ld_array,
    //     mdata.batchsize
    // );

    // Upper panel triangular solves
    marshallBatchedTRSMUData(k_st, k_end, perm_c_supno);

    magmablas_dtrsm_vbatched(
        MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
        mdata.dev_diag_dim_array, mdata.dev_panel_dim_array, 1.0,
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array,
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array,
        mdata.batchsize, A_gpu.magma_queue);

    // Lower panel triangular solves
    marshallBatchedTRSMLData(k_st, k_end, perm_c_supno);

    magmablas_dtrsm_vbatched(
        MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
        mdata.dev_panel_dim_array, mdata.dev_diag_dim_array, 1.0,
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array,
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array,
        mdata.batchsize, A_gpu.magma_queue);

    // Wajih: This should be converted to a single bcast if possible
    // Hackathon change: assuming individual matrices in a local batch are not distributed
    // for (int k0 = k_st; k0 < k_end; k0++)
    // {
    //     int k = perm_c_supno[k0];
    //     int offset = 0;
    //     /*======= Panel Broadcast  ======*/
    //     dPanelBcastGPU(k, offset);
    // }

    // Initialize the schur complement update marshall data
    initSCUMarshallData(k_st, k_end, perm_c_supno);
    xSCUMarshallData &sc_mdata = A_gpu.sc_marshall_data;

    // Keep marshalling while there are batches to be processed
    int done_i = 0;
    int total_batches = 0;
    while (done_i == 0)
    {
        done_i = marshallSCUBatchedDataOuter(k_st, k_end, perm_c_supno);
        int done_j = 0;
        while (done_i == 0 && done_j == 0)
        {
            done_j = marshallSCUBatchedDataInner(k_st, k_end, perm_c_supno);
            if (done_j != 1)
            {
                total_batches++;
                magmablas_dgemm_vbatched_max_nocheck(
                    MagmaNoTrans, MagmaNoTrans, sc_mdata.dev_m_array, sc_mdata.dev_n_array, sc_mdata.dev_k_array,
                    1.0, sc_mdata.dev_A_ptrs, sc_mdata.dev_lda_array, sc_mdata.dev_B_ptrs, sc_mdata.dev_ldb_array,
                    0.0, sc_mdata.dev_C_ptrs, sc_mdata.dev_ldc_array, sc_mdata.batchsize,
                    sc_mdata.max_m, sc_mdata.max_n, sc_mdata.max_k, A_gpu.magma_queue);

                scatterGPU_batchDriver(
                    sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_jst, sc_mdata.dev_jend,
                    sc_mdata.max_ilen, sc_mdata.max_jlen, sc_mdata.dev_C_ptrs, sc_mdata.dev_ldc_array,
                    A_gpu.maxSuperSize, ldt, sc_mdata.dev_gpu_lpanels, sc_mdata.dev_gpu_upanels,
                    dA_gpu, sc_mdata.batchsize, A_gpu.cuStreams[0]);
            }
        }
    }
    // printf("SCU batches = %d\n", total_batches);
#endif
}

#if 0
//// This is not used anymore //////
template <typename Ftype>
int xLUstruct_t<Ftype>::dsparseTreeFactorBatchGPU(
    sForest_t *sforest,
    diagFactBufs_type<Ftype> **dFBufs, // size maxEtree level
    gEtreeInfo_t *gEtreeInfo, // global etree info
    int tag_ub)
{
#if 0// def HAVE_MAGMA
    int nnodes = sforest->nNodes; // number of nodes in the tree
    int topoLvl, k_st, k_end, k0, k, offset, ksupc;
    if (nnodes < 1)
    {
        return 1;
    }

    int_t *perm_c_supno = sforest->nodeList; // list of nodes in the order of factorization
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t *myIperm = treeTopoInfo->myIperm;
    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Enter dsparseTreeFactorBatchGPU()");
#endif
    printf("Using level-based scheduling on GPU\n");
    fflush(stdout);

    /* For all the leaves at level 0 */
    topoLvl = 0;
    k_st = eTreeTopLims[topoLvl];
    k_end = eTreeTopLims[topoLvl + 1];
    printf("level %d: k_st %d, k_end %d\n", topoLvl, k_st, k_end);
    fflush(stdout);

#if 0
    //ToDo: make this batched 
    for (k0 = k_st; k0 < k_end; k0++)
    {
        k = perm_c_supno[k0];
        offset = k0 - k_st;
        // dDiagFactorPanelSolveGPU(k, offset, dFBufs);
        dDFactPSolveGPU(k, 0, offset, dFBufs);

        /*======= Panel Broadcast  ======*/
        dPanelBcastGPU(k, offset); // does this only if (UidxSendCounts[k] > 0)
        //donePanelSolve[k0]=1;

        /*======= Schurcomplement Update ======*/
        /* UidxSendCounts are computed in LUstruct_v100 constructor in LUpanels.cpp */
        if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0) {
            // k_upanel.checkCorrectness();
            int streamId = 0;
                upanel_t k_upanel = getKUpanel(k,offset);
                lpanel_t k_lpanel = getKLpanel(k,offset);
            dSchurComplementUpdateGPU( streamId,
                        k, k_lpanel, k_upanel);
            // cudaStreamSynchronize(cuStream); // there is sync inside the kernel
        }
    }
    
    /* Main loop over all the internal levels */
    for (topoLvl = 1; topoLvl < maxTopoLevel; ++topoLvl) {
      
        k_st = eTreeTopLims[topoLvl];
        k_end = eTreeTopLims[topoLvl + 1];

    printf("level %d: k_st %d, k_end %d\n", topoLvl, k_st, k_end); fflush(stdout);
    
	/* loop over all the nodes at level topoLvl */
        for (k0 = k_st; k0 < k_end; ++k0) { /* ToDo: batch this */
            k = perm_c_supno[k0];
            offset = k0 - k_st;
            // offset = getBufferOffset(k0, k1, winSize, winParity, halfWin);
            //ksupc = SuperSize(k);

            dDFactPSolveGPU(k, 0, offset, dFBufs);

                /*======= Panel Broadcast  ======*/
            dPanelBcastGPU(k, offset); // does this only if (UidxSendCounts[k] > 0)
	    
            /*======= Schurcomplement Update ======*/
            if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
            {
                // k_upanel.checkCorrectness();
                int streamId = 0;

		//#define NDEBUG
#ifndef NDEBUG
                checkGPU(); // ??
#endif
                upanel_t k_upanel = getKUpanel(k,offset);
                lpanel_t k_lpanel = getKLpanel(k,offset);
                dSchurComplementUpdateGPU(streamId,
					  k, k_lpanel, k_upanel);
// cudaStreamSynchronize(cuStream); // there is sync inside the kernel
#if 0 // Sherry commented out 7/4/23
#ifndef NDEBUG
                dSchurComplementUpdate(k, k_lpanel, k_upanel);   // ?? why do this on CPU ?
n                cudaStreamSynchronize(cuStream);
                checkGPU();
#endif
#endif		
            }
            // MPI_Barrier(grid3d->comm);
        } /* end for k0= k_st:k_end */
    } /* end for topoLvl = 0:maxTopoLevel */
#else
    printf("Using batched code\n");
    double t0 = SuperLU_timer_();

    // Copy the nodelist to the GPU
    gpuErrchk(cudaMemcpy(A_gpu.dperm_c_supno, perm_c_supno, sizeof(int) * sforest->nNodes, cudaMemcpyHostToDevice));

    // Solve level by level
    dFactBatchSolve(k_st, k_end, perm_c_supno);
    for (topoLvl = 1; topoLvl < maxTopoLevel; ++topoLvl)
    {

        k_st = eTreeTopLims[topoLvl];
        k_end = eTreeTopLims[topoLvl + 1];
        dFactBatchSolve(k_st, k_end, perm_c_supno);
    }
    printf("Batch time = %.3f\n", SuperLU_timer_() - t0);
#endif

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Exit dsparseTreeFactorBatchGPU()");
#endif

#else
    printf("MAGMA is required for batched execution!\n");
    fflush(stdout);
    exit(0);

#endif /* match ifdef have_magma */

    return 0;
} /* end dsparseTreeFactorBatchGPU */

#endif /* match #if 0 */
//////////////////////////////////////////////////////////////////////////////////////////////////////////


// TODO: needs to be merged as a single factorization function
template <typename Ftype>
int_t xLUstruct_t<Ftype>::dsparseTreeFactorGPUBaseline(
    sForest_t *sforest,
    diagFactBufs_type<Ftype> **dFBufs, // size maxEtree level
    gEtreeInfo_t *gEtreeInfo, // global etree info

    int tag_ub)
{
    int_t nnodes = sforest->nNodes; // number of nodes in the tree
    if (nnodes < 1)
    {
        return 1;
    }

    printf("Using New code V100 with GPU acceleration\n");
#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Enter dsparseTreeFactorGPUBaseline()");
#endif

    int_t *perm_c_supno = sforest->nodeList; // list of nodes in the order of factorization
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t *myIperm = treeTopoInfo->myIperm;
    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;

    /*main loop over all the levels*/
    int_t numLA = getNumLookAhead(options);

    // start the pipeline

    for (int_t topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)
    {
        /* code */
        int_t k_st = eTreeTopLims[topoLvl];
        int_t k_end = eTreeTopLims[topoLvl + 1];
        for (int_t k0 = k_st; k0 < k_end; ++k0)
        {
            int_t k = perm_c_supno[k0];
            int_t offset = k0 - k_st;
            int_t ksupc = SuperSize(k);
            cublasHandle_t cubHandle = A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            /*=======   Diagonal Factorization      ======*/
            if (iam == procIJ(k, k))
            {
// #define NDEBUG
#ifndef NDEBUG
                lPanelVec[g2lCol(k)].checkGPU();
                lPanelVec[g2lCol(k)].diagFactor(k, dFBufs[offset]->BlockUFactor, ksupc,
                                                thresh, xsup, options, stat, info);
                lPanelVec[g2lCol(k)].packDiagBlock(dFBufs[offset]->BlockLFactor, ksupc);
#endif
                lPanelVec[g2lCol(k)].diagFactorPackDiagBlockGPU(k,
                                                                dFBufs[offset]->BlockUFactor, ksupc, // CPU pointers
                                                                dFBufs[offset]->BlockLFactor, ksupc, // CPU pointers
                                                                thresh, xsup, options, stat, info);
// cudaStreamSynchronize(cuStream);
#ifndef NDEBUG
                cudaStreamSynchronize(cuStream);
                lPanelVec[g2lCol(k)].checkGPU();
#endif
            }

            /*=======   Diagonal Broadcast          ======*/
            if (myrow == krow(k))
                MPI_Bcast((void *)dFBufs[offset]->BlockLFactor, ksupc * ksupc,
                          get_mpi_type<Ftype>(), kcol(k), (grid->rscp).comm);
            if (mycol == kcol(k))
                MPI_Bcast((void *)dFBufs[offset]->BlockUFactor, ksupc * ksupc,
                          get_mpi_type<Ftype>(), krow(k), (grid->cscp).comm);

            /*=======   Panel Update                ======*/
            if (myrow == krow(k))
            {
#ifndef NDEBUG
                uPanelVec[g2lRow(k)].checkGPU();
#endif
                cudaMemcpy(A_gpu.dFBufs[0], dFBufs[offset]->BlockLFactor,
                           ksupc * ksupc * sizeof(Ftype), cudaMemcpyHostToDevice);
                uPanelVec[g2lRow(k)].panelSolveGPU(
                    cubHandle, cuStream,
                    ksupc, A_gpu.dFBufs[0], ksupc);
                cudaStreamSynchronize(cuStream); // synchronize befpre broadcast
#ifndef NDEBUG
                uPanelVec[g2lRow(k)].panelSolve(ksupc, dFBufs[offset]->BlockLFactor, ksupc);
                cudaStreamSynchronize(cuStream);
                uPanelVec[g2lRow(k)].checkGPU();
#endif
            }

            if (mycol == kcol(k))
            {
                cudaMemcpy(A_gpu.dFBufs[0], dFBufs[offset]->BlockUFactor,
                           ksupc * ksupc * sizeof(Ftype), cudaMemcpyHostToDevice);
                lPanelVec[g2lCol(k)].panelSolveGPU(
                    cubHandle, cuStream,
                    ksupc, A_gpu.dFBufs[0], ksupc);
                cudaStreamSynchronize(cuStream);
#ifndef NDEBUG
                lPanelVec[g2lCol(k)].panelSolve(ksupc, dFBufs[offset]->BlockUFactor, ksupc);
                cudaStreamSynchronize(cuStream);
                lPanelVec[g2lCol(k)].checkGPU();
#endif
            }

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
                // assuming GPU direct is available
                MPI_Bcast(k_upanel.gpuPanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
                MPI_Bcast(k_upanel.gpuPanel.val, UvalSendCounts[k], get_mpi_type<Ftype>(), krow(k), grid3d->cscp.comm);
                // copy the index to cpu
                cudaMemcpy(k_upanel.index, k_upanel.gpuPanel.index,
                           sizeof(int_t) * UidxSendCounts[k], cudaMemcpyDeviceToHost);

#ifndef NDEBUG
                MPI_Bcast(k_upanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
                MPI_Bcast(k_upanel.val, UvalSendCounts[k], get_mpi_type<Ftype>(), krow(k), grid3d->cscp.comm);
#endif
            }

            if (LidxSendCounts[k] > 0)
            {
                MPI_Bcast(k_lpanel.gpuPanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
                MPI_Bcast(k_lpanel.gpuPanel.val, LvalSendCounts[k], get_mpi_type<Ftype>(), kcol(k), grid3d->rscp.comm);
                cudaMemcpy(k_lpanel.index, k_lpanel.gpuPanel.index,
                           sizeof(int_t) * LidxSendCounts[k], cudaMemcpyDeviceToHost);

#ifndef NDEBUG
                MPI_Bcast(k_lpanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
                MPI_Bcast(k_lpanel.val, LvalSendCounts[k], get_mpi_type<Ftype>(), kcol(k), grid3d->rscp.comm);
#endif
            }

            /*=======   Schurcomplement Update      ======*/

            if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
            {
                // k_upanel.checkCorrectness();
                int streamId = 0;
#ifndef NDEBUG
                checkGPU();
#endif
                dSchurComplementUpdateGPU(
                    streamId,
                    k, k_lpanel, k_upanel);
// cudaStreamSynchronize(cuStream); // there is sync inside the kernel
#ifndef NDEBUG
                dSchurComplementUpdate(k, k_lpanel, k_upanel);
                cudaStreamSynchronize(cuStream);
                checkGPU();
#endif
            }
            // MPI_Barrier(grid3d->comm);

        } /*for k0= k_st:k_end */

    } /*for topoLvl = 0:maxTopoLevel*/

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Exit dsparseTreeFactorGPUBaseline()");
#endif

    return 0;
} /* dsparseTreeFactorGPUBaseline */

#endif  /* match #if HAVE_CUDA  */

