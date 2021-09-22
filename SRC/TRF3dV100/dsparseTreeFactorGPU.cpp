#include <cstdio>
#include "superlu_ddefs.h"
#include "lupanels.hpp"
#include "lupanels_GPU.cuh"

// int_t LUstruct_v100::dDiagFactorPanelSolve(k, offset,dFBufs)
// {

//     return 0;
// }
int_t LUstruct_v100::dsparseTreeFactorGPU(
    sForest_t *sforest,
    commRequests_t **comReqss, // lists of communication requests // size maxEtree level
    dscuBufs_t *scuBufs,        // contains buffers for schur complement update
    packLUInfo_t *packLUInfo,
    msgs_t **msgss,           // size=num Look ahead
    dLUValSubBuf_t **LUvsbs,  // size=num Look ahead
    ddiagFactBufs_t **dFBufs,  // size maxEtree level
    gEtreeInfo_t *gEtreeInfo, // global etree info
    int_t *gIperm_c_supno,
    int tag_ub)
{
    int_t nnodes = sforest->nNodes; // number of nodes in the tree
    if (nnodes < 1)
    {
        return 1;
    }

    printf("Using New code V100 with GPU acceleration\n");
#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Enter dsparseTreeFactor_ASYNC()");
#endif

    int_t *perm_c_supno = sforest->nodeList; // list of nodes in the order of factorization
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t *myIperm = treeTopoInfo->myIperm;
    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;

    /*main loop over all the levels*/
    int_t numLA = getNumLookAhead(options);
    
    // start the pipeline 
    int_t* donePanelBcast = intMalloc_dist(nnodes);
    int_t* donePanelSolve = intMalloc_dist(nnodes);
    int_t* localNumChildrenLeft = intMalloc_dist(nnodes);
    
    //TODO: not needed, remove after testing  
    for(int_t i=0;i<nnodes;i++)
    {
        donePanelBcast[i] =0; 
        donePanelSolve[i]=0;
        localNumChildrenLeft[i]=0;
    }

    for(int_t k0=0;k0<nnodes;k0++)
    {
        int_t k = perm_c_supno[k0];
        int_t k_parent = gEtreeInfo->setree[k];
        int_t ik = myIperm[k_parent];
        if(ik >-1 && ik<nnodes)
        localNumChildrenLeft[ik]++;
    }

    // start the pipeline 
    int_t topoLvl =0;
    int_t k_st = eTreeTopLims[topoLvl];
    int_t k_end = eTreeTopLims[topoLvl + 1];
    


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
            cublasHandle_t cubHandle= A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            /*=======   Diagonal Factorization      ======*/
            if (iam == procIJ(k, k))
            {
                #define NDEBUG
                #ifndef NDEBUG
                lPanelVec[g2lCol(k)].checkGPU();
                lPanelVec[g2lCol(k)].diagFactor(k, dFBufs[offset]->BlockUFactor, ksupc,
                                                thresh, xsup, options, stat, info);
                lPanelVec[g2lCol(k)].packDiagBlock(dFBufs[offset]->BlockLFactor, ksupc);
                #endif 
                lPanelVec[g2lCol(k)].diagFactorPackDiagBlockGPU( k,
                                     dFBufs[offset]->BlockUFactor, ksupc,     // CPU pointers
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
                          MPI_DOUBLE, kcol(k), (grid->rscp).comm);
            if (mycol == kcol(k))
                MPI_Bcast((void *)dFBufs[offset]->BlockUFactor, ksupc * ksupc,
                          MPI_DOUBLE, krow(k), (grid->cscp).comm);

            /*=======   Panel Update                ======*/
            if (myrow == krow(k))
            {
                #ifndef NDEBUG
                uPanelVec[g2lRow(k)].checkGPU();
                #endif 
                cudaMemcpy(A_gpu.dFBufs[0],dFBufs[offset]->BlockLFactor, 
                ksupc*ksupc*sizeof(double), cudaMemcpyHostToDevice);
                uPanelVec[g2lRow(k)].panelSolveGPU(
                    cubHandle, cuStream, 
                    ksupc, A_gpu.dFBufs[0], ksupc);
                cudaStreamSynchronize(cuStream);    // synchronize befpre broadcast 
                #ifndef NDEBUG
                uPanelVec[g2lRow(k)].panelSolve(ksupc, dFBufs[offset]->BlockLFactor, ksupc);
                cudaStreamSynchronize (cuStream);
                uPanelVec[g2lRow(k)].checkGPU();
                #endif 
            }
                

            if (mycol == kcol(k))
            {
                cudaMemcpy(A_gpu.dFBufs[0],dFBufs[offset]->BlockUFactor, 
                    ksupc*ksupc*sizeof(double), cudaMemcpyHostToDevice);
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
                A_gpu.UidxRecvBufs[0], A_gpu.UvalRecvBufs[0]) ;
            lpanel_t k_lpanel(LidxRecvBufs[0], LvalRecvBufs[0],
                A_gpu.LidxRecvBufs[0], A_gpu.LvalRecvBufs[0]);
            if (myrow == krow(k))
            {
                k_upanel= uPanelVec[g2lRow(k)];
            }
            if (mycol == kcol(k))
                k_lpanel = lPanelVec[g2lCol(k)];

            if(UidxSendCounts[k]>0)
            {
                // assuming GPU direct is available 
                MPI_Bcast(k_upanel.gpuPanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
                MPI_Bcast(k_upanel.gpuPanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
                // copy the index to cpu 
                cudaMemcpy(k_upanel.index, k_upanel.gpuPanel.index, 
                    sizeof(int_t)*UidxSendCounts[k], cudaMemcpyDeviceToHost);
                
                #ifndef NDEBUG
                MPI_Bcast(k_upanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
                MPI_Bcast(k_upanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
                #endif 
            }

            if(LidxSendCounts[k]>0)
            {
                MPI_Bcast(k_lpanel.gpuPanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
                MPI_Bcast(k_lpanel.gpuPanel.val, LvalSendCounts[k], MPI_DOUBLE, kcol(k), grid3d->rscp.comm);
                cudaMemcpy(k_lpanel.index, k_lpanel.gpuPanel.index, 
                    sizeof(int_t)*LidxSendCounts[k], cudaMemcpyDeviceToHost);
                
                #ifndef NDEBUG
                MPI_Bcast(k_lpanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
                MPI_Bcast(k_lpanel.val, LvalSendCounts[k], MPI_DOUBLE, kcol(k), grid3d->rscp.comm);
                #endif 
            }
            

            /*=======   Schurcomplement Update      ======*/
            #warning single node only 
            // dSchurComplementUpdate(k, lPanelVec[g2lCol(k)], uPanelVec[g2lRow(k)]);
            // dSchurComplementUpdate(k, lPanelVec[g2lCol(k)], k_upanel);
            if(UidxSendCounts[k]>0 && LidxSendCounts[k]>0)
            {
                // k_upanel.checkCorrectness();
                int streamId =0; 
                #ifndef NDEBUG
                checkGPU();
                #endif 

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
    CHECK_MALLOC(grid3d->iam, "Exit dsparseTreeFactor_ASYNC()");
#endif

    return 0;
} /* dsparseTreeFactor_ASYNC */


//TODO: needs to be merged as a single factorization function 
int_t LUstruct_v100::dsparseTreeFactorGPUBaseline(
    sForest_t *sforest,
    commRequests_t **comReqss, // lists of communication requests // size maxEtree level
    dscuBufs_t *scuBufs,        // contains buffers for schur complement update
    packLUInfo_t *packLUInfo,
    msgs_t **msgss,           // size=num Look ahead
    dLUValSubBuf_t **LUvsbs,  // size=num Look ahead
    ddiagFactBufs_t **dFBufs,  // size maxEtree level
    gEtreeInfo_t *gEtreeInfo, // global etree info
    int_t *gIperm_c_supno,
    int tag_ub)
{
    int_t nnodes = sforest->nNodes; // number of nodes in the tree
    if (nnodes < 1)
    {
        return 1;
    }

    printf("Using New code V100 with GPU acceleration\n");
#if (DEBUGlevel >= 1)
    CHECK_MALLOC(grid3d->iam, "Enter dsparseTreeFactor_ASYNC()");
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
            cublasHandle_t cubHandle= A_gpu.cuHandles[0];
            cudaStream_t cuStream = A_gpu.cuStreams[0];
            /*=======   Diagonal Factorization      ======*/
            if (iam == procIJ(k, k))
            {
                #define NDEBUG
                #ifndef NDEBUG
                lPanelVec[g2lCol(k)].checkGPU();
                lPanelVec[g2lCol(k)].diagFactor(k, dFBufs[offset]->BlockUFactor, ksupc,
                                                thresh, xsup, options, stat, info);
                lPanelVec[g2lCol(k)].packDiagBlock(dFBufs[offset]->BlockLFactor, ksupc);
                #endif 
                lPanelVec[g2lCol(k)].diagFactorPackDiagBlockGPU( k,
                                     dFBufs[offset]->BlockUFactor, ksupc,     // CPU pointers
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
                          MPI_DOUBLE, kcol(k), (grid->rscp).comm);
            if (mycol == kcol(k))
                MPI_Bcast((void *)dFBufs[offset]->BlockUFactor, ksupc * ksupc,
                          MPI_DOUBLE, krow(k), (grid->cscp).comm);

            /*=======   Panel Update                ======*/
            if (myrow == krow(k))
            {
                #ifndef NDEBUG
                uPanelVec[g2lRow(k)].checkGPU();
                #endif 
                cudaMemcpy(A_gpu.dFBufs[0],dFBufs[offset]->BlockLFactor, 
                ksupc*ksupc*sizeof(double), cudaMemcpyHostToDevice);
                uPanelVec[g2lRow(k)].panelSolveGPU(
                    cubHandle, cuStream, 
                    ksupc, A_gpu.dFBufs[0], ksupc);
                cudaStreamSynchronize(cuStream);    // synchronize befpre broadcast 
                #ifndef NDEBUG
                uPanelVec[g2lRow(k)].panelSolve(ksupc, dFBufs[offset]->BlockLFactor, ksupc);
                cudaStreamSynchronize (cuStream);
                uPanelVec[g2lRow(k)].checkGPU();
                #endif 
            }
                

            if (mycol == kcol(k))
            {
                cudaMemcpy(A_gpu.dFBufs[0],dFBufs[offset]->BlockUFactor, 
                    ksupc*ksupc*sizeof(double), cudaMemcpyHostToDevice);
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
                A_gpu.UidxRecvBufs[0], A_gpu.UvalRecvBufs[0]) ;
            lpanel_t k_lpanel(LidxRecvBufs[0], LvalRecvBufs[0],
                A_gpu.LidxRecvBufs[0], A_gpu.LvalRecvBufs[0]);
            if (myrow == krow(k))
            {
                k_upanel= uPanelVec[g2lRow(k)];
            }
            if (mycol == kcol(k))
                k_lpanel = lPanelVec[g2lCol(k)];

            if(UidxSendCounts[k]>0)
            {
                // assuming GPU direct is available 
                MPI_Bcast(k_upanel.gpuPanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
                MPI_Bcast(k_upanel.gpuPanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
                // copy the index to cpu 
                cudaMemcpy(k_upanel.index, k_upanel.gpuPanel.index, 
                    sizeof(int_t)*UidxSendCounts[k], cudaMemcpyDeviceToHost);
                
                #ifndef NDEBUG
                MPI_Bcast(k_upanel.index, UidxSendCounts[k], mpi_int_t, krow(k), grid3d->cscp.comm);
                MPI_Bcast(k_upanel.val, UvalSendCounts[k], MPI_DOUBLE, krow(k), grid3d->cscp.comm);
                #endif 
            }

            if(LidxSendCounts[k]>0)
            {
                MPI_Bcast(k_lpanel.gpuPanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
                MPI_Bcast(k_lpanel.gpuPanel.val, LvalSendCounts[k], MPI_DOUBLE, kcol(k), grid3d->rscp.comm);
                cudaMemcpy(k_lpanel.index, k_lpanel.gpuPanel.index, 
                    sizeof(int_t)*LidxSendCounts[k], cudaMemcpyDeviceToHost);
                
                #ifndef NDEBUG
                MPI_Bcast(k_lpanel.index, LidxSendCounts[k], mpi_int_t, kcol(k), grid3d->rscp.comm);
                MPI_Bcast(k_lpanel.val, LvalSendCounts[k], MPI_DOUBLE, kcol(k), grid3d->rscp.comm);
                #endif 
            }
            

            /*=======   Schurcomplement Update      ======*/
            #warning single node only 
            // dSchurComplementUpdate(k, lPanelVec[g2lCol(k)], uPanelVec[g2lRow(k)]);
            // dSchurComplementUpdate(k, lPanelVec[g2lCol(k)], k_upanel);
            if(UidxSendCounts[k]>0 && LidxSendCounts[k]>0)
            {
                // k_upanel.checkCorrectness();
                int streamId =0; 
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
    CHECK_MALLOC(grid3d->iam, "Exit dsparseTreeFactor_ASYNC()");
#endif

    return 0;
} /* dsparseTreeFactor_ASYNC */
