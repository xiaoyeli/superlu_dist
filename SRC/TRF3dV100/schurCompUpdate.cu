#include "superlu_ddefs.h"
#ifdef GPU_ACC
#include "lupanels_GPU.cuh"
#include "lupanels.hpp"

cudaError_t checkCudaLocal(cudaError_t result)
{
// #if defined(DEBUG) || defined(_DEBUG)
    // printf("Checking cuda\n");
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
// #endif
    return result;
}

#define NDEBUG
__device__
int_t lpanelGPU_t::find(int_t k)
{
    int threadId = threadIdx.x;
    __shared__ int idx; 
    __shared__ int found;
    if(!threadId)
    {
        idx =-1;
        found=0;
    }

    int nThreads = blockDim.x; 
    int blocksPerThreads = CEILING( nblocks(),    nThreads);
    __syncthreads();
    for(int blk =blocksPerThreads*threadIdx.x; 
        blk< blocksPerThreads*(threadIdx.x +1);
        blk++)
    {
        // if(found) break; 

        if(blk< nblocks())
        {
            if(k == gid(blk))
            {
                idx = blk;
                found =1;
            }
        }
        
    }
    __syncthreads();
    return idx;
}


__device__
int_t upanelGPU_t::find(int_t k)
{
    int threadId = threadIdx.x;
    __shared__ int idx; 
    __shared__ int found;
    if(!threadId)
    {
        idx =-1;
        found=0;
    }
    __syncthreads();
    
    int nThreads = blockDim.x; 
    int blocksPerThreads = CEILING( nblocks(),    nThreads);
    
    for(int blk =blocksPerThreads*threadIdx.x; 
        blk< blocksPerThreads*(threadIdx.x +1);
        blk++)
    {
        // if(found) break; 

        if(blk< nblocks())
        {
            if(k == gid(blk))
            {
                idx = blk;
                found =1;
            }
        }
        
        
    }
    __syncthreads();
    return idx;
}


__device__
int computeIndirectMapGPU(int* rcS2D,  int_t srcLen, int_t *srcVec,
                                         int_t dstLen, int_t *dstVec,
                                         int *dstIdx)
{
    int threadId = threadIdx.x;
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        if(threadId < srcLen)
            rcS2D[threadId] = srcVec[threadId];
        __syncthreads();
        return 0; 
    }
    
    
    if(threadId < dstLen)
        dstIdx[dstVec[threadId]] = threadId;
    __syncthreads();
    
    if(threadId < srcLen)
        rcS2D[threadId] = dstIdx[srcVec[threadId]];
    __syncthreads();
    
    return 0;
}


__global__
void scatterGPU(
    int iSt,  int jSt, 
    double* gemmBuff, int LDgemmBuff,
    lpanelGPU_t lpanel, upanelGPU_t upanel, 
    LUstructGPU_t* dA)
{

    // calculate gi,gj
    int ii = iSt + blockIdx.x; 
    int jj = jSt + blockIdx.y; 
    int threadId = threadIdx.x;

    int gi = lpanel.gid(ii);
    int gj = upanel.gid(jj);
    #ifndef NDEBUG
    if(!threadId )
    printf("Scattering to (%d, %d) \n",gi, gj);
    #endif 
    double *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    int li, lj; 
    if (gj > gi) // its in upanel
    {
        li = dA->g2lRow(gi);
        lj = dA->uPanelVec[li].find(gj);
        Dst = dA->uPanelVec[li].blkPtr(lj);
        lddst = dA->supersize(gi);
        dstRowLen = dA->supersize(gi);
        dstRowList = NULL;
        dstColLen = dA->uPanelVec[li].nbcol(lj);
        dstColList = dA->uPanelVec[li].colList(lj);
        
    }
    else
    {
        lj = dA->g2lCol(gj);
        li = dA->lPanelVec[lj].find(gi);
        Dst = dA->lPanelVec[lj].blkPtr(li);
        lddst = dA->lPanelVec[lj].LDA();
        dstRowLen = dA->lPanelVec[lj].nbrow(li);
        dstRowList = dA->lPanelVec[lj].rowList(li);
        // if(!threadId )
        // printf("Scattering to (%d, %d) by %d li=%d\n",gi, gj,threadId,li);
        dstColLen = dA->supersize(gj);
        dstColList = NULL;
        
    }

    
    // compute source row to dest row mapping
    int maxSuperSize = dA->maxSuperSize; 
    extern __shared__ int baseSharedPtr[]; 
    int* rowS2D = baseSharedPtr;
    int* colS2D = &rowS2D[maxSuperSize];
    int* dstIdx = &colS2D[maxSuperSize];

    int nrows = lpanel.nbrow(ii);
    int ncols = upanel.nbcol(jj);
    // lpanel.rowList(ii), upanel.colList(jj)
    
    computeIndirectMapGPU(rowS2D,  nrows, lpanel.rowList(ii),
        dstRowLen, dstRowList, dstIdx);
    
// compute source col to dest col mapping
    computeIndirectMapGPU(colS2D, ncols, upanel.colList(jj),
        dstColLen, dstColList, dstIdx);
    

    int nThreads = blockDim.x; 
    int colsPerThreadBlock = nThreads/ nrows;
    
    
    

    int rowOff = lpanel.stRow(ii) - lpanel.stRow(iSt);
    int colOff = upanel.stCol(jj) - upanel.stCol(jSt);
    double* Src = &gemmBuff[ rowOff+ colOff* LDgemmBuff];
    int ldsrc = LDgemmBuff; 
    // TODO: this seems inefficient 
    if (threadId < nrows * colsPerThreadBlock)
	{
		/* 1D threads are logically arranged in 2D shape. */
		int i = threadId % nrows;
		int j = threadId / nrows;

        #pragma unroll 4
        while(j<ncols)
        {   

            #define ATOMIC_SCATTER
            //Atomic Scatter is need if I want to perform multiple Schur Complement
            // update concurrently  
            #ifdef  ATOMIC_SCATTER
            atomicAdd(&Dst[rowS2D[i] + lddst * colS2D[j]], -Src[i + ldsrc * j]);
            #else 
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
            #endif 
            j += colsPerThreadBlock;
        }
		
	}

    __syncthreads();
} 


int_t LUstruct_v100::dSchurComplementUpdateGPU(
    int streamId, 
    int_t k, lpanel_t &lpanel, upanel_t &upanel)
{
    
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int iSt =st_lb;
    int iEnd =iSt; 
    
    

    int nrows = lpanel.stRow(nlb) - lpanel.stRow(st_lb);
    int ncols = upanel.nzcols(); 

    int maxGemmRows = nrows;  
    int maxGemmCols = ncols;
    // entire gemm doesn't fit in gemm buffer
    if(nrows* ncols > A_gpu.gemmBufferSize)
    {
        int maxGemmOpSize = (int) sqrt(A_gpu.gemmBufferSize);
        int numberofRowChunks = (nrows +maxGemmOpSize-1)/maxGemmOpSize;
        maxGemmRows =   nrows /numberofRowChunks;
        maxGemmCols = A_gpu.gemmBufferSize/ maxGemmRows; 
    }
    
    while(iEnd< nlb)
    {
        iSt = iEnd;
        iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
        
        assert(iEnd>iSt);
        int jSt =0;
        int jEnd =0; 
        while(jEnd< nub)
        {
            jSt = jEnd; 
            jEnd = upanel.getEndBlock(jSt, maxGemmCols);
            assert(jEnd>jSt);
            cublasHandle_t handle = A_gpu.cuHandles[streamId];
            cudaStream_t cuStream = A_gpu.cuStreams[streamId];
            cublasSetStream(handle, cuStream);
            int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
            int gemm_n = upanel.stCol(jEnd) - upanel.stCol(jSt);
            int gemm_k = supersize(k);
            double alpha = 1.0;
            double beta = 0.0; 
#ifndef NDEBUG
            printf("m=%d, n=%d, k=%d\n", gemm_m,gemm_n,gemm_k);
#endif
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        gemm_m, gemm_n, gemm_k, &alpha,
                        lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                        upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
                        A_gpu.gpuGemmBuffs[streamId], gemm_m);


            // setting up scatter 
            dim3 dimBlock(ldt); // 1d thread
            dim3 dimGrid(iEnd - iSt, jEnd - jSt);
            size_t sharedMemorySize=3* A_gpu.maxSuperSize * sizeof(int_t); 

            scatterGPU<<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
                iSt, jSt, 
                A_gpu.gpuGemmBuffs[streamId], gemm_m,
                lpanel.gpuPanel, upanel.gpuPanel, dA_gpu);   
            
            
		}
    }
    checkCudaLocal(cudaStreamSynchronize(A_gpu.cuStreams[streamId]));
    return 0;
}




int_t LUstruct_v100::lookAheadUpdateGPU(
    int streamId, 
    int_t k, int_t laIdx, lpanel_t &lpanel, upanel_t &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();
    
    int_t laILoc = lpanel.find(laIdx);
    int_t laJLoc = upanel.find(laIdx);

    int iSt =st_lb;
    int jSt =0;

    /* call look ahead update on Lpanel*/
    if(laJLoc != GLOBAL_BLOCK_NOT_FOUND)
    dSchurCompUpdatePartGPU(
        iSt, nlb, laJLoc, laJLoc+1, 
        k, lpanel, upanel,
        A_gpu.lookAheadLHandle[streamId], A_gpu.lookAheadLStream[streamId],
        A_gpu.lookAheadLGemmBuffer[streamId]);

    /* call look ahead update on Upanel*/
    if(laILoc != GLOBAL_BLOCK_NOT_FOUND)
    {
        dSchurCompUpdatePartGPU(
            laILoc, laILoc+1,  jSt, laJLoc,  
            k, lpanel, upanel,
            A_gpu.lookAheadUHandle[streamId], A_gpu.lookAheadUStream[streamId],
            A_gpu.lookAheadUGemmBuffer[streamId]);
        dSchurCompUpdatePartGPU(
            laILoc, laILoc+1,  laJLoc+1,nub,   
            k, lpanel, upanel,
            A_gpu.lookAheadUHandle[streamId], A_gpu.lookAheadUStream[streamId],
            A_gpu.lookAheadUGemmBuffer[streamId]);
    }
    
    
    // checkCudaLocal(cudaStreamSynchronize(A_gpu.lookAheadLStream[streamId]));
    // checkCudaLocal(cudaStreamSynchronize(A_gpu.lookAheadUStream[streamId]));
    
    return 0;
}

int_t LUstruct_v100::SyncLookAheadUpdate(int streamId)
{
    checkCudaLocal(cudaStreamSynchronize(A_gpu.lookAheadLStream[streamId]));
    checkCudaLocal(cudaStreamSynchronize(A_gpu.lookAheadUStream[streamId]));
    
    return 0;
}

int_t LUstruct_v100::dSchurCompUpdateExcludeOneGPU(
    int streamId, 
    int_t k, int_t ex,  // suypernodes to be excluded 
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

    dSchurCompUpLimitedMem(
        streamId, 
        st_lb, exILoc, 0, exJLoc,
        k, lpanel, upanel);
    
    dSchurCompUpLimitedMem(
        streamId, 
        st_lb, exILoc, exJLoc+1, nub,
        k, lpanel, upanel);

    int_t nextStI = exILoc +1;
    if(exILoc == GLOBAL_BLOCK_NOT_FOUND)
        nextStI = st_lb; 
    /*
    for j we don't need to change since, if exJLoc == GLOBAL_BLOCK_NOT_FOUND =-1
    then exJLoc+1 =0 will work out correctly as starting j
    */
    dSchurCompUpLimitedMem(
        streamId, 
        nextStI, nlb, 0, exJLoc,
        k, lpanel, upanel);
    
    dSchurCompUpLimitedMem(
        streamId, 
        nextStI, nlb, exJLoc+1, nub,
        k, lpanel, upanel);

    // checkCudaLocal(cudaStreamSynchronize(A_gpu.cuStreams[streamId]));
    return 0;

}
        
int_t LUstruct_v100::dSchurCompUpdatePartGPU(
    int_t  iSt, int_t iEnd, int_t jSt, int_t  jEnd, 
    int_t k, lpanel_t &lpanel, upanel_t &upanel,
    cublasHandle_t handle, cudaStream_t cuStream,
    double* gemmBuff )
{
    if(iSt >= iEnd || jSt >= jEnd )
        return 0;

    cublasSetStream(handle, cuStream);
    int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    int gemm_n = upanel.stCol(jEnd) - upanel.stCol(jSt);
    int gemm_k = supersize(k);
    double alpha = 1.0;
    double beta = 0.0; 
#ifndef NDEBUG
    printf("m=%d, n=%d, k=%d\n", gemm_m,gemm_n,gemm_k);
#endif
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                gemm_m, gemm_n, gemm_k, &alpha,
                lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
                gemmBuff, gemm_m);


    // setting up scatter 
    dim3 dimBlock(ldt); // 1d thread
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize=3* A_gpu.maxSuperSize * sizeof(int_t); 

    scatterGPU<<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
        iSt, jSt, 
        gemmBuff, gemm_m,
        lpanel.gpuPanel, upanel.gpuPanel, dA_gpu);   

    return 0; 
}



int_t LUstruct_v100::dSchurCompUpLimitedMem(
    int streamId, 
    int_t lStart, int_t lEnd,
    int_t uStart, int_t uEnd,
    int_t k, lpanel_t &lpanel, upanel_t &upanel)
{

    if(lStart >= lEnd || uStart >= uEnd )
        return 0;
    int iSt =lStart;
    int iEnd =iSt; 
    int nrows = lpanel.stRow(lEnd) - lpanel.stRow(lStart);
    int ncols = upanel.stCol(uEnd) - upanel.stCol(uStart);

    int maxGemmRows = nrows;  
    int maxGemmCols = ncols;
    // entire gemm doesn't fit in gemm buffer
    if(nrows* ncols > A_gpu.gemmBufferSize)
    {
        int maxGemmOpSize = (int) sqrt(A_gpu.gemmBufferSize);
        int numberofRowChunks = (nrows +maxGemmOpSize-1)/maxGemmOpSize;
        maxGemmRows =   nrows /numberofRowChunks;
        maxGemmCols = A_gpu.gemmBufferSize/ maxGemmRows; 
    }
    
    while(iEnd< lEnd)
    {
        iSt = iEnd;
        iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
        if(iEnd>lEnd)
            iEnd=lEnd;
        
        assert(iEnd>iSt);
        int jSt =uStart;
        int jEnd =uStart; 
        while(jEnd< uEnd)
        {
            jSt = jEnd; 
            jEnd = upanel.getEndBlock(jSt, maxGemmCols);
            if(jEnd>uEnd)
                jEnd=uEnd;
            
            cublasHandle_t handle = A_gpu.cuHandles[streamId];
            cudaStream_t cuStream = A_gpu.cuStreams[streamId];
            dSchurCompUpdatePartGPU(iSt,  iEnd,  jSt,   jEnd, 
                 k, lpanel,  upanel, handle, cuStream, A_gpu.gpuGemmBuffs[streamId]);
            
            
		}
    }
    
    return 0;
}


int getMPIProcsPerGPU()
{
    if (!(getenv("MPI_PROCESS_PER_GPU")))
    {
        return 1;
    }
    else
    {
        int devCount;
        cudaGetDeviceCount(&devCount);
        int envCount = atoi(getenv("MPI_PROCESS_PER_GPU"));
        envCount = SUPERLU_MAX(envCount, 0);
        return SUPERLU_MIN(envCount, devCount);
    }
}

#define USABLE_GPU_MEM_FRACTION 0.9

size_t getGPUMemPerProcs(MPI_Comm baseCommunicator)
{

    size_t mfree, mtotal;
    //TODO: shared memory communicator should be part of
    // LU struct 
    // MPI_Comm sharedComm;
    // MPI_Comm_split_type(baseCommunicator, MPI_COMM_TYPE_SHARED,
    //                     0, MPI_INFO_NULL, &sharedComm);
    // MPI_Barrier(sharedComm);
    cudaMemGetInfo(&mfree, &mtotal);
    // MPI_Barrier(sharedComm);
    // MPI_Comm_free(&sharedComm);
#if 0
    printf("Total memory %zu & free memory %zu\n", mtotal, mfree);
#endif
    return (size_t)(USABLE_GPU_MEM_FRACTION * (double)mfree) / getMPIProcsPerGPU();
}

int_t LUstruct_v100::setLUstruct_GPU()
{

    A_gpu.Pr = Pr;
    A_gpu.Pc = Pc;
    A_gpu.maxSuperSize = ldt;

    /*Mapping to device*/
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // How many GPUs?
    int device_id = grid3d->iam % deviceCount;
    cudaSetDevice(device_id);

    size_t useableGPUMem = getGPUMemPerProcs(grid3d->comm);
    /**
     *  Memory is divided into two parts data memory and buffer memory 
     *  data memory is used for useful data
     *  bufferMemory is used for buffers
     * */
    size_t memReqData = 0;

    /*Memory for XSUP*/
    memReqData += (nsupers + 1) * sizeof(int_t);

    size_t totalNzvalSize = 0;
    /*Memory for lapenlPanel Data*/
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            memReqData += lPanelVec[i].totalSize();
            totalNzvalSize += lPanelVec[i].nzvalSize();
        }
    }
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            memReqData += uPanelVec[i].totalSize();
            totalNzvalSize += uPanelVec[i].nzvalSize();
        }
    }
    memReqData += CEILING(nsupers, Pc) * sizeof(lpanelGPU_t);
    memReqData += CEILING(nsupers, Pr) * sizeof(upanelGPU_t);

    memReqData += sizeof(LUstructGPU_t);
    // Per stream data
    // TODO: estimate based on ancestor size
    A_gpu.gemmBufferSize = SUPERLU_MIN(get_max_buffer_size(), totalNzvalSize);
    size_t dataPerStream = 3 * sizeof(double) * maxLvalCount 
                + 3 * sizeof(double) * maxUvalCount 
                + 2 * sizeof(int_t) * maxLidxCount 
                + 2 * sizeof(int_t) * maxUidxCount 
                + A_gpu.gemmBufferSize * sizeof(double) + ldt * ldt * sizeof(double);
    if (memReqData + 2 * dataPerStream > useableGPUMem)
    {
        printf("Not enough memory on GPU: available = %zu, required for 2 streams =%zu, exiting\n", useableGPUMem, memReqData + 2 * dataPerStream);
        exit(-1);
    }


    int_t maxNumberOfStream = (useableGPUMem - memReqData) / dataPerStream;

    int numberOfStreams = SUPERLU_MIN(getNumLookAhead(options), maxNumberOfStream);
    numberOfStreams = SUPERLU_MIN(numberOfStreams, MAX_CUDA_STREAMS);
    int rNumberOfStreams;
    MPI_Allreduce(&numberOfStreams, &rNumberOfStreams, 1,
                  MPI_INT, MPI_MIN, grid3d->comm);
    A_gpu.numCudaStreams = rNumberOfStreams;

    if (!grid3d->iam)
        printf("Using %d CUDA LookAhead streams\n", rNumberOfStreams);
    size_t totalMemoryRequired = memReqData + numberOfStreams * dataPerStream;


    
#if 0
    // upanelGPU_t *uPanelVec_GPU = new upanelGPU_t[CEILING(nsupers, Pr)];
    // lpanelGPU_t *lPanelVec_GPU = new lpanelGPU_t[CEILING(nsupers, Pc)];
    // void *gpuBasePtr, *gpuCurrentPtr;
    // cudaMalloc(&gpuBasePtr, totalMemoryRequired);
    // gpuCurrentPtr = gpuBasePtr;

    // A_gpu.xsup = (int_t *)gpuCurrentPtr;
    // gpuCurrentPtr = (int_t *)gpuCurrentPtr + (nsupers + 1);
    // cudaMemcpy(A_gpu.xsup, xsup, (nsupers + 1) * sizeof(int_t), cudaMemcpyHostToDevice);

    // for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    // {
    //     if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
    //     {
    //         lPanelVec_GPU[i] = lPanelVec[i].copyToGPU(gpuCurrentPtr);
    //         gpuCurrentPtr = (char *)gpuCurrentPtr + lPanelVec[i].totalSize();
    //     }
    // }
    // A_gpu.lPanelVec = (lpanelGPU_t *)gpuCurrentPtr;
    // gpuCurrentPtr = (char *)gpuCurrentPtr + CEILING(nsupers, Pc) * sizeof(lpanelGPU_t);
    // cudaMemcpy(A_gpu.lPanelVec, lPanelVec_GPU,
    //            CEILING(nsupers, Pc) * sizeof(lpanelGPU_t), cudaMemcpyHostToDevice);

    // for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    // {
    //     if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
    //     {
    //         uPanelVec_GPU[i] = uPanelVec[i].copyToGPU(gpuCurrentPtr);
    //         gpuCurrentPtr = (char *)gpuCurrentPtr + uPanelVec[i].totalSize();
    //     }
    // }
    // A_gpu.uPanelVec = (upanelGPU_t *)gpuCurrentPtr;
    // gpuCurrentPtr = (char *)gpuCurrentPtr + CEILING(nsupers, Pr) * sizeof(upanelGPU_t);
    // cudaMemcpy(A_gpu.uPanelVec, uPanelVec_GPU,
    //            CEILING(nsupers, Pr) * sizeof(upanelGPU_t), cudaMemcpyHostToDevice);

    // for (int stream = 0; stream < A_gpu.numCudaStreams; stream++)
    // {

    //     cudaStreamCreate(&A_gpu.cuStreams[stream]);
    //     cublasCreate(&A_gpu.cuHandles[stream]);
    //     A_gpu.LvalRecvBufs[stream] = (double *)gpuCurrentPtr;
    //     gpuCurrentPtr = (double *)gpuCurrentPtr + maxLvalCount;
    //     A_gpu.UvalRecvBufs[stream] = (double *)gpuCurrentPtr;
    //     gpuCurrentPtr = (double *)gpuCurrentPtr + maxUvalCount;
    //     A_gpu.LidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
    //     gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxLidxCount;
    //     A_gpu.UidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
    //     gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxUidxCount;

    //     A_gpu.gpuGemmBuffs[stream] = (double *)gpuCurrentPtr;
    //     gpuCurrentPtr = (double *)gpuCurrentPtr + A_gpu.gemmBufferSize;
    //     A_gpu.dFBufs[stream] = (double *)gpuCurrentPtr;
    //     gpuCurrentPtr = (double *)gpuCurrentPtr + ldt * ldt;

    //     /*lookAhead buffers and stream*/
    //     cublasCreate(&A_gpu.lookAheadLHandle[stream]);
    //     cudaStreamCreate(&A_gpu.lookAheadLStream[stream]);
    //     A_gpu.lookAheadLGemmBuffer[stream] = (double *)gpuCurrentPtr;
    //     gpuCurrentPtr = (double *)gpuCurrentPtr + maxLvalCount;
    //     cublasCreate(&A_gpu.lookAheadUHandle[stream]);
    //     cudaStreamCreate(&A_gpu.lookAheadUStream[stream]);
    //     A_gpu.lookAheadUGemmBuffer[stream] = (double *)gpuCurrentPtr;
    //     gpuCurrentPtr = (double *)gpuCurrentPtr + maxUvalCount;
    // }
    // // cudaCheckError();
    // // allocate
    // dA_gpu = (LUstructGPU_t *)gpuCurrentPtr;

    // cudaMemcpy(dA_gpu, &A_gpu, sizeof(LUstructGPU_t), cudaMemcpyHostToDevice);
    // gpuCurrentPtr = (LUstructGPU_t *)gpuCurrentPtr + 1;

#else
    cudaMalloc(&A_gpu.xsup, (nsupers + 1) * sizeof(int_t));
    cudaMemcpy(A_gpu.xsup, xsup, (nsupers + 1) * sizeof(int_t), cudaMemcpyHostToDevice);

    double tLsend, tUsend; 
    #if 0
    tLsend = SuperLU_timer_();
    upanelGPU_t *uPanelVec_GPU = copyUpanelsToGPU();
    tLsend = SuperLU_timer_() - tLsend;
    tUsend = SuperLU_timer_();
    lpanelGPU_t *lPanelVec_GPU = copyLpanelsToGPU();
    tUsend = SuperLU_timer_() - tUsend;
    #else 

    upanelGPU_t *uPanelVec_GPU = new upanelGPU_t[CEILING(nsupers, Pr)];
    lpanelGPU_t *lPanelVec_GPU = new lpanelGPU_t[CEILING(nsupers, Pc)];
    tLsend = SuperLU_timer_();
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
            lPanelVec_GPU[i] = lPanelVec[i].copyToGPU();
    }
    tLsend = SuperLU_timer_() - tLsend;
    tUsend = SuperLU_timer_();
    // cudaCheckError();
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            uPanelVec_GPU[i] = uPanelVec[i].copyToGPU();
    }
    tUsend = SuperLU_timer_() - tUsend;
    #endif 

    printf("Time to send Lpanel=%g  and U panels =%g \n", tLsend,tUsend);
    
    cudaMalloc(&A_gpu.lPanelVec, CEILING(nsupers, Pc) * sizeof(lpanelGPU_t));
    cudaMemcpy(A_gpu.lPanelVec, lPanelVec_GPU,
               CEILING(nsupers, Pc) * sizeof(lpanelGPU_t), cudaMemcpyHostToDevice);
    cudaMalloc(&A_gpu.uPanelVec, CEILING(nsupers, Pr) * sizeof(upanelGPU_t));
    cudaMemcpy(A_gpu.uPanelVec, uPanelVec_GPU,
               CEILING(nsupers, Pr) * sizeof(upanelGPU_t), cudaMemcpyHostToDevice);
    // cudaCheckError();

    // assert(A_gpu.numCudaStreams < options->num_lookaheads);

    // cudaMalloc(&A_gpu.LvalRecvBufs, sizeof(double*)*A_gpu.numCudaStreams);
    for (int stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {

        cudaStreamCreate(&A_gpu.cuStreams[stream]);
        cublasCreate(&A_gpu.cuHandles[stream]);
        cudaMalloc(&A_gpu.LvalRecvBufs[stream], sizeof(double) * maxLvalCount);
        cudaMalloc(&A_gpu.UvalRecvBufs[stream], sizeof(double) * maxUvalCount);
        cudaMalloc(&A_gpu.LidxRecvBufs[stream], sizeof(int_t) * maxLidxCount);
        cudaMalloc(&A_gpu.UidxRecvBufs[stream], sizeof(int_t) * maxUidxCount);

        cudaMalloc(&A_gpu.gpuGemmBuffs[stream], A_gpu.gemmBufferSize * sizeof(double));
        cudaMalloc(&A_gpu.dFBufs[stream], ldt * ldt * sizeof(double));

        /*lookAhead buffers and stream*/
        cublasCreate(&A_gpu.lookAheadLHandle[stream]);
        cudaStreamCreate(&A_gpu.lookAheadLStream[stream]);
        cudaMalloc(&A_gpu.lookAheadLGemmBuffer[stream], sizeof(double) * maxLvalCount);
        cublasCreate(&A_gpu.lookAheadUHandle[stream]);
        cudaStreamCreate(&A_gpu.lookAheadUStream[stream]);
        cudaMalloc(&A_gpu.lookAheadUGemmBuffer[stream], sizeof(double) * maxUvalCount);
    }

    // allocate
    cudaMalloc(&dA_gpu, sizeof(LUstructGPU_t));
    cudaMemcpy(dA_gpu, &A_gpu, sizeof(LUstructGPU_t), cudaMemcpyHostToDevice);

#endif
    // cudaCheckError();
    return 0; 
}

int_t LUstruct_v100::copyLUGPUtoHost()
{

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
            lPanelVec[i].copyFromGPU();

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            uPanelVec[i].copyFromGPU();
    return 0;
}

int_t LUstruct_v100::checkGPU()
{

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
        lPanelVec[i].checkGPU();

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        uPanelVec[i].checkGPU();

    std::cout << "Checking LU struct completed succesfully"
              << "\n";
    return 0;
}


lpanelGPU_t* LUstruct_v100::copyLpanelsToGPU()
{
    lpanelGPU_t *lPanelVec_GPU = new lpanelGPU_t[CEILING(nsupers, Pc)];

    // TODO: set gpuLvalSize, gpuLidxSize
    gpuLvalSize =0;
    gpuLidxSize =0; 
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            gpuLvalSize +=sizeof(double)*lPanelVec[i].nzvalSize();
            gpuLidxSize +=sizeof(int_t)*lPanelVec[i].indexSize();
        }
    }


    double* valBuffer = (double*) SUPERLU_MALLOC(gpuLvalSize);
    int_t* idxBuffer  = (int_t*) SUPERLU_MALLOC(gpuLidxSize);

    // allocate memory buffer on GPU 
    cudaMalloc(&gpuLvalBasePtr, gpuLvalSize);
    cudaMalloc(&gpuLidxBasePtr, gpuLidxSize);

    size_t valOffset=0;
    size_t idxOffset=0;

    // do a memcpy to CPU buffer
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            if(lPanelVec[i].isEmpty())
            {
                lpanelGPU_t ithLpanel(NULL,NULL);
                lPanelVec[i].gpuPanel = ithLpanel; 
                lPanelVec_GPU[i] = ithLpanel;     
            }
            else
            {
                lpanelGPU_t ithLpanel(&gpuLidxBasePtr[idxOffset], &gpuLvalBasePtr[valOffset]);
                lPanelVec[i].gpuPanel = ithLpanel; 
                lPanelVec_GPU[i] = ithLpanel; 
                memcpy(&valBuffer[valOffset], lPanelVec[i].val, sizeof(double)*lPanelVec[i].nzvalSize() );
                memcpy(&idxBuffer[idxOffset], lPanelVec[i].index, sizeof(int_t)*lPanelVec[i].indexSize() );

                valOffset+= lPanelVec[i].nzvalSize();
                idxOffset+= lPanelVec[i].indexSize();

            }
        }
    }

    // do a cudaMemcpy to GPU
    cudaMemcpy(gpuLvalBasePtr,valBuffer, gpuLvalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuLidxBasePtr,idxBuffer, gpuLidxSize, cudaMemcpyHostToDevice);

    SUPERLU_FREE(valBuffer);
    SUPERLU_FREE(idxBuffer);
    return lPanelVec_GPU;
}


upanelGPU_t* LUstruct_v100::copyUpanelsToGPU()
{
    upanelGPU_t *uPanelVec_GPU = new upanelGPU_t[CEILING(nsupers, Pr)];

    gpuUvalSize =0;
    gpuUidxSize =0; 
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            gpuUvalSize +=sizeof(double)*uPanelVec[i].nzvalSize();
            gpuUidxSize +=sizeof(int_t)*uPanelVec[i].indexSize();
        }
    }

    // TODO: set gpuUvalSize, gpuUidxSize
    double* valBuffer = (double*) SUPERLU_MALLOC(gpuUvalSize);
    int_t* idxBuffer  = (int_t*) SUPERLU_MALLOC(gpuUidxSize);

    // allocate memory buffer on GPU 
    cudaMalloc(&gpuUvalBasePtr, gpuUvalSize);
    cudaMalloc(&gpuUidxBasePtr, gpuUidxSize);

    size_t valOffset=0;
    size_t idxOffset=0;

    // do a memcpy to CPU buffer
   for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            if(uPanelVec[i].isEmpty())
            {
                upanelGPU_t ithupanel(NULL,NULL);
                uPanelVec[i].gpuPanel = ithupanel; 
                uPanelVec_GPU[i] = ithupanel;     
            }
            else
            {
                upanelGPU_t ithupanel(&gpuUidxBasePtr[idxOffset], &gpuUvalBasePtr[valOffset]);
                uPanelVec[i].gpuPanel = ithupanel; 
                uPanelVec_GPU[i] = ithupanel; 
                memcpy(&valBuffer[valOffset], uPanelVec[i].val, sizeof(double)*uPanelVec[i].nzvalSize() );
                memcpy(&idxBuffer[idxOffset], uPanelVec[i].index, sizeof(int_t)*uPanelVec[i].indexSize() );

                valOffset+= uPanelVec[i].nzvalSize();
                idxOffset+= uPanelVec[i].indexSize();

            }
        }
    }

    // do a cudaMemcpy to GPU
    double tLsend = SuperLU_timer_();
    cudaMemcpy(gpuUvalBasePtr,valBuffer, gpuUvalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuUidxBasePtr,idxBuffer, gpuUidxSize, cudaMemcpyHostToDevice);
    tLsend = SuperLU_timer_() - tLsend;
    printf("cudaMemcpy time U =%g \n", tLsend);
    SUPERLU_FREE(valBuffer);
    SUPERLU_FREE(idxBuffer);
    return uPanelVec_GPU;
}
#endif