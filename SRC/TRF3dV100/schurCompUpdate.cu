#include "superlu_ddefs.h"
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
    
    if (gj > gi) // its in upanel
    {
        int li = dA->g2lRow(gi);
        int lj = dA->uPanelVec[li].find(gj);
        Dst = dA->uPanelVec[li].blkPtr(lj);
        lddst = dA->supersize(gi);
        dstRowLen = dA->supersize(gi);
        dstRowList = NULL;
        dstColLen = dA->uPanelVec[li].nbcol(lj);
        dstColList = dA->uPanelVec[li].colList(lj);
        
    }
    else
    {
        int lj = dA->g2lCol(gj);
        int li = dA->lPanelVec[lj].find(gi);
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


