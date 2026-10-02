#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#ifdef HAVE_CUDA
#include <thrust/system/cuda/execution_policy.h>
#elif defined(HAVE_HIP)
#include <thrust/system/hip/execution_policy.h>
#endif
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/extrema.h>

#include "batch_factorize.h"
#include <cstdio>
#ifndef gpuMemsetAsync
#ifdef HAVE_CUDA
#define gpuMemsetAsync cudaMemsetAsync
#elif defined(HAVE_HIP)
#define gpuMemsetAsync hipMemsetAsync
#endif
#endif
#include <vector>
#include <iostream>


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Marshalling routines for batched execution 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
inline void marshallBatchedLUData(TBatchFactorizeWorkspace<T>* ws, int_t k_st, int_t k_end)
{
    TBatchLUMarshallData<T>& mdata = ws->marshall_data;
    LocalLU_type<T>& d_localLU = ws->d_localLU;

    mdata.batchsize = k_end - k_st;

    TMarshallLUFunc<T> func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array, 
        d_localLU.Lnzval_bc_ptr, d_localLU.Lrowind_bc_ptr, ws->perm_c_supno, ws->xsup
    );

    thrust::for_each(
        gpu_thrust_par, thrust::counting_iterator<int_t>(0),
        thrust::counting_iterator<int_t>(mdata.batchsize), func
    );
}

template<class T>
inline void marshallBatchedTRSMUData(TBatchFactorizeWorkspace<T>* ws, int_t k_st, int_t k_end)
{
    TBatchLUMarshallData<T>& mdata = ws->marshall_data;
    LocalLU_type<T>& d_localLU = ws->d_localLU;

    mdata.batchsize = k_end - k_st;

    TMarshallTRSMUFunc<T> func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array, 
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, mdata.dev_panel_dim_array, 
        d_localLU.Unzval_br_new_ptr, d_localLU.Ucolind_br_ptr, d_localLU.Lnzval_bc_ptr, 
        d_localLU.Lrowind_bc_ptr, ws->perm_c_supno, ws->xsup
    );

    thrust::for_each(
        gpu_thrust_par, thrust::counting_iterator<int_t>(0),
        thrust::counting_iterator<int_t>(mdata.batchsize), func
    );
}

template<class T>
inline void marshallBatchedTRSMLData(TBatchFactorizeWorkspace<T>* ws, int_t k_st, int_t k_end)
{
    TBatchLUMarshallData<T>& mdata = ws->marshall_data;
    LocalLU_type<T>& d_localLU = ws->d_localLU;
    
    mdata.batchsize = k_end - k_st;

    TMarshallTRSMLFunc<T> func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array, 
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, mdata.dev_panel_dim_array,
        d_localLU.Lnzval_bc_ptr, d_localLU.Lrowind_bc_ptr, ws->perm_c_supno, ws->xsup
    );

    thrust::for_each(
        gpu_thrust_par, thrust::counting_iterator<int_t>(0),
        thrust::counting_iterator<int_t>(mdata.batchsize), func
    );
}

template<class T>
inline void marshallBatchedSCUData(TBatchFactorizeWorkspace<T>* ws, int_t k_st, int_t k_end)
{
    TBatchSCUMarshallData<T>& sc_mdata = ws->sc_marshall_data;
    LocalLU_type<T>& d_localLU = ws->d_localLU;

    sc_mdata.batchsize = k_end - k_st;
    
    thrust::counting_iterator<int_t> start(0), end(sc_mdata.batchsize);
    
    TMarshallSCUFunc<T> func(
        k_st, sc_mdata.dev_A_ptrs, sc_mdata.dev_lda_array, sc_mdata.dev_B_ptrs, sc_mdata.dev_ldb_array, 
        sc_mdata.dev_C_ptrs, sc_mdata.dev_ldc_array, sc_mdata.dev_m_array, sc_mdata.dev_n_array, sc_mdata.dev_k_array,
        sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_jst, sc_mdata.dev_jend, d_localLU.Unzval_br_new_ptr, d_localLU.Ucolind_br_ptr, 
        d_localLU.Lnzval_bc_ptr, d_localLU.Lrowind_bc_ptr, ws->perm_c_supno, ws->xsup, ws->gemm_buff_ptrs
    );

    thrust::for_each(gpu_thrust_par, start, end, func);

    // Set the max dims in the marshalled data 
    sc_mdata.max_m = thrust::reduce(gpu_thrust_par, sc_mdata.dev_m_array, sc_mdata.dev_m_array + sc_mdata.batchsize, 0, thrust::maximum<BatchDim_t>());
    sc_mdata.max_n = thrust::reduce(gpu_thrust_par, sc_mdata.dev_n_array, sc_mdata.dev_n_array + sc_mdata.batchsize, 0, thrust::maximum<BatchDim_t>());
    sc_mdata.max_k = thrust::reduce(gpu_thrust_par, sc_mdata.dev_k_array, sc_mdata.dev_k_array + sc_mdata.batchsize, 0, thrust::maximum<BatchDim_t>());
    sc_mdata.max_ilen = thrust::transform_reduce(gpu_thrust_par, start, end, element_diff<BatchDim_t>(sc_mdata.dev_ist, sc_mdata.dev_iend), 0, thrust::maximum<BatchDim_t>());
    sc_mdata.max_jlen = thrust::transform_reduce(gpu_thrust_par, start, end, element_diff<BatchDim_t>(sc_mdata.dev_jst, sc_mdata.dev_jend), 0, thrust::maximum<BatchDim_t>());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility routines
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct BatchLDataSizeAssign_Func {
    int_t** Lrowind_bc_ptr;
    int64_t *d_lblock_gid_offsets, *d_lblock_start_offsets;

    BatchLDataSizeAssign_Func(int_t** Lrowind_bc_ptr, int64_t* d_lblock_gid_offsets, int64_t* d_lblock_start_offsets)
    {
        this->Lrowind_bc_ptr = Lrowind_bc_ptr;
        this->d_lblock_gid_offsets = d_lblock_gid_offsets;
        this->d_lblock_start_offsets = d_lblock_start_offsets;
    }

    __device__ void operator()(const int_t &i) const
    {   
        if(i == 0)
            d_lblock_gid_offsets[i] = d_lblock_start_offsets[i] = 0;
        else
        {
            int_t *Lrowind_bc = Lrowind_bc_ptr[i - 1];
            d_lblock_gid_offsets[i] = (Lrowind_bc ? Lrowind_bc[0] : 0);
            d_lblock_start_offsets[i] = (Lrowind_bc ? Lrowind_bc[0] + 1 : 0);
        }
    }
};

struct BatchLDataAssign_Func {
    int_t **Lrowind_bc_ptr, **d_lblock_gid_ptrs, **d_lblock_start_ptrs;

    BatchLDataAssign_Func(int_t** Lrowind_bc_ptr, int_t** d_lblock_gid_ptrs, int_t** d_lblock_start_ptrs)
    {
        this->Lrowind_bc_ptr = Lrowind_bc_ptr;
        this->d_lblock_gid_ptrs = d_lblock_gid_ptrs;
        this->d_lblock_start_ptrs = d_lblock_start_ptrs;
    }

    __device__ void operator()(const int_t &i) const
    {   
        int_t *Lrowind_bc = Lrowind_bc_ptr[i];
        if(!Lrowind_bc)
            d_lblock_gid_ptrs[i] = d_lblock_start_ptrs[i] = NULL;
        else
        {   
            int_t *block_gids = d_lblock_gid_ptrs[i], *block_starts = d_lblock_start_ptrs[i];
            int_t nblocks = Lrowind_bc[0], Lptr = BC_HEADER, psum = 0;
            for(int_t b = 0; b < nblocks; b++)
            {
                block_gids[b] = Lrowind_bc[Lptr];
                int_t nrows = Lrowind_bc[Lptr + 1];
                block_starts[b] = psum;
                psum += nrows;
                Lptr += nrows + LB_DESCRIPTOR;
            }
            block_starts[nblocks] = psum;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device functions and kernels 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ inline int_t find_entry_index_flat(int_t *index_list, int_t index, int_t n)
{
    int threadId = threadIdx.x;
    __shared__ int_t idx;
    
    if (!threadId)    
        idx = -1;

    __syncthreads();

    int nThreads = blockDim.x;
    int blocksPerThreads = CEILING(n, nThreads);

    for (int_t blk = blocksPerThreads * threadIdx.x;
         blk < blocksPerThreads * (threadIdx.x + 1);
         blk++)
    {
        if (blk < n)
        {
            if(index == index_list[blk])
                idx = blk;
        }
    }
    __syncthreads();
    return idx;
}

__device__ inline int computeIndirectMapGPU_flat(int_t *rcS2D, int_t srcLen, int_t *srcVec, int_t src_first_index,
                                     int_t dstLen, int_t *dstVec, int_t dst_first_index, int_t *dstIdx)
{
    int threadId = threadIdx.x;
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        if (threadId < srcLen)
            rcS2D[threadId] = srcVec[threadId] - src_first_index;
        __syncthreads();
        return 0;
    }

    if (threadId < dstLen)
        dstIdx[dstVec[threadId] - dst_first_index] = threadId;
    __syncthreads();

    if (threadId < srcLen)
        rcS2D[threadId] = dstIdx[srcVec[threadId] - src_first_index];
    __syncthreads();

    return 0;
}

template<class T>
__global__ void scatterGPU_batch_flat(
    int_t k_st, int_t maxSuperSize, T **gemmBuff_ptrs, BatchDim_t *LDgemmBuff_batch,
    T **Unzval_br_new_ptr, int_t** Ucolind_br_ptr, T** Lnzval_bc_ptr, 
    int_t** Lrowind_bc_ptr, int_t** lblock_gid_ptrs, int_t **lblock_start_ptrs, 
    int_t *dperm_c_supno, int_t *xsup
)
{
    int batch_index = blockIdx.z;
    int_t k = dperm_c_supno[k_st + batch_index];
    
    T* gemmBuff = gemmBuff_ptrs[batch_index];
    int_t *Ucolind_br = Ucolind_br_ptr[k];
    int_t *Lrowind_bc = Lrowind_bc_ptr[k];
    int_t *lblock_gid = lblock_gid_ptrs[k];
    int_t *lblock_start = lblock_start_ptrs[k];

    if(!Ucolind_br || !Lrowind_bc || !gemmBuff || !lblock_gid || !lblock_start)
        return;

    int_t L_blocks = Lrowind_bc[0];    
    int_t U_blocks = Ucolind_br[0];
    BatchDim_t LDgemmBuff = LDgemmBuff_batch[batch_index];

    int_t ii = 1 + blockIdx.x;
    int_t jj = blockIdx.y;

    if(ii >= L_blocks || jj >= U_blocks)
        return;

    // calculate gi, gj
    int threadId = threadIdx.x;

    int_t gi = lblock_gid[ii];
    int_t gj = Ucolind_br[UB_DESCRIPTOR_NEWUCPP + jj];
    
    T *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    int_t dst_row_first_index, dst_col_first_index;
    int_t li = 0, lj = 0;

    if (gj > gi) // its in upanel
    {
        int_t* U_index_i = Ucolind_br_ptr[gi];
        int_t nub = U_index_i[0];
        lddst = U_index_i[2];
        lj = find_entry_index_flat(U_index_i + UB_DESCRIPTOR_NEWUCPP, gj, nub);
        li = gi;
        int_t col_offset = U_index_i[UB_DESCRIPTOR_NEWUCPP + nub + lj];
        Dst = Unzval_br_new_ptr[gi] + lddst * col_offset;
        dstRowLen = lddst;
        dstRowList = NULL;
        dst_row_first_index = 0;
        dstColLen = U_index_i[UB_DESCRIPTOR_NEWUCPP + nub + lj + 1] - col_offset;
        dstColList = U_index_i + UB_DESCRIPTOR_NEWUCPP + 2 * nub + 1 + col_offset;
        dst_col_first_index = xsup[gj];
    }
    else
    {
        int_t* L_index_j = Lrowind_bc_ptr[gj], *ljblock_start = lblock_start_ptrs[gj];
        int_t nlb = L_index_j[0];
        lddst = L_index_j[1];
        li = find_entry_index_flat(lblock_gid_ptrs[gj], gi, nlb);
        lj = gj;
        int_t row_offset = ljblock_start[li];
        Dst = Lnzval_bc_ptr[gj] + row_offset;
        dstRowLen = ljblock_start[li + 1] - row_offset;
        dstRowList = L_index_j + BC_HEADER + (li + 1) * LB_DESCRIPTOR + row_offset;
        dst_row_first_index = xsup[gi];
        dstColLen = SuperSize(gj);
        dstColList = NULL;
        dst_col_first_index = 0;
    }

    // compute source row to dest row mapping
    extern __shared__ int_t baseSharedPtr[];
    int_t *rowS2D = baseSharedPtr;
    int_t *colS2D = &rowS2D[maxSuperSize];
    int_t *dstIdx = &colS2D[maxSuperSize];

    int_t ublock_start = Ucolind_br[UB_DESCRIPTOR_NEWUCPP + U_blocks + jj];
    int_t nrows = lblock_start[ii + 1] - lblock_start[ii];
    int_t ncols = Ucolind_br[UB_DESCRIPTOR_NEWUCPP + U_blocks + jj + 1] - ublock_start;

    int_t *lpanel_row_list = Lrowind_bc + BC_HEADER + (ii + 1) * LB_DESCRIPTOR + lblock_start[ii];
    int_t *upanel_col_list = Ucolind_br + UB_DESCRIPTOR_NEWUCPP + 2 * U_blocks + 1 + ublock_start;
    int_t lpanel_first_index = xsup[gi];
    int_t upanel_first_index = xsup[gj];

    computeIndirectMapGPU_flat(rowS2D, nrows, lpanel_row_list, lpanel_first_index,
                          dstRowLen, dstRowList, dst_row_first_index, dstIdx);

    // compute source col to dest col mapping
    computeIndirectMapGPU_flat(colS2D, ncols, upanel_col_list, upanel_first_index,
                          dstColLen, dstColList, dst_col_first_index, dstIdx);

    int nThreads = blockDim.x;
    int colsPerThreadBlock = nThreads / nrows;

    int_t rowOff = lblock_start[ii] - lblock_start[1];
    int_t colOff = ublock_start;

    T *Src = &gemmBuff[rowOff + colOff * LDgemmBuff];
    int_t ldsrc = LDgemmBuff;

    // TODO: this seems inefficient
    if (threadId < nrows * colsPerThreadBlock)
    {
        /* 1D threads are logically arranged in 2D shape. */
        int i = threadId % nrows;
        int j = threadId / nrows;

#pragma unroll 4
        while (j < ncols)
        {

#define ATOMIC_SCATTER
// Atomic Scatter is need if I want to perform multiple Schur Complement
//  update concurrently
#ifdef ATOMIC_SCATTER
             atomicAddT(&Dst[rowS2D[i] + lddst * colS2D[j]], -Src[i + ldsrc * j]);
#else
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
#endif
            j += colsPerThreadBlock;
        }
    }

    __syncthreads();
}

template<class T>
inline void scatterGPU_batchDriver_flat(
    int_t k_st, int_t maxSuperSize, T **gemmBuff_ptrs, BatchDim_t *LDgemmBuff_batch,
    T **Unzval_br_new_ptr, int_t** Ucolind_br_ptr, T** Lnzval_bc_ptr, 
    int_t** Lrowind_bc_ptr, int_t** lblock_gid_ptrs, int_t **lblock_start_ptrs, 
    int_t *dperm_c_supno, int_t *xsup, int_t ldt, BatchDim_t max_ilen, BatchDim_t max_jlen, 
    BatchDim_t batchCount, gpuStream_t cuStream
)
{
    const BatchDim_t op_increment = 65535;

    /* A tree level can contain only supernodes with no off-diagonal blocks, in
       which case marshallBatchedSCUData's transform_reduce returns 0 for
       max_ilen / max_jlen (its init value).  dim3(0, 0, n) is an illegal launch
       configuration, so cudaLaunchKernel fails with
       cudaErrorInvalidConfiguration; thrust then reports that sticky error on
       its next call as the misleading "parallel_for failed:
       cudaErrorInvalidDevice".  There is nothing to scatter in that case. */
    if (max_ilen == 0 || max_jlen == 0 || batchCount == 0)
        return;

    for(BatchDim_t op_start = 0; op_start < batchCount; op_start += op_increment)
	{
		BatchDim_t batch_size = std::min(op_increment, batchCount - op_start);
    
        dim3 dimBlock(ldt); // 1d thread
        dim3 dimGrid(max_ilen, max_jlen, batch_size);
        size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);

        /* The kernel indexes the marshalled GEMM buffers by blockIdx.z, which
           restarts at 0 in every chunk, so those arrays must be advanced by
           op_start just as k_st is.  Without this, a tree level wider than
           op_increment (65535, the gridDim.z limit) scatters the wrong
           Schur-complement blocks from the second chunk onwards -- silently,
           since the launch itself is valid. */
        scatterGPU_batch_flat<<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            k_st + op_start, maxSuperSize, gemmBuff_ptrs + op_start,
            LDgemmBuff_batch + op_start, Unzval_br_new_ptr,
            Ucolind_br_ptr, Lnzval_bc_ptr, Lrowind_bc_ptr, lblock_gid_ptrs, lblock_start_ptrs, 
            dperm_c_supno, xsup
        );
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void computeLBlockData(TBatchFactorizeWorkspace<T>* ws, int_t nsupers)
{
    LocalLU_type<T>& d_localLU = ws->d_localLU;

    // Allocate memory for the offsets and the pointers 
    gpuErrchk( gpuMalloc(&ws->d_lblock_gid_offsets, sizeof(int64_t) * (nsupers + 1)) );
    gpuErrchk( gpuMalloc(&ws->d_lblock_start_offsets, sizeof(int64_t) * (nsupers + 1)) );
    gpuErrchk( gpuMalloc(&ws->d_lblock_gid_ptrs, sizeof(int_t*) * nsupers) );
    gpuErrchk( gpuMalloc(&ws->d_lblock_start_ptrs, sizeof(int_t*) * nsupers) );

    // Initialize to the block counts for each panel
    thrust::for_each(
        gpu_thrust_par, thrust::counting_iterator<int_t>(0), 
        thrust::counting_iterator<int_t>(nsupers + 1), BatchLDataSizeAssign_Func(
            d_localLU.Lrowind_bc_ptr, ws->d_lblock_gid_offsets, ws->d_lblock_start_offsets
    ) );
    
    // Do an inclusive scan to compute offsets and get the total amount of blocks 
    ws->total_l_blocks = *(thrust::device_ptr<int64_t>(
        thrust::inclusive_scan(
            gpu_thrust_par, ws->d_lblock_gid_offsets + 1, 
            ws->d_lblock_gid_offsets + nsupers + 1, ws->d_lblock_gid_offsets + 1
    ) ) - 1);

    ws->total_start_size = *(thrust::device_ptr<int64_t>(
        thrust::inclusive_scan(
            gpu_thrust_par, ws->d_lblock_start_offsets + 1, 
            ws->d_lblock_start_offsets + nsupers + 1, ws->d_lblock_start_offsets + 1
    ) ) - 1);

    // Allocate the block data 
    gpuErrchk( gpuMalloc(&ws->d_lblock_gid_dat, sizeof(int_t) * ws->total_l_blocks) );
    gpuErrchk( gpuMalloc(&ws->d_lblock_start_dat, sizeof(int_t) * ws->total_start_size) );

    // Generate the pointers 
    generateOffsetPointers(ws->d_lblock_gid_dat, ws->d_lblock_gid_offsets, ws->d_lblock_gid_ptrs, nsupers);
    generateOffsetPointers(ws->d_lblock_start_dat, ws->d_lblock_start_offsets, ws->d_lblock_start_ptrs, nsupers);

    // Now copy the data over from d_localLU
    thrust::for_each(
        gpu_thrust_par, thrust::counting_iterator<int_t>(0), 
        thrust::counting_iterator<int_t>(nsupers), BatchLDataAssign_Func(
        d_localLU.Lrowind_bc_ptr, ws->d_lblock_gid_ptrs, ws->d_lblock_start_ptrs
    ) );
}

template<class T>
void batchAllocateGemmBuffers(
    TBatchFactorizeWorkspace<T>* ws, LUStruct_type<T> *LUstruct, trf3dpartitionType<T> *trf3Dpartition, 
    gridinfo3d_t *grid3d
)
{
    int_t mxLeafNode = trf3Dpartition->mxLeafNode;

    // TODO: is this necessary if this is being done on a single node?
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    std::vector<int64_t> gemmCsizes(mxLeafNode, 0);
	int_t mx_fsize = 0;
	
	for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl) 
    {
	    int_t treeId = trf3Dpartition->myTreeIdxs[ilvl];
	    sForest_t* sforest = trf3Dpartition->sForests[treeId];
	    if (sforest)
        {
            int_t *perm_c_supno = sforest->nodeList;
            mx_fsize = max(mx_fsize, sforest->nNodes);

            int_t maxTopoLevel = sforest->topoInfo.numLvl;
            for (int_t topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl) 
            {
                int_t k_st = sforest->topoInfo.eTreeTopLims[topoLvl];
                int_t k_end = sforest->topoInfo.eTreeTopLims[topoLvl + 1];
            
                for (int_t k0 = k_st; k0 < k_end; ++k0) 
                {
                    int_t offset = k0 - k_st;
                    int_t k = perm_c_supno[k0];
                    int_t* L_data = LUstruct->Llu->Lrowind_bc_ptr[k];
                    int_t* U_data = LUstruct->Llu->Ucolind_br_ptr[k];
                    if(L_data && U_data)
                    {
                        int_t Csize = L_data[1] * U_data[1];
                        gemmCsizes[offset] = SUPERLU_MAX(gemmCsizes[offset], Csize);
                    }
                }
		    }
	    }
	}
    // Allocate the gemm buffers 
    gpuErrchk( gpuMalloc(&(ws->gemm_buff_ptrs), sizeof(T*) * mxLeafNode) );
    gpuErrchk( gpuMalloc(&(ws->gemm_buff_offsets), sizeof(int64_t) * (mxLeafNode + 1)) );

    // Copy the host offset to the device 
    gpuErrchk( gpuMemcpy( ws->gemm_buff_offsets + 1, gemmCsizes.data(), mxLeafNode * sizeof(int64_t), gpuMemcpyHostToDevice) );
    *(thrust::device_ptr<int64_t>(ws->gemm_buff_offsets)) = 0;

    int64_t total_entries = *(thrust::device_ptr<int64_t>(
        thrust::inclusive_scan(
            gpu_thrust_par, ws->gemm_buff_offsets + 1, 
            ws->gemm_buff_offsets + mxLeafNode + 1, ws->gemm_buff_offsets + 1
    ) ) - 1);

    // Allocate the base memory and generate the pointers on the device
    gpuErrchk(gpuMalloc(&(ws->gemm_buff_base), sizeof(T) * total_entries));
    generateOffsetPointers(ws->gemm_buff_base, ws->gemm_buff_offsets, ws->gemm_buff_ptrs, mxLeafNode);

    // Allocate GPU copy for the node list 
    gpuErrchk(gpuMalloc(&(ws->perm_c_supno), sizeof(int_t) * mx_fsize));
}

template<class T>
void copyHostLUDataToGPU(TBatchFactorizeWorkspace<T>* ws, LocalLU_type<T>* host_Llu, int_t nsupers)
{
    LocalLU_type<T>& d_localLU = ws->d_localLU;

    // Allocate data, offset and ptr arrays for the indices and lower triangular blocks 
    d_localLU.Lrowind_bc_cnt = host_Llu->Lrowind_bc_cnt;
    gpuErrchk( gpuMalloc(&(d_localLU.Lrowind_bc_dat), d_localLU.Lrowind_bc_cnt * sizeof(int_t)) );
    gpuErrchk( gpuMalloc(&(d_localLU.Lrowind_bc_offset), nsupers * sizeof(long int)) );
    gpuErrchk( gpuMalloc(&(d_localLU.Lrowind_bc_ptr), nsupers * sizeof(int_t*)) );

    d_localLU.Lnzval_bc_cnt = host_Llu->Lnzval_bc_cnt;
    gpuErrchk( gpuMalloc(&(d_localLU.Lnzval_bc_dat), d_localLU.Lnzval_bc_cnt * sizeof(T)) );
    gpuErrchk( gpuMalloc(&(d_localLU.Lnzval_bc_offset), nsupers * sizeof(long int)) );
    gpuErrchk( gpuMalloc(&(d_localLU.Lnzval_bc_ptr), nsupers * sizeof(T*)) );

    // Allocate data, offset and ptr arrays for the indices and upper triangular blocks 
    d_localLU.Ucolind_br_cnt = host_Llu->Ucolind_br_cnt;
    gpuErrchk( gpuMalloc(&(d_localLU.Ucolind_br_dat), d_localLU.Ucolind_br_cnt * sizeof(int_t)) );
    gpuErrchk( gpuMalloc(&(d_localLU.Ucolind_br_offset), nsupers * sizeof(int64_t)) );
    gpuErrchk( gpuMalloc(&(d_localLU.Ucolind_br_ptr), nsupers * sizeof(int_t*)) );

    d_localLU.Unzval_br_new_cnt = host_Llu->Unzval_br_new_cnt;
    gpuErrchk( gpuMalloc(&(d_localLU.Unzval_br_new_dat), d_localLU.Unzval_br_new_cnt * sizeof(T)) );
    gpuErrchk( gpuMalloc(&(d_localLU.Unzval_br_new_offset), nsupers * sizeof(int64_t)) );
    gpuErrchk( gpuMalloc(&(d_localLU.Unzval_br_new_ptr), nsupers * sizeof(T*)) );

    // Copy the index and nzval data over to the GPU 
    gpuErrchk( gpuMemcpy(d_localLU.Lrowind_bc_dat, host_Llu->Lrowind_bc_dat, d_localLU.Lrowind_bc_cnt * sizeof(int_t), gpuMemcpyHostToDevice) );
    gpuErrchk( gpuMemcpy(d_localLU.Lrowind_bc_offset, host_Llu->Lrowind_bc_offset, nsupers * sizeof(long int), gpuMemcpyHostToDevice) );
    gpuErrchk( gpuMemcpy(d_localLU.Lnzval_bc_dat, host_Llu->Lnzval_bc_dat, d_localLU.Lnzval_bc_cnt * sizeof(T), gpuMemcpyHostToDevice) );
    gpuErrchk( gpuMemcpy(d_localLU.Lnzval_bc_offset, host_Llu->Lnzval_bc_offset, nsupers * sizeof(long int), gpuMemcpyHostToDevice) );
    
    gpuErrchk( gpuMemcpy(d_localLU.Ucolind_br_dat, host_Llu->Ucolind_br_dat, d_localLU.Ucolind_br_cnt * sizeof(int_t), gpuMemcpyHostToDevice) );
    gpuErrchk( gpuMemcpy(d_localLU.Ucolind_br_offset, host_Llu->Ucolind_br_offset, nsupers * sizeof(int64_t), gpuMemcpyHostToDevice) );
    gpuErrchk( gpuMemcpy(d_localLU.Unzval_br_new_dat, host_Llu->Unzval_br_new_dat, d_localLU.Unzval_br_new_cnt * sizeof(T), gpuMemcpyHostToDevice) );
    gpuErrchk( gpuMemcpy(d_localLU.Unzval_br_new_offset, host_Llu->Unzval_br_new_offset, nsupers * sizeof(int64_t), gpuMemcpyHostToDevice) );
    
    // Generate the pointers using the offsets 
    generateOffsetPointers(d_localLU.Lrowind_bc_dat, d_localLU.Lrowind_bc_offset, d_localLU.Lrowind_bc_ptr, nsupers);
    generateOffsetPointers(d_localLU.Lnzval_bc_dat, d_localLU.Lnzval_bc_offset, d_localLU.Lnzval_bc_ptr, nsupers);
    generateOffsetPointers(d_localLU.Ucolind_br_dat, d_localLU.Ucolind_br_offset, d_localLU.Ucolind_br_ptr, nsupers);
    generateOffsetPointers(d_localLU.Unzval_br_new_dat, d_localLU.Unzval_br_new_offset, d_localLU.Unzval_br_new_ptr, nsupers);

    // Copy the L data for global ids and block offsets into a more parallel friendly data structure 
    computeLBlockData(ws, nsupers);
}

/* EMT study instrumentation: SLU_BATCH_PROF=1 times every stage of every tree
   level with a device synchronization on both sides (so it perturbs the run;
   leave unset for production timings).  Stages: 0 marshal(all four), 1 getrf,
   2 info reduce, 3 trsm U, 4 trsm L, 5 gemm, 6 scatter. */
#define SLU_BPROF_NSTAGE 7
static int slu_bprof = -1;
static double slu_bprof_tot[SLU_BPROF_NSTAGE];
static inline int slu_bprof_on()
{
    if ( slu_bprof < 0 ) { const char *e = getenv("SLU_BATCH_PROF"); slu_bprof = e ? atoi(e) : 0; }
    return slu_bprof;
}
#define BPROF_TIC()  if (prof) { gpuErrchk(gpuDeviceSynchronize()); t0 = SuperLU_timer_(); }
#define BPROF_TOC(i) if (prof) { gpuErrchk(gpuDeviceSynchronize()); tp[i] += SuperLU_timer_() - t0; }

template<class T>
void TFactBatchSolve(TBatchFactorizeWorkspace<T>* ws, int_t k_st, int_t k_end)
{
#ifdef HAVE_MAGMA
    LocalLU_type<T>& d_localLU = ws->d_localLU;
    TBatchLUMarshallData<T>& mdata = ws->marshall_data;
    TBatchSCUMarshallData<T>& sc_mdata = ws->sc_marshall_data;

    const T t_one = one<T>(), t_zero = zeroT<T>();
    const int prof = slu_bprof_on();
    double tp[SLU_BPROF_NSTAGE] = {0.0}, t0 = 0.0;

    // Diagonal block batched LU decomposition   
    BPROF_TIC();
    marshallBatchedLUData<T>(ws, k_st, k_end);
    BPROF_TOC(0);
    long long prof_maxdiag = 0, prof_maxLpanel = 0, prof_maxUpanel = 0;
    if (prof) prof_maxdiag = thrust::reduce(gpu_thrust_par, mdata.dev_diag_dim_array, mdata.dev_diag_dim_array + mdata.batchsize, 0, thrust::maximum<BatchDim_t>());
    
    // TODO: This should be replaced by the user defined tolerances
    /* magma_*getrf_nopiv_expert_vbatched (dtol_array == NULL) replaces every
       diagonal pivot with |pivot| < eps by sign(pivot)*eps and counts the
       replacements in info.  This is an absolute threshold, applied whatever
       options->ReplaceTinyPivot says.  SLU_BATCH_PIVTOL overrides it for
       experiments (0 disables replacement); the default is unchanged. */
    static double eps = -1.0;
    if ( eps < 0.0 ) {
        const char *e = getenv("SLU_BATCH_PIVTOL");
        eps = e ? atof(e) : 1e-6;
    }

    BPROF_TIC();
    int_t info = magma_getrf_nopiv_vbatched(
        mdata.dev_diag_dim_array, mdata.dev_diag_dim_array, 
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, 
        NULL, eps, mdata.dev_info_array, mdata.batchsize, 
        ws->magma_queue
    );
    BPROF_TOC(1);
    
    BPROF_TIC();
    int max_info = thrust::reduce(gpu_thrust_par, mdata.dev_info_array, mdata.dev_info_array + mdata.batchsize, 0, thrust::maximum<BatchDim_t>());
    printf("Factor info = %d max_info = %d\n", info, max_info);
    BPROF_TOC(2);

    // Upper panel batched triangular solves
    BPROF_TIC();
    marshallBatchedTRSMUData<T>(ws, k_st, k_end);
    BPROF_TOC(0);
    if (prof) prof_maxUpanel = thrust::reduce(gpu_thrust_par, mdata.dev_panel_dim_array, mdata.dev_panel_dim_array + mdata.batchsize, 0, thrust::maximum<BatchDim_t>());

    BPROF_TIC();
    magmablas_trsm_vbatched_nocheck(
        MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
        mdata.dev_diag_dim_array, mdata.dev_panel_dim_array, t_one, 
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, 
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, 
        mdata.batchsize, ws->magma_queue
    );
    BPROF_TOC(3);

    // Lower panel batched triangular solves
    BPROF_TIC();
    marshallBatchedTRSMLData<T>(ws, k_st, k_end);
    BPROF_TOC(0);
    if (prof) prof_maxLpanel = thrust::reduce(gpu_thrust_par, mdata.dev_panel_dim_array, mdata.dev_panel_dim_array + mdata.batchsize, 0, thrust::maximum<BatchDim_t>());

    BPROF_TIC();
    magmablas_trsm_vbatched_nocheck(
        MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
        mdata.dev_panel_dim_array, mdata.dev_diag_dim_array, t_one, 
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, 
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, 
        mdata.batchsize, ws->magma_queue
    );
    BPROF_TOC(4);

    // Batched schur complement updates 
    BPROF_TIC();
    marshallBatchedSCUData<T>(ws, k_st, k_end);
    BPROF_TOC(0);
    
    BPROF_TIC();
    magmablas_gemm_vbatched_max_nocheck (
        MagmaNoTrans, MagmaNoTrans, sc_mdata.dev_m_array, sc_mdata.dev_n_array, sc_mdata.dev_k_array,
        t_one, sc_mdata.dev_A_ptrs, sc_mdata.dev_lda_array, sc_mdata.dev_B_ptrs, sc_mdata.dev_ldb_array,
        t_zero, sc_mdata.dev_C_ptrs, sc_mdata.dev_ldc_array, sc_mdata.batchsize,
        sc_mdata.max_m, sc_mdata.max_n, sc_mdata.max_k, ws->magma_queue
    );
    BPROF_TOC(5);
    
    if (getenv("SLU_DEBUG_BATCH_SCU"))
    {
        printf("[scu] k_st %lld batchsize %lld  max_m %lld max_n %lld max_k %lld  max_ilen %lld max_jlen %lld%s\n",
               (long long)k_st, (long long)sc_mdata.batchsize,
               (long long)sc_mdata.max_m, (long long)sc_mdata.max_n, (long long)sc_mdata.max_k,
               (long long)sc_mdata.max_ilen, (long long)sc_mdata.max_jlen,
               (sc_mdata.max_ilen == 0 || sc_mdata.max_jlen == 0) ? "   <-- ZERO GRID DIM (skipped)" : "");
        fflush(stdout);
    }

    BPROF_TIC();
    scatterGPU_batchDriver_flat<T>(
        k_st, ws->maxSuperSize, sc_mdata.dev_C_ptrs, sc_mdata.dev_ldc_array,
        d_localLU.Unzval_br_new_ptr, d_localLU.Ucolind_br_ptr, d_localLU.Lnzval_bc_ptr, 
        d_localLU.Lrowind_bc_ptr, ws->d_lblock_gid_ptrs, ws->d_lblock_start_ptrs, 
        ws->perm_c_supno, ws->xsup, ws->ldt, sc_mdata.max_ilen, sc_mdata.max_jlen, 
        sc_mdata.batchsize, ws->stream
    );
    BPROF_TOC(6);

    if (prof) {
        double tl = 0.0;
        for (int i = 0; i < SLU_BPROF_NSTAGE; ++i) { tl += tp[i]; slu_bprof_tot[i] += tp[i]; }
        printf("[bprof] level k_st %lld nsup %lld | maxdiag %lld maxLpanel %lld maxUpanel %lld gemm m/n/k %lld/%lld/%lld scatter grid %lldx%lldx%lld x%lld thr | ms: marshal %.3f getrf %.3f info %.3f trsmU %.3f trsmL %.3f gemm %.3f scatter %.3f | level %.3f\n",
               (long long)k_st, (long long)mdata.batchsize, prof_maxdiag, prof_maxLpanel, prof_maxUpanel,
               (long long)sc_mdata.max_m, (long long)sc_mdata.max_n, (long long)sc_mdata.max_k,
               (long long)sc_mdata.max_ilen, (long long)sc_mdata.max_jlen, (long long)sc_mdata.batchsize, (long long)ws->ldt,
               1e3*tp[0], 1e3*tp[1], 1e3*tp[2], 1e3*tp[3], 1e3*tp[4], 1e3*tp[5], 1e3*tp[6], 1e3*tl);
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main factorization routiunes
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
int sparseTreeFactorBatchGPUT(TBatchFactorizeWorkspace<T>* ws, sForest_t *sforest)
{
    int_t nnodes = sforest->nNodes; 

    if(nnodes < 1)
        return 1;
    
    // Host list of nodes in the order of factorization copied to the GPU 
    int_t *perm_c_supno = sforest->nodeList; 
    gpuErrchk(gpuMemcpy(ws->perm_c_supno, perm_c_supno, sizeof(int_t) * nnodes, gpuMemcpyHostToDevice));

    // Tree containing the supernode limits per level 
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;

    if (getenv("SLU_DEBUG_LEVELW")) {
        int_t leafw = eTreeTopLims[1] - eTreeTopLims[0], mx = 0, mxl = -1;
        for (int_t l = 0; l < maxTopoLevel; ++l) {
            int_t w = eTreeTopLims[l + 1] - eTreeTopLims[l];
            if (w > mx) { mx = w; mxl = l; }
        }
        printf("[levelw] forest: nodes %lld levels %lld leafWidth %lld widest %lld (level %lld)\n",
               (long long)nnodes, (long long)maxTopoLevel, (long long)leafw,
               (long long)mx, (long long)mxl);
        fflush(stdout);
    }

    if (slu_bprof_on()) for (int i = 0; i < SLU_BPROF_NSTAGE; ++i) slu_bprof_tot[i] = 0.0;
    double bprof_t0 = SuperLU_timer_();

    for(int_t topoLvl = 0; topoLvl < maxTopoLevel; topoLvl++)
        TFactBatchSolve<T>(ws, eTreeTopLims[topoLvl], eTreeTopLims[topoLvl + 1]);

    /* The last scatter kernel is asynchronous; without this the caller's
       timer stops before the factorization has finished on the device. */
    if (getenv("SLU_BATCH_SYNC") || slu_bprof_on()) gpuErrchk(gpuDeviceSynchronize());
    if (getenv("SLU_BATCH_SYNC") || slu_bprof_on())
        printf("[bsync] synchronized batch factorization wall %.3f ms\n", 1e3*(SuperLU_timer_() - bprof_t0));
    if (slu_bprof_on()) {
        double *t = slu_bprof_tot;
        printf("[bprof] TOTAL levels %lld nsup %lld | ms: marshal %.3f getrf %.3f info %.3f trsmU %.3f trsmL %.3f gemm %.3f scatter %.3f | sum %.3f\n",
               (long long)maxTopoLevel, (long long)nnodes, 1e3*t[0], 1e3*t[1], 1e3*t[2], 1e3*t[3], 1e3*t[4], 1e3*t[5], 1e3*t[6],
               1e3*(t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]));
        fflush(stdout);
    }

    return 0;
}

template<class T>
TBatchFactorizeWorkspace<T>* getBatchFactorizeWorkspaceT(
    int_t nsupers, int_t ldt, trf3dpartitionType<T> *trf3Dpartition, LUStruct_type<T> *LUstruct, 
    gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info,
    int convertU = 1
)
{
#ifdef HAVE_MAGMA
    TBatchFactorizeWorkspace<T>* ws = new TBatchFactorizeWorkspace<T>();
    
    int device_id;
    gpuErrchk( gpuGetDevice(&device_id) );

    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t n = xsup[nsupers];
    gridinfo_t *grid = &(grid3d->grid2d);

    double tic = SuperLU_timer_();

    /* convertU == 0: the caller already holds U in the row-block layout
       (device-resident reuse builds it once, with the slot map). */
    if (convertU) pconvert_flatten_skyline2UROWDATA(options, grid, LUstruct, stat, n);

    double convert_time = SuperLU_timer_() - tic;

    if (slu_bprof_on()) {
        /* supernode size / panel size statistics (host data, after the U conversion) */
        long long hist[10] = {0}, sumc = 0, maxc = 0, sumlr = 0, maxlr = 0, sumuc = 0, maxuc = 0, nnzLU = 0, nL = 0, nU = 0;
        for (int_t k = 0; k < nsupers; ++k) {
            long long c = xsup[k + 1] - xsup[k];
            int b = 0; while ((1LL << b) < c) ++b;   /* bins 1,2,3-4,5-8,... */
            hist[b > 9 ? 9 : b]++; sumc += c; if (c > maxc) maxc = c;
            int_t *Lidx = LUstruct->Llu->Lrowind_bc_ptr[k], *Uidx = LUstruct->Llu->Ucolind_br_ptr[k];
            if (Lidx) { long long r = Lidx[1]; sumlr += r; if (r > maxlr) maxlr = r; nnzLU += r * c; ++nL; }
            if (Uidx) { long long u = Uidx[1], ur = Uidx[2]; sumuc += u; if (u > maxuc) maxuc = u; nnzLU += u * ur; ++nU; }
        }
        printf("[bprof] supernodes %lld  cols avg %.2f max %lld | L rows avg %.2f max %lld | U cols avg %.2f max %lld (in %lld of them) | dense L+U entries %lld | ldt(maxsup) %lld\n",
               (long long)nsupers, (double)sumc / nsupers, maxc, nL ? (double)sumlr / nL : 0.0, maxlr,
               nU ? (double)sumuc / nU : 0.0, maxuc, nU, nnzLU, (long long)ldt);
        printf("[bprof] supernode size histogram  1:%lld 2:%lld 3-4:%lld 5-8:%lld 9-16:%lld 17-32:%lld 33-64:%lld 65-128:%lld 129-256:%lld >256:%lld\n",
               hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8], hist[9]);
    }

    // TODO: determine if ldt is supposed to be the same as maxSuperSize?
    ws->ldt = ws->maxSuperSize = ldt;
    ws->nsupers = nsupers;

    // Set up device handles 
    gpuErrchk( gpuStreamCreate(&ws->stream) );
    gpublasCreate( &ws->cuhandle );
#ifdef HAVE_CUDA
    magma_queue_create_from_cuda(device_id, ws->stream, ws->cuhandle, NULL, &ws->magma_queue);
#elif defined(HAVE_HIP)
    magma_queue_create_from_hip(device_id, ws->stream, ws->cuhandle, NULL, &ws->magma_queue);
#endif

    // Copy the xsup to the GPU 
    tic = SuperLU_timer_();
    gpuErrchk(gpuMalloc(&ws->xsup, (nsupers + 1) * sizeof(int_t)));
    gpuErrchk(gpuMemcpy(ws->xsup, xsup, (nsupers + 1) * sizeof(int_t), gpuMemcpyHostToDevice));

    // Copy the flattened LU data over to the GPU 
    // TODO: I currently have to make a GPU friendly copy of the globa ids of blocks within L
    // and compute block offsets. Can this be avoided with a change to the L index structure?
    copyHostLUDataToGPU<T>(ws, LUstruct->Llu, nsupers);

    double copy_time = SuperLU_timer_() - tic;

    // Allocate marhsalling workspace
    tic = SuperLU_timer_();
    if (getenv("SLU_DEBUG_LEVELW")) {
        printf("[levelw] marshall arrays sized to mxLeafNode = %lld\n",
               (long long) trf3Dpartition->mxLeafNode); fflush(stdout);
    }
    ws->marshall_data.setBatchSize(trf3Dpartition->mxLeafNode);
    ws->sc_marshall_data.setBatchSize(trf3Dpartition->mxLeafNode);

    // Determine buffer sizes for schur complement updates and supernode lists 
    batchAllocateGemmBuffers<T>(ws, LUstruct, trf3Dpartition, grid3d);
    double ws_time = SuperLU_timer_() - tic;
    
    printf("\tSky2UROWDATA Convert time = %.4f\n", convert_time);
    printf("\tH2D Copy time = %.4f\n", copy_time);
    printf("\tWorkspace alloc time = %.4f\n", ws_time);

    return ws;
#endif
}

template<class T>
void copyGPULUDataToHostT(
    TBatchFactorizeWorkspace<T>* ws, LUStruct_type<T> *LUstruct, gridinfo3d_t *grid3d,
    SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
)
{
    LocalLU_type<T>& d_localLU = ws->d_localLU;
    LocalLU_type<T>* host_Llu = LUstruct->Llu;

    double tic = SuperLU_timer_();
    
    // Only need to copy the nzval data arrays when moving from the GPU to the Host 
    gpuErrchk( gpuMemcpy(host_Llu->Lnzval_bc_dat, d_localLU.Lnzval_bc_dat, d_localLU.Lnzval_bc_cnt * sizeof(T), gpuMemcpyDeviceToHost) );
    gpuErrchk( gpuMemcpy(host_Llu->Unzval_br_new_dat, d_localLU.Unzval_br_new_dat, d_localLU.Unzval_br_new_cnt * sizeof(T), gpuMemcpyDeviceToHost) );
    
    double copy_time = SuperLU_timer_() - tic;

    // Convert the host data from block row to skyline 
    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t n = xsup[ws->nsupers];
    gridinfo_t *grid = &(grid3d->grid2d);

    tic = SuperLU_timer_();
    pconvertUROWDATA2skyline(options, grid, LUstruct, stat, n);
    double convert_time = SuperLU_timer_() - tic;

    printf("\tD2H Copy time = %.4f\n", copy_time);
    printf("\tConvert time = %.4f\n", convert_time);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device-resident reuse: refill the device L/U values straight from A
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void scatterAvalsKernel(const T* a, const int64_t* map, int_t nnz, int64_t Lcnt, T* L, T* U)
{
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz) return;
    int64_t m = map[p];
    if (m < 0) return;
    if (m < Lcnt) L[m] = a[p];
    else          U[m - Lcnt] = a[p];
}

/* One-time setup on the first (DOFACT) call: device copies of the slot map and
   of A's values, and pinned registration of the three host buffers that cross
   the bus every step (A values in, L and U values out). */
template<class T>
int batchDevResSetupT(TBatchFactorizeWorkspace<T>* ws, LUStruct_type<T> *LUstruct, T *avals, int_t nnz, const int64_t *map_host)
{
    LocalLU_type<T>* Llu = LUstruct->Llu;
    ws->nnz_a = nnz;
    gpuErrchk(gpuMalloc(&ws->d_avals, sizeof(T) * nnz));
    gpuErrchk(gpuMalloc(&ws->d_map, sizeof(int64_t) * nnz));
    gpuErrchk(gpuMemcpy(ws->d_map, map_host, sizeof(int64_t) * nnz, gpuMemcpyHostToDevice));
    ws->h_map = (int64_t *) malloc(sizeof(int64_t) * nnz);
    memcpy(ws->h_map, map_host, sizeof(int64_t) * nnz);
#ifdef HAVE_CUDA
    void  *ptrs[3]  = { avals, Llu->Lnzval_bc_dat, Llu->Unzval_br_new_dat };
    size_t sizes[3] = { sizeof(T) * (size_t) nnz, sizeof(T) * (size_t) Llu->Lnzval_bc_cnt, sizeof(T) * (size_t) Llu->Unzval_br_new_cnt };
    for (int i = 0; i < 3; ++i) {
        if (cudaHostRegister(ptrs[i], sizes[i], cudaHostRegisterDefault) == cudaSuccess) ws->pinned[i] = ptrs[i];
        else { cudaGetLastError(); printf("[devres] cudaHostRegister of buffer %d failed; using pageable copies\n", i); }
    }
#endif
    return 0;
}

/* Every call: upload A's values, zero the device factors, scatter. */
template<class T>
int batchDevResRefillT(TBatchFactorizeWorkspace<T>* ws, const T *avals, int_t nnz)
{
    LocalLU_type<T>& d = ws->d_localLU;
    if (nnz != ws->nnz_a) { fprintf(stderr, "batchDevResRefill: nnz changed (%lld -> %lld)\n", (long long) ws->nnz_a, (long long) nnz); return -1; }
    double t0 = SuperLU_timer_();
    gpuErrchk(gpuMemcpyAsync(ws->d_avals, avals, sizeof(T) * nnz, gpuMemcpyHostToDevice, ws->stream));
    gpuErrchk(gpuStreamSynchronize(ws->stream));
    double t1 = SuperLU_timer_();
    gpuErrchk(gpuMemsetAsync(d.Lnzval_bc_dat, 0, sizeof(T) * d.Lnzval_bc_cnt, ws->stream));
    gpuErrchk(gpuMemsetAsync(d.Unzval_br_new_dat, 0, sizeof(T) * d.Unzval_br_new_cnt, ws->stream));
    int nthreads = 256;
    int64_t nblocks = ((int64_t) nnz + nthreads - 1) / nthreads;
    scatterAvalsKernel<T><<<(unsigned) nblocks, nthreads, 0, ws->stream>>>(ws->d_avals, ws->d_map, nnz, (int64_t) d.Lnzval_bc_cnt, d.Lnzval_bc_dat, d.Unzval_br_new_dat);
    gpuErrchk(gpuGetLastError());
    gpuErrchk(gpuStreamSynchronize(ws->stream));
    double t2 = SuperLU_timer_();
    printf("\tDevRes refill time = %.4f (A upload %.4f, zero+scatter %.4f)\n", t2 - t0, t1 - t0, t2 - t1);
    return 0;
}

template<class T>
__global__ void gatherUvalsKernel(const T* src, const int64_t* map, int64_t n, T* dst)
{
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int64_t m = map[j];
    dst[j] = (m < 0) ? T(0) : src[m];
}

/* Inverse of every diagonal block, one thread block per local block column:
   Linv = inv(unit lower part), Uinv = inv(upper part), both knsupc x knsupc
   column-major, by column-wise substitution (what pdCompute_Diag_Inv gets
   from dtrtri).  Thread j owns column j of both inverses. */
template<class T>
__global__ void diagInvKernel(int_t nsupers, int npcol, int mycol, const int_t* xsup,
                              const long int* Lrowind_off, const int_t* Lrowind_dat,
                              const long int* Lnzval_off, const T* Lnzval,
                              const long int* Linv_off, T* Linv, const long int* Uinv_off, T* Uinv)
{
    int_t lk = blockIdx.x;
    int_t k = (int_t) lk * npcol + mycol;
    if (k >= nsupers) return;
    if (Lrowind_off[lk] < 0 || Linv_off[lk] < 0 || Uinv_off[lk] < 0 || Lnzval_off[lk] < 0) return;
    int n = (int)(xsup[k + 1] - xsup[k]);
    int nsupr = (int) Lrowind_dat[Lrowind_off[lk] + 1];
    const T* L = Lnzval + Lnzval_off[lk];
    T* Li = Linv + Linv_off[lk];
    T* Ui = Uinv + Uinv_off[lk];
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        /* column j of inv(L), L unit lower */
        for (int i = 0; i < j; ++i) Li[j * n + i] = T(0);
        Li[j * n + j] = T(1);
        for (int i = j + 1; i < n; ++i) {
            T s = T(0);
            for (int l = j; l < i; ++l) s += L[l * nsupr + i] * Li[j * n + l];
            Li[j * n + i] = -s;
        }
        /* column j of inv(U), U upper non-unit */
        for (int i = j + 1; i < n; ++i) Ui[j * n + i] = T(0);
        for (int i = j; i >= 0; --i) {
            T s = (i == j) ? T(1) : T(0);
            for (int l = i + 1; l <= j; ++l) s -= L[l * nsupr + i] * Ui[j * n + l];
            Ui[j * n + i] = s / L[i * nsupr + i];
        }
    }
}

template<class T>
int batchDevResSolveSetupT(TBatchFactorizeWorkspace<T>* ws, const int64_t *umap_host, int64_t ucnt)
{
    ws->ucnt = ucnt;
    if (umap_host && ucnt > 0) {   /* column-wise U solve: gather map */
        gpuErrchk(gpuMalloc(&ws->d_umap, sizeof(int64_t) * ucnt));
        gpuErrchk(gpuMemcpy(ws->d_umap, umap_host, sizeof(int64_t) * ucnt, gpuMemcpyHostToDevice));
    }                              /* else: row-data U solve, same layout as the factorization's U */
    ws->solve_ready = 1;
    return 0;
}

/* Reuse step: hand the device factors to the GPU triangular solve without
   touching the host: L values by device copy, U values gathered into the
   solve's column-wise layout, diagonal inverses computed on the device. */
template<class T>
int batchDevResSolveRefreshT(TBatchFactorizeWorkspace<T>* ws, LUStruct_type<T> *LUstruct, int_t nsupers, int npcol, int mycol)
{
    LocalLU_type<T>* Llu = LUstruct->Llu;
    LocalLU_type<T>& d = ws->d_localLU;
    if (!ws->solve_ready) return -1;
    double t0 = SuperLU_timer_();
    gpuErrchk(gpuMemcpyAsync(Llu->d_Lnzval_bc_dat, d.Lnzval_bc_dat, sizeof(T) * d.Lnzval_bc_cnt, gpuMemcpyDeviceToDevice, ws->stream));
    gpuErrchk(gpuStreamSynchronize(ws->stream));
    double t1 = SuperLU_timer_();
    if (ws->d_umap) {
        int nthreads = 256;
        int64_t nblocks = (ws->ucnt + nthreads - 1) / nthreads;
        gatherUvalsKernel<T><<<(unsigned) nblocks, nthreads, 0, ws->stream>>>(d.Unzval_br_new_dat, ws->d_umap, ws->ucnt, Llu->d_Unzval_bc_dat);
        gpuErrchk(gpuGetLastError());
        gpuErrchk(gpuStreamSynchronize(ws->stream));
    } else {
        /* row-data U solve reads the same layout the factorization produced */
        if (Llu->d_Unzval_br_new_dat == NULL) { fprintf(stderr, "solve refresh: d_Unzval_br_new_dat not allocated\n"); return -1; }
        gpuErrchk(gpuMemcpyAsync(Llu->d_Unzval_br_new_dat, d.Unzval_br_new_dat, sizeof(T) * (size_t) d.Unzval_br_new_cnt, gpuMemcpyDeviceToDevice, ws->stream));
        gpuErrchk(gpuStreamSynchronize(ws->stream));
    }
    double t2 = SuperLU_timer_();
    {
        int_t nsupers_j = (nsupers + npcol - 1) / npcol;
        diagInvKernel<T><<<(unsigned) nsupers_j, 32, 0, ws->stream>>>(nsupers, npcol, mycol, Llu->d_xsup,
            Llu->d_Lrowind_bc_offset, Llu->d_Lrowind_bc_dat, Llu->d_Lnzval_bc_offset, Llu->d_Lnzval_bc_dat,
            Llu->d_Linv_bc_offset, Llu->d_Linv_bc_dat, Llu->d_Uinv_bc_offset, Llu->d_Uinv_bc_dat);
        gpuErrchk(gpuGetLastError());
        gpuErrchk(gpuStreamSynchronize(ws->stream));
    }
    double t3 = SuperLU_timer_();
    printf("\tDevRes solve refresh time = %.4f (L copy %.4f, U gather %.4f, diag inv %.4f)\n", t3 - t0, t1 - t0, t2 - t1, t3 - t2);
    return 0;
}

/* Stage 4 kernels: device L/U slot <- R[i]*C[j] * (caller's value), either
   from one contiguous staging array or through the per-system pointers. */
template<class T>
__global__ void scatterSysValsKernel(const T* vals, const int64_t* map2, const double* scale2, int_t nnz, int64_t Lcnt, T* L, T* U)
{
    int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnz) return;
    int64_t m = map2[q];
    if (m < 0) return;
    T v = scale2[q] * vals[q];
    if (m < Lcnt) L[m] = v; else U[m - Lcnt] = v;
}
template<class T>
__global__ void scatterSysPtrsKernel(T* const* Aptrs, const int* ent_sys, const int* ent_idx, const int64_t* map2, const double* scale2,
                                     int_t nnz, int64_t Lcnt, T* L, T* U)
{
    int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnz) return;
    int64_t m = map2[q];
    if (m < 0) return;
    T v = scale2[q] * Aptrs[ent_sys[q]][ent_idx[q]];
    if (m < Lcnt) L[m] = v; else U[m - Lcnt] = v;
}

template<class T>
int batchDevResSetupAT(TBatchFactorizeWorkspace<T>* ws, int nsys, int_t nnz2, const int64_t *posmap,
                       const double *scale2, const int *ent_sys, const int *ent_idx, double anorm)
{
    if (!ws->h_map) return -1;
    std::vector<int64_t> map2(nnz2);
    int64_t nmiss = 0;
    for (int_t q = 0; q < nnz2; ++q) {
        int64_t pos = posmap[q];
        map2[q] = (pos >= 0 && pos < ws->nnz_a) ? ws->h_map[pos] : -1;
        if (map2[q] < 0) ++nmiss;
    }
    ws->nnz2 = nnz2; ws->nsys = nsys; ws->anorm_cache = anorm;
    gpuErrchk(gpuMalloc(&ws->d_map2, sizeof(int64_t) * nnz2));
    gpuErrchk(gpuMalloc(&ws->d_scale2, sizeof(double) * nnz2));
    gpuErrchk(gpuMalloc(&ws->d_ent_sys, sizeof(int) * nnz2));
    gpuErrchk(gpuMalloc(&ws->d_ent_idx, sizeof(int) * nnz2));
    gpuErrchk(gpuMalloc(&ws->d_Aptrs, sizeof(double*) * nsys));
    gpuErrchk(gpuMemcpy(ws->d_map2, map2.data(), sizeof(int64_t) * nnz2, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ws->d_scale2, scale2, sizeof(double) * nnz2, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ws->d_ent_sys, ent_sys, sizeof(int) * nnz2, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ws->d_ent_idx, ent_idx, sizeof(int) * nnz2, gpuMemcpyHostToDevice));
#ifdef HAVE_CUDA
    if (cudaHostAlloc((void**)&ws->h_avals_cat, sizeof(T) * (size_t) nnz2, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError(); ws->h_avals_cat = (T*) malloc(sizeof(T) * (size_t) nnz2);
    }
#else
    ws->h_avals_cat = (T*) malloc(sizeof(T) * (size_t) nnz2);
#endif
    printf("[devres] A map: %lld entries from %d systems, unmapped %lld\n", (long long) nnz2, nsys, (long long) nmiss);
    return 0;
}

template<class T>
int batchDevResRefillAT(TBatchFactorizeWorkspace<T>* ws, int from_device, T **Aptrs, const int *nnzd)
{
    LocalLU_type<T>& d = ws->d_localLU;
    if (!ws->d_map2) return -1;
    double t0 = SuperLU_timer_(), t1;
    gpuErrchk(gpuMemsetAsync(d.Lnzval_bc_dat, 0, sizeof(T) * d.Lnzval_bc_cnt, ws->stream));
    gpuErrchk(gpuMemsetAsync(d.Unzval_br_new_dat, 0, sizeof(T) * d.Unzval_br_new_cnt, ws->stream));
    int nthreads = 256;
    int64_t nblocks = ((int64_t) ws->nnz2 + nthreads - 1) / nthreads;
    if (from_device) {
        gpuErrchk(gpuMemcpyAsync(ws->d_Aptrs, Aptrs, sizeof(T*) * ws->nsys, gpuMemcpyHostToDevice, ws->stream));
        t1 = SuperLU_timer_();
        scatterSysPtrsKernel<T><<<(unsigned) nblocks, nthreads, 0, ws->stream>>>(ws->d_Aptrs, ws->d_ent_sys, ws->d_ent_idx, ws->d_map2, ws->d_scale2,
                                                                                  ws->nnz2, (int64_t) d.Lnzval_bc_cnt, d.Lnzval_bc_dat, d.Unzval_br_new_dat);
    } else {
        size_t off = 0;
        for (int s = 0; s < ws->nsys; ++s) { memcpy(ws->h_avals_cat + off, Aptrs[s], sizeof(T) * (size_t) nnzd[s]); off += nnzd[s]; }
        if ((int_t) off != ws->nnz2) { fprintf(stderr, "batchDevResRefillA: nnz changed (%lld -> %lld)\n", (long long) ws->nnz2, (long long) off); return -1; }
        gpuErrchk(gpuMemcpyAsync(ws->d_avals, ws->h_avals_cat, sizeof(T) * (size_t) ws->nnz2, gpuMemcpyHostToDevice, ws->stream));
        gpuErrchk(gpuStreamSynchronize(ws->stream));
        t1 = SuperLU_timer_();
        scatterSysValsKernel<T><<<(unsigned) nblocks, nthreads, 0, ws->stream>>>(ws->d_avals, ws->d_map2, ws->d_scale2,
                                                                                  ws->nnz2, (int64_t) d.Lnzval_bc_cnt, d.Lnzval_bc_dat, d.Unzval_br_new_dat);
    }
    gpuErrchk(gpuGetLastError());
    gpuErrchk(gpuStreamSynchronize(ws->stream));
    double t2 = SuperLU_timer_();
    ws->a_prefilled = 1;
    printf("\tDevRes A refill time = %.4f (%s %.4f, zero+scatter %.4f)\n", t2 - t0, from_device ? "pointers" : "gather+upload", t1 - t0, t2 - t1);
    return 0;
}

template<class T>
void batchDevResFreeT(TBatchFactorizeWorkspace<T>* ws)
{
    if (ws->h_map) { free(ws->h_map); ws->h_map = nullptr; }
    if (ws->d_map2)    { gpuErrchk(gpuFree(ws->d_map2));    ws->d_map2 = nullptr; }
    if (ws->d_scale2)  { gpuErrchk(gpuFree(ws->d_scale2));  ws->d_scale2 = nullptr; }
    if (ws->d_ent_sys) { gpuErrchk(gpuFree(ws->d_ent_sys)); ws->d_ent_sys = nullptr; }
    if (ws->d_ent_idx) { gpuErrchk(gpuFree(ws->d_ent_idx)); ws->d_ent_idx = nullptr; }
    if (ws->d_Aptrs)   { gpuErrchk(gpuFree(ws->d_Aptrs));   ws->d_Aptrs = nullptr; }
#ifdef HAVE_CUDA
    if (ws->h_avals_cat) { if (cudaFreeHost(ws->h_avals_cat) != cudaSuccess) { cudaGetLastError(); free(ws->h_avals_cat); } ws->h_avals_cat = nullptr; }
#else
    if (ws->h_avals_cat) { free(ws->h_avals_cat); ws->h_avals_cat = nullptr; }
#endif
#ifdef HAVE_CUDA
    for (int i = 0; i < 3; ++i) if (ws->pinned[i]) { cudaHostUnregister(ws->pinned[i]); ws->pinned[i] = nullptr; }
#endif
    if (ws->d_avals) { gpuErrchk(gpuFree(ws->d_avals)); ws->d_avals = nullptr; }
    if (ws->d_map)   { gpuErrchk(gpuFree(ws->d_map));   ws->d_map = nullptr; }
    if (ws->d_umap)  { gpuErrchk(gpuFree(ws->d_umap));  ws->d_umap = nullptr; }
}

template<class T>
void freeBatchFactorizeWorkspaceT(TBatchFactorizeWorkspace<T>* ws)
{
    batchDevResFreeT<T>(ws);
    gpuErrchk( gpuFree(ws->d_lblock_gid_dat) );
    gpuErrchk( gpuFree(ws->d_lblock_gid_offsets) );
    gpuErrchk( gpuFree(ws->d_lblock_gid_ptrs) );
    gpuErrchk( gpuFree(ws->d_lblock_start_dat) );
    gpuErrchk( gpuFree(ws->d_lblock_start_offsets) );
    gpuErrchk( gpuFree(ws->d_lblock_start_ptrs) );
    gpuErrchk( gpuFree(ws->gemm_buff_base) );
    gpuErrchk( gpuFree(ws->gemm_buff_offsets) );
    gpuErrchk( gpuFree(ws->gemm_buff_ptrs) );
    gpuErrchk( gpuFree(ws->perm_c_supno) );
    gpuErrchk( gpuFree(ws->xsup) );

    LocalLU_type<T>& d_localLU = ws->d_localLU;
    gpuErrchk( gpuFree(d_localLU.Lrowind_bc_dat) );
    gpuErrchk( gpuFree(d_localLU.Lrowind_bc_offset) );
    gpuErrchk( gpuFree(d_localLU.Lrowind_bc_ptr) );
    gpuErrchk( gpuFree(d_localLU.Lnzval_bc_dat) );
    gpuErrchk( gpuFree(d_localLU.Lnzval_bc_offset) );
    gpuErrchk( gpuFree(d_localLU.Lnzval_bc_ptr) );
    gpuErrchk( gpuFree(d_localLU.Ucolind_br_dat) );
    gpuErrchk( gpuFree(d_localLU.Ucolind_br_offset) );
    gpuErrchk( gpuFree(d_localLU.Ucolind_br_ptr) );
    gpuErrchk( gpuFree(d_localLU.Unzval_br_new_dat) );
    gpuErrchk( gpuFree(d_localLU.Unzval_br_new_offset) );
    gpuErrchk( gpuFree(d_localLU.Unzval_br_new_ptr) );
#ifdef HAVE_MAGMA
    magma_queue_destroy(ws->magma_queue);
#endif
    gpublasDestroy( ws->cuhandle );
    gpuErrchk( gpuStreamDestroy(ws->stream) );
    //YL: not sure why the destructor TBatchLUMarshallData and TBatchSCUMarshallData are not called. Calling them explicitly here. 
    ws->marshall_data.DeleteTBatchLUMarshallData();
    ws->sc_marshall_data.DeleteTBatchSCUMarshallData();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU-resident RHS / solution for the batched interface
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void vbatchStackRhsKernel(int_t m_big, int nrhs, const int_t* rowsys, const int_t* rowloc, const int_t* rhsdst,
                                     const double* rscale, double* const* RHSptrs, const int* ldRHS, double* b)
{
    int_t g = (int_t)((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (g >= m_big) return;
    int_t d = rowsys[g], i = rowloc[g];
    const double* rhs = RHSptrs[d];
    for (int k = 0; k < nrhs; ++k)
        b[(int64_t)k * m_big + rhsdst[g]] = rscale[g] * rhs[(int64_t)k * ldRHS[d] + i];
}

__global__ void vbatchUnstackXKernel(int_t m_big, int nrhs, const int_t* rowsys, const int_t* rowloc, const int_t* xsrc,
                                     const double* cscale, double* const* Xptrs, const int* ldX, const double* b)
{
    int_t g = (int_t)((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (g >= m_big) return;
    int_t d = rowsys[g], i = rowloc[g];
    double* x = Xptrs[d];
    for (int k = 0; k < nrhs; ++k)
        x[(int64_t)k * ldX[d] + i] = cscale[g] * b[(int64_t)k * m_big + xsrc[g]];
}

extern "C" {

/* Build (once, on the DOFACT call) the row maps that fold the per-system
   equilibration and permutations of the RHS and solution into two gathers. */
int dvbatch_gpures_setup(dvbatch_ctx_t *ctx, int batchCount, int *m, int **RpivPtr, int **CpivPtr,
                         double **ReqPtr, double **CeqPtr, DiagScale_t *DiagScale)
{
    int_t m_big = ctx->m_big;
    std::vector<int_t> rowsys(m_big), rowloc(m_big), rhsdst(m_big), xsrc(m_big);
    std::vector<double> rscale(m_big), cscale(m_big);
    int_t off = 0;
    for (int d = 0; d < batchCount; ++d) {
        int rowequ = (DiagScale[d] == ROW || DiagScale[d] == BOTH);
        int colequ = (DiagScale[d] == COL || DiagScale[d] == BOTH);
        for (int i = 0; i < m[d]; ++i) {
            int_t g = off + i;
            rowsys[g] = d; rowloc[g] = i;
            rhsdst[g] = off + CpivPtr[d][RpivPtr[d][i]];
            xsrc[g]   = off + CpivPtr[d][i];
            rscale[g] = rowequ ? ReqPtr[d][i] : 1.0;
            cscale[g] = colequ ? CeqPtr[d][i] : 1.0;
        }
        off += m[d];
    }
    gpuErrchk(gpuMalloc(&ctx->d_b, sizeof(double) * (size_t) m_big * ctx->nrhs));
    gpuErrchk(gpuMalloc(&ctx->d_rowsys, sizeof(int_t) * m_big));
    gpuErrchk(gpuMalloc(&ctx->d_rowloc, sizeof(int_t) * m_big));
    gpuErrchk(gpuMalloc(&ctx->d_rhsdst, sizeof(int_t) * m_big));
    gpuErrchk(gpuMalloc(&ctx->d_xsrc,   sizeof(int_t) * m_big));
    gpuErrchk(gpuMalloc(&ctx->d_rscale, sizeof(double) * m_big));
    gpuErrchk(gpuMalloc(&ctx->d_cscale, sizeof(double) * m_big));
    gpuErrchk(gpuMalloc(&ctx->d_RHSptrs, sizeof(double*) * batchCount));
    gpuErrchk(gpuMalloc(&ctx->d_Xptrs,   sizeof(double*) * batchCount));
    gpuErrchk(gpuMalloc(&ctx->d_ldRHS, sizeof(int) * batchCount));
    gpuErrchk(gpuMalloc(&ctx->d_ldX,   sizeof(int) * batchCount));
    gpuErrchk(gpuMemcpy(ctx->d_rowsys, rowsys.data(), sizeof(int_t) * m_big, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ctx->d_rowloc, rowloc.data(), sizeof(int_t) * m_big, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ctx->d_rhsdst, rhsdst.data(), sizeof(int_t) * m_big, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ctx->d_xsrc,   xsrc.data(),   sizeof(int_t) * m_big, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ctx->d_rscale, rscale.data(), sizeof(double) * m_big, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ctx->d_cscale, cscale.data(), sizeof(double) * m_big, gpuMemcpyHostToDevice));
    return 0;
}

int dvbatch_gpures_stack(dvbatch_ctx_t *ctx, int batchCount, double **RHSptr, int *ldRHS, int nrhs)
{
    double t0 = SuperLU_timer_();
    gpuErrchk(gpuMemcpy(ctx->d_RHSptrs, RHSptr, sizeof(double*) * batchCount, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ctx->d_ldRHS, ldRHS, sizeof(int) * batchCount, gpuMemcpyHostToDevice));
    int nthreads = 256;
    int64_t nblocks = ((int64_t) ctx->m_big + nthreads - 1) / nthreads;
    vbatchStackRhsKernel<<<(unsigned) nblocks, nthreads>>>(ctx->m_big, nrhs, ctx->d_rowsys, ctx->d_rowloc, ctx->d_rhsdst,
                                                          ctx->d_rscale, ctx->d_RHSptrs, ctx->d_ldRHS, ctx->d_b);
    gpuErrchk(gpuGetLastError());
    gpuErrchk(gpuDeviceSynchronize());
    printf("\tGPURES stack RHS time = %.4f\n", SuperLU_timer_() - t0);
    return 0;
}

int dvbatch_gpures_unstack(dvbatch_ctx_t *ctx, int batchCount, double **Xptr, int *ldX, int nrhs)
{
    double t0 = SuperLU_timer_();
    gpuErrchk(gpuMemcpy(ctx->d_Xptrs, Xptr, sizeof(double*) * batchCount, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(ctx->d_ldX, ldX, sizeof(int) * batchCount, gpuMemcpyHostToDevice));
    int nthreads = 256;
    int64_t nblocks = ((int64_t) ctx->m_big + nthreads - 1) / nthreads;
    vbatchUnstackXKernel<<<(unsigned) nblocks, nthreads>>>(ctx->m_big, nrhs, ctx->d_rowsys, ctx->d_rowloc, ctx->d_xsrc,
                                                          ctx->d_cscale, ctx->d_Xptrs, ctx->d_ldX, ctx->d_b);
    gpuErrchk(gpuGetLastError());
    gpuErrchk(gpuDeviceSynchronize());
    printf("\tGPURES unstack X time = %.4f\n", SuperLU_timer_() - t0);
    return 0;
}

void dvbatch_gpures_free(dvbatch_ctx_t *ctx)
{
    void **p[] = { (void**)&ctx->d_b, (void**)&ctx->d_rowsys, (void**)&ctx->d_rowloc, (void**)&ctx->d_rhsdst, (void**)&ctx->d_xsrc,
                   (void**)&ctx->d_rscale, (void**)&ctx->d_cscale, (void**)&ctx->d_RHSptrs, (void**)&ctx->d_Xptrs,
                   (void**)&ctx->d_ldRHS, (void**)&ctx->d_ldX };
    for (size_t i = 0; i < sizeof(p) / sizeof(p[0]); ++i) if (*p[i]) { gpuErrchk(gpuFree(*p[i])); *p[i] = NULL; }
}

} /* extern "C" */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C interface
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef TBatchFactorizeWorkspace<float >        sBatchFactorizeWorkspace;
typedef TBatchFactorizeWorkspace<double>        dBatchFactorizeWorkspace;
typedef TBatchFactorizeWorkspace<doublecomplex> zBatchFactorizeWorkspace;

extern "C" {
//single   
int ssparseTreeFactorBatchGPU(sBatchFactorizeWorkspace* ws, sForest_t *sforest)
{
    return sparseTreeFactorBatchGPUT<float>(ws, sforest);
}

sBatchFactorizeWorkspace* sgetBatchFactorizeWorkspace(
    int_t nsupers, int_t ldt, strf3Dpartition_t *trf3Dpartition, sLUstruct_t *LUstruct, 
    gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info
)
{ 
    return getBatchFactorizeWorkspaceT<float>(nsupers, ldt, trf3Dpartition, LUstruct, grid3d, options, stat, info); 
}

void scopyGPULUDataToHost(
    sBatchFactorizeWorkspace* ws, sLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
)
{ 
    copyGPULUDataToHostT<float>(ws, LUstruct, grid3d, SCT_, options, stat); 
}

void sfreeBatchFactorizeWorkspace(sBatchFactorizeWorkspace* ws)
{ 
    freeBatchFactorizeWorkspaceT<float>(ws); 
}

//double  
int dsparseTreeFactorBatchGPU(dBatchFactorizeWorkspace* ws, sForest_t *sforest)
{
    return sparseTreeFactorBatchGPUT<double>(ws, sforest);
}

dBatchFactorizeWorkspace* dgetBatchFactorizeWorkspace(
    int_t nsupers, int_t ldt, dtrf3Dpartition_t *trf3Dpartition, dLUstruct_t *LUstruct, 
    gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info
)
{ 
    return getBatchFactorizeWorkspaceT<double>(nsupers, ldt, trf3Dpartition, LUstruct, grid3d, options, stat, info); 
}

void dcopyGPULUDataToHost(
    dBatchFactorizeWorkspace* ws, dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
)
{ 
    copyGPULUDataToHostT<double>(ws, LUstruct, grid3d, SCT_, options, stat); 
}

void dfreeBatchFactorizeWorkspace(dBatchFactorizeWorkspace* ws)
{ 
    freeBatchFactorizeWorkspaceT<double>(ws); 
}

dBatchFactorizeWorkspace* dgetBatchFactorizeWorkspaceEx(
    int_t nsupers, int_t ldt, dtrf3Dpartition_t *trf3Dpartition, dLUstruct_t *LUstruct, 
    gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info, int convertU
)
{ 
    return getBatchFactorizeWorkspaceT<double>(nsupers, ldt, trf3Dpartition, LUstruct, grid3d, options, stat, info, convertU); 
}

int dbatchDevResSetup(dBatchFactorizeWorkspace* ws, dLUstruct_t *LUstruct, double *avals, int_t nnz, const int64_t *map_host)
{ 
    return batchDevResSetupT<double>(ws, LUstruct, avals, nnz, map_host); 
}

int dbatchDevResRefill(dBatchFactorizeWorkspace* ws, const double *avals, int_t nnz)
{ 
    return batchDevResRefillT<double>(ws, avals, nnz); 
}

void dbatchDevResFree(dBatchFactorizeWorkspace* ws)
{ 
    freeBatchFactorizeWorkspaceT<double>(ws);   /* unregisters the pinned host buffers first */
}

int dbatchDevResSolveSetup(dBatchFactorizeWorkspace* ws, const int64_t *umap_host, int64_t ucnt)
{ 
    return batchDevResSolveSetupT<double>(ws, umap_host, ucnt); 
}

int dbatchDevResSetupA(dBatchFactorizeWorkspace* ws, int nsys, int_t nnz2, const int64_t *posmap,
                       const double *scale2, const int *ent_sys, const int *ent_idx, double anorm)
{ 
    return batchDevResSetupAT<double>(ws, nsys, nnz2, posmap, scale2, ent_sys, ent_idx, anorm); 
}

int dbatchDevResRefillA(dBatchFactorizeWorkspace* ws, int from_device, double **Aptrs, const int *nnzd)
{ 
    return batchDevResRefillAT<double>(ws, from_device, Aptrs, nnzd); 
}

int dbatchDevResAReady(dBatchFactorizeWorkspace* ws)
{ 
    return ws && ws->d_map2 != nullptr; 
}

int dbatchDevResAPrefilled(dBatchFactorizeWorkspace* ws)
{ 
    int f = ws ? ws->a_prefilled : 0;
    if (ws) ws->a_prefilled = 0;
    return f;
}

double dbatchDevResAnorm(dBatchFactorizeWorkspace* ws)
{ 
    return ws ? ws->anorm_cache : 0.0; 
}

int dbatchDevResSolveReady(dBatchFactorizeWorkspace* ws)
{ 
    return ws && ws->solve_ready; 
}

int dbatchDevResSolveRefresh(dBatchFactorizeWorkspace* ws, dLUstruct_t *LUstruct, int_t nsupers, int npcol, int mycol)
{ 
    return batchDevResSolveRefreshT<double>(ws, LUstruct, nsupers, npcol, mycol); 
}

//doublecomplex 
int zsparseTreeFactorBatchGPU(zBatchFactorizeWorkspace* ws, sForest_t *sforest)
{
    return sparseTreeFactorBatchGPUT<doublecomplex>(ws, sforest);
}

zBatchFactorizeWorkspace* zgetBatchFactorizeWorkspace(
    int_t nsupers, int_t ldt, ztrf3Dpartition_t *trf3Dpartition, zLUstruct_t *LUstruct, 
    gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info
)
{ 
    return getBatchFactorizeWorkspaceT<doublecomplex>(nsupers, ldt, trf3Dpartition, LUstruct, grid3d, options, stat, info); 
}

void zcopyGPULUDataToHost(
    zBatchFactorizeWorkspace* ws, zLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
)
{ 
    copyGPULUDataToHostT<doublecomplex>(ws, LUstruct, grid3d, SCT_, options, stat); 
}

void zfreeBatchFactorizeWorkspace(zBatchFactorizeWorkspace* ws)
{ 
    freeBatchFactorizeWorkspaceT<doublecomplex>(ws); 
}

}