/*! @file
 * \brief Precision-independent utility routines for GPU
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology, Oak Ridge National Laboratory
 * December 12, 2021 version 7.2.0
 *
 * Last update: December 12, 2021  remove dependence on CUB/scan
 * </pre>
 */
 
#include <cuda.h>
#include <cuda_runtime.h>

/*error reporting functions */
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

__device__ int dnextpow2(int v)

{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}


typedef int pfx_dtype ; 
__device__ void incScan(pfx_dtype *inOutArr, pfx_dtype *temp, int n)
{
    // extern __shared__ pfx_dtype temp[];
    int n_original = n;
    n = (n & (n - 1)) == 0? n: dnextpow2(n);
    int thread_id = threadIdx.x;
    int offset = 1;
    if(2*thread_id  < n_original)
        temp[2*thread_id] = inOutArr[2*thread_id]; 
    else 
        temp[2*thread_id] =0;


    if(2*thread_id+1 <n_original)
        temp[2*thread_id+1] = inOutArr[2*thread_id+1];
    else 
        temp[2*thread_id+1] =0;
    
    for (int d = n>>1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (thread_id < d)
        {
            int ai = offset*(2*thread_id+1)-1;
            int bi = offset*(2*thread_id+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    if (thread_id == 0) { temp[n - 1] = 0; } 
    for (int d = 1; d < n; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (thread_id < d)
        {
            int ai = offset*(2*thread_id+1)-1;
            int bi = offset*(2*thread_id+2)-1;
            pfx_dtype t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    if(2*thread_id  < n_original)
    inOutArr[2*thread_id] = temp[2*thread_id]+ inOutArr[2*thread_id]; // write results to device memory
    if(2*thread_id+1  < n_original)
    inOutArr[2*thread_id+1] = temp[2*thread_id+1]+ inOutArr[2*thread_id+1];
    __syncthreads();
    
} /* end incScan */ 


#if 0 // Not used
__global__ void gExScan(pfx_dtype *inArr, int n)
{
    extern __shared__ pfx_dtype temp[];
    incScan(inArr, temp, n);
    
}
#endif
