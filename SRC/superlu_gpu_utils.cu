#include "superlu_defs.h"
#include "gpu_api_utils.h"

// /*error reporting functions */
// gpuError_t checkGPU(gpuError_t result)
// {
// #if defined(DEBUG) || defined(_DEBUG)
//     if (result != gpuSuccess) {
//         fprintf(stderr, "GPU Runtime Error: %s\n", gpuGetErrorString(result));
//         assert(result == gpuSuccess);
//     }
// #endif
//     return result;
// }

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
    int thread_id = threadIdx_x;
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

