#pragma once 
// GPU related functions
#ifdef HAVE_CUDA
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
#ifdef HAVE_MAGMA
  #include "magma.h"
#endif 
#endif

#ifdef HAVE_CUDA
cudaError_t checkCudaLocal(cudaError_t result);

#define gpuErrchk(ans)                                                                                                 \
{                                                                                                                  \
    gpuAssert((ans), __FILE__, __LINE__);                                                                          \
}

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s(%d) %s %d\n", cudaGetErrorString(code), (int)code, file, line);
        exit(-1);
    }
}

#define gpuCusolverErrchk(ans)                                                                                         \
{                                                                                                                  \
    gpuCusolverAssert((ans), __FILE__, __LINE__);                                                                  \
}
inline void gpuCusolverAssert(cusolverStatus_t code, const char *file, int line)
{
    if (code != CUSOLVER_STATUS_SUCCESS)
        printf("cuSolverAssert: %d %s %d\n", code, file, line);
}
#endif