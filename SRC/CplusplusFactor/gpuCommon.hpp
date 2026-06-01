#pragma once
// GPU related functions
// Note: gpu_wrapper.h is expected to be included before this file
// (via superlu_defs.h -> gpu_api_utils.h -> gpu_wrapper.h)

#ifdef HAVE_CUDA
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
#ifdef HAVE_MAGMA
  #include "magma.h"
#endif
#endif

#if defined(HAVE_CUDA) || defined(HAVE_HIP)

#define gpuErrchk(ans)                                                                                                 \
{                                                                                                                  \
    gpuAssert((ans), __FILE__, __LINE__);                                                                          \
}

inline void gpuAssert(gpuError_t code, const char *file, int line)
{
    if (code != gpuSuccess)
    {
        fprintf(stderr, "GPUassert: %s(%d) %s %d\n", gpuGetErrorString(code), (int)code, file, line);
        fflush(stderr);
        fprintf(stdout, "GPUassert: %s(%d) %s %d\n", gpuGetErrorString(code), (int)code, file, line);
        fflush(stdout);
        exit(-1);
    }
}

#endif

#ifdef HAVE_CUDA
cudaError_t checkCudaLocal(cudaError_t result);

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