#pragma once
#include <cassert>
#include <algorithm>
#include <cmath>
#include "superlu_defs.h"
#include "superlu_dist_config.h"
// Note: gpu_wrapper.h (with gpuMalloc, gpuMemcpy, etc.) is included via
// superlu_defs.h -> gpu_api_utils.h -> gpu_wrapper.h

#define EPSILON 1e-3
#include "luAuxStructTemplated.hpp"  // threshPivValType and other type helpers
#include "cublas_cusolver_wrappers.hpp"
#include "xgstrf2.hpp"   // type-generic diagonal block factorization (replaces dgstrf2)

// #include "lupanels.hpp"

#include <complex>

template<typename T>
int checkArr(const T *A, const T *B, int n)
{
    double nrmA = 0;
    for (int i = 0; i < n; i++) {
        // For complex numbers, std::norm gives the squared magnitude.
        nrmA += sqnorm(A[i]);
    }
    nrmA = std::sqrt(nrmA);

    for (int i = 0; i < n; i++) {
        // Use std::abs for both real and complex numbers to get the magnitude.
        // assert(std::abs(A[i] - B[i]) <= EPSILON * nrmA / n);
        assert(std::sqrt(sqnorm(A[i] - B[i])) <= EPSILON * nrmA / n);
    }

    return 0;
}

template <typename T>
xlpanelGPU_t<T> xlpanel_t<T>::copyToGPU()
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    gpuErrchk(gpuMalloc(&gpuPanel.index, idxSize));
    gpuErrchk(gpuMalloc(&gpuPanel.val, valSize));

    gpuErrchk(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
xlpanelGPU_t<T> xlpanel_t<T>::copyToGPU(void* basePtr)
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    gpuPanel.index = (int_t*) basePtr;
    gpuErrchk(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize;
    gpuPanel.val = (T *) basePtr;

    gpuErrchk(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
int_t xlpanel_t<T>::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    gpuErrchk(gpuMemcpy(val, gpuPanel.val,  valSize, gpuMemcpyDeviceToHost));
    return 0;
}

template <typename T>
int_t xupanel_t<T>::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    gpuErrchk(gpuMemcpy(val, gpuPanel.val,  valSize, gpuMemcpyDeviceToHost));
    return 0;
}

template <typename T>
int xupanel_t<T>::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    gpuErrchk(gpuMemcpy(gpuPanel.val, val,  valSize, gpuMemcpyHostToDevice));
}

template <typename T>
int xlpanel_t<T>::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    gpuErrchk(gpuMemcpy(gpuPanel.val, val,  valSize, gpuMemcpyHostToDevice));
}

template <typename T>
xupanelGPU_t<T> xupanel_t<T>::copyToGPU()
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    gpuErrchk(gpuMalloc(&gpuPanel.index, idxSize));
    gpuErrchk(gpuMalloc(&gpuPanel.val, valSize));

    gpuErrchk(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));
    gpuErrchk(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));
    return gpuPanel;
}


template <typename T>
xupanelGPU_t<T> xupanel_t<T>::copyToGPU(void* basePtr)
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    gpuPanel.index = (int_t*) basePtr;
    gpuErrchk(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize;
    gpuPanel.val = (T *) basePtr;

    gpuErrchk(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
int xlpanel_t<T>::checkGPU()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(T) * nzvalSize();

    std::vector<T> tmpArr(nzvalSize());
    gpuErrchk(gpuMemcpy(tmpArr.data(), gpuPanel.val, valSize, gpuMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());

    return 0;
}

template <typename T>
int_t xlpanel_t<T>::panelSolveGPU(gpublasHandle_t handle, gpuStream_t cuStream,
                              int_t ksupsz,
                              T *DiagBlk, // device pointer
                              int_t LDD)
{
    if (isEmpty())
        return 0;
    T *lPanelStPtr = blkPtrGPU(0); // &val[blkPtrOffset(0)];
    int_t len = nzrows();
    if (haveDiag())
    {
        lPanelStPtr = blkPtrGPU(1); // &val[blkPtrOffset(1)];
        len -= nbrow(0);
    }

    T alpha = one<T>();

    gpublasSetStream(handle, cuStream);
    gpublasStatus_t cbstatus =
        myCublasTrsm<T>(handle,
                    GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER,
                    GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT,
                    len, ksupsz, &alpha, DiagBlk, LDD,
                    lPanelStPtr, LDA());

    return 0;
}

template <typename T>
int_t xlpanel_t<T>::diagFactorPackDiagBlockGPU(int_t k,
                                           T *UBlk, int_t LDU,     // CPU pointers
                                           T *DiagLBlk, int_t LDD, // CPU pointers
                                           threshPivValType<T> thresh, int_t *xsup,
                                           superlu_dist_options_t *options,
                                           SuperLUStat_t *stat, int *info)
{
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(T);
    size_t spitch = LDA() * sizeof(T);
    size_t width = kSupSize * sizeof(T);
    size_t height = kSupSize;
    T *val = blkPtrGPU(0);

    /* Use a regular heap buffer for the D2H copy and xgstrf2 computation.
       On AMD ROCm, gpuMallocHost (hipHostMalloc) returns a pointer in the GPU
       virtual address space.  CPU random-access operations (like the inner loops
       of xgstrf2) on such Write-Combining pinned memory cause SIGBUS / crashes.
       We therefore copy from GPU into a plain std::vector on the CPU heap, run
       xgstrf2 there, then memcpy the result into DiagLBlk (which may be pinned
       by the caller) before doing the H2D DMA back to the GPU. */
    std::vector<T> localDiagBuf(kSupSize * LDD);

    gpuErrchk(gpuMemcpy2D(localDiagBuf.data(), dpitch, val, spitch,
                 width, height, gpuMemcpyDeviceToHost));

    // xgstrf2 writes each U row into BlockUfactor (UBlk) AND leaves U in
    // diagBlk's upper triangle (rows are never modified after being read).
    // So localDiagBuf ends up in standard LAPACK combined L/U format, which is
    // exactly what the downstream TRSM panel solves expect.
    // However xgstrf2 unconditionally dereferences BlockUfactor (reads pivot
    // from ujrow[0], writes U rows), so passing NULL causes a crash or silent
    // corruption.  Allocate a scratch buffer when the caller passed NULL.
    T *tmpUBlk = UBlk;
    bool tmpOwned = false;
    if (!tmpUBlk) {
        tmpUBlk = new T[kSupSize * kSupSize]();
        LDU = kSupSize;
        tmpOwned = true;
    }

    /* Use a local info variable for xgstrf2.  The CUDA path (diagFactorCuSolver)
       calls cusolverDnDgetrf with devIpiv=NULL and NEVER reads the GPU-side dInfo,
       so zero pivots detected inside the dense diagonal-block factorization do NOT
       propagate to the caller's *info.  Mirror that behaviour here: zero pivots
       recorded by xgstrf2 stay local and do not set the global factorisation status.
       Callers that need to know about local singularity can inspect local_info. */
    int local_info = 0;
    xgstrf2(k, localDiagBuf.data(), LDD, tmpUBlk, LDU,
            thresh, xsup, options, stat, &local_info);

    if (tmpOwned) { delete[] tmpUBlk; tmpUBlk = nullptr; }

    /* Copy xgstrf2 result into DiagLBlk so the caller can see the factored block.
       DiagLBlk may be pinned memory (gpuMallocHost), but a plain memcpy into it
       is fine — it is only the random-access CPU read/write pattern of xgstrf2
       that is unsafe on WC pinned pages. */
    memcpy(DiagLBlk, localDiagBuf.data(), (size_t)kSupSize * LDD * sizeof(T));

    /* Use row-by-row gpuMemcpy (1D) instead of gpuMemcpy2D H2D to avoid
       a hang in hipMemcpy2D HostToDevice on AMD CDNA (gfx908, ROCm 6.2.x).
       DiagLBlk is pinned (gpuMallocHost) for reliable DMA. */
    for (size_t row = 0; row < height; row++) {
        gpuErrchk(gpuMemcpy((char*)val + row * spitch,
                             (char*)DiagLBlk + row * dpitch,
                             width, gpuMemcpyHostToDevice));
    }

    return 0;
}

#ifdef HAVE_CUDA
template <typename T>
int_t xlpanel_t<T>::diagFactorCuSolver(int_t k,
                                     cusolverDnHandle_t cusolverH, cudaStream_t cuStream,
                                    T *dWork, int* dInfo,  // GPU pointers
                                    T *dDiagBuf, int_t LDD, // GPU pointers
                                    threshPivValType<T> thresh, int_t *xsup,
                                    superlu_dist_options_t *options,
                                    SuperLUStat_t *stat, int *info)
{
    // cudaStream_t stream = NULL;
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(T);
    size_t spitch = LDA() * sizeof(T);
    size_t width = kSupSize * sizeof(T);
    size_t height = kSupSize;
    T *val = blkPtrGPU(0);

    gpuCusolverErrchk(cusolverDnSetStream(cusolverH, cuStream));
    gpuCusolverErrchk(myCusolverGetrf<T>(cusolverH, kSupSize, kSupSize, val, LDA(), dWork, NULL, dInfo));

    gpuErrchk(gpuMemcpy2DAsync(dDiagBuf, dpitch, val, spitch,
                 width, height, gpuMemcpyDeviceToDevice, cuStream));
    gpuErrchk(gpuStreamSynchronize(cuStream));
    return 0;
}
#endif /* HAVE_CUDA */

#if defined(HAVE_HIP)
/* ---- device helpers for the diagonal-block LU kernel (self-contained) ---- */
template <typename T> __device__ inline double dgf_sqnorm(const T &v);
template <> __device__ inline double dgf_sqnorm<double>(const double &v) { return v * v; }
template <> __device__ inline double dgf_sqnorm<float>(const float &v) { return (double)v * (double)v; }
template <> __device__ inline double dgf_sqnorm<doublecomplex>(const doublecomplex &v) { return v.r * v.r + v.i * v.i; }

/* setDiagToThreshold equivalents, matching the host versions in
   luAuxStructTemplated.hpp (real: sign-preserving; complex: {thresh,0}). */
template <typename T> __device__ inline void dgf_setThresh(T *p, double thresh);
template <> __device__ inline void dgf_setThresh<double>(double *p, double t) { *p = (*p < 0) ? -t : t; }
template <> __device__ inline void dgf_setThresh<float>(float *p, double t) { *p = (*p < 0) ? -(float)t : (float)t; }
template <> __device__ inline void dgf_setThresh<doublecomplex>(doublecomplex *p, double t) { p->r = t; p->i = 0.0; }

/* Single-thread-block, no-pivot LU of an n x n diagonal block (column-major,
   leading dimension lda), performed in place — the GPU-native equivalent of
   the CPU xgstrf2.  Replaces exactly-zero pivots (and, when replaceTiny is
   set, tiny pivots |d|<thresh) with a thresholded value, exactly like
   xgstrf2 / SuperLU's CPU diagonal factorization.  This is required because
   rocsolver_*getrf_npvt divides by zero pivots and yields NaN, whereas the
   CUDA cuSolver path tolerates them.  tinyCount (device int, may be NULL) is
   atomically incremented for each replaced pivot. */
template <typename T>
__global__ void diagBlkLU_kernel(T *A, int n, int lda, double thresh,
                                 int replaceTiny, int *tinyCount)
{
    int tid = threadIdx.x;
    int nt  = blockDim.x;
    for (int j = 0; j < n; ++j)
    {
        if (tid == 0)
        {
            T *pa = &A[(size_t)j * lda + j];
            double mag = dgf_sqnorm<T>(*pa);
            if (mag == 0.0 || (replaceTiny && sqrt(mag) < thresh))
            {
                dgf_setThresh<T>(pa, thresh);
                if (tinyCount) atomicAdd(tinyCount, 1);
            }
        }
        __syncthreads();

        T pivot = A[(size_t)j * lda + j];
        /* scale column j below the diagonal by 1/pivot */
        for (int i = j + 1 + tid; i < n; i += nt)
            A[(size_t)j * lda + i] = A[(size_t)j * lda + i] / pivot;
        __syncthreads();

        /* rank-1 update of trailing submatrix: A[i,l] -= A[i,j]*A[j,l] */
        int m = n - j - 1;
        for (int idx = tid; idx < m * m; idx += nt)
        {
            int ii = j + 1 + (idx % m);
            int ll = j + 1 + (idx / m);
            A[(size_t)ll * lda + ii] =
                A[(size_t)ll * lda + ii] - A[(size_t)j * lda + ii] * A[(size_t)ll * lda + j];
        }
        __syncthreads();
    }
}

/* HIP equivalent of diagFactorCuSolver.

   The diagonal block is factored entirely on the GPU by diagBlkLU_kernel
   (no-pivot LU with tiny/zero-pivot replacement, mirroring SuperLU's CPU
   xgstrf2).  We do NOT use rocsolver_*getrf_npvt here: although it is the
   direct analogue of cusolverDn*getrf, it divides by zero pivots and emits
   NaN for the (near-)singular diagonal blocks that arise from ill-conditioned
   matrices, whereas cuSolver tolerates them.  The custom kernel reproduces
   cuSolver/xgstrf2's tolerant behaviour while staying fully on-device.

   The factorization and the D2D copy into dDiagBuf both run on cuStream, so
   they are correctly ordered with the rest of the panel's GPU work. */
template <typename T>
int_t xlpanel_t<T>::diagFactorGPU(int_t k,
                                  gpuStream_t cuStream,
                                  T *dDiagBuf, int_t LDD, // GPU pointers
                                  threshPivValType<T> thresh, int_t *xsup,
                                  superlu_dist_options_t *options,
                                  SuperLUStat_t *stat, int *info)
{
    if (isEmpty()) return 0;

    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(T);
    size_t spitch = LDA() * sizeof(T);
    size_t width  = kSupSize * sizeof(T);
    size_t height = kSupSize;
    T *val = blkPtrGPU(0);

    int replaceTiny = (options->ReplaceTinyPivot == YES) ? 1 : 0;
    int nThreads = kSupSize < 32 ? (kSupSize > 0 ? kSupSize * 8 : 1) : 256;
    if (nThreads > 256) nThreads = 256;
    diagBlkLU_kernel<T><<<1, nThreads, 0, cuStream>>>(
        val, kSupSize, (int)LDA(), (double)thresh, replaceTiny, /*tinyCount=*/nullptr);

    /* D2D copy to dDiagBuf on the same stream — correctly ordered after the LU. */
    gpuErrchk(gpuMemcpy2DAsync(dDiagBuf, dpitch, val, spitch,
                               width, height, gpuMemcpyDeviceToDevice, cuStream));
    gpuErrchk(gpuStreamSynchronize(cuStream));
    return 0;
}
#endif /* HAVE_HIP */

template <typename T>
int_t xupanel_t<T>::panelSolveGPU(gpublasHandle_t handle, gpuStream_t cuStream,
                              int_t ksupsz, T *DiagBlk, int_t LDD)
{
    if (isEmpty())
        return 0;

    T alpha = one<T>();

    gpublasSetStream(handle, cuStream);
    gpublasStatus_t cbstatus =
        myCublasTrsm<T>(handle,
                    GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER,
                    GPUBLAS_OP_N, GPUBLAS_DIAG_UNIT,
                    ksupsz, nzcols(), &alpha, DiagBlk, LDD,
                    blkPtrGPU(0), LDA());

    return 0;
}

template <typename T>
int xupanel_t<T>::checkGPU()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(T) * nzvalSize();

    std::vector<T> tmpArr(nzvalSize());
    gpuErrchk(gpuMemcpy(tmpArr.data(), gpuPanel.val, valSize, gpuMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());

    return 0;
}
