#pragma once
// GPU BLAS/Solver wrappers for both CUDA (cuBLAS/cuSolver) and HIP (hipBLAS)
// Headers are included via gpu_wrapper.h (through superlu_defs.h chain)

#if defined(HAVE_CUDA) || defined(HAVE_HIP)

#include <cstring>

/* A doublecomplex is only 8-byte aligned, but cuDoubleComplex and
   hipDoubleComplex are 16-byte aligned types, and the GPU BLAS may read a host
   scalar of that type with an aligned 16-byte load (cuBLAS 13 does in Zscal):
   a doublecomplex scalar at an address equal to 8 mod 16, cast to the GPU type,
   faults inside the BLAS.  The doublecomplex wrappers below therefore pass
   aligned copies of their scalars. */
static inline gpuDoubleComplex slu_gpu_zscalar(const doublecomplex *z)
{
    gpuDoubleComplex c;
    std::memcpy(&c, z, sizeof(c));
    return c;
}

// ============================================================
// cuSolver wrappers — CUDA only (no HIP/rocSOLVER equivalent here)
// ============================================================
#ifdef HAVE_CUDA
#include <cusolverDn.h>

template <typename Ftype>
cusolverStatus_t myCusolverGetrf(cusolverDnHandle_t handle, int m, int n, Ftype *A, int lda, Ftype *Workspace, int *devIpiv, int *devInfo);

template <>
cusolverStatus_t myCusolverGetrf<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo)
{
    return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
cusolverStatus_t myCusolverGetrf<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, int *devIpiv, int *devInfo)
{
    return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
cusolverStatus_t myCusolverGetrf<cuComplex>(cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, cuComplex *Workspace, int *devIpiv, int *devInfo)
{
    return cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
cusolverStatus_t myCusolverGetrf<cuDoubleComplex>(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *Workspace, int *devIpiv, int *devInfo)
{
    return cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
cusolverStatus_t myCusolverGetrf<doublecomplex>(
    cusolverDnHandle_t handle, int m, int n, doublecomplex *A, int lda,
    doublecomplex *Workspace, int *devIpiv, int *devInfo)
{
    return cusolverDnZgetrf(
        handle, m, n, reinterpret_cast<cuDoubleComplex *>(A), lda,
        reinterpret_cast<cuDoubleComplex *>(Workspace), devIpiv, devInfo);
}
#endif /* HAVE_CUDA */
/* HIP note: the diagonal-block factorization is done by a custom on-device
   kernel (diagBlkLU_kernel in lupanels_GPU_impl.hpp) with tiny/zero-pivot
   replacement, so no rocSOLVER/rocBLAS solver wrapper is needed here.  We rely
   only on hipBLAS (TRSM/GEMM) via the generic wrappers below. */


// ============================================================
// Generic BLAS wrapper declarations using gpu_wrapper.h types
// ============================================================

template <typename Ftype>
gpublasStatus_t myCublasTrsm(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const Ftype *alpha,
    const Ftype *A, int lda, Ftype *B, int ldb);

template <typename Ftype>
gpublasStatus_t myCublasScal(gpublasHandle_t handle, int n,
    const Ftype *alpha, Ftype *x, int incx);

template <typename Ftype>
gpublasStatus_t myCublasAxpy(gpublasHandle_t handle, int n,
    const Ftype *alpha, const Ftype *x, int incx, Ftype *y, int incy);

template <typename Ftype>
gpublasStatus_t myCublasGemm(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const Ftype *alpha, const Ftype *A, int lda,
    const Ftype *B, int ldb,
    const Ftype *beta, Ftype *C, int ldc);


// ============================================================
// CUDA specializations
// ============================================================
#ifdef HAVE_CUDA

/* --- myCublasTrsm --- */
template <>
gpublasStatus_t myCublasTrsm<double>(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const double *alpha,
    const double *A, int lda, double *B, int ldb)
{
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
gpublasStatus_t myCublasTrsm<float>(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const float *alpha,
    const float *A, int lda, float *B, int ldb)
{
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
gpublasStatus_t myCublasTrsm<cuComplex>(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const cuComplex *alpha,
    const cuComplex *A, int lda, cuComplex *B, int ldb)
{
    return cublasCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
gpublasStatus_t myCublasTrsm<cuDoubleComplex>(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb)
{
    return cublasZtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
gpublasStatus_t myCublasTrsm<doublecomplex>(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const doublecomplex *alpha,
    const doublecomplex *A, int lda, doublecomplex *B, int ldb)
{
    const gpuDoubleComplex a = slu_gpu_zscalar(alpha);
    return cublasZtrsm(handle, side, uplo, trans, diag, m, n,
        &a,
        reinterpret_cast<const cuDoubleComplex *>(A), lda,
        reinterpret_cast<cuDoubleComplex *>(B), ldb);
}

/* --- myCublasScal --- */
template <>
gpublasStatus_t myCublasScal<double>(gpublasHandle_t handle, int n,
    const double *alpha, double *x, int incx)
{
    return cublasDscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myCublasScal<float>(gpublasHandle_t handle, int n,
    const float *alpha, float *x, int incx)
{
    return cublasSscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myCublasScal<cuComplex>(gpublasHandle_t handle, int n,
    const cuComplex *alpha, cuComplex *x, int incx)
{
    return cublasCscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myCublasScal<cuDoubleComplex>(gpublasHandle_t handle, int n,
    const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx)
{
    return cublasZscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myCublasScal<doublecomplex>(gpublasHandle_t handle, int n,
    const doublecomplex *alpha, doublecomplex *x, int incx)
{
    const gpuDoubleComplex a = slu_gpu_zscalar(alpha);
    return cublasZscal(handle, n,
        &a,
        reinterpret_cast<cuDoubleComplex *>(x), incx);
}

/* --- myCublasAxpy --- */
template <>
gpublasStatus_t myCublasAxpy<double>(gpublasHandle_t handle, int n,
    const double *alpha, const double *x, int incx, double *y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myCublasAxpy<float>(gpublasHandle_t handle, int n,
    const float *alpha, const float *x, int incx, float *y, int incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myCublasAxpy<cuComplex>(gpublasHandle_t handle, int n,
    const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy)
{
    return cublasCaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myCublasAxpy<cuDoubleComplex>(gpublasHandle_t handle, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx,
    cuDoubleComplex *y, int incy)
{
    return cublasZaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myCublasAxpy<doublecomplex>(gpublasHandle_t handle, int n,
    const doublecomplex *alpha, const doublecomplex *x, int incx,
    doublecomplex *y, int incy)
{
    const gpuDoubleComplex a = slu_gpu_zscalar(alpha);
    return cublasZaxpy(handle, n,
        &a,
        reinterpret_cast<const cuDoubleComplex *>(x), incx,
        reinterpret_cast<cuDoubleComplex *>(y), incy);
}

/* --- myCublasGemm --- */
template <>
gpublasStatus_t myCublasGemm<double>(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const double *alpha, const double *A, int lda,
    const double *B, int ldb,
    const double *beta, double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myCublasGemm<float>(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const float *alpha, const float *A, int lda,
    const float *B, int ldb,
    const float *beta, float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myCublasGemm<cuComplex>(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    const cuComplex *beta, cuComplex *C, int ldc)
{
    return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myCublasGemm<cuDoubleComplex>(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myCublasGemm<doublecomplex>(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const doublecomplex *alpha, const doublecomplex *A, int lda,
    const doublecomplex *B, int ldb,
    const doublecomplex *beta, doublecomplex *C, int ldc)
{
    const gpuDoubleComplex a = slu_gpu_zscalar(alpha);
    const gpuDoubleComplex b = slu_gpu_zscalar(beta);
    return cublasZgemm(handle, transa, transb, m, n, k,
        &a,
        reinterpret_cast<const cuDoubleComplex *>(A), lda,
        reinterpret_cast<const cuDoubleComplex *>(B), ldb,
        &b,
        reinterpret_cast<cuDoubleComplex *>(C), ldc);
}

#elif defined(HAVE_HIP) /* ============ HIP specializations ============ */

/* --- myCublasTrsm for HIP --- */
template <>
gpublasStatus_t myCublasTrsm<double>(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const double *alpha,
    const double *A, int lda, double *B, int ldb)
{
    return hipblasDtrsm(handle, side, uplo, trans, diag, m, n, alpha,
                        const_cast<double *>(A), lda, B, ldb);
}

template <>
gpublasStatus_t myCublasTrsm<float>(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const float *alpha,
    const float *A, int lda, float *B, int ldb)
{
    return hipblasStrsm(handle, side, uplo, trans, diag, m, n, alpha,
                        const_cast<float *>(A), lda, B, ldb);
}

template <>
gpublasStatus_t myCublasTrsm<doublecomplex>(gpublasHandle_t handle,
    gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n, const doublecomplex *alpha,
    const doublecomplex *A, int lda, doublecomplex *B, int ldb)
{
    const gpuDoubleComplex a = slu_gpu_zscalar(alpha);
    return hipblasZtrsm(handle, side, uplo, trans, diag, m, n,
        &a,
        reinterpret_cast<gpuDoubleComplex *>(const_cast<doublecomplex *>(A)), lda,
        reinterpret_cast<gpuDoubleComplex *>(B), ldb);
}

/* --- myCublasScal for HIP --- */
template <>
gpublasStatus_t myCublasScal<double>(gpublasHandle_t handle, int n,
    const double *alpha, double *x, int incx)
{
    return hipblasDscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myCublasScal<float>(gpublasHandle_t handle, int n,
    const float *alpha, float *x, int incx)
{
    return hipblasSscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myCublasScal<doublecomplex>(gpublasHandle_t handle, int n,
    const doublecomplex *alpha, doublecomplex *x, int incx)
{
    const gpuDoubleComplex a = slu_gpu_zscalar(alpha);
    return hipblasZscal(handle, n,
        &a,
        reinterpret_cast<gpuDoubleComplex *>(x), incx);
}

/* --- myCublasAxpy for HIP --- */
template <>
gpublasStatus_t myCublasAxpy<double>(gpublasHandle_t handle, int n,
    const double *alpha, const double *x, int incx, double *y, int incy)
{
    return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myCublasAxpy<float>(gpublasHandle_t handle, int n,
    const float *alpha, const float *x, int incx, float *y, int incy)
{
    return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myCublasAxpy<doublecomplex>(gpublasHandle_t handle, int n,
    const doublecomplex *alpha, const doublecomplex *x, int incx,
    doublecomplex *y, int incy)
{
    const gpuDoubleComplex a = slu_gpu_zscalar(alpha);
    return hipblasZaxpy(handle, n,
        &a,
        reinterpret_cast<const gpuDoubleComplex *>(x), incx,
        reinterpret_cast<gpuDoubleComplex *>(y), incy);
}

/* --- myCublasGemm for HIP --- */
template <>
gpublasStatus_t myCublasGemm<double>(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const double *alpha, const double *A, int lda,
    const double *B, int ldb,
    const double *beta, double *C, int ldc)
{
    return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myCublasGemm<float>(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const float *alpha, const float *A, int lda,
    const float *B, int ldb,
    const float *beta, float *C, int ldc)
{
    return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myCublasGemm<doublecomplex>(gpublasHandle_t handle,
    gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const doublecomplex *alpha, const doublecomplex *A, int lda,
    const doublecomplex *B, int ldb,
    const doublecomplex *beta, doublecomplex *C, int ldc)
{
    const gpuDoubleComplex a = slu_gpu_zscalar(alpha);
    const gpuDoubleComplex b = slu_gpu_zscalar(beta);
    return hipblasZgemm(handle, transa, transb, m, n, k,
        &a,
        reinterpret_cast<const gpuDoubleComplex *>(A), lda,
        reinterpret_cast<const gpuDoubleComplex *>(B), ldb,
        &b,
        reinterpret_cast<gpuDoubleComplex *>(C), ldc);
}

#endif /* HAVE_CUDA / HAVE_HIP */

#endif /* HAVE_CUDA || HAVE_HIP */
