#pragma once
#include <cublas_v2.h>

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

template <typename Ftype>
cublasStatus_t myCublasTrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const Ftype *alpha, const Ftype *A, int lda, Ftype *B, int ldb);

template <>
cublasStatus_t myCublasTrsm<double>(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb)
{
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
cublasStatus_t myCublasTrsm<float>(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb)
{
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
cublasStatus_t myCublasTrsm<cuComplex>(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb)
{
    return cublasCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
cublasStatus_t myCublasTrsm<cuDoubleComplex>(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb)
{
    return cublasZtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <typename Ftype>
cublasStatus_t myCublasScal(cublasHandle_t handle, int n, const Ftype *alpha, Ftype *x, int incx);

template <typename Ftype>
cublasStatus_t myCublasAxpy(cublasHandle_t handle, int n, const Ftype *alpha, const Ftype *x, int incx, Ftype *y, int incy);

template <>
cublasStatus_t myCublasScal<double>(cublasHandle_t handle, int n, const double *alpha, double *x, int incx)
{
    return cublasDscal(handle, n, alpha, x, incx);
}

template <>
cublasStatus_t myCublasScal<float>(cublasHandle_t handle, int n, const float *alpha, float *x, int incx)
{
    return cublasSscal(handle, n, alpha, x, incx);
}

template <>
cublasStatus_t myCublasAxpy<double>(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
cublasStatus_t myCublasAxpy<float>(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
cublasStatus_t myCublasScal<cuComplex>(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx)
{
    return cublasCscal(handle, n, alpha, x, incx);
}

template <>
cublasStatus_t myCublasScal<cuDoubleComplex>(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx)
{
    return cublasZscal(handle, n, alpha, x, incx);
}

template <>
cublasStatus_t myCublasAxpy<cuComplex>(cublasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy)
{
    return cublasCaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
cublasStatus_t myCublasAxpy<cuDoubleComplex>(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
{
    return cublasZaxpy(handle, n, alpha, x, incx, y, incy);
}

template <typename Ftype>
cublasStatus_t myCublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const Ftype *alpha, const Ftype *A, int lda, const Ftype *B, int ldb, const Ftype *beta, Ftype *C, int ldc);

template <>
cublasStatus_t myCublasGemm<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t myCublasGemm<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t myCublasGemm<cuComplex>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
{
    return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t myCublasGemm<cuDoubleComplex>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t myCublasGemm<doublecomplex>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const doublecomplex *alpha, const doublecomplex *A, int lda, const doublecomplex *B, int ldb, const doublecomplex *beta, doublecomplex *C, int ldc)
{
    // return cublasZgemm(handle, transa, transb, m, n, k, 
    // alpha, A, lda, B, ldb, beta, C, ldc);
    // cast doublecomplex to cuDoubleComplex
    return cublasZgemm(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A), lda,
        reinterpret_cast<const cuDoubleComplex *>(B), ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C), ldc);
    
}

template <>
cusolverStatus_t myCusolverGetrf<doublecomplex>(
    cusolverDnHandle_t handle, int m, int n, doublecomplex *A, int lda,
    doublecomplex *Workspace, int *devIpiv, int *devInfo)
{
    // return cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
    // cast doublecomplex to cuDoubleComplex
    return cusolverDnZgetrf(
        handle, m, n, reinterpret_cast<cuDoubleComplex *>(A), lda,
        reinterpret_cast<cuDoubleComplex *>(Workspace), devIpiv, devInfo);
}

// now creating the wrappers for the other functions 
template <>
cublasStatus_t myCublasTrsm<doublecomplex>(cublasHandle_t handle,
                                           cublasSideMode_t side, cublasFillMode_t uplo,
                                           cublasOperation_t trans, cublasDiagType_t diag,
                                           int m, int n,
                                           const doublecomplex *alpha,
                                           const doublecomplex *A, int lda,
                                           doublecomplex *B, int ldb) {
    // Your implementation here
    // You can use cublasZtrsm function because it's for cuDoubleComplex type
    return cublasZtrsm(handle, side, uplo, trans, diag, m, n, 
                       reinterpret_cast<const cuDoubleComplex*>(alpha), 
                       reinterpret_cast<const cuDoubleComplex*>(A), lda, 
                       reinterpret_cast<cuDoubleComplex*>(B), ldb);
}

template <>
cublasStatus_t myCublasScal<doublecomplex>(cublasHandle_t handle, int n, 
                                           const doublecomplex *alpha, 
                                           doublecomplex *x, int incx) {
    // Your implementation here
    // You can use cublasZscal function because it's for cuDoubleComplex type
    return cublasZscal(handle, n, reinterpret_cast<const cuDoubleComplex*>(alpha), 
                       reinterpret_cast<cuDoubleComplex*>(x), incx);
}

template <>
cublasStatus_t myCublasAxpy<doublecomplex>(cublasHandle_t handle, int n, 
                                           const doublecomplex *alpha, 
                                           const doublecomplex *x, int incx, 
                                           doublecomplex *y, int incy) {
    // Your implementation here
    // You can use cublasZaxpy function because it's for cuDoubleComplex type
    return cublasZaxpy(handle, n, reinterpret_cast<const cuDoubleComplex*>(alpha), 
                       reinterpret_cast<const cuDoubleComplex*>(x), incx, 
                       reinterpret_cast<cuDoubleComplex*>(y), incy);
}


// cublasStatus_t myCublasScal<doublecomplex> 
// cublasStatus_t myCublasAxpy<doublecomplex>
// cublasStatus_t myCublasGemm<doublecomplex>