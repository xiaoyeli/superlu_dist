#pragma once 
#include <cublas_v2.h>

template <typename Ftype>
cublasStatus_t myCublasTrsm(cublasHandle_t handle, ... /* other parameters */, Ftype alpha, Ftype *DiagBlk, ... /* other parameters */);

template <>
cublasStatus_t myCublasTrsm<double>(cublasHandle_t handle, ... /* other parameters */, double alpha, double *DiagBlk, ... /* other parameters */) {
    return cublasDtrsm(handle, ... /* other parameters */, alpha, DiagBlk, ... /* other parameters */);
}

template <>
cublasStatus_t myCublasTrsm<float>(cublasHandle_t handle, ... /* other parameters */, float alpha, float *DiagBlk, ... /* other parameters */) {
    return cublasStrsm(handle, ... /* other parameters */, alpha, DiagBlk, ... /* other parameters */);
}

template <typename Ftype>
cusolverStatus_t myCusolverGetrf(cusolverDnHandle_t handle, int m, int n, Ftype *A, int lda, Ftype *Workspace, int *devIpiv, int *devInfo);

template <>
cusolverStatus_t myCusolverGetrf<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo) {
    return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
cusolverStatus_t myCusolverGetrf<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, int *devIpiv, int *devInfo) {
    return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}
