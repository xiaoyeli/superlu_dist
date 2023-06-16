#include <cassert>
#include <algorithm>
#include <cmath>
#include "superlu_defs.h"
#include "superlu_dist_config.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "cublas_v2.h"


#include "lupanels.hpp"

//TODO: make expsilon a enviroment variable 
// #define EPSILON 1e-3
#define EPSILON 1e-6

#if 0
// int checkArr(double *A, double *B, int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         assert(fabs(A[i] - B[i]) <= EPSILON * std::min(fabs(A[i]), fabs(B[i])));
//     }

//     return 0;
// }
#else 
int checkArr(double *A, double *B, int n)
{
    double nrmA = 0;
    for (int i = 0; i < n; i++) 
        nrmA += A[i]*A[i];
    nrmA = sqrt(nrmA);
    for (int i = 0; i < n; i++)
    {
        assert(fabs(A[i] - B[i]) <= EPSILON * nrmA/n );
    }

    return 0;
}
#endif 
lpanelGPU_t lpanel_t::copyToGPU()
{

    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(double) * nzvalSize();


    gpuErrchk(cudaMalloc(&gpuPanel.index, idxSize));
    gpuErrchk(cudaMalloc(&gpuPanel.val, valSize));

    
    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpuPanel.val, val, valSize, cudaMemcpyHostToDevice));

    return gpuPanel;
}

lpanelGPU_t lpanel_t::copyToGPU(void* basePtr)
{

    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(double) * nzvalSize();

    gpuPanel.index = (int_t*) basePtr;
    // cudaMalloc(&gpuPanel.index, idxSize);
    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize; 
    gpuPanel.val = (double *) basePtr; 
    // cudaMalloc(&gpuPanel.val, valSize);
    
    gpuErrchk(cudaMemcpy(gpuPanel.val, val, valSize, cudaMemcpyHostToDevice));

    return gpuPanel;
}

int_t lpanel_t::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(double) * nzvalSize();
    gpuErrchk(cudaMemcpy(val, gpuPanel.val,  valSize, cudaMemcpyDeviceToHost));
}

int_t upanel_t::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(double) * nzvalSize();
    gpuErrchk(cudaMemcpy(val, gpuPanel.val,  valSize, cudaMemcpyDeviceToHost));
}

int upanel_t::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(double) * nzvalSize();
    gpuErrchk(cudaMemcpy(gpuPanel.val, val,  valSize, cudaMemcpyHostToDevice));
}

int lpanel_t::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(double) * nzvalSize();
    gpuErrchk(cudaMemcpy(gpuPanel.val, val,  valSize, cudaMemcpyHostToDevice));
}

upanelGPU_t upanel_t::copyToGPU()
{

    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(double) * nzvalSize();

    gpuErrchk(cudaMalloc(&gpuPanel.index, idxSize));
    gpuErrchk(cudaMalloc(&gpuPanel.val, valSize));

    
    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpuPanel.val, val, valSize, cudaMemcpyHostToDevice));
    return gpuPanel;
}

upanelGPU_t upanel_t::copyToGPU(void* basePtr)
{

    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(double) * nzvalSize();

    gpuPanel.index = (int_t*) basePtr;
    // cudaMalloc(&gpuPanel.index, idxSize);
    gpuErrchk(cudaMemcpy(gpuPanel.index, index, idxSize, cudaMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize; 
    gpuPanel.val = (double *) basePtr; 
    // cudaMalloc(&gpuPanel.val, valSize);
    
    gpuErrchk(cudaMemcpy(gpuPanel.val, val, valSize, cudaMemcpyHostToDevice));

    return gpuPanel;
}


int lpanel_t::checkGPU()
{

    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(double) * nzvalSize();

   std::vector<double> tmpArr(nzvalSize());
    gpuErrchk(cudaMemcpy(tmpArr.data(), gpuPanel.val, valSize, cudaMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());

    return 0;
}

int_t lpanel_t::panelSolveGPU(cublasHandle_t handle, cudaStream_t cuStream,
                              int_t ksupsz,
                              double *DiagBlk, // device pointer
                              int_t LDD)
{
    if (isEmpty())
        return 0;
    double *lPanelStPtr = blkPtrGPU(0); // &val[blkPtrOffset(0)];
    int_t len = nzrows();
    if (haveDiag())
    {
        /* code */
        lPanelStPtr = blkPtrGPU(1); // &val[blkPtrOffset(1)];
        len -= nbrow(0);
    }

    double alpha = 1.0;

    cublasSetStream(handle, cuStream);
    cublasStatus_t cbstatus =
        cublasDtrsm(handle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    len, ksupsz, &alpha, DiagBlk, LDD,
                    lPanelStPtr, LDA());

    return 0;
}

int_t lpanel_t::diagFactorPackDiagBlockGPU(int_t k,
                                           double *UBlk, int_t LDU,     // CPU pointers
                                           double *DiagLBlk, int_t LDD, // CPU pointers
                                           double thresh, int_t *xsup,
                                           superlu_dist_options_t *options,
                                           SuperLUStat_t *stat, int *info)
{
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(double);
    size_t spitch = LDA() * sizeof(double);
    size_t width = kSupSize * sizeof(double);
    size_t height = kSupSize;
    double *val = blkPtrGPU(0);

    gpuErrchk(cudaMemcpy2D(DiagLBlk, dpitch, val, spitch,
                 width, height, cudaMemcpyDeviceToHost));

    // call dgetrf2
    dgstrf2(k, DiagLBlk, LDD, UBlk, LDU,
            thresh, xsup, options, stat, info);

    //copy back to device
    gpuErrchk(cudaMemcpy2D(val, spitch, DiagLBlk, dpitch,
                 width, height, cudaMemcpyHostToDevice));

    return 0;
}

int_t lpanel_t::diagFactorCuSolver(int_t k,
                                     cusolverDnHandle_t cusolverH, cudaStream_t cuStream,
                                    double *dWork, int* dInfo,  // GPU pointers 
                                    double *dDiagBuf, int_t LDD, // GPU pointers
                                    double thresh, int_t *xsup,
                                    superlu_dist_options_t *options,
                                    SuperLUStat_t *stat, int *info)
{
    // cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(double);
    size_t spitch = LDA() * sizeof(double);
    size_t width = kSupSize * sizeof(double);
    size_t height = kSupSize;
    double *val = blkPtrGPU(0);
    // cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork)
    
    // call the cusolver 
    // cublasSetStream(handle, cuStream);
    //  cusolverDnSetStream(cuStream);
    gpuCusolverErrchk(cusolverDnSetStream(cusolverH, cuStream));
    gpuCusolverErrchk(cusolverDnDgetrf(cusolverH, kSupSize, kSupSize, val, LDA(), dWork, NULL, dInfo));

    // Device to Device Copy
    gpuErrchk(cudaMemcpy2DAsync(dDiagBuf, dpitch, val, spitch,
                 width, height, cudaMemcpyDeviceToDevice, cuStream));
    gpuErrchk(cudaStreamSynchronize(cuStream));
    return 0;
}

int_t upanel_t::panelSolveGPU(cublasHandle_t handle, cudaStream_t cuStream,
                              int_t ksupsz, double *DiagBlk, int_t LDD)
{
    if (isEmpty())
        return 0;

    double alpha = 1.0;
    
    cublasSetStream(handle, cuStream);
    cublasStatus_t cbstatus =
        cublasDtrsm(handle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                    ksupsz, nzcols(), &alpha, DiagBlk, LDD,
                    blkPtrGPU(0), LDA());
}

int upanel_t::checkGPU()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(double) * nzvalSize();

    // double *tmpArr = new double[nzvalSize()];
    std::vector<double> tmpArr(nzvalSize());
    gpuErrchk(cudaMemcpy(tmpArr.data(), gpuPanel.val, valSize, cudaMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());
    // delete tmpArr;

    return 0;
}

#if 0
lpanelGPU_t::lpanelGPU_t(lpanel_t &lpanel) : lpanel_CPU(lpanel)
{
    size_t idxSize = sizeof(int_t) * lpanel.indexSize();
    size_t valSize = sizeof(double) * lpanel.nzvalSize();

    
    cudaMalloc(&index, idxSize);
    cudaMemcpy(index, lpanel.index, idxSize, cudaMemcpyHostToDevice);

    cudaMalloc(&val, valSize);
    cudaMemcpy(val, lpanel.val, valSize, cudaMemcpyHostToDevice);
}

int lpanelGPU_t::check(lpanel_t &lpanel)
{
    // both should be simulatnously empty or non empty
    assert(isEmpty() == lpanel.isEmpty());

    size_t valSize = sizeof(double) * lpanel.nzvalSize();

    double *tmpArr = double[lpanel.nzvalSize()];
    cudaMemcpy(tmpArr, val, valSize, cudaMemcpyDeviceToHost);

    int out = checkArr(tmpArr, lpanel.val, lpanel.nzvalSize());
    delete tmpArr;
    return 0;
}

int_t lpanelGPU_t::panelSolve(cublasHandle_t handle, cudaStream_t cuStream,
                              int_t ksupsz, double *DiagBlk, int_t LDD)
{

    if (lpanel_CPU.isEmpty())
        return 0;
    double *lPanelStPtr = &val[lpanel_CPU.blkPtrOffset(0)];
    int_t len = lpanel_CPU.nzrows();
    if (lpanel_CPU.haveDiag())
    {
        /* code */
        lPanelStPtr = &val[lpanel_CPU.blkPtrOffset(1)];
        len -= lpanel_CPU.nbrow(0);
    }

    double alpha = 1.0;
    
    cublasStatus_t cbstatus = cublasDtrsm(handle,
                                          CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                          𝖢𝖴BLAS_OP_𝖭, CUBLAS_DIAG_NON_UNIT,
                                          len, ksupsz, alpha, DiagBlk, LDD,
                                          lPanelStPtr, lpanel_CPU.LDA());

    // if (isEmpty()) return 0;
    // double *lPanelStPtr = blkPtr(0);
    // int_t len = nzrows();
    // if (haveDiag())
    // {
    //     /* code */
    //     lPanelStPtr = blkPtr(1);
    //     len -= nbrow(0);
    // }
    // double alpha = 1.0;
    // superlu_dtrsm("R", "U", "N", "N",
    //               len, ksupsz, alpha, DiagBlk, LDD,
    //               lPanelStPtr, LDA());
}

int_t lpanelGPU_t::diagFactorPackDiagBlock(int_t k,
                                           double *UBlk, int_t LDU,
                                           double *DiagLBlk, int_t LDD,
                                           double thresh, int_t *xsup,
                                           superlu_dist_options_t *options, SuperLUStat_t *stat, int *info)
{
    // pack and transfer to CPU
    // cudaMemcpy2D
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(double);
    size_t spitch = lpanel_CPU.LDA() * sizeof(double);
    size_t width = kSupSize * sizeof(double);
    size_t height = kSupSize;

    cudaMemcpy2D(DiagLBlk, dpitch, val, spitch,
                 width, height, cudaMemcpyDeviceToHost);

    // call dgetrf2
    dgstrf2(k, DiagLBlk, LDD, UBlk, LDU,
            thresh, xsup, options, stat, info);

    //copy back to device
    cudaMemcpy2D(val, spitch, DiagLBlk, dpitch,
                 width, height, cudaMemcpyHostToDevice);

    return 0;
}


upanelGPU_t::upanelGPU_t(upanel_t &upanel) : upanel_CPU(upanel)
{
    size_t idxSize = sizeof(int_t) * upanel.indexSize();
    size_t valSize = sizeof(double) * upanel.nzvalSize();

    cudaMalloc(&index, idxSize);
    cudaMemcpy(index, upanel.index, idxSize, cudaMemcpyHostToDevice);

    cudaMalloc(&val, valSize);
    cudaMemcpy(val, upanel.val, valSize, cudaMemcpyHostToDevice);
}


int upanelGPU_t::check(upanel_t &upanel)
{
    // both should be simulatnously empty or non empty
    assert(isEmpty() == upanel.isEmpty());

    size_t valSize = sizeof(double) * upanel.nzvalSize();

    double *tmpArr = double[upanel.nzvalSize()];
    cudaMemcpy(tmpArr, val, valSize, cudaMemcpyDeviceToHost);

    int out = checkArr(tmpArr, upanel.val, upanel.nzvalSize());
    delete tmpArr;
    return 0;
}



int_t upanelGPU_t::panelSolve(cublasHandle_t handle, cudaStream_t cuStream,
                              int_t ksupsz, double *DiagBlk, int_t LDD)
{
    if (upanel_CPU.isEmpty())
        return 0;

    double alpha = 1.0;
    
    cublasStatus_t cbstatus =
        cublasDtrsm(handle,
                    𝖢𝖴BLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    𝖢𝖴BLAS_OP_𝖭, CUBLAS_DIAG_UNIT,
                    ksupsz, upanel_CPU.nzcols(), alpha, DiagBlk, LDD,
                    val, upanel_CPU.LDA());

    // superlu_dtrsm("L", "L", "N", "U",
    //               ksupsz, nzcols(), 1.0, DiagBlk, LDD, val, LDA());
    return 0;
}
#endif
#endif