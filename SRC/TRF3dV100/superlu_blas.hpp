#pragma once 
#include "superlu_defs.h"
#include "superlu_zdefs.h"
#include "superlu_sdefs.h"
//#include "lupanels.hpp"

template<typename T>
void superlu_trsm(const char *side, const char *uplo, const char *transa, const char *diag,
               int m, int n, T alpha, T *A, int lda, T *B, int ldb);

// Specialization for double
template<>
void superlu_trsm<double>(const char *side, const char *uplo, const char *transa, const char *diag,
                       int m, int n, double alpha, double *A, int lda, double *B, int ldb) {
    superlu_dtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

// Specialization for float
template<>
void superlu_trsm<float>(const char *side, const char *uplo, const char *transa, const char *diag,
                      int m, int n, float alpha, float *A, int lda, float *B, int ldb) {
    superlu_strsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}


template<typename T>
void superlu_gemm(const char *transa, const char *transb, int m, int n, int k, T alpha,
               T *A, int lda, T *B, int ldb, T beta, T *C, int ldc);

// Specialization for double
template<>
void superlu_gemm<double>(const char *transa, const char *transb, int m, int n, int k, double alpha,
                       double *A, int lda, double *B, int ldb, double beta, 
                       double *C, int ldc) {
    superlu_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specialization for float
template<>
void superlu_gemm<float>(const char *transa, const char *transb, int m, int n, int k, float alpha,
                      float *A, int lda, float *B, int ldb, float beta,
                      float *C, int ldc) {
    superlu_sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// create variant for superlu_dscal, superlu_sscal, superlu_cscal, superlu_zscal
template<typename T>
void superlu_scal(int n, T alpha, T *x, int incx);

// Specialization for double
template<>
void superlu_scal<double>(int n, double alpha, double *x, int incx) {
    superlu_dscal(n, alpha, x, incx);
}

// Specialization for float
template<>
void superlu_scal<float>(int n, float alpha, float *x, int incx) {
    superlu_sscal(n, alpha, x, incx);
}


// Specialization for double complex

// create variant for superlu_daxpy, superlu_saxpy, superlu_caxpy, superlu_zaxpy
template<typename T>
void superlu_axpy(int n, T alpha, T *x, int incx, T *y, int incy);

// Specialization for double
template<>
void superlu_axpy<double>(int n, double alpha, double *x, int incx, double *y, int incy) {
    superlu_daxpy(n, alpha, x, incx, y, incy);
}

// Specialization for float
template<>
void superlu_axpy<float>(int n, float alpha, float *x, int incx, float *y, int incy) {
    superlu_saxpy(n, alpha, x, incx, y, incy);
}

template<typename T>
void superlu_ger(const int m, const int n, const T alpha,
                 const T *x, const int incx, const T *y,
                 const int incy, T *a, const int lda);

// Specialization for double
template<>
void superlu_ger<double>(const int m, const int n, const double alpha,
                         const double *x, const int incx, const double *y,
                         const int incy, double *a, const int lda) {
    superlu_dger(m, n, alpha, x, incx, y, incy, a, lda);
}

// Specialization for float
template<>
void superlu_ger<float>(const int m, const int n, const float alpha,
                        const float *x, const int incx, const float *y,
                        const int incy, float *a, const int lda) {
    superlu_sger(m, n, alpha, x, incx, y, incy, a, lda);
}




 
// #ifdef enable_complex16
// Specialization for double complex
template<>
void superlu_axpy<doublecomplex>(int n, doublecomplex alpha, doublecomplex *x, int incx, doublecomplex *y, int incy) {
    superlu_zaxpy(n, alpha, x, incx, y, incy);
}

template<>
void superlu_scal<doublecomplex>(int n, doublecomplex alpha, doublecomplex *x, int incx) {
    superlu_zscal(n, alpha, x, incx);
}   


template<>
void superlu_gemm<doublecomplex>(const char *transa, const char *transb, int m, int n, int k, doublecomplex alpha,
                      doublecomplex *A, int lda, doublecomplex *B, int ldb, doublecomplex beta,
                      doublecomplex *C, int ldc) {
    superlu_zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template<>
void superlu_trsm<doublecomplex>(const char *side, const char *uplo, const char *transa, const char *diag,
                      int m, int n, doublecomplex alpha, doublecomplex *A, int lda, doublecomplex *B, int ldb) {
    superlu_ztrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

template<>
void superlu_ger<doublecomplex>(const int m, const int n, const doublecomplex alpha,
                         const doublecomplex *x, const int incx, const doublecomplex *y,
                         const int incy, doublecomplex *a, const int lda) {
    superlu_zger(m, n, alpha, x, incx, y, incy, a, lda);
}
// #endif 
