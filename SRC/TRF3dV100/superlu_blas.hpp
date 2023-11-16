#pragma once 
#include "lupanels.hpp"


template<typename T>
void superlu_trsm(const char *side, const char *uplo, const char *transa, const char *diag,
               int m, int n, T alpha, const T *A, int lda, T *B, int ldb);

// Specialization for double
template<>
void superlu_trsm<double>(const char *side, const char *uplo, const char *transa, const char *diag,
                       int m, int n, double alpha, const double *A, int lda, double *B, int ldb) {
    superlu_dtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

// Specialization for float
template<>
void superlu_trsm<float>(const char *side, const char *uplo, const char *transa, const char *diag,
                      int m, int n, float alpha, const float *A, int lda, float *B, int ldb) {
    superlu_strsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}


template<typename T>
void superlu_gemm(const char *transa, const char *transb, int m, int n, int k, T alpha,
               const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc);

// Specialization for double
template<>
void superlu_gemm<double>(const char *transa, const char *transb, int m, int n, int k, double alpha,
                       const double *A, int lda, const double *B, int ldb, double beta, 
                       double *C, int ldc) {
    superlu_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specialization for float
template<>
void superlu_gemm<float>(const char *transa, const char *transb, int m, int n, int k, float alpha,
                      const float *A, int lda, const float *B, int ldb, float beta,
                      float *C, int ldc) {
    superlu_sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
