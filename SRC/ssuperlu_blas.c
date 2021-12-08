/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Wrapper functions to call BLAS.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Oak Ridge National Lab
 * December 6, 2020
 */

#include "superlu_sdefs.h"

#ifdef _CRAY
_fcd ftcs = _cptofcd("N", strlen("N"));
_fcd ftcs1 = _cptofcd("L", strlen("L"));
_fcd ftcs2 = _cptofcd("N", strlen("N"));
_fcd ftcs3 = _cptofcd("U", strlen("U"));
#endif

int superlu_sgemm(const char *transa, const char *transb,
                  int m, int n, int k, float alpha, float *a,
                  int lda, float *b, int ldb, float beta, float *c, int ldc)
{
#ifdef _CRAY
    _fcd ftcs = _cptofcd(transa, strlen(transa));
    _fcd ftcs1 = _cptofcd(transb, strlen(transb));
    return SGEMM(ftcs, ftcs1, &m, &n, &k,
                 &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#elif defined(USE_VENDOR_BLAS)
    sgemm_(transa, transb, &m, &n, &k,
           &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
    return 0;
#else
    return sgemm_(transa, transb, &m, &n, &k,
                  &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}

int superlu_strsm(const char *sideRL, const char *uplo,
                  const char *transa, const char *diag,
                  const int m, const int n,
                  const float alpha, const float *a,
                  const int lda, float *b, const int ldb)

{
#if defined(USE_VENDOR_BLAS)
    strsm_(sideRL, uplo, transa, diag,
           &m, &n, &alpha, a, &lda, b, &ldb,
           1, 1, 1, 1);
    return 0;
#else
    return strsm_(sideRL, uplo, transa, diag,
                  &m, &n, &alpha, a, &lda, b, &ldb);
#endif
}

int superlu_sger(const int m, const int n, const float alpha,
                 const float *x, const int incx, const float *y,
                 const int incy, float *a, const int lda)
{
#ifdef _CRAY
    SGER(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
#else
    sger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
#endif

    return 0;
}

int superlu_sscal(const int n, const float alpha, float *x, const int incx)
{
    sscal_(&n, &alpha, x, &incx);
    return 0;
}

int superlu_saxpy(const int n, const float alpha,
    const float *x, const int incx, float *y, const int incy)
{
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    return 0;
}

int superlu_sgemv(const char *trans, const int m,
                  const int n, const float alpha, const float *a,
                  const int lda, const float *x, const int incx,
                  const float beta, float *y, const int incy)
{
#ifdef USE_VENDOR_BLAS
    sgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy, 1);
#else
    sgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#endif
    
    return 0;
}

int superlu_strsv(char *uplo, char *trans, char *diag,
                  int n, float *a, int lda, float *x, int incx)
{
#ifdef _CRAY
    // _fcd ftcs = _cptofcd("N", strlen("N"));
    STRSV(_cptofcd(uplo, strlen(uplo)), _cptofcd(trans, strlen(trans)), _cptofcd(diag, strlen(diag)), 
         &n, a, &lda, x, &incx);
#elif defined (USE_VENDOR_BLAS)
    strsv_(uplo, trans, diag, &n, a, &lda, x, &incx, 1, 1, 1);
#else
    strsv_(uplo, trans, diag, &n, a, &lda, x, &incx);
#endif
    
    return 0;
}

