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

#include "superlu_ddefs.h"

#ifdef _CRAY
_fcd ftcs = _cptofcd("N", strlen("N"));
_fcd ftcs1 = _cptofcd("L", strlen("L"));
_fcd ftcs2 = _cptofcd("N", strlen("N"));
_fcd ftcs3 = _cptofcd("U", strlen("U"));
#endif

int superlu_dgemm(const char *transa, const char *transb,
                  int m, int n, int k, double alpha, double *a,
                  int lda, double *b, int ldb, double beta, double *c, int ldc)
{
#ifdef _CRAY
    _fcd ftcs = _cptofcd(transa, strlen(transa));
    _fcd ftcs1 = _cptofcd(transb, strlen(transb));
    return SGEMM(ftcs, ftcs1, &m, &n, &k,
                 &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#elif defined(USE_VENDOR_BLAS)
    dgemm_(transa, transb, &m, &n, &k,
           &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
    return 0;
#else
    return dgemm_(transa, transb, &m, &n, &k,
                  &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}

int superlu_dtrsm(const char *sideRL, const char *uplo,
                  const char *transa, const char *diag,
                  const int m, const int n,
                  const double alpha, const double *a,
                  const int lda, double *b, const int ldb)

{
#if defined(USE_VENDOR_BLAS)
    dtrsm_(sideRL, uplo, transa, diag,
           &m, &n, &alpha, a, &lda, b, &ldb,
           1, 1, 1, 1);
    return 0;
#else
    return dtrsm_(sideRL, uplo, transa, diag,
                  &m, &n, &alpha, a, &lda, b, &ldb);
#endif
}

int superlu_dger(const int m, const int n, const double alpha,
                 const double *x, const int incx, const double *y,
                 const int incy, double *a, const int lda)
{
#ifdef _CRAY
    SGER(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
#else
    dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
#endif

    return 0;
}

int superlu_dscal(const int n, const double alpha, double *x, const int incx)
{
    dscal_(&n, &alpha, x, &incx);
    return 0;
}

int superlu_daxpy(const int n, const double alpha,
    const double *x, const int incx, double *y, const int incy)
{
    daxpy_(&n, &alpha, x, &incx, y, &incy);
    return 0;
}

int superlu_dgemv(const char *trans, const int m,
                  const int n, const double alpha, const double *a,
                  const int lda, const double *x, const int incx,
                  const double beta, double *y, const int incy)
{
#ifdef USE_VENDOR_BLAS
    dgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy, 1);
#else
    dgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#endif
    
    return 0;
}

int superlu_dtrsv(char *uplo, char *trans, char *diag,
                  int n, double *a, int lda, double *x, int incx)
{
#ifdef _CRAY
    // _fcd ftcs = _cptofcd("N", strlen("N"));
    STRSV(_cptofcd(uplo, strlen(uplo)), _cptofcd(trans, strlen(trans)), _cptofcd(diag, strlen(diag)), 
         &n, a, &lda, x, &incx);
#elif defined (USE_VENDOR_BLAS)
    dtrsv_(uplo, trans, diag, &n, a, &lda, x, &incx, 1, 1, 1);
#else
    dtrsv_(uplo, trans, diag, &n, a, &lda, x, &incx);
#endif
    
    return 0;
}

