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

#include "superlu_zdefs.h"

#ifdef _CRAY
_fcd ftcs = _cptofcd("N", strlen("N"));
_fcd ftcs1 = _cptofcd("L", strlen("L"));
_fcd ftcs2 = _cptofcd("N", strlen("N"));
_fcd ftcs3 = _cptofcd("U", strlen("U"));
#endif

int superlu_zgemm(const char *transa, const char *transb,
                  int m, int n, int k, doublecomplex alpha, doublecomplex *a,
                  int lda, doublecomplex *b, int ldb, doublecomplex beta, doublecomplex *c, int ldc)
{
#ifdef _CRAY
    _fcd ftcs = _cptofcd(transa, strlen(transa));
    _fcd ftcs1 = _cptofcd(transb, strlen(transb));
    return CGEMM(ftcs, ftcs1, &m, &n, &k,
                 &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#elif defined(USE_VENDOR_BLAS)
    zgemm_(transa, transb, &m, &n, &k,
           &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
    return 0;
#else
    return zgemm_(transa, transb, &m, &n, &k,
                  &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}

int superlu_ztrsm(const char *sideRL, const char *uplo,
                  const char *transa, const char *diag,
                  const int m, const int n,
                  const doublecomplex alpha, const doublecomplex *a,
                  const int lda, doublecomplex *b, const int ldb)

{
#if defined(USE_VENDOR_BLAS)
    ztrsm_(sideRL, uplo, transa, diag,
           &m, &n, &alpha, a, &lda, b, &ldb,
           1, 1, 1, 1);
    return 0;
#else
    return ztrsm_(sideRL, uplo, transa, diag,
                  &m, &n, &alpha, a, &lda, b, &ldb);
#endif
}

int superlu_zger(const int m, const int n, const doublecomplex alpha,
                 const doublecomplex *x, const int incx, const doublecomplex *y,
                 const int incy, doublecomplex *a, const int lda)
{
#ifdef _CRAY
    CGERU(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
#else
    zgeru_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
#endif

    return 0;
}

int superlu_zscal(const int n, const doublecomplex alpha, doublecomplex *x, const int incx)
{
    zscal_(&n, &alpha, x, &incx);
    return 0;
}

int superlu_zaxpy(const int n, const doublecomplex alpha,
    const doublecomplex *x, const int incx, doublecomplex *y, const int incy)
{
    zaxpy_(&n, &alpha, x, &incx, y, &incy);
    return 0;
}

int superlu_zgemv(const char *trans, const int m,
                  const int n, const doublecomplex alpha, const doublecomplex *a,
                  const int lda, const doublecomplex *x, const int incx,
                  const doublecomplex beta, doublecomplex *y, const int incy)
{
#ifdef USE_VENDOR_BLAS
    zgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy, 1);
#else
    zgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#endif
    
    return 0;
}

int superlu_ztrsv(char *uplo, char *trans, char *diag,
                  int n, doublecomplex *a, int lda, doublecomplex *x, int incx)
{
#ifdef _CRAY
    // _fcd ftcs = _cptofcd("N", strlen("N"));
    CTRSV(_cptofcd(uplo, strlen(uplo)), _cptofcd(trans, strlen(trans)), _cptofcd(diag, strlen(diag)), 
         &n, a, &lda, x, &incx);
#elif defined (USE_VENDOR_BLAS)
    ztrsv_(uplo, trans, diag, &n, a, &lda, x, &incx, 1, 1, 1);
#else
    ztrsv_(uplo, trans, diag, &n, a, &lda, x, &incx);
#endif
    
    return 0;
}

