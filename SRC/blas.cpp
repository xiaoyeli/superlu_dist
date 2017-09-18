/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Authors: Jack Poulson and Lin Lin
	 
   This file is part of PEXSI. All rights reserved.

	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:

	 (1) Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 (2) Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 (3) Neither the name of the University of California, Lawrence Berkeley
	 National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
	 be used to endorse or promote products derived from this software without
	 specific prior written permission.

	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	 You are under no obligation whatsoever to provide any bug fixes, patches, or
	 upgrades to the features, functionality or performance of the source code
	 ("Enhancements") to anyone; however, if you choose to make your Enhancements
	 available either publicly, or directly to Lawrence Berkeley National
	 Laboratory, without imposing a separate written license agreement for such
	 Enhancements, then you hereby grant the following license: a non-exclusive,
	 royalty-free perpetual license to install, use, modify, prepare derivative
	 works, incorporate into other computer software, distribute, and sublicense
	 such enhancements or derivative works thereof, in binary and source code form.
*/
/// @file blas.cpp
/// @brief Thin interface to BLAS
/// @date 2012-09-12
#include "blas.hpp"

namespace PEXSI {
namespace blas {

extern "C" {

//------------------------------------------------------------------------//
// Level 1 BLAS                                                           //
//------------------------------------------------------------------------//
void BLAS(saxpy)
( const Int* n, const float* alpha, const float* x, const Int* incx,
                                          float* y, const Int* incy );
void BLAS(daxpy)
( const Int* n, const double* alpha, const double* x, const Int* incx,
                                           double* y, const Int* incy );
void BLAS(caxpy)
( const Int* n,
  const scomplex* alpha,
  const scomplex* x, const Int* incx,
        scomplex* y, const Int* incy );
void BLAS(zaxpy)
( const Int* n,
  const dcomplex* alpha,
  const dcomplex* x, const Int* incx,
        dcomplex* y, const Int* incy );

void BLAS(scopy)
( const Int* n, const float* x, const Int* incx,
                      float* y, const Int* incy );
void BLAS(dcopy)
( const Int* n, const double* x, const Int* incx,
                      double* y, const Int* incy );
void BLAS(ccopy)
( const Int* n, const scomplex* x, const Int* incx,
                      scomplex* y, const Int* incy );
void BLAS(zcopy)
( const Int* n, const dcomplex* x, const Int* incx,
                      dcomplex* y, const Int* incy );

float BLAS(sdot)
( const Int* n, const float* x, const Int* incx,
                const float* y, const Int* incy );
double BLAS(ddot)
( const Int* n, const double* x, const Int* incx,
                const double* y, const Int* incy );
// To avoid the compatibility issue, we simply handroll our own complex dots
float BLAS(snrm2)
( const Int* n, const float* x, const Int* incx );
double BLAS(dnrm2)
( const Int* n, const double* x, const Int* incx );
float BLAS(scnrm2)
( const Int* n, const scomplex* x, const Int* incx );
double BLAS(dznrm2)
( const Int* n, const dcomplex* x, const Int* incx );

void BLAS(sscal)
( const Int* n, const float* alpha, float* x, const Int* incx );
void BLAS(dscal)
( const Int* n, const double* alpha, double* x, const Int* incx );
void BLAS(cscal)
( const Int* n, const scomplex* alpha, scomplex* x,
  const Int* incx );
void BLAS(zscal)
( const Int* n, const dcomplex* alpha, dcomplex* x,
  const Int* incx );

//------------------------------------------------------------------------//
// Level 2 BLAS                                                           //
//------------------------------------------------------------------------//
void BLAS(sgemv)
( const char* trans, const Int* m, const Int* n,
  const float* alpha, const float* A, const Int* lda,
                      const float* x, const Int* incx,
  const float* beta,        float* y, const Int* incy );
void BLAS(dgemv)
( const char* trans, const Int* m, const Int* n,
  const double* alpha, const double* A, const Int* lda,
                       const double* x, const Int* incx,
  const double* beta,        double* y, const Int* incy );
void BLAS(cgemv)
( const char* trans, const Int* m, const Int* n,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* x, const Int* incx,
  const scomplex* beta,
        scomplex* y, const Int* incy );
void BLAS(zgemv)
( const char* trans, const Int* m, const Int* n,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* x, const Int* incx,
  const dcomplex* beta,
        dcomplex* y, const Int* incy );

void BLAS(sger)
( const Int* m, const Int* n,
  const float* alpha, const float* x, const Int* incx,
                      const float* y, const Int* incy,
                            float* A, const Int* lda  );
void BLAS(dger)
( const Int* m, const Int* n,
  const double* alpha, const double* x, const Int* incx,
                       const double* y, const Int* incy,
                             double* A, const Int* lda  );
void BLAS(cgerc)
( const Int* m, const Int* n,
  const scomplex* alpha,
  const scomplex* x, const Int* incx,
  const scomplex* y, const Int* incy,
        scomplex* A, const Int* lda  );
void BLAS(zgerc)
( const Int* m, const Int* n,
  const dcomplex* alpha,
  const dcomplex* x, const Int* incx,
  const dcomplex* y, const Int* incy,
        dcomplex* A, const Int* lda  );

void BLAS(cgeru)
( const Int* m, const Int* n,
  const scomplex* alpha,
  const scomplex* x, const Int* incx,
  const scomplex* y, const Int* incy,
        scomplex* A, const Int* lda  );
void BLAS(zgeru)
( const Int* m, const Int* n,
  const dcomplex* alpha,
  const dcomplex* x, const Int* incx,
  const dcomplex* y, const Int* incy,
        dcomplex* A, const Int* lda  );

void BLAS(chemv)
( const char* uplo, const Int* m,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* x, const Int* incx,
  const scomplex* beta,
        scomplex* y, const Int* incy );
void BLAS(zhemv)
( const char* uplo, const Int* m,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* x, const Int* incx,
  const dcomplex* beta,
        dcomplex* y, const Int* incy );

void BLAS(cher)
( const char* uplo, const Int* m,
  const scomplex* alpha,
  const scomplex* x, const Int* incx,
        scomplex* A, const Int* lda  );
void BLAS(zher)
( const char* uplo, const Int* m,
  const dcomplex* alpha,
  const dcomplex* x, const Int* incx,
        dcomplex* A, const Int* lda  );

void BLAS(cher2)
( const char* uplo, const Int* m,
  const scomplex* alpha,
  const scomplex* x, const Int* incx,
  const scomplex* y, const Int* incy,
        scomplex* A, const Int* lda  );
void BLAS(zher2)
( const char* uplo, const Int* m,
  const dcomplex* alpha,
  const dcomplex* x, const Int* incx,
  const dcomplex* y, const Int* incy,
        dcomplex* A, const Int* lda  );

void BLAS(ssymv)
( const char* uplo, const Int* m,
  const float* alpha, const float* A, const Int* lda,
                      const float* x, const Int* incx,
  const float* beta,        float* y, const Int* incy );
void BLAS(dsymv)
( const char* uplo, const Int* m,
  const double* alpha, const double* A, const Int* lda,
                       const double* x, const Int* incx,
  const double* beta,        double* y, const Int* incy );
// 'csymv' is an auxiliary LAPACK routine, but we will treat it as BLAS
void LAPACK(csymv)
( const char* uplo, const Int* m,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* x, const Int* incx,
  const scomplex* beta,
        scomplex* y, const Int* incy );
// 'zsymv' is an auxiliary LAPACK routine, but we will treat it as BLAS
void LAPACK(zsymv)
( const char* uplo, const Int* m,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* x, const Int* incx,
  const dcomplex* beta,
        dcomplex* y, const Int* incy );

void BLAS(ssyr)
( const char* uplo, const Int* m,
  const float* alpha, const float* x, const Int* incx,
                            float* A, const Int* lda  );
void BLAS(dsyr)
( const char* uplo, const Int* m,
  const double* alpha, const double* x, const Int* incx,
                             double* A, const Int* lda  );
// 'csyr' is an auxilliary LAPACK routine, but we will treat it as BLAS
void LAPACK(csyr)
( const char* uplo, const Int* m,
  const scomplex* alpha,
  const scomplex* x, const Int* incx,
        scomplex* A, const Int* lda  );
// 'zsyr' is an auxilliary LAPACK routine, but we will treat it as BLAS
void LAPACK(zsyr)
( const char* uplo, const Int* m,
  const dcomplex* alpha,
  const dcomplex* x, const Int* incx,
        dcomplex* A, const Int* lda  );

void BLAS(ssyr2)
( const char* uplo, const Int* m,
  const float* alpha, const float* x, const Int* incx,
                      const float* y, const Int* incy,
                            float* A, const Int* lda  );
void BLAS(dsyr2)
( const char* uplo, const Int* m,
  const double* alpha, const double* x, const Int* incx,
                       const double* y, const Int* incy,
                             double* A, const Int* lda  );

void BLAS(strmv)
( const char* uplo, const char* trans, const char* diag, const Int* m,
  const float* A, const Int* lda, float* x, const Int* incx );
void BLAS(dtrmv)
( const char* uplo, const char* trans, const char* diag, const Int* m,
  const double* A, const Int* lda, double* x, const Int* incx );
void BLAS(ctrmv)
( const char* uplo, const char* trans, const char* diag, const Int* m,
  const scomplex* A, const Int* lda,
        scomplex* x, const Int* incx );
void BLAS(ztrmv)
( const char* uplo, const char* trans, const char* diag, const Int* m,
  const dcomplex* A, const Int* lda,
        dcomplex* x, const Int* incx );

void BLAS(strsv)
( const char* uplo, const char* trans, const char* diag, const Int* m,
  const float* A, const Int* lda, float* x, const Int* incx );
void BLAS(dtrsv)
( const char* uplo, const char* trans, const char* diag, const Int* m,
  const double* A, const Int* lda, double* x, const Int* incx );
void BLAS(ctrsv)
( const char* uplo, const char* trans, const char* diag, const Int* m,
  const scomplex* A, const Int* lda,
        scomplex* x, const Int* incx );
void BLAS(ztrsv)
( const char* uplo, const char* trans, const char* diag, const Int* m,
  const dcomplex* A, const Int* lda,
        dcomplex* x, const Int* incx );

//------------------------------------------------------------------------//
// Level 3 BLAS                                                           //
//------------------------------------------------------------------------//
void BLAS(sgemm)
( const char* transA, const char* transB,
  const Int* m, const Int* n, const Int* k,
  const float* alpha, const float* A, const Int* lda,
                      const float* B, const Int* ldb,
  const float* beta,        float* C, const Int* ldc );
void BLAS(dgemm)
( const char* transA, const char* transB,
  const Int* m, const Int* n, const Int* k,
  const double* alpha, const double* A, const Int* lda,
                       const double* B, const Int* ldb,
  const double* beta,        double* C, const Int* ldc );
void BLAS(cgemm)
( const char* transA, const char* transB,
  const Int* m, const Int* n, const Int* k,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* B, const Int* ldb,
  const scomplex* beta,
        scomplex* C, const Int* ldc );
void BLAS(zgemm)
( const char* transA, const char* transB,
  const Int* m, const Int* n, const Int* k,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* B, const Int* ldb,
  const dcomplex* beta,
        dcomplex* C, const Int* ldc );

void BLAS(chemm)
( const char* side, const char* uplo,
  const Int* m, const Int* n,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* B, const Int* ldb,
  const scomplex* beta,
        scomplex* C, const Int* ldc );
void BLAS(zhemm)
( const char* side, const char* uplo,
  const Int* m, const Int* n,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* B, const Int* ldb,
  const dcomplex* beta,
        dcomplex* C, const Int* ldc );

void BLAS(cher2k)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* B, const Int* ldb,
  const scomplex* beta,
        scomplex* C, const Int* ldc );
void BLAS(zher2k)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* B, const Int* ldb,
  const dcomplex* beta,
        dcomplex* C, const Int* ldc );

void BLAS(cherk)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* beta,
        scomplex* C, const Int* ldc );
void BLAS(zherk)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* beta,
        dcomplex* C, const Int* ldc );

void BLAS(ssymm)
( const char* side, const char* uplo,
  const Int* m, const Int* n,
  const float* alpha, const float* A, const Int* lda,
                      const float* B, const Int* ldb,
  const float* beta,        float* C, const Int* ldc );
void BLAS(dsymm)
( const char* side, const char* uplo,
  const Int* m, const Int* n,
  const double* alpha, const double* A, const Int* lda,
                       const double* B, const Int* ldb,
  const double* beta,        double* C, const Int* ldc );
void BLAS(csymm)
( const char* side, const char* uplo,
  const Int* m, const Int* n,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* B, const Int* ldb,
  const scomplex* beta,
        scomplex* C, const Int* ldc );
void BLAS(zsymm)
( const char* side, const char* uplo,
  const Int* m, const Int* n,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* B, const Int* ldb,
  const dcomplex* beta,
        dcomplex* C, const Int* ldc );

void BLAS(ssyr2k)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const float* alpha, const float* A, const Int* lda,
                      const float* B, const Int* ldb,
  const float* beta,        float* C, const Int* ldc );
void BLAS(dsyr2k)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const double* alpha, const double* A, const Int* lda,
                       const double* B, const Int* ldb,
  const double* beta,        double* C, const Int* ldc );
void BLAS(csyr2k)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* B, const Int* ldb,
  const scomplex* beta,
        scomplex* C, const Int* ldc );
void BLAS(zsyr2k)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* B, const Int* ldb,
  const dcomplex* beta,
        dcomplex* C, const Int* ldc );

void BLAS(ssyrk)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const float* alpha, const float* A, const Int* lda,
  const float* beta,        float* C, const Int* ldc );
void BLAS(dsyrk)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const double* alpha, const double* A, const Int* lda,
  const double* beta,        double* C, const Int* ldc );
void BLAS(csyrk)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
  const scomplex* beta,
        scomplex* C, const Int* ldc );
void BLAS(zsyrk)
( const char* uplo, const char* trans,
  const Int* n, const Int* k,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
  const dcomplex* beta,
        dcomplex* C, const Int* ldc );

void BLAS(strmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const Int* m, const Int* n,
  const float* alpha, const float* A, const Int* lda,
                            float* B, const Int* ldb );
void BLAS(dtrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const Int* m, const Int* n,
  const double* alpha, const double* A, const Int* lda,
                             double* B, const Int* ldb );
void BLAS(ctrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const Int* m, const Int* n,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
        scomplex* B, const Int* ldb );
void BLAS(ztrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const Int* m, const Int* n,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
        dcomplex* B, const Int* ldb );

void BLAS(strsm)
( const char* side, const char* uplo, const char* transA, const char* diag,
  const Int* m, const Int* n,
  const float* alpha, const float* A, const Int* lda,
                            float* B, const Int* ldb );
void BLAS(dtrsm)
( const char* side, const char* uplo, const char* transA, const char* diag,
  const Int* m, const Int* n,
  const double* alpha, const double* A, const Int* lda,
                             double* B, const Int* ldb );
void BLAS(ctrsm)
( const char* side, const char* uplo, const char* transA, const char* diag,
  const Int* m, const Int* n,
  const scomplex* alpha,
  const scomplex* A, const Int* lda,
        scomplex* B, const Int* ldb );
void BLAS(ztrsm)
( const char* side, const char* uplo, const char* transA, const char* diag,
  const Int* m, const Int* n,
  const dcomplex* alpha,
  const dcomplex* A, const Int* lda,
        dcomplex* B, const Int* ldb );

} // extern "C"

//----------------------------------------------------------------------------//
// Level 1 BLAS                                                               //
//----------------------------------------------------------------------------//
void Axpy
( Int n, float alpha, const float* x, Int incx, float* y, Int incy )
{ BLAS(saxpy)( &n, &alpha, x, &incx, y, &incy ); }

void Axpy
( Int n, double alpha, const double* x, Int incx, double* y, Int incy )
{ BLAS(daxpy)( &n, &alpha, x, &incx, y, &incy ); }

void Axpy
( Int n, scomplex alpha, const scomplex* x, Int incx, scomplex* y, Int incy )
{ BLAS(caxpy)( &n, &alpha, x, &incx, y, &incy ); }

void Axpy
( Int n, dcomplex alpha, const dcomplex* x, Int incx, dcomplex* y, Int incy )
{ BLAS(zaxpy)( &n, &alpha, x, &incx, y, &incy ); }


void Copy( Int n, const int* x, Int incx, int* y, Int incy )
{ for(int i = 0; i < n; i++) { *y = *x; x+=incx; y+=incy; } }

void Copy( Int n, const float* x, Int incx, float* y, Int incy )
{ BLAS(scopy)( &n, x, &incx, y, &incy ); }

void Copy( Int n, const double* x, Int incx, double* y, Int incy )
{ BLAS(dcopy)( &n, x, &incx, y, &incy ); }

void Copy( Int n, const scomplex* x, Int incx, scomplex* y, Int incy )
{ BLAS(ccopy)( &n, x, &incx, y, &incy ); }

void Copy( Int n, const dcomplex* x, Int incx, dcomplex* y, Int incy )
{ BLAS(zcopy)( &n, x, &incx, y, &incy ); }

float Dot( Int n, const float* x, Int incx, const float* y, Int incy )
{ return BLAS(sdot)( &n, x, &incx, y, &incy ); }

double Dot( Int n, const double* x, Int incx, const double* y, Int incy )
{ return BLAS(ddot)( &n, x, &incx, y, &incy ); }

scomplex Dot( Int n, const scomplex* x, Int incx, const scomplex* y, Int incy )
{ 
    scomplex alpha = 0;
    for( Int i=0; i<n; ++i ) 
        alpha += conj(x[i*incx])*y[i*incy];
    return alpha;
}

dcomplex Dot( Int n, const dcomplex* x, Int incx, const dcomplex* y, Int incy )
{
    dcomplex alpha = 0;
    for( Int i=0; i<n; ++i ) 
        alpha += conj(x[i*incx])*y[i*incy];
    return alpha;
}

float Dotc( Int n, const float* x, Int incx, const float* y, Int incy )
{ return BLAS(sdot)( &n, x, &incx, y, &incy ); }

double Dotc( Int n, const double* x, Int incx, const double* y, Int incy )
{ return BLAS(ddot)( &n, x, &incx, y, &incy ); }

scomplex Dotc( Int n, const scomplex* x, Int incx, const scomplex* y, Int incy )
{ 
    scomplex alpha = 0;
    for( Int i=0; i<n; ++i ) 
        alpha += conj(x[i*incx])*y[i*incy];
    return alpha;
}

dcomplex Dotc( Int n, const dcomplex* x, Int incx, const dcomplex* y, Int incy )
{ 
    dcomplex alpha = 0;
    for( Int i=0; i<n; ++i ) 
        alpha += conj(x[i*incx])*y[i*incy];
    return alpha;
}

float Dotu( Int n, const float* x, Int incx, const float* y, Int incy )
{ return BLAS(sdot)( &n, x, &incx, y, &incy ); }

double Dotu( Int n, const double* x, Int incx, const double* y, Int incy )
{ return BLAS(ddot)( &n, x, &incx, y, &incy ); }

scomplex Dotu( Int n, const scomplex* x, Int incx, const scomplex* y, Int incy )
{
    scomplex alpha = 0;
    for( Int i=0; i<n; ++i ) 
        alpha += x[i*incx]*y[i*incy];
    return alpha;
}

dcomplex Dotu( Int n, const dcomplex* x, Int incx, const dcomplex* y, Int incy )
{
    dcomplex alpha = 0;
    for( Int i=0; i<n; ++i ) 
        alpha += x[i*incx]*y[i*incy];
    return alpha;
}

float Nrm2( Int n, const float* x, Int incx )
{ return BLAS(snrm2)( &n, x, &incx ); }

double Nrm2( Int n, const double* x, Int incx )
{ return BLAS(dnrm2)( &n, x, &incx ); }

float Nrm2( Int n, const scomplex* x, Int incx )
{ return BLAS(scnrm2)( &n, x, &incx ); }

double Nrm2( Int n, const dcomplex* x, Int incx )
{ return BLAS(dznrm2)( &n, x, &incx ); }

void Scal( Int n, float alpha, float* x, Int incx )
{ BLAS(sscal)( &n, &alpha, x, &incx ); }

void Scal( Int n, double alpha, double* x, Int incx )
{ BLAS(dscal)( &n, &alpha, x, &incx ); }

void Scal( Int n, scomplex alpha, scomplex* x, Int incx )
{ BLAS(cscal)( &n, &alpha, x, &incx ); }

void Scal( Int n, dcomplex alpha, dcomplex* x, Int incx )
{ BLAS(zscal)( &n, &alpha, x, &incx ); }

//----------------------------------------------------------------------------//
// Level 2 BLAS                                                               //
//----------------------------------------------------------------------------//
void Gemv
( char trans, Int m, Int n,
  float alpha, const float* A, Int lda, const float* x, Int incx,
  float beta,        float* y, Int incy )
{
    const char fixedTrans = ( trans == 'C' ? 'T' : trans );
    BLAS(sgemv)
    ( &fixedTrans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

void Gemv
( char trans, Int m, Int n,
  double alpha, const double* A, Int lda, const double* x, Int incx,
  double beta,        double* y, Int incy )
{
    const char fixedTrans = ( trans == 'C' ? 'T' : trans );
    BLAS(dgemv)
    ( &fixedTrans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

void Gemv
( char trans, Int m, Int n,
  scomplex alpha, const scomplex* A, Int lda, const scomplex* x, Int incx,
  scomplex beta,        scomplex* y, Int incy )
{ BLAS(cgemv)( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy ); }

void Gemv
( char trans, Int m, Int n,
  dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* x, Int incx,
  dcomplex beta,        dcomplex* y, Int incy )
{ BLAS(zgemv)( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy ); }

void Ger
( Int m, Int n,
  float alpha, const float* x, Int incx, const float* y, Int incy,
                     float* A, Int lda )
{ BLAS(sger)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Ger
( Int m, Int n,
  double alpha, const double* x, Int incx, const double* y, Int incy,
                      double* A, Int lda  )
{ BLAS(dger)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Ger
( Int m, Int n,
  scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
                        scomplex* A, Int lda )
{ BLAS(cgerc)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Ger
( Int m, Int n,
  dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
                        dcomplex* A, Int lda )
{ BLAS(zgerc)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Gerc
( Int m, Int n,
  float alpha, const float* x, Int incx, const float* y, Int incy,
                     float* A, Int lda )
{ BLAS(sger)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Gerc
( Int m, Int n,
  double alpha, const double* x, Int incx, const double* y, Int incy,
                      double* A, Int lda )
{ BLAS(dger)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Gerc
( Int m, Int n,
  scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
                        scomplex* A, Int lda )
{ BLAS(cgerc)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Gerc
( Int m, Int n,
  dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
                        dcomplex* A, Int lda )
{ BLAS(zgerc)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Geru
( Int m, Int n,
  float alpha, const float* x, Int incx, const float* y, Int incy,
                     float* A, Int lda )
{ BLAS(sger)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Geru
( Int m, Int n,
  double alpha, const double* x, Int incx, const double* y, Int incy,
                      double* A, Int lda )
{ BLAS(dger)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Geru
( Int m, Int n,
  scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
                        scomplex* A, Int lda )
{ BLAS(cgeru)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Geru
( Int m, Int n,
  dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
                        dcomplex* A, Int lda )
{ BLAS(zgeru)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda ); }

void Hemv
( char uplo, Int m,
  float alpha, const float* A, Int lda, const float* x, Int incx,
  float beta,        float* y, Int incy )
{ BLAS(ssymv)( &uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy ); }

void Hemv
( char uplo, Int m,
  double alpha, const double* A, Int lda, const double* x, Int incx,
  double beta,        double* y, Int incy )
{ BLAS(dsymv)( &uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy ); }

void Hemv
( char uplo, Int m,
  scomplex alpha, const scomplex* A, Int lda, const scomplex* x, Int incx,
  scomplex beta,        scomplex* y, Int incy )
{ BLAS(chemv)( &uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy ); }

void Hemv
( char uplo, Int m,
  dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* x, Int incx,
  dcomplex beta,        dcomplex* y, Int incy )
{ BLAS(zhemv)( &uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy ); }

void Her
( char uplo, Int m,
  float alpha, const float* x, Int incx, float* A, Int lda )
{ BLAS(ssyr)( &uplo, &m, &alpha, x, &incx, A, &lda ); }

void Her
( char uplo, Int m,
  double alpha, const double* x, Int incx, double* A, Int lda )
{ BLAS(dsyr)( &uplo, &m, &alpha, x, &incx, A, &lda ); }

void Her
( char uplo, Int m,
  scomplex alpha, const scomplex* x, Int incx, scomplex* A, Int lda )
{ BLAS(cher)( &uplo, &m, &alpha, x, &incx, A, &lda ); }

void Her
( char uplo, Int m,
  dcomplex alpha, const dcomplex* x, Int incx, dcomplex* A, Int lda )
{ BLAS(zher)( &uplo, &m, &alpha, x, &incx, A, &lda ); }

void Her2
( char uplo, Int m,
  float alpha, const float* x, Int incx, const float* y, Int incy,
                     float* A, Int lda )
{ BLAS(ssyr2)( &uplo, &m, &alpha, x, &incx, y, &incy, A, &lda ); }

void Her2
( char uplo, Int m,
  double alpha, const double* x, Int incx, const double* y, Int incy,
                      double* A, Int lda )
{ BLAS(dsyr2)( &uplo, &m, &alpha, x, &incx, y, &incy, A, &lda ); }

void Her2
( char uplo, Int m,
  scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
                        scomplex* A, Int lda )
{ BLAS(cher2)( &uplo, &m, &alpha, x, &incx, y, &incy, A, &lda ); }

void Her2
( char uplo, Int m,
  dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
                        dcomplex* A, Int lda )
{ BLAS(zher2)( &uplo, &m, &alpha, x, &incx, y, &incy, A, &lda ); }

void Symv
( char uplo, Int m,
  float alpha, const float* A, Int lda, const float* x, Int incx,
  float beta,        float* y, Int incy )
{ BLAS(ssymv)( &uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy ); }

void Symv
( char uplo, Int m,
  double alpha, const double* A, Int lda, const double* x, Int incx,
  double beta,        double* y, Int incy )
{ BLAS(dsymv)( &uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy ); }

void Symv
( char uplo, Int m,
  scomplex alpha, const scomplex* A, Int lda, const scomplex* x, Int incx,
  scomplex beta,        scomplex* y, Int incy )
{
    // Recall that 'csymv' is an LAPACK auxiliary routine
    LAPACK(csymv)( &uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

void Symv
( char uplo, Int m,
  dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* x, Int incx,
  dcomplex beta,        dcomplex* y, Int incy )
{
    // Recall that 'zsymv' is an LAPACK auxiliary routine
    LAPACK(zsymv)( &uplo, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

void Syr
( char uplo, Int m,
  float alpha, const float* x, Int incx, float* A, Int lda  )
{ BLAS(ssyr)( &uplo, &m, &alpha, x, &incx, A, &lda ); }

void Syr
( char uplo, Int m,
  double alpha, const double* x, Int incx, double* A, Int lda )
{ BLAS(dsyr)( &uplo, &m, &alpha, x, &incx, A, &lda ); }

void Syr
( char uplo, Int m,
  scomplex alpha, const scomplex* x, Int incx, scomplex* A, Int lda )
{
    // Recall that 'csyr' is an LAPACK auxiliary routine
    LAPACK(csyr)( &uplo, &m, &alpha, x, &incx, A, &lda ); 
}

void Syr
( char uplo, Int m,
  dcomplex alpha, const dcomplex* x, Int incx, dcomplex* A, Int lda )
{
    // Recall that 'zsyr' is an LAPACK auxiliary routine
    LAPACK(zsyr)( &uplo, &m, &alpha, x, &incx, A, &lda ); 
}

void Syr2
( char uplo, Int m,
  float alpha, const float* x, Int incx, const float* y, Int incy,
                     float* A, Int lda )
{ BLAS(ssyr2)( &uplo, &m, &alpha, x, &incx, y, &incy, A, &lda ); }

void Syr2
( char uplo, Int m,
  double alpha, const double* x, Int incx, const double* y, Int incy,
                      double* A, Int lda )
{ BLAS(dsyr2)( &uplo, &m, &alpha, x, &incx, y, &incy, A, &lda ); }

void Syr2
( char uplo, Int m,
  scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
                        scomplex* A, Int lda )
{
    // csyr2 doesn't exist, so we route through csyr2k. However, csyr2k expects 
    // contiguous access of 'x', so we treat x and y as a row vectors where 
    // their leading dimensions are 'incx' and 'incy'. Thus we must perform 
    // A += x' y + y' x
    const char trans = 'T';
    const Int k = 1;
    const scomplex beta = 1.;
    BLAS(csyr2k)
    ( &uplo, &trans, &m, &k, &alpha, x, &incx, y, &incy, &beta, A, &lda );
}

void Syr2
( char uplo, Int m,
  dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
                        dcomplex* A, Int lda )
{
    // zsyr2 doesn't exist, so we route through zsyr2k. However, zsyr2k expects 
    // contiguous access of 'x', so we treat x and y as a row vectors where 
    // their leading dimensions are 'incx' and 'incy'. Thus we must perform 
    // A += x' y + y' x
    const char trans = 'T';
    const Int k = 1;
    const dcomplex beta = 1.;
    BLAS(zsyr2k)
    ( &uplo, &trans, &m, &k, &alpha, x, &incx, y, &incy, &beta, A, &lda );
}

void Trmv
( char uplo, char trans, char diag, Int m,
  const float* A, Int lda, float* x, Int incx )
{ BLAS(strmv)( &uplo, &trans, &diag, &m, A, &lda, x, &incx ); }

void Trmv
( char uplo, char trans, char diag, Int m,
  const double* A, Int lda, double* x, Int incx )
{ BLAS(dtrmv)( &uplo, &trans, &diag, &m, A, &lda, x, &incx ); }

void Trmv
( char uplo, char trans, char diag, Int m,
  const scomplex* A, Int lda, scomplex* x, Int incx )
{ BLAS(ctrmv)( &uplo, &trans, &diag, &m, A, &lda, x, &incx ); }

void Trmv
( char uplo, char trans, char diag, Int m,
  const dcomplex* A, Int lda, dcomplex* x, Int incx )
{ BLAS(ztrmv)( &uplo, &trans, &diag, &m, A, &lda, x, &incx ); }

void Trsv
( char uplo, char trans, char diag, Int m,
  const float* A, Int lda, float* x, Int incx )
{ BLAS(strsv)( &uplo, &trans, &diag, &m, A, &lda, x, &incx ); }

void Trsv
( char uplo, char trans, char diag, Int m,
  const double* A, Int lda, double* x, Int incx )
{ BLAS(dtrsv)( &uplo, &trans, &diag, &m, A, &lda, x, &incx ); }

void Trsv
( char uplo, char trans, char diag, Int m,
  const scomplex* A, Int lda, scomplex* x, Int incx )
{ BLAS(ctrsv)( &uplo, &trans, &diag, &m, A, &lda, x, &incx ); }

void Trsv
( char uplo, char trans, char diag, Int m,
  const dcomplex* A, Int lda, dcomplex* x, Int incx )
{ BLAS(ztrsv)( &uplo, &trans, &diag, &m, A, &lda, x, &incx ); }

//----------------------------------------------------------------------------//
// Level 3 BLAS                                                               //
//----------------------------------------------------------------------------//
void Gemm
( char transA, char transB, Int m, Int n, Int k, 
  float alpha, const float* A, Int lda, const float* B, Int ldb,
  float beta,        float* C, Int ldc )
{
    const char fixedTransA = ( transA == 'C' ? 'T' : transA );
    const char fixedTransB = ( transB == 'C' ? 'T' : transB );
    BLAS(sgemm)( &fixedTransA, &fixedTransB, &m, &n, &k,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Gemm
( char transA, char transB,
  Int m, Int n, Int k, 
  double alpha, const double* A, Int lda, const double* B, Int ldb,
  double beta,        double* C, Int ldc )
{
    const char fixedTransA = ( transA == 'C' ? 'T' : transA );
    const char fixedTransB = ( transB == 'C' ? 'T' : transB );
    BLAS(dgemm)( &fixedTransA, &fixedTransB, &m, &n, &k,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Gemm
( char transA, char transB, Int m, Int n, Int k, 
  scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
  scomplex beta,        scomplex* C, Int ldc )
{
    BLAS(cgemm)( &transA, &transB, &m, &n, &k,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Gemm
( char transA, char transB, Int m, Int n, Int k, 
  dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
  dcomplex beta,        dcomplex* C, Int ldc )
{
    BLAS(zgemm)( &transA, &transB, &m, &n, &k,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Hemm
( char side, char uplo, Int m, Int n,
  float alpha, const float* A, Int lda, const float* B, Int ldb,
  float beta,        float* C, Int ldc )
{
    BLAS(ssymm)( &side, &uplo, &m, &n,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Hemm
( char side, char uplo, Int m, Int n,
  double alpha, const double* A, Int lda, const double* B, Int ldb,
  double beta,        double* C, Int ldc )
{
    BLAS(dsymm)( &side, &uplo, &m, &n,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Hemm
( char side, char uplo, Int m, Int n,
  scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
  scomplex beta,        scomplex* C, Int ldc )
{
    BLAS(chemm)( &side, &uplo, &m, &n,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Hemm
( char side, char uplo, Int m, Int n,
  dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
  dcomplex beta,        dcomplex* C, Int ldc )
{
    BLAS(zhemm)( &side, &uplo, &m, &n,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Her2k
( char uplo, char trans, Int n, Int k,
  float alpha, const float* A, Int lda, const float* B, Int ldb,
  float beta,        float* C, Int ldc )
{
    const char transFixed = ( trans == 'C' ? 'T' : trans );
    BLAS(ssyr2k)
    ( &uplo, &transFixed, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Her2k
( char uplo, char trans, Int n, Int k,
  double alpha, const double* A, Int lda, const double* B, Int ldb,
  double beta,        double* C, Int ldc )
{
    const char transFixed = ( trans == 'C' ? 'T' : trans );
    BLAS(dsyr2k)
    ( &uplo, &transFixed, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Her2k
( char uplo, char trans, Int n, Int k,
  scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
  scomplex beta,        scomplex* C, Int ldc )
{
    BLAS(cher2k)
    ( &uplo, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Her2k
( char uplo, char trans, Int n, Int k,
  dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
  dcomplex beta,        dcomplex* C, Int ldc )
{
    BLAS(zher2k)
    ( &uplo, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Herk
( char uplo, char trans, Int n, Int k,
  float alpha, const float* A, Int lda,
  float beta,        float* C, Int ldc )
{
    const char transFixed = ( trans == 'C' ? 'T' : trans );
    BLAS(ssyrk)( &uplo, &transFixed, &n, &k, &alpha, A, &lda, &beta, C, &ldc );
}

void Herk
( char uplo, char trans, Int n, Int k,
  double alpha, const double* A, Int lda,
  double beta,        double* C, Int ldc )
{
    const char transFixed = ( trans == 'C' ? 'T' : trans );
    BLAS(dsyrk)( &uplo, &transFixed, &n, &k, &alpha, A, &lda, &beta, C, &ldc );
}

void Herk
( char uplo, char trans, Int n, Int k,
  scomplex alpha, const scomplex* A, Int lda,
  scomplex beta,        scomplex* C, Int ldc )
{ BLAS(cherk)( &uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc ); }

void Herk
( char uplo, char trans, Int n, Int k,
  dcomplex alpha, const dcomplex* A, Int lda,
  dcomplex beta,        dcomplex* C, Int ldc )
{ BLAS(zherk)( &uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc ); }

void Symm
( char side, char uplo, Int m, Int n,
  float alpha, const float* A, Int lda, const float* B, Int ldb,
  float beta,        float* C, Int ldc )
{
    BLAS(ssymm)( &side, &uplo, &m, &n,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Symm
( char side, char uplo, Int m, Int n,
  double alpha, const double* A, Int lda, const double* B, Int ldb,
  double beta,        double* C, Int ldc )
{
    BLAS(dsymm)( &side, &uplo, &m, &n,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Symm
( char side, char uplo, Int m, Int n,
  scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
  scomplex beta,        scomplex* C, Int ldc )
{
    BLAS(csymm)( &side, &uplo, &m, &n,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Symm
( char side, char uplo, Int m, Int n,
  dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
  dcomplex beta,        dcomplex* C, Int ldc )
{
    BLAS(zsymm)( &side, &uplo, &m, &n,
                 &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Syr2k
( char uplo, char trans, Int n, Int k,
  float alpha, const float* A, Int lda, const float* B, Int ldb,
  float beta,        float* C, Int ldc )
{
    BLAS(ssyr2k)
    ( &uplo, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Syr2k
( char uplo, char trans, Int n, Int k,
  double alpha, const double* A, Int lda, const double* B, Int ldb,
  double beta,        double* C, Int ldc )
{
    BLAS(dsyr2k)
    ( &uplo, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Syr2k
( char uplo, char trans, Int n, Int k,
  scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
  scomplex beta,        scomplex* C, Int ldc )
{
    BLAS(csyr2k)
    ( &uplo, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Syr2k
( char uplo, char trans, Int n, Int k,
  dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
  dcomplex beta,        dcomplex* C, Int ldc )
{
    BLAS(zsyr2k)
    ( &uplo, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Syrk
( char uplo, char trans, Int n, Int k,
  float alpha, const float* A, Int lda,
  float beta,        float* C, Int ldc )
{ BLAS(ssyrk)( &uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc ); }

void Syrk
( char uplo, char trans, Int n, Int k,
  double alpha, const double* A, Int lda,
  double beta,        double* C, Int ldc )
{ BLAS(dsyrk)( &uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc ); }

void Syrk
( char uplo, char trans, Int n, Int k,
  scomplex alpha, const scomplex* A, Int lda,
  scomplex beta,        scomplex* C, Int ldc )
{ BLAS(csyrk)( &uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc ); }

void Syrk
( char uplo, char trans, Int n, Int k,
  dcomplex alpha, const dcomplex* A, Int lda,
  dcomplex beta,        dcomplex* C, Int ldc )
{ BLAS(zsyrk)( &uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc ); }

void Trmm
( char side, char uplo, char trans, char unit, Int m, Int n,
  float alpha, const float* A, Int lda, float* B, Int ldb )
{
    const char fixedTrans = ( trans == 'C' ? 'T' : trans );    
    BLAS(strmm)( &side, &uplo, &fixedTrans, &unit, &m, &n,
                 &alpha, A, &lda, B, &ldb );
}

void Trmm
( char side, char uplo, char trans, char unit, Int m, Int n,
  double alpha, const double* A, Int lda, double* B, Int ldb )
{
    const char fixedTrans = ( trans == 'C' ? 'T' : trans );    
    BLAS(dtrmm)( &side, &uplo, &fixedTrans, &unit, &m, &n,
                 &alpha, A, &lda, B, &ldb );
}

void Trmm
( char side, char uplo, char trans, char unit, Int m, Int n,
  scomplex alpha, const scomplex* A, Int lda, scomplex* B, Int ldb )
{
    BLAS(ctrmm)( &side, &uplo, &trans, &unit, &m, &n,
                 &alpha, A, &lda, B, &ldb );
}

void Trmm
( char side, char uplo, char trans, char unit, Int m, Int n,
  dcomplex alpha, const dcomplex* A, Int lda, dcomplex* B, Int ldb )
{
    BLAS(ztrmm)( &side, &uplo, &trans, &unit, &m, &n,
                 &alpha, A, &lda, B, &ldb );
}

void Trsm
( char side, char uplo, char trans, char unit, Int m, Int n,
  float alpha, const float* A, Int lda, float* B, Int ldb )
{
    const char fixedTrans = ( trans == 'C' ? 'T' : trans );
    BLAS(strsm)( &side, &uplo, &fixedTrans, &unit, &m, &n,
                 &alpha, A, &lda, B, &ldb );
} 

void Trsm
( char side, char uplo, char trans, char unit, Int m, Int n,
  double alpha, const double* A, Int lda, double* B, Int ldb )
{
    const char fixedTrans = ( trans == 'C' ? 'T' : trans );
    BLAS(dtrsm)( &side, &uplo, &fixedTrans, &unit, &m, &n,
                 &alpha, A, &lda, B, &ldb );
} 

void Trsm
( char side, char uplo, char trans, char unit, Int m, Int n,
  scomplex alpha, const scomplex* A, Int lda, scomplex* B, Int ldb )
{
    BLAS(ctrsm)( &side, &uplo, &trans, &unit, &m, &n,
                 &alpha, A, &lda, B, &ldb );
} 

void Trsm
( char side, char uplo, char trans, char unit, Int m, Int n,
  dcomplex alpha, const dcomplex* A, Int lda, dcomplex* B, Int ldb )
{
    BLAS(ztrsm)( &side, &uplo, &trans, &unit, &m, &n,
                 &alpha, A, &lda, B, &ldb );
} 

} // namespace blas
} // namespace PEXSI
