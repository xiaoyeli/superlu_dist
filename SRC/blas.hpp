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
/// @file blas.hpp
/// @brief Thin interface to BLAS
/// @date 2012-09-12
#ifndef _PEXSI_BLAS_HPP_ 
#define _PEXSI_BLAS_HPP_
#include "environment.hpp"

namespace PEXSI {

/// @namespace blas
///
/// @brief Thin interface to BLAS
namespace blas {

  typedef  int                    Int;
  typedef  std::complex<float>    scomplex;
  typedef  std::complex<double>   dcomplex;


  // *********************************************************************
  // Level 1 BLAS                                                   //
  // *********************************************************************
  void Axpy
    ( Int n, float alpha, const float* x, Int incx, float* y, Int incy );
  void Axpy
    ( Int n, double alpha, const double* x, Int incx, double* y, Int incy );
  void Axpy
    ( Int n, scomplex alpha, const scomplex* x, Int incx, scomplex* y, Int incy );
  void Axpy
    ( Int n, dcomplex alpha, const dcomplex* x, Int incx, dcomplex* y, Int incy );
  template<typename T>
    void Axpy( Int n, T alpha, const T* x, Int incx, T* y, Int incy );

  void Copy( Int n, const int* x, Int incx, int* y, Int incy );
  void Copy( Int n, const float* x, Int incx, float* y, Int incy );
  void Copy( Int n, const double* x, Int incx, double* y, Int incy );
  void Copy( Int n, const scomplex* x, Int incx, scomplex* y, Int incy );
  void Copy( Int n, const dcomplex* x, Int incx, dcomplex* y, Int incy );
  template<typename T>
    void Copy( Int n, const T* x, Int incx, T* y, Int incy );

  float Dot( Int n, const float* x, Int incx, const float* y, Int incy );
  double Dot( Int n, const double* x, Int incx, const double* y, Int incy );
  scomplex Dot( Int n, const scomplex* x, Int incx, const scomplex* y, Int incy );
  dcomplex Dot( Int n, const dcomplex* x, Int incx, const dcomplex* y, Int incy );
  template<typename T>
    T Dot( Int n, const T* x, Int incx, const T* y, Int incy );

  float Dotc
    ( Int n, const float* x, Int incx, const float* y, Int incy );
  double Dotc
    ( Int n, const double* x, Int incx, const double* y, Int incy );
  scomplex Dotc
    ( Int n, const scomplex* x, Int incx, const scomplex* y, Int incy );
  dcomplex Dotc
    ( Int n, const dcomplex* x, Int incx, const dcomplex* y, Int incy );
  template<typename T>
    T Dotc( Int n, const T* x, Int incx, const T* y, Int incy );

  float Dotu
    ( Int n, const float* x, Int incx, const float* y, Int incy );
  double Dotu
    ( Int n, const double* x, Int incx, const double* y, Int incy );
  scomplex Dotu
    ( Int n, const scomplex* x, Int incx, const scomplex* y, Int incy );
  dcomplex Dotu
    ( Int n, const dcomplex* x, Int incx, const dcomplex* y, Int incy );
  template<typename T>
    T Dotu( Int n, const T* x, Int incx, const T* y, Int incy );

  float Nrm2( Int n, const float* x, Int incx );
  double Nrm2( Int n, const double* x, Int incx );
  float Nrm2( Int n, const scomplex* x, Int incx );
  double Nrm2( Int n, const dcomplex* x, Int incx );
  template<typename F> F Nrm2( Int n, const F* x, Int incx );

  void Scal( Int n, float alpha, float* x, Int incx );
  void Scal( Int n, double alpha, double* x, Int incx );
  void Scal( Int n, scomplex alpha, scomplex* x, Int incx );
  void Scal( Int n, dcomplex alpha, dcomplex* x, Int incx );
  template<typename F> void Scal( Int n, F alpha, F* x, Int incx );

  // *********************************************************************
  // Level 2 BLAS                                                   
  // *********************************************************************
  void Gemv
    ( char trans, Int m, Int n,
      float alpha, const float* A, Int lda, const float* x, Int incx,
      float beta,        float* y, Int incy );
  void Gemv
    ( char trans, Int m, Int n,
      double alpha, const double* A, Int lda, const double* x, Int incx,
      double beta,        double* y, Int incy );
  void Gemv
    ( char trans, Int m, Int n,
      scomplex alpha, const scomplex* A, Int lda, const scomplex* x, Int incx,
      scomplex beta,        scomplex* y, Int incy );
  void Gemv
    ( char trans, Int m, Int n,
      dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* x, Int incx,
      dcomplex beta,        dcomplex* y, Int incy );
  template<typename T>
    void Gemv
    ( char trans, Int m, Int n,
      T alpha, const T* A, Int lda, const T* x, Int incx,
      T beta,        T* y, Int incy );

  void Ger
    ( Int m, Int n,
      float alpha, const float* x, Int incx, const float* y, Int incy,
      float* A, Int lda );
  void Ger
    ( Int m, Int n,
      double alpha, const double* x, Int incx, const double* y, Int incy,
      double* A, Int lda );
  void Ger
    ( Int m, Int n,
      scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
      scomplex* A, Int lda );
  void Ger
    ( Int m, Int n,
      dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
      dcomplex* A, Int lda );
  template<typename T>
    void Ger
    ( char trans, Int m, Int n,
      T alpha, const T* x, Int incx, const T* y, Int incy,
      T beta,        T* A, Int lda );

  void Gerc
    ( Int m, Int n,
      float alpha, const float* x, Int incx, const float* y, Int incy,
      float* A, Int lda );
  void Gerc
    ( Int m, Int n,
      double alpha, const double* x, Int incx, const double* y, Int incy,
      double* A, Int lda );
  void Gerc
    ( Int m, Int n,
      scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
      scomplex* A, Int lda );
  void Gerc
    ( Int m, Int n,
      dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
      dcomplex* A, Int lda );
  template<typename T>
    void Gerc
    ( char trans, Int m, Int n,
      T alpha, const T* x, Int incx, const T* y, Int incy,
      T beta,        T* A, Int lda );

  void Geru
    ( Int m, Int n,
      float alpha, const float* x, Int incx, const float* y, Int incy,
      float* A, Int lda );
  void Geru
    ( Int m, Int n,
      double alpha, const double* x, Int incx, const double* y, Int incy,
      double* A, Int lda );
  void Geru
    ( Int m, Int n,
      scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
      scomplex* A, Int lda );
  void Geru
    ( Int m, Int n,
      dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
      dcomplex* A, Int lda );
  template<typename T>
    void Geru
    ( char trans, Int m, Int n,
      T alpha, const T* x, Int incx, const T* y, Int incy,
      T beta,        T* A, Int lda );

  void Hemv
    ( char uplo, Int m,
      float alpha, const float* A, Int lda, const float* x, Int incx,
      float beta,        float* y, Int incy );
  void Hemv
    ( char uplo, Int m,
      double alpha, const double* A, Int lda, const double* x, Int incx,
      double beta,        double* y, Int incy );
  void Hemv
    ( char uplo, Int m,
      scomplex alpha, const scomplex* A, Int lda, const scomplex* x, Int incx,
      scomplex beta,        scomplex* y, Int incy );
  void Hemv
    ( char uplo, Int m,
      dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* x, Int incx,
      dcomplex beta,        dcomplex* y, Int incy );
  template<typename T>
    void Hemv
    ( char uplo, Int m,
      T alpha, const T* A, Int lda, const T* x, Int incx,
      T beta,        T* y, Int incy );

  void Her
    ( char uplo, Int m,
      float alpha, const float* x, Int incx, float* A, Int lda );
  void Her
    ( char uplo, Int m,
      double alpha, const double* x, Int incx, double* A, Int lda );
  void Her
    ( char uplo, Int m,
      scomplex alpha, const scomplex* x, Int incx, scomplex* A, Int lda );
  void Her
    ( char uplo, Int m,
      dcomplex alpha, const dcomplex* x, Int incx, dcomplex* A, Int lda );
  template<typename T>
    void Hemv( char uplo, Int m, T alpha, const T* x, Int incx, T* A, Int lda );

  void Her2
    ( char uplo, Int m,
      float alpha, const float* x, Int incx, const float* y, Int incy,
      float* A, Int lda );
  void Her2
    ( char uplo, Int m,
      double alpha, const double* x, Int incx, const double* y, Int incy,
      double* A, Int lda );
  void Her2
    ( char uplo, Int m,
      scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
      scomplex* A, Int lda );
  void Her2
    ( char uplo, Int m,
      dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
      dcomplex* A, Int lda );
  template<typename T>
    void Her2
    ( char uplo, Int m,
      T alpha, const T* x, Int incx, const T* y, Int incy, 
      T* A, Int lda );

  void Symv
    ( char uplo, Int m,
      float alpha, const float* A, Int lda, const float* x, Int incx,
      float beta,        float* y, Int incy );
  void Symv
    ( char uplo, Int m, 
      double alpha, const double* A, Int lda, const double* x, Int incx,
      double beta,        double* y, Int incy );
  void Symv
    ( char uplo, Int m,
      scomplex alpha, const scomplex* A, Int lda, const scomplex* x, Int incx,
      scomplex beta,        scomplex* y, Int incy );
  void Symv
    ( char uplo, Int m,
      dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* x, Int incx,
      dcomplex beta,        dcomplex* y, Int incy );
  template<typename T>
    void Symv
    ( char uplo, Int m,
      T alpha, const T* A, Int lda, const T* x, Int incx,
      T beta,        T* y, Int incy );

  void Syr
    ( char uplo, Int m,
      float alpha, const float* x, Int incx, float* A, Int lda );
  void Syr
    ( char uplo, Int m,
      double alpha, const double* x, Int incx, double* A, Int lda );
  void Syr
    ( char uplo, Int m,
      scomplex alpha, const scomplex* x, Int incx, scomplex* A, Int lda ); 
  void Syr
    ( char uplo, Int m,
      dcomplex alpha, const dcomplex* x, Int incx, dcomplex* A, Int lda );
  template<typename T>
    void Syr( char uplo, Int m, T alpha, const T* x, Int incx, T* A, Int lda );

  void Syr2
    ( char uplo, Int m,
      float alpha, const float* x, Int incx, const float* y, Int incy,
      float* A, Int lda );
  void Syr2
    ( char uplo, Int m,
      double alpha, const double* x, Int incx, const double* y, Int incy,
      double* A, Int lda );
  void Syr2
    ( char uplo, Int m,
      scomplex alpha, const scomplex* x, Int incx, const scomplex* y, Int incy,
      scomplex* A, Int lda );
  void Syr2
    ( char uplo, Int m,
      dcomplex alpha, const dcomplex* x, Int incx, const dcomplex* y, Int incy,
      dcomplex* A, Int lda );
  template<typename T>
    void Syr2
    ( char uplo, Int m,
      T alpha, const T* x, Int incx, const T* y, Int incy,
      T* A, Int lda );

  void Trmv
    ( char uplo, char trans, char diag, Int m,
      const float* A, Int lda, float* x, Int incx );
  void Trmv
    ( char uplo, char trans, char diag, Int m,
      const double* A, Int lda, double* x, Int incx );
  void Trmv
    ( char uplo, char trans, char diag, Int m,
      const scomplex* A, Int lda, scomplex* x, Int incx );
  void Trmv
    ( char uplo, char trans, char diag, Int m,
      const dcomplex* A, Int lda, dcomplex* x, Int incx );
  template<typename T>
    void Trmv
    ( char uplo, char trans, char diag, Int m,
      const T* A, Int lda, T* x, Int incx );

  void Trsv
    ( char uplo, char trans, char diag, Int m,
      const float* A, Int lda, float* x, Int incx );
  void Trsv
    ( char uplo, char trans, char diag, Int m,
      const double* A, Int lda, double* x, Int incx );
  void Trsv
    ( char uplo, char trans, char diag, Int m,
      const scomplex* A, Int lda, scomplex* x, Int incx );
  void Trsv
    ( char uplo, char trans, char diag, Int m,
      const dcomplex* A, Int lda, dcomplex* x, Int incx );
  template<typename T>
    void Trsv
    ( char uplo, char trans, char diag, Int m,
      const T* A, Int lda, T* x, Int incx );

  // *********************************************************************
  // Level 3 BLAS                                                  
  // *********************************************************************
  void Gemm
    ( char transA, char transB, Int m, Int n, Int k,
      float alpha, const float* A, Int lda, const float* B, Int ldb,
      float beta,        float* C, Int ldc );
  void Gemm
    ( char transA, char transB, Int m, Int n, Int k,
      double alpha, const double* A, Int lda, const double* B, Int ldb,
      double beta,        double* C, Int ldc );
  void Gemm
    ( char transA, char transB, Int m, Int n, Int k,
      scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
      scomplex beta,        scomplex* C, Int ldc );
  void Gemm
    ( char transA, char transB, Int m, Int n, Int k,
      dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
      dcomplex beta,        dcomplex* C, Int ldc );
  template<typename T>
    void Gemm
    ( char transA, char transB, Int m, Int n, Int k,
      T alpha, const T* A, Int lda, const T* B, Int ldb,
      T beta,        T* C, Int ldc );

  void Hemm
    ( char side, char uplo, Int m, Int n,
      float alpha, const float* A, Int lda, const float* B, Int ldb,
      float beta,        float* C, Int ldc );
  void Hemm
    ( char side, char uplo, Int m, Int n,
      double alpha, const double* A, Int lda, const double* B, Int ldb,
      double beta,        double* C, Int ldc );
  void Hemm
    ( char side, char uplo, Int m, Int n,
      scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
      scomplex beta,        scomplex* C, Int ldc );
  void Hemm
    ( char side, char uplo, Int m, Int n,
      dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
      dcomplex beta,        dcomplex* C, Int ldc );
  template<typename T>
    void Hemm
    ( char side, char uplo, Int m, Int n,
      T alpha, const T* A, Int lda, const T* B, Int ldb,
      T beta,        T* C, Int ldc );

  void Her2k
    ( char uplo, char trans, Int n, Int k,
      float alpha, const float* A, Int lda, const float* B, Int ldb,
      float beta,        float* C, Int ldc );
  void Her2k
    ( char uplo, char trans, Int n, Int k,
      double alpha, const double* A, Int lda, const double* B, Int ldb,
      double beta,        double* C, Int ldc );
  void Her2k
    ( char uplo, char trans, Int n, Int k,
      scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
      scomplex beta,        scomplex* C, Int ldc );
  void Her2k
    ( char uplo, char trans, Int n, Int k,
      dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
      dcomplex beta,        dcomplex* C, Int ldc );
  template<typename T>
    void Her2k
    ( char uplo, char trans, Int n, Int k,
      T alpha, const T* A, Int lda, const T* B, Int ldb,
      T beta,        T* C, Int ldc );

  void Herk
    ( char uplo, char trans, Int n, Int k,
      float alpha, const float* A, Int lda, float beta, float* C, Int ldc );
  void Herk
    ( char uplo, char trans, Int n, Int k,
      double alpha, const double* A, Int lda, double beta, double* C, Int ldc );
  void Herk
    ( char uplo, char trans, Int n, Int k,
      scomplex alpha, const scomplex* A, Int lda,
      scomplex beta,        scomplex* C, Int ldc );
  void Herk
    ( char uplo, char trans, Int n, Int k,
      dcomplex alpha, const dcomplex* A, Int lda,
      dcomplex beta,        dcomplex* C, Int ldc );
  template<typename T>
    void Herk
    ( char uplo, char trans, Int n, Int k,
      T alpha, const T* A, Int lda,
      T beta,        T* C, Int ldc );

  void Symm
    ( char side, char uplo, Int m, Int n,
      float alpha, const float* A, Int lda, const float* B, Int ldb,
      float beta,        float* C, Int ldc );
  void Symm
    ( char side, char uplo, Int m, Int n,
      double alpha, const double* A, Int lda, const double* B, Int ldb,
      double beta,        double* C, Int ldc );
  void Symm
    ( char side, char uplo, Int m, Int n,
      scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
      scomplex beta,        scomplex* C, Int ldc );
  void Symm
    ( char side, char uplo, Int m, Int n,
      dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
      dcomplex beta,        dcomplex* C, Int ldc );
  template<typename T>
    void Symm
    ( char side, char uplo, Int m, Int n,
      T alpha, const T* A, Int lda, const T* B, Int ldb,
      T beta,        T* C, Int ldc );

  void Syr2k
    ( char uplo, char trans, Int n, Int k,
      float alpha, const float* A, Int lda, const float* B, Int ldb,
      float beta,        float* C, Int ldc );
  void Syr2k
    ( char uplo, char trans, Int n, Int k,
      double alpha, const double* A, Int lda, const double* B, Int ldb,
      double beta,        double* C, Int ldc );
  void Syr2k
    ( char uplo, char trans, Int n, Int k,
      scomplex alpha, const scomplex* A, Int lda, const scomplex* B, Int ldb,
      scomplex beta,        scomplex* C, Int ldc );
  void Syr2k
    ( char uplo, char trans, Int n, Int k,
      dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
      dcomplex beta,        dcomplex* C, Int ldc );
  template<typename T>
    void Syr2k
    ( char uplo, char trans, Int n, Int k,
      T alpha, const T* A, Int lda, const T* B, Int ldb,
      T beta,        T* C, Int ldc );

  void Syrk
    ( char uplo, char trans, Int n, Int k,
      float alpha, const float* A, Int lda,
      float beta,        float* C, Int ldc );
  void Syrk
    ( char uplo, char trans, Int n, Int k,
      double alpha, const double* A, Int lda,
      double beta,        double* C, Int ldc );
  void Syrk
    ( char uplo, char trans, Int n, Int k,
      scomplex alpha, const scomplex* A, Int lda,
      scomplex beta,        scomplex* C, Int ldc );
  void Syrk
    ( char uplo, char trans, Int n, Int k,
      dcomplex alpha, const dcomplex* A, Int lda,
      dcomplex beta,        dcomplex* C, Int ldc );
  template<typename T>
    void Syrk
    ( char uplo, char trans, Int n, Int k,
      T alpha, const T* A, Int lda,
      T beta,        T* C, Int ldc );

  void Trmm
    ( char side,  char uplo, char trans, char unit, Int m, Int n,
      float alpha, const float* A, Int lda, float* B, Int ldb );
  void Trmm
    ( char side,  char uplo, char trans, char unit, Int m, Int n,
      double alpha, const double* A, Int lda, double* B, Int ldb );
  void Trmm
    ( char side,  char uplo, char trans, char unit, Int m, Int n,
      scomplex alpha, const scomplex* A, Int lda, scomplex* B, Int ldb );
  void Trmm
    ( char side,  char uplo, char trans, char unit, Int m, Int n,
      dcomplex alpha, const dcomplex* A, Int lda, dcomplex* B, Int ldb );
  template<typename T>
    void Trmm
    ( char side, char uplo, char trans, char unit, Int m, Int n,
      T alpha, const T* A, Int lda, T* B, Int ldb );

  void Trsm
    ( char side,  char uplo, char trans, char unit, Int m, Int n,
      float alpha, const float* A, Int lda, float* B, Int ldb );
  void Trsm
    ( char side,  char uplo, char trans, char unit, Int m, Int n,
      double alpha, const double* A, Int lda, double* B, Int ldb );
  void Trsm
    ( char side,  char uplo, char trans, char unit, Int m, Int n,
      scomplex alpha, const scomplex* A, Int lda, scomplex* B, Int ldb );
  void Trsm
    ( char side,  char uplo, char trans, char unit, Int m, Int n,
      dcomplex alpha, const dcomplex* A, Int lda, dcomplex* B, Int ldb );
  template<typename T>
    void Trsm
    ( char side, char uplo, char trans, char unit, Int m, Int n,
      T alpha, const T* A, Int lda, T* B, Int ldb );

} // namespace blas
} // namespace PEXSI 

#endif //_PEXSI_BLAS_HPP_
