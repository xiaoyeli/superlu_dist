/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Header for dcomplex.c
 *
 * <pre>
 * -- Distributed SuperLU routine (version 1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 * </pre>
 */

/*
 * This header file is to be included in source files z*.c
 */
#ifndef __SUPERLU_DCOMPLEX /* allow multiple inclusions */
#define __SUPERLU_DCOMPLEX

#include "superlu_defs.h"
#include <mpi.h>

#if defined(HAVE_SYCL)
#include <complex>

using doublecomplex = std::complex<double>;

/*
 * These variables will be defined to be MPI datatypes for complex
 * and double complex. I'm too lazy to declare them external in every
 * file that needs them.
 *
 * Use WINOWS_EXPORT_ALL_SYMBOLS on windows to export all symbols when
 * building a shared library.
 * Introduce macro SUPERLU_DIST_EXPORT to correctly export the only
 * remaining data symbol SuperLU_MPI_DOUBLE_COMPLEX.
 */
// extern SUPERLU_DIST_EXPORT MPI_Datatype SuperLU_MPI_DOUBLE_COMPLEX;

/* Macro definitions */

/*! \brief Complex Copy c = a */
#define z_copy(c, a)                                                           \
  { (*c) = (*a); }

/*! \brief Complex Addition c = a + b */
#define z_add(c, a, b)                                                         \
  { (*c) = (*a) + (*b); }

/*! \brief Complex Subtraction c = a - b */
#define z_sub(c, a, b)                                                         \
  { (*c) = (*a) - (*b); }

/*! \brief Complex-Double Multiplication */
#define zd_mult(c, a, b)                                                       \
  { (*c) = (*a) * (b); }

/*! \brief Complex-Complex Multiplication */
#define zz_mult(c, a, b)                                                       \
  { (*c) = (*a) * (*b); }

/*! \brief Complex equality testing, receives pointer so dereference them */
#define z_eq(a, b) ( (*a) == (*b) )

#elif defined(HAVE_CUDA) || defined(HAVE_HIP)

typedef struct {
  double r, i;
} doublecomplex;

double& real(doublecomplex& str) {
  return str.r;
}
double& imag(doublecomplex& str) {
  return str.i;
}

/*
 * These variables will be defined to be MPI datatypes for complex
 * and double complex. I'm too lazy to declare them external in every
 * file that needs them.
 *
 * Use WINOWS_EXPORT_ALL_SYMBOLS on windows to export all symbols when
 * building a shared library.
 * Introduce macro SUPERLU_DIST_EXPORT to correctly export the only
 * remaining data symbol SuperLU_MPI_DOUBLE_COMPLEX.
 */
// extern SUPERLU_DIST_EXPORT MPI_Datatype SuperLU_MPI_DOUBLE_COMPLEX;

/* Macro definitions */

/*! \brief Complex Copy c = a */
#define z_copy(c, a)                                                           \
  {                                                                            \
    (c)->r = (a)->r;                                                           \
    (c)->i = (a)->i;                                                           \
  }

/*! \brief Complex Addition c = a + b */
#define z_add(c, a, b)                                                         \
  {                                                                            \
    (c)->r = (a)->r + (b)->r;                                                  \
    (c)->i = (a)->i + (b)->i;                                                  \
  }

/*! \brief Complex Subtraction c = a - b */
#define z_sub(c, a, b)                                                         \
  {                                                                            \
    (c)->r = (a)->r - (b)->r;                                                  \
    (c)->i = (a)->i - (b)->i;                                                  \
  }

/*! \brief Complex-Double Multiplication */
#define zd_mult(c, a, b)                                                       \
  {                                                                            \
    (c)->r = (a)->r * (b);                                                     \
    (c)->i = (a)->i * (b);                                                     \
  }

/*! \brief Complex-Complex Multiplication */
#define zz_mult(c, a, b)                                                       \
  {                                                                            \
    double cr, ci;                                                             \
    cr = (a)->r * (b)->r - (a)->i * (b)->i;                                    \
    ci = (a)->i * (b)->r + (a)->r * (b)->i;                                    \
    (c)->r = cr;                                                               \
    (c)->i = ci;                                                               \
  }

/*! \brief Complex equality testing */
#define z_eq(a, b) ((a)->r == (b)->r && (a)->i == (b)->i)

#endif // HAVE_SYCL

#ifdef __cplusplus
extern "C" {
#endif

/* Prototypes for functions in dcomplex.c */
void slud_z_div(doublecomplex *, doublecomplex *, doublecomplex *);
double slud_z_abs(doublecomplex *);  /* exact */
double slud_z_abs1(doublecomplex *); /* approximate */

#ifdef __cplusplus
}
#endif

#endif /* __SUPERLU_DCOMPLEX */
