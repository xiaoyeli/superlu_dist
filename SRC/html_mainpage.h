/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! \mainpage SuperLU_DIST Documentation

  <img src="superlu-logo.jpg" align="left" width=50px>
  <div style="clear: both"></div>
 
  SuperLU_DIST is a general purpose distributed-memory parallel library for the
  direct solution of large, sparse, nonsymmetric systems of linear equations. 
  The library is written in C and is callable from either C or Fortran program.
  It uses MPI and OpenMP to support various forms of parallelism, and is 
  GPU capable (CUDA, HIP, ...) . It supports both real and complex datatypes,
   both single and double precision, and 64-bit integer indexing.
  The library routines performs an LU decomposition with static pivoting and
   triangular system solves through forward and back substitution.
   The LU factorization routines can handle non-square matrices but the
   triangular solves are performed only for square matrices.
  
   The matrix may be preordered (before factorization) either through library
   or user supplied routines. This preordering for sparsity is completely separate
   from the factorization. Working precision or extra precision iterative refinement
   subroutines are provided for improved backward stability and forward accuracy.
   Routines are also provided to equilibrate the system,
   calculate the relative backward error, and estimate error bounds for the refined solutions.
 
   The SuperLU main web site is https://portal.nersc.gov/project/sparse/superlu/
 */

 * @image html superlu-logo.jpg  width=40px
