/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Solves a system of linear equations A*X=B using 3D process grid.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 * May 12, 2021
 * October 5, 2021
 * Last update: November 8, 2021  v7.2.0
 */

#include "superlu_zdefs.h"
//#include "TRF3dV100/superlu_summit.h"
#include "superlu_upacked.h"
// #include "pzdistribute3d.h"

// #include "zssvx3dAux.c"

// int_t zgatherAllFactoredLU3d( ztrf3Dpartition_t*  trf3Dpartition,
// 			   zLUstruct_t* LUstruct, gridinfo3d_t* grid3d, SCT_t* SCT );
#include <stdbool.h>

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * PZGSSVX3D solves a system of linear equations A*X=B,
 * by using Gaussian elimination with "static pivoting" to
 * compute the LU factorization of A.
 *
 * Static pivoting is a technique that combines the numerical stability
 * of partial pivoting with the scalability of Cholesky (no pivoting),
 * to run accurately and efficiently on large numbers of processors.
 * See our paper at http://www.nersc.gov/~xiaoye/SuperLU/ for a detailed
 * description of the parallel algorithms.
 *
 * The input matrices A and B are distributed by block rows.
 * Here is a graphical illustration (0-based indexing):
 *
 *                        A                B
 *               0 ---------------       ------
 *                   |           |        |  |
 *                   |           |   P0   |  |
 *                   |           |        |  |
 *                 ---------------       ------
 *        - fst_row->|           |        |  |
 *        |          |           |        |  |
 *       m_loc       |           |   P1   |  |
 *        |          |           |        |  |
 *        -          |           |        |  |
 *                 ---------------       ------
 *                   |    .      |        |. |
 *                   |    .      |        |. |
 *                   |    .      |        |. |
 *                 ---------------       ------
 *
 * where, fst_row is the row number of the first row,
 *        m_loc is the number of rows local to this processor
 * These are defined in the 'SuperMatrix' structure, see supermatrix.h.
 *
 *
 * Here are the options for using this code:
 *
 *   1. Independent of all the other options specified below, the
 *      user must supply
 *
 *      -  B, the matrix of right-hand sides, distributed by block rows,
 *            and its dimensions ldb (local) and nrhs (global)
 *      -  grid, a structure describing the 2D processor mesh
 *      -  options->IterRefine, which determines whether or not to
 *            improve the accuracy of the computed solution using
 *            iterative refinement
 *
 *      On output, B is overwritten with the solution X.
 *
 *   2. Depending on options->Fact, the user has four options
 *      for solving A*X=B. The standard option is for factoring
 *      A "from scratch". (The other options, described below,
 *      are used when A is sufficiently similar to a previously
 *      solved problem to save time by reusing part or all of
 *      the previous factorization.)
 *
 *      -  options->Fact = DOFACT: A is factored "from scratch"
 *
 *      In this case the user must also supply
 *
 *        o  A, the input matrix
 *
 *        as well as the following options to determine what matrix to
 *        factorize.
 *
 *        o  options->Equil,   to specify how to scale the rows and columns
 *                             of A to "equilibrate" it (to try to reduce its
 *                             condition number and so improve the
 *                             accuracy of the computed solution)
 *
 *        o  options->RowPerm, to specify how to permute the rows of A
 *                             (typically to control numerical stability)
 *
 *        o  options->ColPerm, to specify how to permute the columns of A
 *                             (typically to control fill-in and enhance
 *                             parallelism during factorization)
 *
 *        o  options->ReplaceTinyPivot, to specify how to deal with tiny
 *                             pivots encountered during factorization
 *                             (to control numerical stability)
 *
 *      The outputs returned include
 *
 *        o  ScalePermstruct,  modified to describe how the input matrix A
 *                             was equilibrated and permuted:
 *          .  ScalePermstruct->DiagScale, indicates whether the rows and/or
 *                                         columns of A were scaled
 *          .  ScalePermstruct->R, array of row scale factors
 *          .  ScalePermstruct->C, array of column scale factors
 *          .  ScalePermstruct->perm_r, row permutation vector
 *          .  ScalePermstruct->perm_c, column permutation vector
 *
 *          (part of ScalePermstruct may also need to be supplied on input,
 *           depending on options->RowPerm and options->ColPerm as described
 *           later).
 *
 *        o  A, the input matrix A overwritten by the scaled and permuted
 *              matrix diag(R)*A*diag(C)*Pc^T, where
 *              Pc is the row permutation matrix determined by
 *                  ScalePermstruct->perm_c
 *              diag(R) and diag(C) are diagonal scaling matrices determined
 *                  by ScalePermstruct->DiagScale, ScalePermstruct->R and
 *                  ScalePermstruct->C
 *
 *        o  LUstruct, which contains the L and U factorization of A1 where
 *
 *                A1 = Pc*Pr*diag(R)*A*diag(C)*Pc^T = L*U
 *
 *               (Note that A1 = Pc*Pr*Aout, where Aout is the matrix stored
 *                in A on output.)
 *
 *   3. The second value of options->Fact assumes that a matrix with the same
 *      sparsity pattern as A has already been factored:
 *
 *      -  options->Fact = SamePattern: A is factored, assuming that it has
 *            the same nonzero pattern as a previously factored matrix. In
 *            this case the algorithm saves time by reusing the previously
 *            computed column permutation vector stored in
 *            ScalePermstruct->perm_c and the "elimination tree" of A
 *            stored in LUstruct->etree
 *
 *      In this case the user must still specify the following options
 *      as before:
 *
 *        o  options->Equil
 *        o  options->RowPerm
 *        o  options->ReplaceTinyPivot
 *
 *      but not options->ColPerm, whose value is ignored. This is because the
 *      previous column permutation from ScalePermstruct->perm_c is used as
 *      input. The user must also supply
 *
 *        o  A, the input matrix
 *        o  ScalePermstruct->perm_c, the column permutation
 *        o  LUstruct->etree, the elimination tree
 *
 *      The outputs returned include
 *
 *        o  A, the input matrix A overwritten by the scaled and permuted
 *              matrix as described above
 *        o  ScalePermstruct, modified to describe how the input matrix A was
 *                            equilibrated and row permuted
 *        o  LUstruct, modified to contain the new L and U factors
 *
 *   4. The third value of options->Fact assumes that a matrix B with the same
 *      sparsity pattern as A has already been factored, and where the
 *      row permutation of B can be reused for A. This is useful when A and B
 *      have similar numerical values, so that the same row permutation
 *      will make both factorizations numerically stable. This lets us reuse
 *      all of the previously computed structure of L and U.
 *
 *      -  options->Fact = SamePattern_SameRowPerm: A is factored,
 *            assuming not only the same nonzero pattern as the previously
 *            factored matrix B, but reusing B's row permutation.
 *
 *      In this case the user must still specify the following options
 *      as before:
 *
 *        o  options->Equil
 *        o  options->ReplaceTinyPivot
 *
 *      but not options->RowPerm or options->ColPerm, whose values are
 *      ignored. This is because the permutations from ScalePermstruct->perm_r
 *      and ScalePermstruct->perm_c are used as input.
 *
 *      The user must also supply
 *
 *        o  A, the input matrix
 *        o  ScalePermstruct->DiagScale, how the previous matrix was row
 *                                       and/or column scaled
 *        o  ScalePermstruct->R, the row scalings of the previous matrix,
 *                               if any
 *        o  ScalePermstruct->C, the columns scalings of the previous matrix,
 *                               if any
 *        o  ScalePermstruct->perm_r, the row permutation of the previous
 *                                    matrix
 *        o  ScalePermstruct->perm_c, the column permutation of the previous
 *                                    matrix
 *        o  all of LUstruct, the previously computed information about
 *                            L and U (the actual numerical values of L and U
 *                            stored in LUstruct->Llu are ignored)
 *
 *      The outputs returned include
 *
 *        o  A, the input matrix A overwritten by the scaled and permuted
 *              matrix as described above
 *        o  ScalePermstruct,  modified to describe how the input matrix A was
 *                             equilibrated (thus ScalePermstruct->DiagScale,
 *                             R and C may be modified)
 *        o  LUstruct, modified to contain the new L and U factors
 *
 *   5. The fourth and last value of options->Fact assumes that A is
 *      identical to a matrix that has already been factored on a previous
 *      call, and reuses its entire LU factorization
 *
 *      -  options->Fact = Factored: A is identical to a previously
 *            factorized matrix, so the entire previous factorization
 *            can be reused.
 *
 *      In this case all the other options mentioned above are ignored
 *      (options->Equil, options->RowPerm, options->ColPerm,
 *       options->ReplaceTinyPivot)
 *
 *      The user must also supply
 *
 *        o  A, the unfactored matrix, only in the case that iterative
 *              refinment is to be done (specifically A must be the output
 *              A from the previous call, so that it has been scaled and permuted)
 *        o  all of ScalePermstruct
 *        o  all of LUstruct, including the actual numerical values of
 *           L and U
 *
 *      all of which are unmodified on output.
 *
 * Arguments
 * =========
 *
 * options (input) superlu_dist_options_t* (global)
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *         The following fields should be defined for this structure:
 *
 *         o Fact (fact_t)
 *           Specifies whether or not the factored form of the matrix
 *           A is supplied on entry, and if not, how the matrix A should
 *           be factorized based on the previous history.
 *
 *           = DOFACT: The matrix A will be factorized from scratch.
 *                 Inputs:  A
 *                          options->Equil, RowPerm, ColPerm, ReplaceTinyPivot
 *                 Outputs: modified A
 *                             (possibly row and/or column scaled and/or
 *                              permuted)
 *                          all of ScalePermstruct
 *                          all of LUstruct
 *
 *           = SamePattern: the matrix A will be factorized assuming
 *             that a factorization of a matrix with the same sparsity
 *             pattern was performed prior to this one. Therefore, this
 *             factorization will reuse column permutation vector
 *             ScalePermstruct->perm_c and the elimination tree
 *             LUstruct->etree
 *                 Inputs:  A
 *                          options->Equil, RowPerm, ReplaceTinyPivot
 *                          ScalePermstruct->perm_c
 *                          LUstruct->etree
 *                 Outputs: modified A
 *                             (possibly row and/or column scaled and/or
 *                              permuted)
 *                          rest of ScalePermstruct (DiagScale, R, C, perm_r)
 *                          rest of LUstruct (GLU_persist, Llu)
 *
 *           = SamePattern_SameRowPerm: the matrix A will be factorized
 *             assuming that a factorization of a matrix with the same
 *             sparsity	pattern and similar numerical values was performed
 *             prior to this one. Therefore, this factorization will reuse
 *             both row and column scaling factors R and C, and the
 *             both row and column permutation vectors perm_r and perm_c,
 *             distributed data structure set up from the previous symbolic
 *             factorization.
 *                 Inputs:  A
 *                          options->Equil, ReplaceTinyPivot
 *                          all of ScalePermstruct
 *                          all of LUstruct
 *                 Outputs: modified A
 *                             (possibly row and/or column scaled and/or
 *                              permuted)
 *                          modified LUstruct->Llu
 *           = FACTORED: the matrix A is already factored.
 *                 Inputs:  all of ScalePermstruct
 *                          all of LUstruct
 *
 *         o Equil (yes_no_t)
 *           Specifies whether to equilibrate the system.
 *           = NO:  no equilibration.
 *           = YES: scaling factors are computed to equilibrate the system:
 *                      diag(R)*A*diag(C)*inv(diag(C))*X = diag(R)*B.
 *                  Whether or not the system will be equilibrated depends
 *                  on the scaling of the matrix A, but if equilibration is
 *                  used, A is overwritten by diag(R)*A*diag(C) and B by
 *                  diag(R)*B.
 *
 *         o RowPerm (rowperm_t)
 *           Specifies how to permute rows of the matrix A.
 *           = NATURAL:   use the natural ordering.
 *           = LargeDiag_MC64: use the Duff/Koster algorithm to permute rows of
 *                        the original matrix to make the diagonal large
 *                        relative to the off-diagonal.
 *           = LargeDiag_HPWM: use the parallel approximate-weight perfect
 *                        matching to permute rows of the original matrix
 *                        to make the diagonal large relative to the
 *                        off-diagonal.
 *           = MY_PERMR:  use the ordering given in ScalePermstruct->perm_r
 *                        input by the user.
 *
 *         o ColPerm (colperm_t)
 *           Specifies what type of column permutation to use to reduce fill.
 *           = NATURAL:       natural ordering.
 *           = MMD_AT_PLUS_A: minimum degree ordering on structure of A'+A.
 *           = MMD_ATA:       minimum degree ordering on structure of A'*A.
 *           = MY_PERMC:      the ordering given in ScalePermstruct->perm_c.
 *
 *         o ReplaceTinyPivot (yes_no_t)
 *           = NO:  do not modify pivots
 *           = YES: replace tiny pivots by sqrt(epsilon)*norm(A) during
 *                  LU factorization.
 *
 *         o IterRefine (IterRefine_t)
 *           Specifies how to perform iterative refinement.
 *           = NO:     no iterative refinement.
 *           = SLU_DOUBLE: accumulate residual in double precision.
 *           = SLU_EXTRA:  accumulate residual in extra precision.
 *
 *         NOTE: all options must be indentical on all processes when
 *               calling this routine.
 *
 * A (input) SuperMatrix* (local); A resides on all 3D processes.
 *         On entry, matrix A in A*X=B, of dimension (A->nrow, A->ncol).
 *           The number of linear equations is A->nrow. The type of A must be:
 *           Stype = SLU_NR_loc; Dtype = SLU_Z; Mtype = SLU_GE.
 *           That is, A is stored in distributed compressed row format.
 *           See supermatrix.h for the definition of 'SuperMatrix'.
 *           This routine only handles square A, however, the LU factorization
 *           routine PZGSTRF can factorize rectangular matrices.
 *
 *	   Internally, A is gathered on 2D processs grid-0, call it A2d.
 *         On exit, A2d may be overwtirren by diag(R)*A*diag(C)*Pc^T,
 *           depending on ScalePermstruct->DiagScale and options->ColPerm:
 *             if ScalePermstruct->DiagScale != NOEQUIL, A2d is overwritten by
 *                diag(R)*A*diag(C).
 *             if options->ColPerm != NATURAL, A2d is further overwritten by
 *                diag(R)*A*diag(C)*Pc^T.
 *           If all the above condition are true, the LU decomposition is
 *           performed on the matrix Pc*Pr*diag(R)*A*diag(C)*Pc^T.
 *
 * ScalePermstruct (input/output) zScalePermstruct_t* (global)
 *         The data structure to store the scaling and permutation vectors
 *         describing the transformations performed to the matrix A.
 *         It contains the following fields:
 *
 *         o DiagScale (DiagScale_t)
 *           Specifies the form of equilibration that was done.
 *           = NOEQUIL: no equilibration.
 *           = ROW:     row equilibration, i.e., A was premultiplied by
 *                      diag(R).
 *           = COL:     Column equilibration, i.e., A was postmultiplied
 *                      by diag(C).
 *           = BOTH:    both row and column equilibration, i.e., A was
 *                      replaced by diag(R)*A*diag(C).
 *           If options->Fact = FACTORED or SamePattern_SameRowPerm,
 *           DiagScale is an input argument; otherwise it is an output
 *           argument.
 *
 *         o perm_r (int*)
 *           Row permutation vector, which defines the permutation matrix Pr;
 *           perm_r[i] = j means row i of A is in position j in Pr*A.
 *           If options->RowPerm = MY_PERMR, or
 *           options->Fact = SamePattern_SameRowPerm, perm_r is an
 *           input argument; otherwise it is an output argument.
 *
 *         o perm_c (int*)
 *           Column permutation vector, which defines the
 *           permutation matrix Pc; perm_c[i] = j means column i of A is
 *           in position j in A*Pc.
 *           If options->ColPerm = MY_PERMC or options->Fact = SamePattern
 *           or options->Fact = SamePattern_SameRowPerm, perm_c is an
 *           input argument; otherwise, it is an output argument.
 *           On exit, perm_c may be overwritten by the product of the input
 *           perm_c and a permutation that postorders the elimination tree
 *           of Pc*A'*A*Pc'; perm_c is not changed if the elimination tree
 *           is already in postorder.
 *
 *         o R (double *) dimension (A->nrow)
 *           The row scale factors for A.
 *           If DiagScale = ROW or BOTH, A is multiplied on the left by
 *                          diag(R).
 *           If DiagScale = NOEQUIL or COL, R is not defined.
 *           If options->Fact = FACTORED or SamePattern_SameRowPerm, R is
 *           an input argument; otherwise, R is an output argument.
 *
 *         o C (double *) dimension (A->ncol)
 *           The column scale factors for A.
 *           If DiagScale = COL or BOTH, A is multiplied on the right by
 *                          diag(C).
 *           If DiagScale = NOEQUIL or ROW, C is not defined.
 *           If options->Fact = FACTORED or SamePattern_SameRowPerm, C is
 *           an input argument; otherwise, C is an output argument.
 *
 * B       (input/output) doublecomplex* (local)
 *         On entry, the right-hand side matrix of dimension (m_loc, nrhs),
 *           where, m_loc is the number of rows stored locally on my
 *           process and is defined in the data structure of matrix A.
 *         On exit, the solution matrix if info = 0;
 *
 * ldb     (input) int (local)
 *         The leading dimension of matrix B.
 *
 * nrhs    (input) int (global)
 *         The number of right-hand sides.
 *         If nrhs = 0, only LU decomposition is performed, the forward
 *         and back substitutions are skipped.
 *
 * grid    (input) gridinfo_t* (global)
 *         The 2D process mesh. It contains the MPI communicator, the number
 *         of process rows (NPROW), the number of process columns (NPCOL),
 *         and my process rank. It is an input argument to all the
 *         parallel routines.
 *         Grid can be initialized by subroutine SUPERLU_GRIDINIT.
 *         See superlu_zdefs.h for the definition of 'gridinfo_t'.
 *
 * LUstruct (input/output) zLUstruct_t*
 *         The data structures to store the distributed L and U factors.
 *         It contains the following fields:
 *
 *         o etree (int*) dimension (A->ncol) (global)
 *           Elimination tree of Pc*(A'+A)*Pc' or Pc*A'*A*Pc'.
 *           It is computed in sp_colorder() during the first factorization,
 *           and is reused in the subsequent factorizations of the matrices
 *           with the same nonzero pattern.
 *           On exit of sp_colorder(), the columns of A are permuted so that
 *           the etree is in a certain postorder. This postorder is reflected
 *           in ScalePermstruct->perm_c.
 *           NOTE:
 *           Etree is a vector of parent pointers for a forest whose vertices
 *           are the integers 0 to A->ncol-1; etree[root]==A->ncol.
 *
 *         o Glu_persist (Glu_persist_t*) (global)
 *           Global data structure (xsup, supno) replicated on all processes,
 *           describing the supernode partition in the factored matrices
 *           L and U:
 *	       xsup[s] is the leading column of the s-th supernode,
 *             supno[i] is the supernode number to which column i belongs.
 *
 *         o Llu (zLocalLU_t*) (local)
 *           The distributed data structures to store L and U factors.
 *           See superlu_ddefs.h for the definition of 'dLocalLU_t'.
 *
 * SOLVEstruct (input/output) zSOLVEstruct_t*
 *         The data structure to hold the communication pattern used
 *         in the phases of triangular solution and iterative refinement.
 *         This pattern should be intialized only once for repeated solutions.
 *         If options->SolveInitialized = YES, it is an input argument.
 *         If options->SolveInitialized = NO and nrhs != 0, it is an output
 *         argument. See superlu_zdefs.h for the definition of 'zSOLVEstruct_t'.
 *
 * berr    (output) double*, dimension (nrhs) (global)
 *         The componentwise relative backward error of each solution
 *         vector X(j) (i.e., the smallest relative change in
 *         any element of A or B that makes X(j) an exact solution).
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics on runtime and floating-point operation count.
 *        See util_dist.h for the definition of 'SuperLUStat_t'.
 *
 * info    (output) int*
 *         = 0: successful exit
 *         < 0: if info = -i, the i-th argument had an illegal value
 *         > 0: if info = i, and i is
 *             <= A->ncol: U(i,i) is exactly zero. The factorization has
 *                been completed, but the factor U is exactly singular,
 *                so the solution could not be computed.
 *             > A->ncol: number of bytes allocated when memory allocation
 *                failure occurred, plus A->ncol.
 *
 * See superlu_zdefs.h for the definitions of varioous data types.
 * </pre>
 */

// Sherry: TODO - this should be moved to a utility file.
int zwriteLUtoDisk(int nsupers, int_t *xsup, zLUstruct_t *LUstruct)
{

	if (getenv("LUFILE"))
	{
		FILE *fp = fopen(getenv("LUFILE"), "w");
		printf("writing to %s", getenv("LUFILE"));
		for (int i = 0; i < nsupers; i++)
		{
			if (LUstruct->Llu->Lrowind_bc_ptr[i])
			{
				int_t *lsub = LUstruct->Llu->Lrowind_bc_ptr[i];
				doublecomplex *nzval = LUstruct->Llu->Lnzval_bc_ptr[i];

				int_t len = lsub[1]; /* LDA of the nzval[] */
				int_t len2 = SuperSize(i) * len;
				fwrite(nzval, sizeof(doublecomplex), len2, fp); // assume fp will be incremented
			}

			if (LUstruct->Llu->Ufstnz_br_ptr[i])
			{
				int_t *usub = LUstruct->Llu->Ufstnz_br_ptr[i];
				doublecomplex *nzval = LUstruct->Llu->Unzval_br_ptr[i];
				int_t lenv = usub[1];

				fwrite(nzval, sizeof(doublecomplex), lenv, fp); // assume fp will be incremented
			}
		}

		fclose(fp);
	}
	else
	{
		printf("Please set environment variable LUFILE to write\n..bye bye");
		exit(0);
	}

	return 0;
}

#define EPSILON 1e-3

// Sherry: TODO - this is used for debugging
static int zcheckArr(doublecomplex *A, doublecomplex *B, int n)
{
	for (int i = 0; i < n; i++)
	{
		doublecomplex temp;
        z_sub(&temp, &A[i], &B[i]);
		assert(slud_z_abs(&temp) <= EPSILON * SUPERLU_MIN(slud_z_abs(&A[i]), slud_z_abs(&B[i])));
	}

	return 0;
}

// Sherry: TODO - this should be moved to a utility file.
int zcheckLUFromDisk(int nsupers, int_t *xsup, zLUstruct_t *LUstruct)
{
	zLocalLU_t *Llu = LUstruct->Llu;

	doublecomplex *Lval_buf = doublecomplexMalloc_dist(Llu->bufmax[1]); // DOUBLE_ALLOC(Llu->bufmax[1]);
	doublecomplex *Uval_buf = doublecomplexMalloc_dist(Llu->bufmax[3]); // DOUBLE_ALLOC(Llu->bufmax[3]);

	if (getenv("LUFILE"))
	{
		FILE *fp = fopen(getenv("LUFILE"), "r");
		printf("reading from %s", getenv("LUFILE"));
		for (int i = 0; i < nsupers; i++)
		{
			if (LUstruct->Llu->Lrowind_bc_ptr[i])
			{
				int_t *lsub = LUstruct->Llu->Lrowind_bc_ptr[i];
				doublecomplex *nzval = LUstruct->Llu->Lnzval_bc_ptr[i];

				int_t len = lsub[1]; /* LDA of the nzval[] */
				int_t len2 = SuperSize(i) * len;
				fread(Lval_buf, sizeof(doublecomplex), len2, fp); // assume fp will be incremented
				zcheckArr(nzval, Lval_buf, len2);
			}

			if (LUstruct->Llu->Ufstnz_br_ptr[i])
			{
				int_t *usub = LUstruct->Llu->Ufstnz_br_ptr[i];
				doublecomplex *nzval = LUstruct->Llu->Unzval_br_ptr[i];
				int_t lenv = usub[1];

				fread(Uval_buf, sizeof(doublecomplex), lenv, fp); // assume fp will be incremented
				zcheckArr(nzval, Uval_buf, lenv);
			}
		}
		printf("CHecking LU from  %s is succesful ", getenv("LUFILE"));
		fclose(fp);
	}
	else
	{
		printf("Please set environment variable LUFILE to read\n..bye bye");
		exit(0);
	}

	return 0;
}

// Sherry: TODO - this should be moved to a utility file.
/*! \brief Dump the factored matrix L using matlab triple-let format
 */
void zDumpLblocks3D(int_t nsupers, gridinfo3d_t *grid3d,
		  Glu_persist_t *Glu_persist, zLocalLU_t *Llu)
{
    register int c, extra, gb, j, i, lb, nsupc, nsupr, len, nb, ncb;
    int k, mycol, r, n, nmax;
    int_t nnzL;
    int_t *xsup = Glu_persist->xsup;
    int_t *index;
    doublecomplex *nzval;
	char filename[256];
	FILE *fp, *fopen();
	gridinfo_t *grid = &(grid3d->grid2d);
	int iam = grid->iam;
	int iam3d = grid3d->iam;

	// assert(grid->npcol*grid->nprow==1);

	// count nonzeros in the first pass
	nnzL = 0;
	n = 0;
    ncb = nsupers / grid->npcol;
    extra = nsupers % grid->npcol;
    mycol = MYCOL( iam, grid );
    if ( mycol < extra ) ++ncb;
    for (lb = 0; lb < ncb; ++lb) {
	index = Llu->Lrowind_bc_ptr[lb];
	if ( index ) { /* Not an empty column */
	    nzval = Llu->Lnzval_bc_ptr[lb];
	    nb = index[0];
	    nsupr = index[1];
	    gb = lb * grid->npcol + mycol;
	    nsupc = SuperSize( gb );
	    for (c = 0, k = BC_HEADER, r = 0; c < nb; ++c) {
		len = index[k+1];

		for (j = 0; j < nsupc; ++j) {
		for (i=0; i<len; ++i){

		if(index[k+LB_DESCRIPTOR+i]+1>=xsup[gb]+j+1){
			nnzL ++;
			nmax = SUPERLU_MAX(n,index[k+LB_DESCRIPTOR+i]+1);
			n = nmax;
		}

		}
		}
		k += LB_DESCRIPTOR + len;
		r += len;
	    }
	}
    }
	MPI_Allreduce(MPI_IN_PLACE,&nnzL,1,mpi_int_t,MPI_SUM,grid->comm);
	MPI_Allreduce(MPI_IN_PLACE,&n,1,mpi_int_t,MPI_MAX,grid->comm);

	snprintf(filename, sizeof(filename), "%s-%d", "L", iam3d);
    printf("Dumping L factor to --> %s\n", filename);
 	if ( !(fp = fopen(filename, "w")) ) {
			ABORT("File open failed");
		}

	if(grid->iam==0){
		fprintf(fp, "%d %d " IFMT "\n", n,n,nnzL);
	}

     ncb = nsupers / grid->npcol;
    extra = nsupers % grid->npcol;
    mycol = MYCOL( iam, grid );
    if ( mycol < extra ) ++ncb;
    for (lb = 0; lb < ncb; ++lb) {
	index = Llu->Lrowind_bc_ptr[lb];
	if ( index ) { /* Not an empty column */
	    nzval = Llu->Lnzval_bc_ptr[lb];
	    nb = index[0];
	    nsupr = index[1];
	    gb = lb * grid->npcol + mycol;
	    nsupc = SuperSize( gb );
	    for (c = 0, k = BC_HEADER, r = 0; c < nb; ++c) {
		len = index[k+1];

		for (j = 0; j < nsupc; ++j) {
		for (i=0; i<len; ++i){
			fprintf(fp, IFMT IFMT " %e %e\n", index[k+LB_DESCRIPTOR+i]+1, xsup[gb]+j+1, nzval[r +i+ j*nsupr].r,nzval[r +i+ j*nsupr].i);
		}
		}
		k += LB_DESCRIPTOR + len;
		r += len;
	    }
	}
    }
 	fclose(fp);

} /* zDumpLblocks3D */



void pzgssvx3d(superlu_dist_options_t *options, SuperMatrix *A,
			   zScalePermstruct_t *ScalePermstruct,
			   doublecomplex B[], int ldb, int nrhs, gridinfo3d_t *grid3d,
			   zLUstruct_t *LUstruct, zSOLVEstruct_t *SOLVEstruct,
			   double *berr, SuperLUStat_t *stat, int *info)
{
    NRformat_loc *Astore = A->Store;
    SuperMatrix GA; /* Global A in NC format */
    NCformat *GAstore;
    doublecomplex *a_GA;

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    Glu_freeable_t *Glu_freeable = NULL;
	/* The nonzero structures of L and U factors, which are
	   replicated on all processrs.
	   (lsub, xlsub) contains the compressed subscript of
	   supernodes in L.
	   (usub, xusub) contains the compressed subscript of
	   nonzero segments in U.
	   If options->Fact != SamePattern_SameRowPerm, they are
	   computed by SYMBFACT routine, and then used by PDDISTRIBUTE
	   routine. They will be freed after PDDISTRIBUTE routine.
	   If options->Fact == SamePattern_SameRowPerm, these
	   structures are not used.                                  */
    yes_no_t parSymbFact = options->ParSymbFact;
    fact_t Fact;
    doublecomplex *a;
    int_t *colptr, *rowind;
    int_t *perm_r;			/* row permutations from partial pivoting */
    int_t *perm_c;			/* column permutation vector */
    int_t *etree;			/* elimination tree */
    int_t *rowptr, *colind; /* Local A in NR */
    int colequ, Equil, factored, job, notran, rowequ, need_value;
    int_t i, j, irow, m, n, nnz;
    int_t nnz_loc, m_loc, fst_row, icol;
    int iam, iinfo, permc_spec;
    int ldx; /* LDA for matrix X (local). */
    char equed[1], norm[1];
    double *C, *R, *C1, *R1, amax, anorm, colcnd, rowcnd;
    doublecomplex *X, *b_col, *b_work, *x_col;
    double t;
    float GA_mem_use;	/* memory usage by global A */
    float dist_mem_use; /* memory usage during distribution */
    superlu_dist_mem_usage_t num_mem_usage, symb_mem_usage;
    float flinfo; /* track memory usage of parallel symbolic factorization */
    bool Solve3D = true;
    int_t nsupers;
#if (PRNTlevel >= 2)
    double dmin, dsum, dprod;
#endif

    ztrf3Dpartition_t *trf3Dpartition=LUstruct->trf3Dpart;
    int gpu3dVersion = 1;  // default is to use C++ code in CplusplusFactor/ directory
#ifdef GPU_ACC
    if (getenv("GPU3DVERSION")) {
       gpu3dVersion = atoi(getenv("GPU3DVERSION"));
    }

    LUgpu_Handle LUgpu;
#endif

    LUstruct->dt = 'z';

    // get the 2d grid
    gridinfo_t *grid = &(grid3d->grid2d);
    iam = grid->iam;

    /* Test the options choices. */
    *info = 0;
    Fact = options->Fact;

    validateInput_pzgssvx3d(options, A, ldb, nrhs, grid3d, info);

    /* Initialization. */

    options->Algo3d = YES;

    /* definition of factored seen by each process layer */
    factored = (Fact == FACTORED);

    /* Save the inputs: ldb -> ldb3d, and B -> B3d, Astore -> Astore3d,
       so that the names {ldb, B, and Astore} can be used internally.
       B3d and Astore3d will be assigned back to B and Astore on return.*/
    int ldb3d = ldb;
    NRformat_loc *Astore3d = (NRformat_loc *)A->Store;
    NRformat_loc3d *A3d = SOLVEstruct->A3d;

    /* B3d is aliased to B;
       B2d is allocated;
       B is then aliased to B2d for the following 2D solve;
    */
    zGatherNRformat_loc3d_allgrid(Fact, (NRformat_loc *)A->Store,
				     B, ldb, nrhs, grid3d, &A3d);

    B = (doublecomplex *)A3d->B2d; /* B is now pointing to B2d,
			   allocated in zGatherNRformat_loc3d.  */
    // PrintDouble5("after gather B=B2d", ldb, B);

    SOLVEstruct->A3d = A3d; /* This structure need to be persistent across
				   multiple calls of pzgssvx3d()   */

    NRformat_loc *Astore0 = A3d->A_nfmt; // on all grids
    NRformat_loc *A_orig = A->Store;
//////

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Enter pzgssvx3d()");
#endif

    /* Perform preprocessing steps on process layer zero, including:
       gather 3D matrices {A, B} onto 2D grid-0, preprocessing steps:
	   - equilibration,
	   - ordering,
	   - symbolic factorization,
	   - distribution of L & U                                      */

    m = A->nrow;
    n = A->ncol;
    // checkNRFMT(Astore0, (NRformat_loc *) A->Store);

    // On input, A->Store is on 3D, now A->Store is re-assigned to 2D store
    A->Store = Astore0; // on all grids
    ldb = Astore0->m_loc;

    /* The following code now works on all grids */
    Astore = (NRformat_loc *)A->Store;
    nnz_loc = Astore->nnz_loc;
    m_loc = Astore->m_loc;
    fst_row = Astore->fst_row;
    a = (doublecomplex *)Astore->nzval;
    rowptr = Astore->rowptr;
    colind = Astore->colind;

    /* Structures needed for parallel symbolic factorization */
    int_t *sizes, *fstVtxSep;
    int noDomains, nprocs_num;
    MPI_Comm symb_comm; /* communicator for symbolic factorization */
    int col, key;		/* parameters for creating a new communicator */
    Pslu_freeable_t Pslu_freeable;

    sizes = NULL;
    fstVtxSep = NULL;
    symb_comm = MPI_COMM_NULL;

    Equil = (!factored && options->Equil == YES);
    notran = (options->Trans == NOTRANS);
    iam = grid->iam;

    if (grid3d->zscp.Iam == 0) { /* on 2D grid-0 */
	/* The following code now works on 2D grid-0 */

	job = 5;
	/* Extract equilibration status from a previous factorization */
	if (factored || (Fact == SamePattern_SameRowPerm && Equil)) {
	   rowequ = (ScalePermstruct->DiagScale == ROW) ||
			 (ScalePermstruct->DiagScale == BOTH);
	   colequ = (ScalePermstruct->DiagScale == COL) ||
			 (ScalePermstruct->DiagScale == BOTH);
	} else {
	   rowequ = colequ = FALSE;
	}

	/* Not factored & ask for equilibration, then alloc R & C */
	if (Equil && Fact != SamePattern_SameRowPerm)
	     zallocScalePermstruct_RC(ScalePermstruct, m, n);

	/* The following arrays are replicated on all processes. */
	perm_r = ScalePermstruct->perm_r;
	perm_c = ScalePermstruct->perm_c;
	etree = LUstruct->etree;
	R = ScalePermstruct->R;
	C = ScalePermstruct->C;

	/* ------------------------------------------------------------
	   Diagonal scaling to equilibrate the matrix.
	   ------------------------------------------------------------ */
	if (Equil) {
	    zscaleMatrixDiagonally(Fact, ScalePermstruct,
				  A, stat, grid, &rowequ, &colequ, &iinfo);
	    if (iinfo < 0)
		return; // Sherry: TOO - return a number in INFO

	} /* end if Equil ... LAPACK style, not involving MC64 */

	if (!factored) { /* Skip this if already factored. */
	    /*
	     * Gather A from the distributed compressed row format to
	     * global A in compressed column format.
	     * Numerical values are gathered only when a row permutation
	     ** for large diagonal is sought after.
	     */
	    if (Fact != SamePattern_SameRowPerm &&
			(parSymbFact == NO || options->RowPerm != NO))
	    {
		int_t need_value = (options->RowPerm == LargeDiag_MC64);
		pzCompRow_loc_to_CompCol_global(need_value, A, grid, &GA);
		GAstore = (NCformat *)GA.Store;
		nnz = GAstore->nnz;
		GA_mem_use = (nnz + n + 1) * sizeof(int_t) + need_value * nnz * sizeof(doublecomplex);
		if (!need_value)
		    assert(GAstore->nzval == NULL);
	    }

	    /* ------------------------------------------------------------
	       Find the row permutation for A.
	       ------------------------------------------------------------ */
	    zperform_row_permutation(options, Fact, ScalePermstruct, LUstruct,
				m, n, grid, A, &GA, stat, job, Equil,
				&rowequ, &colequ, &iinfo);

	} /* end if (!factored) */

	/* Compute norm(A), which will be used to adjust small diagonal. */
	if (!factored || options->IterRefine)
	    anorm = zcomputeA_Norm(notran, A, grid);

	/* ------------------------------------------------------------
	   Perform ordering and symbolic factorization
	   ------------------------------------------------------------ */
	if (!factored) {
	    t = SuperLU_timer_();
	    /*
	     * Get column permutation vector perm_c[], according to permc_spec:
	     *   permc_spec = NATURAL:  natural ordering
	     *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
	     *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
	     *   permc_spec = METIS_AT_PLUS_A: METIS on structure of A'+A
	     *   permc_spec = PARMETIS: parallel METIS on structure of A'+A
	     *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
	     */
	    permc_spec = options->ColPerm;

	    if (parSymbFact == YES || permc_spec == PARMETIS) {
		if(grid3d->npdep!=1){
    		    fprintf(stderr, "Error: ParMETIS and Parallel Symbolic Factorization are not yet supported with grid3d->npdep>1.\n");
			return; // or exit(-1); if you want to terminate the program
		}
		nprocs_num = grid->nprow * grid->npcol;
		noDomains = (int)(pow(2, ((int)LOG2(nprocs_num))));

		/* create a new communicator for the first noDomains
		   processes in grid->comm */
		key = iam;
		if (iam < noDomains)
			col = 0;
		else
			col = MPI_UNDEFINED;
		MPI_Comm_split(grid->comm, col, key, &symb_comm);

		if (permc_spec == NATURAL || permc_spec == MY_PERMC) {
		    if (permc_spec == NATURAL) {
			for (j = 0; j < n; ++j)	perm_c[j] = j;
		    }
		    if (!(sizes = intMalloc_dist(2 * noDomains)))
			ABORT("SUPERLU_MALLOC fails for sizes.");
		    if (!(fstVtxSep = intMalloc_dist(2 * noDomains)))
			ABORT("SUPERLU_MALLOC fails for fstVtxSep.");
		    for (i = 0; i < 2 * noDomains - 2; ++i) {
			sizes[i] = 0;
			fstVtxSep[i] = 0;
		    }
		    sizes[2 * noDomains - 2] = m;
		    fstVtxSep[2 * noDomains - 2] = 0;
		} else if (permc_spec != PARMETIS) {
		    /* same as before */
		    printf("{%4d,%4d}: pzgssvx3d: invalid ColPerm option when ParSymbfact is used\n",
				  (int)MYROW(grid->iam, grid), (int)MYCOL(grid->iam, grid));
		}
	    } /* end ... use parmetis */

	    if (permc_spec != MY_PERMC && Fact == DOFACT) {
		if (permc_spec == PARMETIS) {
		/* Get column permutation vector in perm_c.                   *
		 * This routine takes as input the distributed input matrix A *
		 * and does not modify it.  It also allocates memory for      *
		 * sizes[] and fstVtxSep[] arrays, that contain information   *
		 * on the separator tree computed by ParMETIS.                */
		    flinfo = get_perm_c_parmetis(A, perm_r, perm_c, nprocs_num,
						 noDomains, &sizes, &fstVtxSep,
						 grid, &symb_comm);
		    if (flinfo > 0)
			ABORT("ERROR in get perm_c parmetis.");
		} else {
		    get_perm_c_dist(iam, permc_spec, &GA, perm_c);
		}
	    }

	    stat->utime[COLPERM] = SuperLU_timer_() - t;

	    /* Compute the elimination tree of Pc*(A'+A)*Pc' or Pc*A'*A*Pc'
	       (a.k.a. column etree), depending on the choice of ColPerm.
	       Adjust perm_c[] to be consistent with a postorder of etree.
	       Permute columns of A to form A*Pc'. */
	    if (Fact != SamePattern_SameRowPerm) {
		if (parSymbFact == NO)
		{
		    /* Allocating Glu_freeable used by symbfact */
		    if (!(Glu_freeable = (Glu_freeable_t *)
			  SUPERLU_MALLOC(sizeof(Glu_freeable_t))))
				ABORT("Malloc fails for Glu_freeable.");
		    permCol_SymbolicFact3d(options, n, &GA, perm_c, etree,
					   Glu_persist, Glu_freeable, stat,
					   &symb_mem_usage,
					   grid3d);

		} /* end serial symbolic factorization */
		else { /* parallel symbolic factorization */
		    //TODO: need a 3D version of symbfact_dist
		    t = SuperLU_timer_();
		    flinfo = symbfact_dist(options, nprocs_num, noDomains,
					  A, perm_c, perm_r,
					  sizes, fstVtxSep, &Pslu_freeable,
					  &(grid->comm), &symb_comm,
					  &symb_mem_usage);
		    stat->utime[SYMBFAC] = SuperLU_timer_() - t;
		    if (flinfo > 0)
		        ABORT("Insufficient memory for parallel symbolic factorization.");
		}

		/* Destroy GA */
		if (parSymbFact == NO || options->RowPerm != NO)
		    Destroy_CompCol_Matrix_dist(&GA);

	    } /* end if Fact not SamePattern_SameRowPerm */
	} /* end if not Factored */
    } /* end 2D process layer 0 */

    MPI_Bcast(&rowequ, 1, MPI_INT, 0, grid3d->zscp.comm);
    MPI_Bcast(&colequ, 1, MPI_INT, 0, grid3d->zscp.comm);

    /* Broadcast Permuted A and symbolic factorization data from 2d to 3d grid*/
    if (Fact != SamePattern_SameRowPerm && !factored) // place the exact conditions later //all the grid must execute this
    {
	if (parSymbFact == NO) {
		if (Glu_freeable == NULL)
		{
		    if (!(Glu_freeable = (Glu_freeable_t *)
				SUPERLU_MALLOC(sizeof(Glu_freeable_t))))
			ABORT("Malloc fails for Glu_freeable.");
		}
		zbcastPermutedSparseA(A, ScalePermstruct, Glu_freeable,
				     LUstruct, grid3d);
		} else {
		    //TODO: need a parmetis version of zbcastPermutedSparseA broadcasting Pslu_freeable
		}
	}

	perm_r = ScalePermstruct->perm_r;
	perm_c = ScalePermstruct->perm_c;
	etree = LUstruct->etree;
	R = ScalePermstruct->R;
	C = ScalePermstruct->C;
	nsupers = getNsupers(n, LUstruct->Glu_persist);
	Astore = (NRformat_loc *)A->Store;
	a = (doublecomplex *)Astore->nzval;
	rowptr = Astore->rowptr;
	colind = Astore->colind;
	Glu_persist = LUstruct->Glu_persist;

	// perform the  3D distribution
	if (!factored)
	{
		/* Apply column permutation to the original distributed A */
		for (j = 0; j < nnz_loc; ++j)
			colind[j] = perm_c[colind[j]];
		// free quauntities used in Parmetis
		if (sizes)
			SUPERLU_FREE(sizes);
		if (fstVtxSep)
			SUPERLU_FREE(fstVtxSep);
		if (symb_comm != MPI_COMM_NULL)
			MPI_Comm_free(&symb_comm);
		if ( Fact != SamePattern_SameRowPerm){
			LUstruct->trf3Dpart = SUPERLU_MALLOC(sizeof(dtrf3Dpartition_t));
			// computes the new partition for 3D factorization here
			trf3Dpartition=LUstruct->trf3Dpart;
			znewTrfPartitionInit(nsupers, LUstruct, grid3d);
		}
	}

	// perform the  3D distribution
	if (!factored)
	{ /* Skip this if already factored. */

		if (parSymbFact == NO || Fact == SamePattern_SameRowPerm)
		{


			/* Distribute Pc*Pr*diag(R)*A*diag(C)*Pc' into L and U storage.
				NOTE: the row permutation Pc*Pr is applied internally in the
				distribution routine. */
			t = SuperLU_timer_();

			dist_mem_use = pzdistribute3d_Yang(options, n, A, ScalePermstruct,
											Glu_freeable, LUstruct, grid3d);
			stat->utime[DIST] = SuperLU_timer_() - t;

			/* Deallocate storage used in symbolic factorization. */
			if (Fact != SamePattern_SameRowPerm)
			{
				iinfo = symbfact_SubFree(Glu_freeable);
				SUPERLU_FREE(Glu_freeable);
			}
		}
		else
		{
			/* Distribute Pc*Pr*diag(R)*A*diag(C)*Pc' into L and U storage.
				NOTE: the row permutation Pc*Pr is applied internally in the
				distribution routine. */

			// TODO: need a 3D version of zdist_psymbtonum
			t = SuperLU_timer_();
			dist_mem_use = zdist_psymbtonum(options, n, A, ScalePermstruct,
											&Pslu_freeable, LUstruct, grid);
			if (dist_mem_use > 0)
				ABORT("Not enough memory available for dist_psymbtonum\n");

			stat->utime[DIST] = SuperLU_timer_() - t;

			ABORT("zdist_psymbtonum does not yet work with 3D factorization\n");

		}

		/* Flatten L metadata into one buffer. */
		if ( Fact != SamePattern_SameRowPerm ) {
			pzflatten_LDATA(options, n, LUstruct, grid, stat);
		}


		if(Fact != SamePattern_SameRowPerm){
			// checkDist3DLUStruct(LUstruct, grid3d);
			// zeros out the Supernodes that are not owned by the grid
			zinit3DLUstructForest(trf3Dpartition->myTreeIdxs, trf3Dpartition->myZeroTrIdxs,
									trf3Dpartition->sForests, LUstruct, grid3d);

			zLUValSubBuf_t *LUvsb = SUPERLU_MALLOC(sizeof(zLUValSubBuf_t));
			zLluBufInit(LUvsb, LUstruct);
			trf3Dpartition->LUvsb = LUvsb;
			trf3Dpartition->iperm_c_supno = create_iperm_c_supno(nsupers, options, LUstruct->Glu_persist, LUstruct->etree, LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Ufstnz_br_ptr, grid3d);
		}


		/*if (!iam) printf ("\tDISTRIBUTE time  %8.2f\n", stat->utime[DIST]); */

		MPI_Bcast(&anorm, 1, MPI_DOUBLE, 0, grid3d->zscp.comm);

		/* Perform numerical factorization in parallel on all process layers.*/

		/* nvshmem related. The nvshmem_malloc has to be called before ztrs_compute_communication_structure, otherwise solve is much slower*/
		#ifdef HAVE_NVSHMEM
			int nc = CEILING( nsupers, grid->npcol);
			int nr = CEILING( nsupers, grid->nprow);
			int flag_bc_size = RDMA_FLAG_SIZE * (nc+1);
			int flag_rd_size = RDMA_FLAG_SIZE * nr * 2;
			int my_flag_bc_size = RDMA_FLAG_SIZE * (nc+1);
			int my_flag_rd_size = RDMA_FLAG_SIZE * nr * 2;
			int maxrecvsz = sp_ienv_dist(3, options)* nrhs + SUPERLU_MAX( XK_H, LSUM_H );
			int ready_x_size = maxrecvsz*nc;
			int ready_lsum_size = 2*maxrecvsz*nr;
			if (get_acc_solve()){
			nv_init_wrapper(grid->comm);
		    zprepare_multiGPU_buffers(flag_bc_size,flag_rd_size,ready_x_size,ready_lsum_size,my_flag_bc_size,my_flag_rd_size);
			}
		#endif





		SCT_t *SCT = (SCT_t *)SUPERLU_MALLOC(sizeof(SCT_t));
		slu_SCT_init(SCT);

#if (PRNTlevel >= 1)
		if (grid3d->iam == 0)
		{
			printf("After 3D initialization.\n");
			fflush(stdout);
		}
#endif





		t = SuperLU_timer_();

		/*factorize in grid 1*/
		// if(grid3d->zscp.Iam)
		// get environment variable TRF3DVERSION
#ifdef GPU_ACC
		if (gpu3dVersion == 1)
		{ /* this is the new C++ code in CplusplusFactor/ directory */

#if (PRNTlevel >= 1)
			if (!grid3d->iam)
			    printf("Using pzgstrf3d+gpu version 1 in CplusplusFactor/\n");
#endif
				
#if 0
			pzgstrf3d_upacked(options, m, n, anorm, trf3Dpartition, SCT, LUstruct,
				  grid3d, stat, info);
#else
			int_t ldt = sp_ienv_dist(3, options); /* Size of maximum supernode */
			double s_eps = smach_dist("Epsilon");
			double thresh = s_eps * anorm;

#define TEMPLATED_VERSION
#ifdef TEMPLATED_VERSION
zLUgpu_Handle zLUgpu = zCreateLUgpuHandle(nsupers, ldt, trf3Dpartition, LUstruct, grid3d,
						  SCT, options, stat, thresh, info);

			/* call pzgstrf3d() in C++ code */
			pzgstrf3d_LUv1(zLUgpu);

			zCopyLUGPU2Host(zLUgpu, LUstruct);
			zDestroyLUgpuHandle(zLUgpu);
		    //TODO: zCreateLUgpuHandle,pzgstrf3d_LUpackedInterface,zCopyLUGPU2Host,zDestroyLUgpuHandle haven't been created
#else
			/* call constructor in C++ code */
			LUgpu = zCreateLUgpuHandle(nsupers, ldt, trf3Dpartition, LUstruct, grid3d,
						  SCT, options, stat, thresh, info);

			/* call pzgstrf3d() in C++ code */
			pzgstrf3d_LUpackedInterface(LUgpu);

			copyLUGPU2Host(LUgpu, LUstruct);
			destroyLUgpuHandle(LUgpu);
#endif /* TEMPLATED_VERSION */

			// print other stuff
			// if (!grid3d->zscp.Iam)
			// 	SCT_printSummary(grid, SCT);
			reduceStat(FACT, stat, grid3d);
#endif /* matching #if 0 #else */
		}
		else /* this is the old C code, with less GPU offload */
#endif /* matching ifdef GPU_ACC */
		{

			pzgstrf3d(options, m, n, anorm, trf3Dpartition, SCT, LUstruct,
					  grid3d, stat, info);

			// zDumpLblocks3D(nsupers, grid3d, LUstruct->Glu_persist, LUstruct->Llu);


		}

		if (get_new3dsolve()){
			zbroadcastAncestor3d(trf3Dpartition, LUstruct, grid3d, SCT);
		}

		if ( options->Fact != SamePattern_SameRowPerm) {
			if (get_new3dsolve() && Solve3D==true){
				ztrs_compute_communication_structure(options, n, LUstruct,
							ScalePermstruct, trf3Dpartition->supernodeMask, grid, stat);
			}else{
				int* supernodeMask = int32Malloc_dist(nsupers);
				for(int ii=0; ii<nsupers; ii++)
					supernodeMask[ii]=1;
				ztrs_compute_communication_structure(options, n, LUstruct,
							ScalePermstruct, supernodeMask, grid, stat);
				SUPERLU_FREE(supernodeMask);
			}
		}


		stat->utime[FACT] = SuperLU_timer_() - t;

		/*factorize in grid 1*/
		// if(grid3d->zscp.Iam)
		double tgather = SuperLU_timer_();
		if(Solve3D==false){
		zgatherAllFactoredLU(trf3Dpartition, LUstruct, grid3d, SCT);
		}
		SCT->gatherLUtimer += SuperLU_timer_() - tgather;
		/*print stats for bottom grid*/

		// Write LU to file
		int writeLU = 0;
		if (getenv("WRITELU"))
		{
			writeLU = atoi(getenv("WRITELU"));
		}

		if (writeLU)
		{
			if (!grid3d->zscp.Iam)
				zwriteLUtoDisk(nsupers, LUstruct->Glu_persist->xsup, LUstruct);
		}

		int checkLU = 0;
		if (getenv("CHECKLU"))
		{
			checkLU = atoi(getenv("CHECKLU"));
		}

		if (checkLU)
		{
			if (!grid3d->zscp.Iam)
				zcheckLUFromDisk(nsupers, LUstruct->Glu_persist->xsup, LUstruct);
		}

#if (PRNTlevel >= 1)
		if (!grid3d->zscp.Iam)
		{
			slu_SCT_print(grid, SCT);
			slu_SCT_print3D(grid3d, SCT);
		}
		slu_SCT_printComm3D(grid3d, SCT);

		/*print memory usage*/
		z3D_printMemUse(trf3Dpartition, LUstruct, grid3d);

		SCT->gatherLUtimer += SuperLU_timer_() - tgather;
		/*print stats for bottom grid*/
		/*print forest weight and costs*/
		printForestWeightCost(trf3Dpartition->sForests, SCT, grid3d);
		/*reduces stat from all the layers*/
#endif

		slu_SCT_free(SCT);

	} /* end if not Factored ... factor on all process layers */

	if (grid3d->zscp.Iam == 0 )
	{ // only process layer 0
		if (!factored)
		{
			if (options->PrintStat)
			{
				int_t TinyPivots;
				float for_lu, total, avg, loc_max;
				float mem_stage[3];
				struct { float val; int rank; } local_struct, global_struct;

				MPI_Reduce( &stat->TinyPivots, &TinyPivots, 1, mpi_int_t,
						   MPI_SUM, 0, grid->comm );
				stat->TinyPivots = TinyPivots;

				/*-- Compute high watermark of all stages --*/
				if (parSymbFact == TRUE)
				{
					/* The memory used in the redistribution routine
				   includes the memory used for storing the symbolic
				   structure and the memory allocated for numerical
				   factorization */
					mem_stage[0] = (-flinfo); /* symbfact step */
					mem_stage[1] = (-dist_mem_use);      /* distribution step */
					loc_max = SUPERLU_MAX( mem_stage[0], mem_stage[1]);
					if (options->RowPerm != NO )
						loc_max = SUPERLU_MAX(loc_max, GA_mem_use);
				}
				else
				{
					mem_stage[0] = symb_mem_usage.total + GA_mem_use; /* symbfact step */
					mem_stage[1] = symb_mem_usage.for_lu + dist_mem_use + num_mem_usage.for_lu;            /* distribution step */
					loc_max = SUPERLU_MAX(mem_stage[0], mem_stage[1] );
				}

				zQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);
				mem_stage[2] = num_mem_usage.total;  /* numerical factorization step */

				loc_max = SUPERLU_MAX(loc_max, mem_stage[2] ); /* local max of 3 stages */

				local_struct.val = loc_max;
				local_struct.rank = grid->iam;
				MPI_Reduce( &local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid->comm );
				int all_highmark_rank = global_struct.rank;
				float all_highmark_mem = global_struct.val * 1e-6;

				MPI_Reduce( &loc_max, &avg,
						   1, MPI_FLOAT, MPI_SUM, 0, grid->comm );
				MPI_Reduce( &num_mem_usage.for_lu, &for_lu,
						   1, MPI_FLOAT, MPI_SUM, 0, grid->comm );
				MPI_Reduce( &num_mem_usage.total, &total,
						   1, MPI_FLOAT, MPI_SUM, 0, grid->comm );

				/*-- Compute memory usage of numerical factorization --*/
				local_struct.val = num_mem_usage.for_lu;
				MPI_Reduce(&local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid->comm);
				int lu_max_rank = global_struct.rank;
				float lu_max_mem = global_struct.val * 1e-6;

				local_struct.val = stat->peak_buffer;
				MPI_Reduce( &local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid->comm );
	        	int buffer_peak_rank = global_struct.rank;
	        	float buffer_peak = global_struct.val*1e-6;
				if (iam == 0)
				{
					printf("\n** Memory Usage **********************************\n");
					printf("** Total highmark (MB):\n"
						   "    Sum-of-all : %8.2f | Avg : %8.2f  | Max : %8.2f\n",
						   avg * 1e-6,
						   avg / grid->nprow / grid->npcol * 1e-6,
						   all_highmark_mem);
					printf("    Max at rank %d, different stages (MB):\n"
						   "\t. symbfact        %8.2f\n"
						   "\t. distribution    %8.2f\n"
						   "\t. numfact         %8.2f\n",
						   all_highmark_rank, mem_stage[0] * 1e-6, mem_stage[1] * 1e-6, mem_stage[2] * 1e-6);
					printf("** NUMfact space (MB): (sum-of-all-processes)\n"
						   "    L\\U :        %8.2f |  Total : %8.2f\n",
						   for_lu * 1e-6, total * 1e-6);
					printf("\t. max at rank %d, max L+U memory (MB): %8.2f\n"
						   "\t. max at rank %d, peak buffer (MB):    %8.2f\n",
						   lu_max_rank, lu_max_mem,
						   buffer_peak_rank, buffer_peak);
					printf("**************************************************\n\n");
					printf("** number of Tiny Pivots: %8d\n\n", stat->TinyPivots);
					fflush(stdout);
				}
			} /* end printing stats */

		} /* end if not Factored */
    }

		if(Solve3D){

			if ( options->Fact == DOFACT || options->Fact == SamePattern ) {
			/* Need to reset the solve's communication pattern,
			because perm_r[] and/or perm_c[] is changed.    */
			if ( options->SolveInitialized == YES ) { /* Initialized before */
				zSolveFinalize(options, SOLVEstruct); /* Clean up structure */
				pzgstrs_delete_device_lsum_x(SOLVEstruct);
				options->SolveInitialized = NO;   /* Reset the solve state */
			}
			}

			if (get_new3dsolve()){


			if (options->DiagInv == YES && (Fact != FACTORED))
			{
				pzCompute_Diag_Inv(n, LUstruct, grid, stat, info);

				// The following #ifdef GPU_ACC block frees and reallocates GPU data for trisolve. The data seems to be overwritten by pzgstrf3d.
				int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
#if defined(GPU_ACC)

				pzconvertU(options, grid, LUstruct, stat, n);

				// checkGPU(gpuFree(LUstruct->Llu->d_xsup));
				// checkGPU(gpuFree(LUstruct->Llu->d_bcols_masked));
				// checkGPU(gpuFree(LUstruct->Llu->d_LRtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_LBtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_URtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_UBtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lrowind_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lindval_loc_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lrowind_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lindval_loc_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lnzval_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Linv_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Uinv_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_ilsum));
				// checkGPU(gpuFree(LUstruct->Llu->d_grid));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lnzval_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Linv_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Uinv_bc_dat));

				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_xsup, (n + 1) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_xsup, LUstruct->Glu_persist->xsup, (n + 1) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc( (void**)&LUstruct->Llu->d_bcols_masked, LUstruct->Llu->nbcol_masked * sizeof(int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_bcols_masked, LUstruct->Llu->bcols_masked, LUstruct->Llu->nbcol_masked * sizeof(int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_LRtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_LBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_URtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_UBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_LRtree_ptr, LUstruct->Llu->LRtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_LBtree_ptr, LUstruct->Llu->LBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_URtree_ptr, LUstruct->Llu->URtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_UBtree_ptr, LUstruct->Llu->UBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lrowind_bc_dat, (LUstruct->Llu->Lrowind_bc_cnt) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lrowind_bc_dat, LUstruct->Llu->Lrowind_bc_dat, (LUstruct->Llu->Lrowind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lindval_loc_bc_dat, (LUstruct->Llu->Lindval_loc_bc_cnt) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lindval_loc_bc_dat, LUstruct->Llu->Lindval_loc_bc_dat, (LUstruct->Llu->Lindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lrowind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lrowind_bc_offset, LUstruct->Llu->Lrowind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lindval_loc_bc_offset, LUstruct->Llu->Lindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lnzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_offset, LUstruct->Llu->Lnzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Linv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_offset, LUstruct->Llu->Linv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Uinv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_offset, LUstruct->Llu->Uinv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_ilsum, (CEILING(nsupers, grid->nprow) + 1) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_ilsum, LUstruct->Llu->ilsum, (CEILING(nsupers, grid->nprow) + 1) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lnzval_bc_dat, (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(doublecomplex)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Linv_bc_dat, (LUstruct->Llu->Linv_bc_cnt) * sizeof(doublecomplex)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Uinv_bc_dat, (LUstruct->Llu->Uinv_bc_cnt) * sizeof(doublecomplex)));
				// checkGPU(gpuMalloc( (void**)&LUstruct->Llu->d_grid, sizeof(gridinfo_t)));
    			// checkGPU(gpuMemcpy(LUstruct->Llu->d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));
#endif
if (get_acc_solve()){
#ifdef GPU_ACC
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_dat, LUstruct->Llu->Linv_bc_dat,
								   (LUstruct->Llu->Linv_bc_cnt) * sizeof(doublecomplex), gpuMemcpyHostToDevice));
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_dat, LUstruct->Llu->Uinv_bc_dat,
								   (LUstruct->Llu->Uinv_bc_cnt) * sizeof(doublecomplex), gpuMemcpyHostToDevice));
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_dat, LUstruct->Llu->Lnzval_bc_dat,
								   (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(doublecomplex), gpuMemcpyHostToDevice));
#endif
}
			}
			}
		}else{ /* if(Solve3D) */

			if (grid3d->zscp.Iam == 0){  /* on 2D grid-0 */

			if ( options->Fact == DOFACT || options->Fact == SamePattern ) {
			/* Need to reset the solve's communication pattern,
			because perm_r[] and/or perm_c[] is changed.    */
			if ( options->SolveInitialized == YES ) { /* Initialized before */
				zSolveFinalize(options, SOLVEstruct); /* Clean up structure */
				pzgstrs_delete_device_lsum_x(SOLVEstruct);
				options->SolveInitialized = NO;   /* Reset the solve state */
			}
			}

#if (defined(GPU_ACC))
			if (options->DiagInv == NO && get_acc_solve())
			{
				if (iam == 0)
				{
					printf("!!WARNING: GPU trisolve requires setting options->DiagInv==YES\n");
					printf("           otherwise, use CPU trisolve\n");
					fflush(stdout);
				}
				// exit(0);  // Sherry: need to return an error flag
			}
#endif

			if (options->DiagInv == YES && (Fact != FACTORED))
			{
				pzCompute_Diag_Inv(n, LUstruct, grid, stat, info);

				// The following #ifdef GPU_ACC block frees and reallocates GPU data for trisolve. The data seems to be overwritten by pzgstrf3d.
				int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
#ifdef GPU_ACC

				pzconvertU(options, grid, LUstruct, stat, n);

				// checkGPU(gpuFree(LUstruct->Llu->d_xsup));
				// checkGPU(gpuFree(LUstruct->Llu->d_bcols_masked));
				// checkGPU(gpuFree(LUstruct->Llu->d_LRtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_LBtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_URtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_UBtree_ptr));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lrowind_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lindval_loc_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lrowind_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lindval_loc_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lnzval_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Linv_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_Uinv_bc_offset));
				// checkGPU(gpuFree(LUstruct->Llu->d_ilsum));
				// checkGPU(gpuFree(LUstruct->Llu->d_grid));
				// checkGPU(gpuFree(LUstruct->Llu->d_Lnzval_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Linv_bc_dat));
				// checkGPU(gpuFree(LUstruct->Llu->d_Uinv_bc_dat));

				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_xsup, (n + 1) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_xsup, LUstruct->Glu_persist->xsup, (n + 1) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc( (void**)&LUstruct->Llu->d_bcols_masked, LUstruct->Llu->nbcol_masked * sizeof(int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_bcols_masked, LUstruct->Llu->bcols_masked, LUstruct->Llu->nbcol_masked * sizeof(int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_LRtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_LBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_URtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_UBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_LRtree_ptr, LUstruct->Llu->LRtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_LBtree_ptr, LUstruct->Llu->LBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_URtree_ptr, LUstruct->Llu->URtree_ptr, CEILING(nsupers, grid->nprow) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_UBtree_ptr, LUstruct->Llu->UBtree_ptr, CEILING(nsupers, grid->npcol) * sizeof(C_Tree), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lrowind_bc_dat, (LUstruct->Llu->Lrowind_bc_cnt) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lrowind_bc_dat, LUstruct->Llu->Lrowind_bc_dat, (LUstruct->Llu->Lrowind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lindval_loc_bc_dat, (LUstruct->Llu->Lindval_loc_bc_cnt) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lindval_loc_bc_dat, LUstruct->Llu->Lindval_loc_bc_dat, (LUstruct->Llu->Lindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lrowind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lrowind_bc_offset, LUstruct->Llu->Lrowind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lindval_loc_bc_offset, LUstruct->Llu->Lindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lnzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_offset, LUstruct->Llu->Lnzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Linv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_offset, LUstruct->Llu->Linv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Uinv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_offset, LUstruct->Llu->Uinv_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(long int), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_ilsum, (CEILING(nsupers, grid->nprow) + 1) * sizeof(int_t)));
				// checkGPU(gpuMemcpy(LUstruct->Llu->d_ilsum, LUstruct->Llu->ilsum, (CEILING(nsupers, grid->nprow) + 1) * sizeof(int_t), gpuMemcpyHostToDevice));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Lnzval_bc_dat, (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(doublecomplex)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Linv_bc_dat, (LUstruct->Llu->Linv_bc_cnt) * sizeof(doublecomplex)));
				// checkGPU(gpuMalloc((void **)&LUstruct->Llu->d_Uinv_bc_dat, (LUstruct->Llu->Uinv_bc_cnt) * sizeof(doublecomplex)));
				// checkGPU(gpuMalloc( (void**)&LUstruct->Llu->d_grid, sizeof(gridinfo_t)));
    			// checkGPU(gpuMemcpy(LUstruct->Llu->d_grid, grid, sizeof(gridinfo_t), gpuMemcpyHostToDevice));
#endif

if (get_acc_solve()){
#ifdef GPU_ACC

				checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_dat, LUstruct->Llu->Linv_bc_dat,
								   (LUstruct->Llu->Linv_bc_cnt) * sizeof(doublecomplex), gpuMemcpyHostToDevice));
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_dat, LUstruct->Llu->Uinv_bc_dat,
								   (LUstruct->Llu->Uinv_bc_cnt) * sizeof(doublecomplex), gpuMemcpyHostToDevice));
				checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_dat, LUstruct->Llu->Lnzval_bc_dat,
								   (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(doublecomplex), gpuMemcpyHostToDevice));
#endif
}
			}
			}
		}


		/* ------------------------------------------------------------
		   Compute the solution matrix X.
		   ------------------------------------------------------------ */
		if ((nrhs > 0) && (*info == 0))
		{
		if (options->SolveInitialized == NO){
			if (get_acc_solve()){
			if (get_new3dsolve() && Solve3D==true){
				pzgstrs_init_device_lsum_x(options, n, m_loc, nrhs, grid,LUstruct, SOLVEstruct,trf3Dpartition->supernodeMask);
			}else{
				int* supernodeMask = int32Malloc_dist(nsupers);
				for(int ii=0; ii<nsupers; ii++)
					supernodeMask[ii]=1;
				pzgstrs_init_device_lsum_x(options, n, m_loc, nrhs, grid,LUstruct, SOLVEstruct,supernodeMask);
				SUPERLU_FREE(supernodeMask);
			}
			}
		}

		stat->utime[SOLVE] = 0.0;
		if(Solve3D){

			// if (!(b_work = doublecomplexMalloc_dist(n)))
			// 	ABORT("Malloc fails for b_work[]");
			/* ------------------------------------------------------
			   Scale the right-hand side if equilibration was performed
			   ------------------------------------------------------*/
			if (notran)
			{
				if (rowequ)
				{
					b_col = B;
					for (j = 0; j < nrhs; ++j)
					{
						irow = fst_row;
						for (i = 0; i < m_loc; ++i)
						{
                        	zd_mult(&b_col[i], &b_col[i], R[irow]);
							++irow;
						}
						b_col += ldb;
					}
				}
			}
			else if (colequ)
			{
				b_col = B;
				for (j = 0; j < nrhs; ++j)
				{
					irow = fst_row;
					for (i = 0; i < m_loc; ++i)
					{
						zd_mult(&b_col[i], &b_col[i], C[irow]);
						++irow;
					}
					b_col += ldb;
				}
			}

			/* Save a copy of the right-hand side. */
			ldx = ldb;
			if (!(X = doublecomplexMalloc_dist(((size_t)ldx) * nrhs)))
				ABORT("Malloc fails for X[]");
			x_col = X;
			b_col = B;
			for (j = 0; j < nrhs; ++j)
			{
				for (i = 0; i < m_loc; ++i)
					x_col[i] = b_col[i];
				x_col += ldx;
				b_col += ldb;
			}

			/* ------------------------------------------------------
			   Solve the linear system.
			   ------------------------------------------------------*/

			if (options->SolveInitialized == NO)
			/* First time */
			/* Inside this routine, SolveInitialized is set to YES.
			For repeated call to pzgssvx3d(), no need to re-initialilze
			the Solve data & communication structures, unless a new
			factorization with Fact == DOFACT or SamePattern is asked for. */
			{
				zSolveInit(options, A, perm_r, perm_c, nrhs, LUstruct,
							grid, SOLVEstruct);
			}
			if (get_new3dsolve()){
				pzgstrs3d_newsolve (options, n, LUstruct,ScalePermstruct, trf3Dpartition, grid3d, X,
				m_loc, fst_row, ldb, nrhs,SOLVEstruct, stat, info);
			}else{
				pzgstrs3d (options, n, LUstruct,ScalePermstruct, trf3Dpartition, grid3d, X,
				m_loc, fst_row, ldb, nrhs,SOLVEstruct, stat, info);
			}
			if (options->IterRefine)
				{
				/* Improve the solution by iterative refinement. */
				int_t *it, *colind_gsmv = SOLVEstruct->A_colind_gsmv;
				zSOLVEstruct_t *SOLVEstruct1; /* Used by refinement */

				t = SuperLU_timer_ ();
				if (options->RefineInitialized == NO || Fact == DOFACT) {
					/* All these cases need to re-initialize gsmv structure */
					if (options->RefineInitialized)
					pzgsmv_finalize (SOLVEstruct->gsmv_comm);
					pzgsmv_init (A, SOLVEstruct->row_to_proc, grid,
						SOLVEstruct->gsmv_comm);

					/* Save a copy of the transformed local col indices
					in colind_gsmv[]. */
					if (colind_gsmv) SUPERLU_FREE (colind_gsmv);
					if (!(it = intMalloc_dist (nnz_loc)))
					ABORT ("Malloc fails for colind_gsmv[]");
					colind_gsmv = SOLVEstruct->A_colind_gsmv = it;
					for (i = 0; i < nnz_loc; ++i) colind_gsmv[i] = colind[i];
					options->RefineInitialized = YES;
				}
				else if (Fact == SamePattern || Fact == SamePattern_SameRowPerm) {
					doublecomplex at;
					int_t k, jcol, p;
					/* Swap to beginning the part of A corresponding to the
					local part of X, as was done in pzgsmv_init() */
					for (i = 0; i < m_loc; ++i) { /* Loop through each row */
					k = rowptr[i];
					for (j = rowptr[i]; j < rowptr[i + 1]; ++j)
						{
						jcol = colind[j];
						p = SOLVEstruct->row_to_proc[jcol];
						if (p == iam)
							{	/* Local */
							at = a[k];
							a[k] = a[j];
							a[j] = at;
							++k;
							}
						}
					}

					/* Re-use the local col indices of A obtained from the
					previous call to pzgsmv_init() */
					for (i = 0; i < nnz_loc; ++i)
					colind[i] = colind_gsmv[i];
				}

				if (nrhs == 1)
					{	/* Use the existing solve structure */
					SOLVEstruct1 = SOLVEstruct;
					}
				else {
				/* For nrhs > 1, since refinement is performed for RHS
			one at a time, the communication structure for pzgstrs
			is different than the solve with nrhs RHS.
			So we use SOLVEstruct1 for the refinement step.
			*/
					if (!(SOLVEstruct1 = (zSOLVEstruct_t *)
						SUPERLU_MALLOC(sizeof(zSOLVEstruct_t))))
						ABORT ("Malloc fails for SOLVEstruct1");
					/* Copy the same stuff */
					SOLVEstruct1->row_to_proc = SOLVEstruct->row_to_proc;
					SOLVEstruct1->inv_perm_c = SOLVEstruct->inv_perm_c;
					SOLVEstruct1->num_diag_procs = SOLVEstruct->num_diag_procs;
					SOLVEstruct1->diag_procs = SOLVEstruct->diag_procs;
					SOLVEstruct1->diag_len = SOLVEstruct->diag_len;
					SOLVEstruct1->gsmv_comm = SOLVEstruct->gsmv_comm;
					SOLVEstruct1->A_colind_gsmv = SOLVEstruct->A_colind_gsmv;

					/* Initialize the *gstrs_comm for 1 RHS. */
					if (!(SOLVEstruct1->gstrs_comm = (pxgstrs_comm_t *)
						SUPERLU_MALLOC (sizeof (pxgstrs_comm_t))))
						ABORT ("Malloc fails for gstrs_comm[]");
					pzgstrs_init (n, m_loc, 1, fst_row, perm_r, perm_c, grid,
							LUstruct->Glu_persist, SOLVEstruct1);
					if (get_acc_solve()){
					int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
					pzgstrs_init_device_lsum_x(options, n, m_loc, 1, grid,LUstruct, SOLVEstruct1,trf3Dpartition->supernodeMask);
					}
					}

				pzgsrfs3d (options, n, A, anorm, LUstruct, ScalePermstruct, grid3d, trf3Dpartition,
					B, ldb, X, ldx, nrhs, SOLVEstruct1, berr, stat, info);

				/* Deallocate the storage associated with SOLVEstruct1 */
				if (nrhs > 1)
					{
					pzgstrs_delete_device_lsum_x(SOLVEstruct1);
					pxgstrs_finalize (SOLVEstruct1->gstrs_comm);
					SUPERLU_FREE (SOLVEstruct1);
					}

				stat->utime[REFINE] = SuperLU_timer_ () - t;
				} /* end IterRefine */
		}else{

			if (grid3d->zscp.Iam == 0){  /* on 2D grid-0 */

			/* ------------------------------------------------------
			   Scale the right-hand side if equilibration was performed
			   ------------------------------------------------------*/
			if (notran)
			{
				if (rowequ)
				{
					b_col = B;
					for (j = 0; j < nrhs; ++j)
					{
						irow = fst_row;
						for (i = 0; i < m_loc; ++i)
						{
                        	zd_mult(&b_col[i], &b_col[i], R[irow]);
							++irow;
						}
						b_col += ldb;
					}
				}
			}
			else if (colequ)
			{
				b_col = B;
				for (j = 0; j < nrhs; ++j)
				{
					irow = fst_row;
					for (i = 0; i < m_loc; ++i)
					{
						zd_mult(&b_col[i], &b_col[i], C[irow]);
						++irow;
					}
					b_col += ldb;
				}
			}

			/* Save a copy of the right-hand side. */
			ldx = ldb;
			if (!(X = doublecomplexMalloc_dist(((size_t)ldx) * nrhs)))
				ABORT("Malloc fails for X[]");
			x_col = X;
			b_col = B;
			for (j = 0; j < nrhs; ++j)
			{
				for (i = 0; i < m_loc; ++i)
					x_col[i] = b_col[i];
				x_col += ldx;
				b_col += ldb;
			}

			/* ------------------------------------------------------
			   Solve the linear system.
			   ------------------------------------------------------*/
			if (options->SolveInitialized == NO)
			/* First time */
			/* Inside this routine, SolveInitialized is set to YES.
			For repeated call to pzgssvx3d(), no need to re-initialilze
			the Solve data & communication structures, unless a new
			factorization with Fact == DOFACT or SamePattern is asked for. */
			{
				zSolveInit(options, A, perm_r, perm_c, nrhs, LUstruct,
							grid, SOLVEstruct);
			}
			pzgstrs(options, n, LUstruct, ScalePermstruct, grid, X, m_loc,
				fst_row, ldb, nrhs, SOLVEstruct, stat, info);

			/* ------------------------------------------------------------
			Use iterative refinement to improve the computed solution and
			compute error bounds and backward error estimates for it.
			------------------------------------------------------------ */
			if (options->IterRefine)
				{
				/* Improve the solution by iterative refinement. */
				int_t *it, *colind_gsmv = SOLVEstruct->A_colind_gsmv;
				zSOLVEstruct_t *SOLVEstruct1; /* Used by refinement */

				t = SuperLU_timer_ ();
				if (options->RefineInitialized == NO || Fact == DOFACT) {
					/* All these cases need to re-initialize gsmv structure */
					if (options->RefineInitialized)
					pzgsmv_finalize (SOLVEstruct->gsmv_comm);
					pzgsmv_init (A, SOLVEstruct->row_to_proc, grid,
						SOLVEstruct->gsmv_comm);

					/* Save a copy of the transformed local col indices
					in colind_gsmv[]. */
					if (colind_gsmv) SUPERLU_FREE (colind_gsmv);
					if (!(it = intMalloc_dist (nnz_loc)))
					ABORT ("Malloc fails for colind_gsmv[]");
					colind_gsmv = SOLVEstruct->A_colind_gsmv = it;
					for (i = 0; i < nnz_loc; ++i) colind_gsmv[i] = colind[i];
					options->RefineInitialized = YES;
				}
				else if (Fact == SamePattern || Fact == SamePattern_SameRowPerm) {
					doublecomplex at;
					int_t k, jcol, p;
					/* Swap to beginning the part of A corresponding to the
					local part of X, as was done in pzgsmv_init() */
					for (i = 0; i < m_loc; ++i) { /* Loop through each row */
					k = rowptr[i];
					for (j = rowptr[i]; j < rowptr[i + 1]; ++j)
						{
						jcol = colind[j];
						p = SOLVEstruct->row_to_proc[jcol];
						if (p == iam)
							{	/* Local */
							at = a[k];
							a[k] = a[j];
							a[j] = at;
							++k;
							}
						}
					}

					/* Re-use the local col indices of A obtained from the
					previous call to pzgsmv_init() */
					for (i = 0; i < nnz_loc; ++i)
					colind[i] = colind_gsmv[i];
				}

				if (nrhs == 1)
					{	/* Use the existing solve structure */
					SOLVEstruct1 = SOLVEstruct;
					}
				else {
				/* For nrhs > 1, since refinement is performed for RHS
			one at a time, the communication structure for pzgstrs
			is different than the solve with nrhs RHS.
			So we use SOLVEstruct1 for the refinement step.
			*/
					if (!(SOLVEstruct1 = (zSOLVEstruct_t *)
						SUPERLU_MALLOC(sizeof(zSOLVEstruct_t))))
						ABORT ("Malloc fails for SOLVEstruct1");
					/* Copy the same stuff */
					SOLVEstruct1->row_to_proc = SOLVEstruct->row_to_proc;
					SOLVEstruct1->inv_perm_c = SOLVEstruct->inv_perm_c;
					SOLVEstruct1->num_diag_procs = SOLVEstruct->num_diag_procs;
					SOLVEstruct1->diag_procs = SOLVEstruct->diag_procs;
					SOLVEstruct1->diag_len = SOLVEstruct->diag_len;
					SOLVEstruct1->gsmv_comm = SOLVEstruct->gsmv_comm;
					SOLVEstruct1->A_colind_gsmv = SOLVEstruct->A_colind_gsmv;

					/* Initialize the *gstrs_comm for 1 RHS. */
					if (!(SOLVEstruct1->gstrs_comm = (pxgstrs_comm_t *)
						SUPERLU_MALLOC (sizeof (pxgstrs_comm_t))))
						ABORT ("Malloc fails for gstrs_comm[]");
					pzgstrs_init (n, m_loc, 1, fst_row, perm_r, perm_c, grid,
							LUstruct->Glu_persist, SOLVEstruct1);
					if (get_acc_solve()){
					int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
					int* supernodeMask = int32Malloc_dist(nsupers);
					for(int ii=0; ii<nsupers; ii++)
						supernodeMask[ii]=1;
					pzgstrs_init_device_lsum_x(options, n, m_loc, 1, grid,LUstruct, SOLVEstruct1,supernodeMask);
					SUPERLU_FREE(supernodeMask);
					}
					}

				pzgsrfs (options, n, A, anorm, LUstruct, ScalePermstruct, grid,
					B, ldb, X, ldx, nrhs, SOLVEstruct1, berr, stat, info);

				/* Deallocate the storage associated with SOLVEstruct1 */
				if (nrhs > 1)
					{
					pzgstrs_delete_device_lsum_x(SOLVEstruct1);
					pxgstrs_finalize (SOLVEstruct1->gstrs_comm);
					SUPERLU_FREE (SOLVEstruct1);
					}

				stat->utime[REFINE] = SuperLU_timer_ () - t;
				} /* end IterRefine */
			}
		}

if (grid3d->zscp.Iam == 0)  /* on 2D grid-0 */
	{
		/* Permute the solution matrix B <= Pc'*X. */
		pzPermute_Dense_Matrix (fst_row, m_loc, SOLVEstruct->row_to_proc,
					SOLVEstruct->inv_perm_c,
					X, ldx, B, ldb, nrhs, grid);
#if ( DEBUGlevel>=2 )
		printf ("\n (%d) .. After pzPermute_Dense_Matrix(): b =\n", iam);
		for (i = 0; i < m_loc; ++i)
		    printf ("\t(%d)\t%4d\t%.10f\n", iam, i + fst_row, B[i]);
#endif
			/* Transform the solution matrix X to a solution of the original
			   system before the equilibration. */
			if (notran)
			{
				if (colequ)
				{
					b_col = B;
					for (j = 0; j < nrhs; ++j)
					{
						irow = fst_row;
						for (i = 0; i < m_loc; ++i)
						{
							zd_mult(&b_col[i], &b_col[i], C[irow]);
							++irow;
						}
						b_col += ldb;
				    }
			    }
		    }
			else if (rowequ)
		    {
			b_col = B;
			for (j = 0; j < nrhs; ++j)
			    {
				irow = fst_row;
				for (i = 0; i < m_loc; ++i)
				    {
					zd_mult(&b_col[i], &b_col[i], R[irow]);
					++irow;
				    }
				b_col += ldb;
			    }
		    }

		// SUPERLU_FREE (b_work);
	}
	if (grid3d->zscp.Iam == 0 || Solve3D)
		SUPERLU_FREE (X);

	} /* end if nrhs > 0 and factor successful */

#if ( PRNTlevel>=1 )
	if (!grid3d->iam) {
	    printf (".. DiagScale = %d\n", ScalePermstruct->DiagScale);
        }
#endif


	if ( grid3d->zscp.Iam == 0 ) { // only process layer 0
	/* Deallocate R and/or C if it was not used. */
	if (Equil && Fact != SamePattern_SameRowPerm)
	    {
		switch (ScalePermstruct->DiagScale) {
		    case NOEQUIL:
			SUPERLU_FREE (R);
			SUPERLU_FREE (C);
			break;
		    case ROW:
			SUPERLU_FREE (C);
			break;
		    case COL:
			SUPERLU_FREE (R);
			break;
	            default: break;
		}
	}

#if 0
	if (!factored && Fact != SamePattern_SameRowPerm && !parSymbFact)
	    Destroy_CompCol_Permuted_dist (&GAC);
#endif

	} /* process layer 0 done solve */

	/* Scatter the solution from 2D grid-0 to 3D grid */
	if (nrhs > 0)
		zScatter_B3d(A3d, grid3d);

	B = A3d->B3d;		 // B is now assigned back to B3d on return
	A->Store = Astore3d; // restore Astore to 3D

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(iam, "Exit pzgssvx3d()");
#endif
}
