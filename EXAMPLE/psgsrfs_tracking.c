/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Improves the computed solution to a system of linear equations and provides error bounds and backward error estimates
 *
 * <pre>
 * -- Distributed SuperLU routine (version 8.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * May 22, 2022
 *
 * </pre>
 */

#include <math.h>
#include "superlu_sdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * PSGSRFS improves the computed solution to a system of linear
 * equations and provides error bounds and backward error estimates
 * for the solution.
 *
 * Arguments
 * =========
 *
 * n      (input) int (global)
 *        The order of the system of linear equations.
 *
 * A      (input) SuperMatrix*
 *	  The original matrix A, or the scaled A if equilibration was done.
 *        A is also permuted into diag(R)*A*diag(C)*Pc'. The type of A can be:
 *        Stype = SLU_NR_loc; Dtype = SLU_S; Mtype = SLU_GE.
 *
 * anorm  (input) float
 *        The norm of the original matrix A, or the scaled A if
 *        equilibration was done.
 *
 * LUstruct (input) sLUstruct_t*
 *        The distributed data structures storing L and U factors.
 *        The L and U factors are obtained from pdgstrf for
 *        the possibly scaled and permuted matrix A.
 *        See superlu_sdefs.h for the definition of 'sLUstruct_t'.
 *
 * ScalePermstruct (input) sScalePermstruct_t* (global)
 *         The data structure to store the scaling and permutation vectors
 *         describing the transformations performed to the matrix A.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh. It contains the MPI communicator, the number
 *        of process rows (NPROW), the number of process columns (NPCOL),
 *        and my process rank. It is an input argument to all the
 *        parallel routines.
 *        Grid can be initialized by subroutine SUPERLU_GRIDINIT.
 *        See superlu_defs.h for the definition of 'gridinfo_t'.
 *
 * B      (input) float* (local)
 *        The m_loc-by-NRHS right-hand side matrix of the possibly
 *        equilibrated system. That is, B may be overwritten by diag(R)*B.
 *
 * ldb    (input) int (local)
 *        Leading dimension of matrix B.
 *
 * X      (input/output) float* (local)
 *        On entry, the solution matrix Y, as computed by PDGSTRS, of the
 *            transformed system A1*Y = Pc*Pr*B. where
 *            A1 = Pc*Pr*diag(R)*A*diag(C)*Pc' and Y = Pc*diag(C)^(-1)*X.
 *        On exit, the improved solution matrix Y.
 *
 *        In order to obtain the solution X to the original system,
 *        Y should be permutated by Pc^T, and premultiplied by diag(C)
 *        if DiagScale = COL or BOTH.
 *        This must be done after this routine is called.
 *
 * ldx    (input) int (local)
 *        Leading dimension of matrix X.
 *
 * nrhs   (input) int
 *        Number of right-hand sides.
 *
 * SOLVEstruct (output) sSOLVEstruct_t* (global)
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * berr   (output) float*, dimension (nrhs)
 *         The componentwise relative backward error of each solution
 *         vector X(j) (i.e., the smallest relative change in
 *         any element of A or B that makes X(j) an exact solution).
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the refinement steps.
 *        See util.h for the definition of SuperLUStat_t.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *
 * Internal Parameters
 * ===================
 *
 * ITMAX is the maximum number of steps of iterative refinement.
 * </pre>
 */

// Tracking the convergence history
void
psgsrfs_tracking(superlu_dist_options_t *options,
		 int n, SuperMatrix *A, float anorm, sLUstruct_t *LUstruct,
		 sScalePermstruct_t *ScalePermstruct, gridinfo_t *grid,
		 float *B, int_t ldb, float *X, int_t ldx, int nrhs,
		 sSOLVEstruct_t *SOLVEstruct,
		 float *berr, SuperLUStat_t *stat, int *info, double *xtrue)
{
#define ITMAX 10

    float *ax, *R, *dx, *temp, *work, *B_col, *X_col;
    int_t count, i, j, lwork, nz;
    int   iam;
    float eps, lstres;
    float s, safmin, safe1, safe2;

    /* Data structures used by matrix-vector multiply routine. */
    psgsmv_comm_t *gsmv_comm = SOLVEstruct->gsmv_comm;
    NRformat_loc *Astore;
    int_t        m_loc, fst_row;


    /* Initialization. */
    Astore = (NRformat_loc *) A->Store;
    m_loc = Astore->m_loc;
    fst_row = Astore->fst_row;
    iam = grid->iam;

    /* Test the input parameters. */
    *info = 0;
    if ( n < 0 ) *info = -1;
    else if ( A->nrow != A->ncol || A->nrow < 0 || A->Stype != SLU_NR_loc
	      || A->Dtype != SLU_S || A->Mtype != SLU_GE )
	*info = -2;
    else if ( ldb < SUPERLU_MAX(0, m_loc) ) *info = -10;
    else if ( ldx < SUPERLU_MAX(0, m_loc) ) *info = -12;
    else if ( nrhs < 0 ) *info = -13;
    if (*info != 0) {
	i = -(*info);
	pxerr_dist("PSGSRFS", grid, i);
	return;
    }

    /* Quick return if possible. */
    if ( n == 0 || nrhs == 0 ) {
	return;
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter psgsrfs()");
#endif

    lwork = 2 * m_loc;  /* For ax/R/dx and temp */
    if ( !(work = floatMalloc_dist(lwork)) )
	ABORT("Malloc fails for work[]");
    ax = R = dx = work;
    temp = ax + m_loc;

#if ( PRNTlevel>=1 )
    float err, normy, errcomp, derr, c_ratio;
    float ferr[5][ITMAX]; /* 0: normwise Ferr for X
			      1: componentwise Ferr for X
			      2: normwise Ferr for Y
			      3: componentwise Ferr for Y
			      4: componentwise Berr
			   */
    float local_norms[3], global_norms[3];  // for MPI reduction
    
    // Compute ytrue
    float *ytrue, *y_col;
    if ( !(ytrue = floatMalloc_dist(2*m_loc)) )
	ABORT("Malloc fails for ytrue[]");
    y_col = ytrue + m_loc;  // unpermuted ytrue
    float ymax = 0.0, ymin = 1.0e+15;
    float local_norm[1], global_norm[1];

    int colequ = ( ScalePermstruct->DiagScale == COL ||
		   ScalePermstruct->DiagScale == BOTH );
    float *C  = ScalePermstruct->C;
    
    /* ytrue = Pc * inv(C) * xtrue */
    if ( colequ ) { 
      for (i = 0; i < m_loc; ++i) 
	y_col[i] = (double) xtrue[i] / (double) C[i + fst_row];
    } else {
      for (i = 0; i < m_loc; ++i) y_col[i] = (double) xtrue[i];
    }
    for (i = 0; i < m_loc; ++i) {
      ymax = SUPERLU_MAX(ymax, y_col[i]);
      ymin = SUPERLU_MIN(ymin, y_col[i]);
    }
    assert(ldx == ldb);

    /* Permute the true solution ytrue <= Pc * ytrue. */
    psPermute_Dense_Matrix(fst_row, m_loc, SOLVEstruct->row_to_proc,
                           ScalePermstruct->perm_c, y_col, ldx,
                           ytrue, ldb, nrhs, grid);
    local_norms[0] = ymax;
    MPI_Reduce( local_norms, global_norms, 1, 
		MPI_FLOAT, MPI_MAX, 0, grid->comm );
    ymax = global_norms[0];
    local_norms[0] = ymin;
    MPI_Reduce( local_norms, global_norms, 1, 
		MPI_FLOAT, MPI_MIN, 0, grid->comm );
    ymin = global_norms[0];
    if (iam == 0) {
      printf("Kappa(y): ymax %e / ymin %e = %e\n", ymax, ymin, ymax / ymin);
      fflush(stdout);
    }
#endif    

    /* NZ = maximum number of nonzero elements in each row of A, plus 1 */
    nz     = A->ncol + 1;
    eps    = smach_dist("Epsilon");
    safmin = smach_dist("Safe minimum");

    /* Set SAFE1 essentially to be the underflow threshold times the
       number of additions in each row. */
    safe1  = nz * safmin;
    safe2  = safe1 / eps;

#if ( DEBUGlevel>=1 )
    if ( !iam ) printf(".. eps = %e\tanorm = %e\tsafe1 = %e\tsafe2 = %e\n",
		       eps, anorm, safe1, safe2);
#endif

    /* Do for each right-hand side ... */
    for (j = 0; j < nrhs; ++j) {
	count = 0;
	lstres = 3.;
	B_col = &B[j*ldb];
	X_col = &X[j*ldx];

	for (count = 0; count < ITMAX; ++count) {
	  //while (1) { /* Loop until stopping criterion is satisfied. */

	    /* Compute residual R = B - op(A) * X,
	       where op(A) = A, A**T, or A**H, depending on TRANS. */

	    /* Matrix-vector multiply. */
	    psgsmv(0, A, grid, gsmv_comm, X_col, ax);

	    /* Compute residual, stored in R[]. */
	    for (i = 0; i < m_loc; ++i) R[i] = B_col[i] - ax[i];

	    /* Compute abs(op(A))*abs(X) + abs(B), stored in temp[]. */
	    psgsmv(1, A, grid, gsmv_comm, X_col, temp);
	    for (i = 0; i < m_loc; ++i) temp[i] += fabs(B_col[i]);

	    s = 0.0;
	    for (i = 0; i < m_loc; ++i) {
		if ( temp[i] > safe2 ) {
		    s = SUPERLU_MAX(s, fabs(R[i]) / temp[i]);
		} else if ( temp[i] != 0.0 ) {
                    /* Adding SAFE1 to the numerator guards against
                       spuriously zero residuals (underflow). */
                    s = SUPERLU_MAX(s, (safe1 + fabs(R[i])) /temp[i]);
                }
                /* If temp[i] is exactly 0.0 (computed by PxGSMV), then
                   we know the true residual also must be exactly 0.0. */
	    }
	    MPI_Allreduce( &s, &berr[j], 1, MPI_FLOAT, MPI_MAX, grid->comm );

#if ( PRNTlevel>=1)
	    // Error from ytrue
	    err = normy = errcomp = 0.0;
	    for (i = 0; i < m_loc; i++) {
	      derr = fabs(X_col[i] - ytrue[i]);
	      err = SUPERLU_MAX(err, derr); // normwise error
	      normy = SUPERLU_MAX(normy, fabs(X_col[i]));
	      // errcomp = SUPERLU_MAX(errcomp, derr / fabs(y_col[i]));
	      c_ratio = derr / fabs(ytrue[i]);  // componentwise error
	      if ( c_ratio > errcomp ) {
		errcomp = c_ratio;
	      }
	    }
	    /* Reduce 3 numbers */
	    local_norms[0] = err;
	    local_norms[1] = normy;
	    local_norms[2] = errcomp;
	    MPI_Allreduce( local_norms, global_norms, 3,
			   MPI_FLOAT, MPI_MAX, grid->comm ); // MPI_FLOAT
	    ferr[2][count] = global_norms[0] / global_norms[1]; // normwise Y error
	    ferr[3][count] = global_norms[2];  // componentwise Y error
	    ferr[4][count] = berr[j];  // componentwise Y error
	    /*if ( !iam ) {
	      printf("(%2d) .. Step " IFMT ": berr[j] = %e\n", iam, count, berr[j]);
	    printf("%2d %20.16e %20.16e %20.16e\n", i,
		   ferr[2][count], ferr[3][count], ferr[4][count]);
	      fflush(stdout);
	      }*/
#endif
	    
	    //if ( berr[j] > eps && berr[j] * 2 <= lstres && count < ITMAX ) {
	    if ( 1 ) {
		/* Compute new dx. */
		psgstrs(options, n, LUstruct, ScalePermstruct, grid,
			dx, m_loc, fst_row, m_loc, 1,
			SOLVEstruct, stat, info);

		/* Update solution. */
		for (i = 0; i < m_loc; ++i) X_col[i] += dx[i];

		lstres = berr[j];
	    } else {
		break;
	    }
	} /* end for count ... */
	
	stat->RefineSteps = count;
	
#if ( PRNTlevel>=1 )
	if (iam == 0) {
	  printf("%%IR %20s%24s%20s\n", 
		 "||Y-Ytrue||/||Y||", "max |Y-Yt|_i / |Y|_i", "Berr");
	  for (i = 0; i < count; ++i) {
	    printf("%2d %20.16e %20.16e %20.16e\n", i,
		   ferr[2][i], ferr[3][i], ferr[4][i]);
	  }
	  printf("Terminate at step %d\n", stat->RefineSteps);
	  fflush(stdout);
	}
#endif	
    } /* for j ... */

    /* Deallocate storage. */
    SUPERLU_FREE(work);
#if ( PRNTlevel>=1 )
    SUPERLU_FREE(ytrue);
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit psgsrfs()");
#endif

} /* PSGSRFS */

