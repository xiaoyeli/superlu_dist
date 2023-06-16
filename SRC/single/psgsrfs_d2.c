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
 * -- Distributed SuperLU routine (version 4.3) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * March 15, 2003
 *
 * Last modified:
 * December 31, 2015
 * </pre>
 */

#include <math.h>
#include "superlu_sdefs.h"
//#include "superlu_ddefs.h"

#define ITMAX 10
#define RHO_THRESH 0.5
#define DZ_THRESH  0.25

float compute_berr(
		   int m_loc, SuperMatrix *A,
		   psgsmv_comm_t *gsmv_comm, gridinfo_t *grid,
		   float *B, float *X, float *R, float *temp,
		   float safe1, float safe2
		   );

/**** DEBUG: Check against Xtrue and Ytrue at each iteration */
void check_accuracy(
		    int count, int fst_row, int m_loc, int nrhs, int colequ,
		    SuperMatrix *A, psgsmv_comm_t *gsmv_comm, float R[], //resid
		    double y_col[], float dy[], float X_col[], int ldx,
		    float B_col[], float C[], double xtrue[], double ytrue[],
		    float temp[], int_t inv_perm_c[],
		    double converge[7][ITMAX], double ferr[5][ITMAX],
		    sSOLVEstruct_t *SOLVEstruct, gridinfo_t *grid,
		    float safe1, float safe2
		    )
{
    // Sherry!!: for debug, use double
    double local_norms[3], global_norms[3];  // for MPI reduction
    double derr, err, xnorm, temperr, tempxnorm;
    double normx, normy;
    double c_ratio, errcomp;  // componentwise error
    int imax, p_imax = -1; // track the largest error index, and process
    float zero = 0.0;
    int i, iam = grid->iam;

    // NOTE: X is in single precision
    for (i = 0; i < m_loc; ++i) temp[i] = y_col[i]; // round to single???
    /* Permute the solution matrix X_col <= Pc'* Y. */
    psPermute_Dense_Matrix(fst_row, m_loc, SOLVEstruct->row_to_proc,
			   inv_perm_c, temp, ldx, X_col, ldx, nrhs, grid);
    if ( colequ ) {  // X_col[] is the current unscaled X 
      for (i = 0; i < m_loc; ++i) X_col[i] *= C[i + fst_row];
    }
	    
    // Error from xtrue
    err = xnorm = errcomp = 0.0;
    for (i = 0; i < m_loc; i++) {
      derr = fabs(X_col[i] - xtrue[i]);
      err = SUPERLU_MAX(err, derr);
      xnorm = SUPERLU_MAX(xnorm, fabs(X_col[i]));
      errcomp = SUPERLU_MAX(errcomp, derr / fabs(X_col[i]));
    }
    /* Reduce 3 numbers */
    local_norms[0] = err;
    local_norms[1] = xnorm;
    local_norms[2] = errcomp;
    MPI_Allreduce( local_norms, global_norms, 3,
		   MPI_DOUBLE, MPI_MAX, grid->comm );
    //MPI_FLOAT, MPI_MAX, grid->comm );
    ferr[0][count] = global_norms[0] / global_norms[1];
    ferr[1][count] = global_norms[2];

    err = compute_berr(m_loc, A, gsmv_comm, grid, B_col, X_col,
		       R, temp, safe1, safe2);

    ferr[4][count] = err; // Berr

    // Error from ytrue
    err = normy = errcomp = 0.0;
    for (i = 0; i < m_loc; i++) {
      derr = fabs(y_col[i] - ytrue[i]);
      err = SUPERLU_MAX(err, derr);     // double, normwise error
      normy = SUPERLU_MAX(normy, fabs(y_col[i]));
      // errcomp = SUPERLU_MAX(errcomp, derr / fabs(y_col[i]));
      c_ratio = derr / fabs(y_col[i]);  // componentwise error
      if ( c_ratio > errcomp ) {
	imax = i;
	errcomp = c_ratio;
      }
    }
    /* Reduce 3 numbers */
    local_norms[0] = err;
    local_norms[1] = normy;
    local_norms[2] = errcomp;
    MPI_Allreduce( local_norms, global_norms, 3,
		   MPI_DOUBLE, MPI_MAX, grid->comm ); // MPI_FLOAT
    ferr[2][count] = global_norms[0] / global_norms[1]; // normwise Y error
    ferr[3][count] = global_norms[2];  // componentwise Y error
    normy = global_norms[1];

#if ( PRNTlevel>=3 )
    // Track largest Y error term, and dyi
    if (local_norms[2] == global_norms[2]) {
      p_imax = iam; // track largest proc
      printf("\n  (1: P%d) imax %d: y %.8e, dy %.8e, y_true %.8e, normx %.8e, normy %.8e, y-errcomp %.8e\n",
	     iam, imax, y_col[imax], dy[imax], ytrue[imax], 
	     xnorm, normy, global_norms[2]);
      fflush(stdout);
    } else p_imax = -1;
#endif

    /* Check the accuracy of the solution. */
    //psinf_norm_error(iam, m_loc, nrhs, X_col, ldb, xtrue, ldx, grid);

} // end check_accuracy
//************** END DEBUG

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * PSGSRFS_D2 improves the computed solution to a system of linear
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
 * anorm  (input) double
 *        The norm of the original matrix A, or the scaled A if
 *        equilibration was done.
 * 
 * LUstruct (input) LUstruct_t*
 *        The distributed data structures storing L and U factors.
 *        The L and U factors are obtained from pdgstrf for
 *        the possibly scaled and permuted matrix A.
 *        See superlu_sdefs.h for the definition of 'LUstruct_t'.
 *
 * ScalePermstruct (input) ScalePermstruct_t* (global)
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
 * SOLVEstruct (output) SOLVEstruct_t* (global)
 *        Contains the information for the communication during the
 *        solution phase.
 *
 * err_bounds (output) float*, dimension (nrhs * 3) (global)
 *         For each right-hand side j, contains the following error bounds:
 *         err_bounds[j + 0*nrhs] : normwise forward error bound
 *         err_bounds[j + 1*nrhs] : componentwise forward error bound
 *         err_bounds[j + 2*nrhs] : componentwise backward error
 *             The componentwise relative backward error of each solution
 *             vector X(j) (i.e., the smallest relative change in
 *             any element of A or B that makes X(j) an exact solution).
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
void
psgsrfs_d2(superlu_dist_options_t *options,
	   int n, SuperMatrix *A, float anorm, sLUstruct_t *LUstruct,
	   sScalePermstruct_t *ScalePermstruct, gridinfo_t *grid,
	   float *B, int_t ldb, float *X, int_t ldx, int nrhs,
	   sSOLVEstruct_t *SOLVEstruct, float *err_bounds,
	   SuperLUStat_t *stat, int *info,
	   double *xtrue  // xtrue[] is used only for checking purpose
	   )
{
    float *resid, *dy, *temp, *Res, *B_col, *X_col, *C;
    int_t *perm_c = ScalePermstruct->perm_c; 
    int_t *inv_perm_c = SOLVEstruct->inv_perm_c; 
    double *ax, *y_col, *ytrue;
    int  iam, count, i, j, nz, m_loc, fst_row, colequ;
    //double eps, lstres;
    //double s, safmin, safe1, safe2;
    float eps, lstres; // working precision
    float s, safmin, safe1, safe2;
    //double eps_d;      // eps_d <= eps^2, at least double precision

    // Sherry: Change the following from float to double ??
    double Cpi;
    double normy, normx, normdx, normdz, prev_normdx, prev_normdz;
    double yi, dyi, dx_x, final_dx_x, final_dz_z;
    double rho_x, rho_x_max, rho_z, rho_z_max, hugeval;
    double local_norms[3], global_norms[3];  // for MPI reduction
    float zero = 0.0;
    
    double converge[7][ITMAX]; /* convergence history */
    double ferr[5][ITMAX]; /* 0: normwise Ferr for X
			      1: componentwise Ferr for X
			      2: normwise Ferr for Y
			      3: componentwise Ferr for Y
			      4: Berr */

    /* Data structures used by matrix-vector multiply routine. */
    psgsmv_comm_t *gsmv_comm = SOLVEstruct->gsmv_comm;
    NRformat_loc *Astore;
    int_t *rowptr;

    typedef enum {UNSTABLE, WORKING, CONVERGED, NoPROGRESS} IRstate_t;
    int x_state, z_state;
    //    int norm_how_stopped, comp_how_stopped;

    /*---- function prototypes ----*/
    extern void psgsmv_d2(int abs, SuperMatrix *A_internal,
		       gridinfo_t *grid, psgsmv_comm_t *gsmv_comm,
		       double x[],  double ax[]);
    extern double  *doubleMalloc_dist(int_t);
    extern void    pdinf_norm_error(int, int_t, int_t, double [], int_t,
				double [], int_t , gridinfo_t *);
    extern int     pdPermute_Dense_Matrix(int_t, int_t, int_t [], int_t[],
				      double [], int, double [], int, int,
				      gridinfo_t *);
    extern void  Printdouble5(char *name, int_t len, double *x);

    /* Initialization. */
    Astore = (NRformat_loc *) A->Store;
    m_loc = Astore->m_loc;
    fst_row = Astore->fst_row;
    rowptr = Astore->rowptr;
    iam = grid->iam;
    stat->RefineSteps = -1;

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
	pxerr_dist("PSGSRFS_D2", grid, i);
	return;
    }

    /* Quick return if possible. */
    if ( n == 0 || nrhs == 0 ) {
	return;
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter psgsrfs_d2()");
#endif

#if ( PRNTlevel>=2)
    if (iam == 0) {
      printf("  calling mixed-precicion IR: psgsrfs_fp64x2()...\n");
      fflush(stdout);
    }
#endif

    /* NZ = maximum number of nonzero elements in each row of A */
    //nz     = A->ncol + 1;  // dense case
    int nzloc = 0;
    for (i = 0; i < m_loc; ++i) {
      nzloc = SUPERLU_MAX( nzloc, rowptr[i+1] - rowptr[i] );
    }
    MPI_Allreduce( &nzloc, &nz, 1, MPI_INT, MPI_MAX, grid->comm );
    
    colequ = ( ScalePermstruct->DiagScale == COL ||
	       ScalePermstruct->DiagScale == BOTH );
    C      = ScalePermstruct->C;
    eps    = smach_dist("Epsilon");
    safmin = smach_dist("Safe minimum");
    hugeval= smach_dist("Overflow");
    //eps_d  = dmach_dist("Epsilon");

    /* Set SAFE1 essentially to be the underflow threshold times the
       number of additions in each row. */
    safe1  = nz * safmin;
    safe2  = safe1 / eps;

    /* for ax and y_col (DOUBLE), optionally: ytrue */
    if ( !(ax = doubleMalloc_dist(3 * m_loc)) )
      ABORT("Malloc fails for ax[]");
    y_col = ax + m_loc;

    if ( !(resid = floatMalloc_dist(3 * m_loc)) )  /* for resid/dy */
      ABORT("Malloc fails for resid[]");
    dy = resid;
    Res = resid + m_loc;  // Keep a copy of the residual
    temp = resid + 2 * m_loc;

#if (PRNTlevel >= 2)  // FOR DEBUG: compute ytrue, but C is single
    ytrue = ax + 2*m_loc;
    double ymax = 0.0, ymin = 1.0e+15;
    double local_norm[1], global_norm[1];
    /* ytrue = Pc * inv(C) * xtrue */
    if ( colequ ) { 
      for (i = 0; i < m_loc; ++i)   // y_col[] is temporarily aliased to ytrue
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
    pdPermute_Dense_Matrix(fst_row, m_loc, SOLVEstruct->row_to_proc,
                           ScalePermstruct->perm_c, y_col, ldx,
                           ytrue, ldb, nrhs, grid);
    local_norms[0] = ymax;
    MPI_Reduce( local_norms, global_norms, 1, 
		MPI_DOUBLE, MPI_MAX, 0, grid->comm );
    ymax = global_norms[0];
    local_norms[0] = ymin;
    MPI_Reduce( local_norms, global_norms, 1, 
		MPI_DOUBLE, MPI_MIN, 0, grid->comm );
    ymin = global_norms[0];
    if (iam == 0) {
      printf("Kappa(y): ymax %e / ymin %e = %e\n", ymax, ymin, ymax / ymin);
      fflush(stdout);
    }
#endif

#if ( PRNTlevel>=2 )
    if (iam==0) {
      printf("colequ %d, nz %d,  nrhs %d, eps = %e, anorm = %e, safe1 = %e, safe2 = %e\tldx = %d\tldb = %d\n",
	     colequ, nz, nrhs, eps, anorm, safe1, safe2, (int)ldx, (int)ldb);
      fflush(stdout);
    }
#endif
  
    /* Do for each right-hand side ... */
    for (j = 0; j < nrhs; ++j) {
	lstres = 3.;
	B_col = &B[j*ldb];
	X_col = &X[j*ldx];
	for (i = 0; i < m_loc; ++i) y_col[i] = (double) X_col[i]; /* in double */

	rho_x = rho_x_max = 0.0;
	rho_z = rho_z_max = 0.0;
	prev_normdx = prev_normdz = hugeval;
	final_dx_x = final_dz_z = hugeval;
	x_state = WORKING;
	z_state = UNSTABLE;

	/* Loop until stopping criterion is satisfied. */
	for (count = 0; count < ITMAX; ++count) {

	    /* Compute residual R = diag(R)*B - op(A1) * Y,
	       where A1 = diag(R)*A*diag(C)*Pc',
	       op(A1) = A1, A1**T, or A1**H, depending on TRANS. */

	    /* Matrix-vector multiply. */
#if 0
	    psgsmv(0, A, grid, gsmv_comm, X_col, ax);
#else
	    // FIX: both y_col and ax are in double 
	    psgsmv_d2(0, A, grid, gsmv_comm, y_col, ax);
#endif

	    /* Compute residual, stored in resid[] in SINGLE */
	    for (i = 0; i < m_loc; ++i) resid[i] = B_col[i] - ax[i];

	    /* Save a copy of resid for BERR calculation */
	    for (i = 0; i < m_loc; ++i) Res[i] = resid[i];

	    //if (iam==1) Printdouble5("\tresid", 5, resid); fflush(stdout);

	    /* Compute new dy: dy is aliased to resid, in single */
	    psgstrs(options, n, LUstruct, ScalePermstruct, grid, dy, m_loc,
		    fst_row, m_loc, 1, SOLVEstruct, stat, info);

	    /* Compute norms: normx, normdx, normdz (normz ~= 1) */
	    normx = normy = 0.0;
	    normdx = normdz = 0.0;
	    for (i = 0; i < m_loc; ++i) {
	        yi = fabs(y_col[i]);
		dyi = fabs( (double) dy[i]);
		if ( yi != zero ) normdz = SUPERLU_MAX( normdz, dyi / yi );
		else rho_z = hugeval;

		normy = SUPERLU_MAX( normy, yi);
		if ( colequ ) { /* get unscaled norm */
		    // Sherry OLD: C[i+fst_row], use inv_perm_c or perm_c ???? 
		    Cpi = C[inv_perm_c[i + fst_row]]; // find the permuted position
		    normx = SUPERLU_MAX( normx, Cpi * yi );
		    normdx = SUPERLU_MAX( normdx, Cpi * dyi );
		} else {
		    normx = normy;
		    normdx = SUPERLU_MAX( normdx, dyi );
		}
	    }

	    /* Reduce 3 numbers */
	    local_norms[0] = normx;
	    local_norms[1] = normdx;
	    local_norms[2] = normdz;
	    MPI_Allreduce( local_norms, global_norms, 3,
			   MPI_DOUBLE, MPI_MAX, grid->comm );
			   //   MPI_FLOAT, MPI_MAX, grid->comm );
	    normx = global_norms[0];
	    normdx = global_norms[1];
	    normdz = global_norms[2];

	    /* In the following, all processes should compute the same
	       values, and make same decision.  */
	    /* Compute ratios */
	    if ( normx != zero ) {
	        dx_x = normdx / normx;
	    } else if ( normdx == zero ) { // bug: change "=" to "=="
	        dx_x = zero;
	    } else {
	        dx_x = hugeval;
	    }

	    rho_x = normdx / prev_normdx;
	    rho_z = normdz / prev_normdz;

	    /* Update x-state */
	    if ( x_state == NoPROGRESS && rho_x <= RHO_THRESH )
	        x_state = WORKING;
	    if ( x_state == WORKING ) {
	        if ( dx_x <= eps ) x_state = CONVERGED;
		else if ( rho_x > RHO_THRESH ) x_state = NoPROGRESS;
		else rho_x_max = SUPERLU_MAX( rho_x_max, rho_x );
		if ( x_state > WORKING ) final_dx_x = dx_x;
	    }

	    /* Update z-state */
	    if ( z_state == UNSTABLE && normdz <= DZ_THRESH )
	        z_state = WORKING;
	    if ( z_state == NoPROGRESS && rho_z <= RHO_THRESH )
	        z_state = WORKING;
	    if ( z_state == WORKING ) {
	        if ( normdz <= eps ) z_state = CONVERGED;
		else if ( normdz > DZ_THRESH ) {
		    z_state = UNSTABLE;
		    rho_z_max = 0.0;
		    final_dz_z = hugeval;
		} else if ( rho_z > RHO_THRESH ) z_state = NoPROGRESS;
		else rho_z_max = SUPERLU_MAX( rho_z_max, rho_z );
		if ( z_state > WORKING ) final_dz_z = normdz;;
	    }

	    converge[0][count] = rho_x;
	    converge[1][count] = rho_z;
	    converge[2][count] = dx_x;
	    converge[3][count] = normdz;
	    converge[4][count] = x_state;
	    converge[5][count] = z_state;

#if ( PRNTlevel>=2 )
	    //**** DEBUG: Check against Xtrue and Ytrue at each iteration
	    //  **** DEBUG: print more stuff
	    check_accuracy(count, fst_row, m_loc, nrhs, colequ,
			   A, gsmv_comm, Res, //resid
			   y_col, dy, X_col, ldx, 
			   B_col, C, xtrue, ytrue,
			   temp, inv_perm_c,
			   converge, ferr,
			   SOLVEstruct, grid, safe1, safe2 );
#endif //************** END DEBUG

	    /* Exit if both normwise and componentwise stopped working, but
	       if componentwise is unstable, let it go at least two iterations. */
	    if ( x_state != WORKING ) {
	      if ( z_state == NoPROGRESS || z_state == CONVERGED ) {
		if (stat->RefineSteps == -1) stat->RefineSteps = count; // only record the 1st time
#if ( PRNTlevel<=2 ) //*********** Otherwisee, let it run to ITMAX steps
		break;
#endif
	      }
	      if ( z_state == UNSTABLE && count > 0 ) {
		if (stat->RefineSteps == -1) stat->RefineSteps = count;
#if ( PRNTlevel<=2 )  //*********** Otherwise, let it run to ITMAX steps
		break;
#endif
	      }
	    } 

	    /* Update solution. */
	    for (i = 0; i < m_loc; ++i)
	        y_col[i] = y_col[i] + (double) dy[i];

	    prev_normdx = normdx;
	    prev_normdz = normdz;

	} /* end for iteration count ... */

	/* Copy the improved solution to return. X_col aliased to X */
	//if (iam==0) {printf("  Copy back y_col to X\n"); fflush(stdout);}
	for (i = 0; i < m_loc; ++i) X[i + j*ldx] = y_col[i]; /* round to single */

	/* Set final_* when count hits ITMAX */
	if ( x_state == WORKING ) final_dx_x = dx_x;
	if ( z_state == WORKING ) final_dz_z = normdz;

	/* Compute forward error bounds */
	float err_lowerbnd = SUPERLU_MAX(1.0, sqrt(nz)) * eps;  // 10.0 seems too loose
	
#if ( PRNTlevel>=2 )
	if (iam==0) {
	  //printf("final x_state %d \tfinal z_state %d\n", x_state, z_state);
	  printf(".. nz %d, fudge err_lowerbnd %e\n", nz, err_lowerbnd);
	  printf("err_lowerbnd %e\trho_x_max %e\trho_zmax %e\n",
		 err_lowerbnd, rho_x_max, rho_z_max);
	  printf("final_dx_x/(1 - rho_x_max)  %e\n", final_dx_x/(1-rho_x_max));
	  fflush(stdout);
	}
#endif
	err_bounds[j       ] = SUPERLU_MAX( final_dx_x / (1 - rho_x_max), 
					    err_lowerbnd);
	err_bounds[j + nrhs] = SUPERLU_MAX( final_dz_z / (1 - rho_z_max),
					    err_lowerbnd);

	/* Compute backward error BERR in err_bounds[j + 2*nrhs] */
        err_bounds[j + 2*nrhs] = compute_berr(m_loc, A, gsmv_comm, grid, 
					      B_col, X_col, Res, temp, 
					      safe1, safe2);

#if ( PRNTlevel>=1 ) //*************** DEBUG: TO BE REMOVED
	/* check normwise error of Y 
	 * Normwise relative error in the jth solution vector:
	 * Need to reduce 2 numbers
	 */
	//pdinf_norm_error(iam, m_loc, nrhs, y_col, ldb, ytrue, ldx, grid);

	if (iam == 0) {
	  printf("%12s%12s%16s%20s%12s%8s\n",
		 "rho_x","rho_z","dx_x","normdz","x_state","z_state");
	  for (i = 0; i < count; ++i)
	    printf("%e %e %20.16e %20.16e %4d  %4d\n",
		   converge[0][i], converge[1][i], converge[2][i],
		   converge[3][i], (int)converge[4][i], (int)converge[5][i]);
	  fflush(stdout);

#if ( PRNTlevel>=2 )	  
	  Printdouble5("Y", 3, y_col);
	  Printfloat5("X", 3, X_col);
	  printf("%%IR %16s %16s %16s %16s %10s\n", 
		 "||X-Xt||/||X||", "max |X-Xt|_i/|X|_i",
		 "||Y-Yt||/||Y||", "max |Y-Yt|_i/|Y|_i", "Berr");
	  for (i = 0; i < count; ++i)
	    printf("%d  %16.8e %16.8e %16.8e %16.8e %16.8e\n", i,
		   ferr[0][i], ferr[1][i], ferr[2][i], ferr[3][i], ferr[4][i]);
	  printf("Terminate at step %d\n", stat->RefineSteps);
	  fflush(stdout);
#endif	  
	}
#endif //***************

    } /* for each RHS j ... */

    /* Deallocate storage. */
    SUPERLU_FREE(ax);
    SUPERLU_FREE(resid);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit psgsrfs_d2()");
#endif

} /* PSGSRFS_D2 */



/* Compute backward error BERR in err_bounds[j + 2*nrhs] */
float compute_berr(int m_loc, SuperMatrix *A,
		   psgsmv_comm_t *gsmv_comm, gridinfo_t *grid,
		   float *B, float *X, float *R, float *temp,
		   float safe1, float safe2
		  )
{
    int i;
    float s = 0.0, berr;

    /* Compute abs(op(A))*abs(X) + abs(B), stored in temp[]. */
    psgsmv(1, A, grid, gsmv_comm, X, temp);
    for (i = 0; i < m_loc; ++i) temp[i] += fabs(B[i]);
    s = 0.0;
    for (i = 0; i < m_loc; ++i) {
        if ( temp[i] > safe2 ) {
	    s = SUPERLU_MAX(s, fabs(R[i]) / temp[i]);
	} else if ( temp[i] != 0.0 ) {
	    /* Adding SAFE1 to the numerator guards against
	       spuriously zero residuals (underflow). */
	    s = SUPERLU_MAX(s, (safe1 + fabs(R[i])) / temp[i]);
	}
	/* If temp[i] is exactly 0.0 (computed by PxGSMV), then
	   we know the true residual also must be exactly 0.0. */
    }
    MPI_Allreduce( &s, &berr, 1, MPI_FLOAT, MPI_MAX, grid->comm );
    return(berr);
}
