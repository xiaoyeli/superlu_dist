/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Double-precision (mixed-precision) iterative refinement for the 3D solver.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab
 *
 * PSGSRFS3D_D2 is the 3D analogue of PSGSRFS_D2: it improves the computed
 * solution to A*X=B and provides error bounds and backward error estimates,
 * computing the residual and accumulating the solution in DOUBLE precision
 * while the correction solves are carried out by the (single-precision) 3D
 * triangular solve psgstrs3d.  The residual matrix-vector products and norm
 * reductions are performed on grid-layer 0 (the only layer that holds the
 * gathered RHS/solution), and the continue/stop decision is broadcast across
 * the Z (replication) dimension so that all layers call psgstrs3d in lockstep.
 * </pre>
 */

#include <math.h>
#include "superlu_sdefs.h"

#define ITMAX 10
#define RHO_THRESH 0.5
#define DZ_THRESH  0.25

/* Backward-error helper, shared with the 2D routine psgsrfs_d2.c */
extern float compute_berr(int m_loc, SuperMatrix *A,
			  psgsmv_comm_t *gsmv_comm, gridinfo_t *grid,
			  float *B, float *X, float *R, float *temp,
			  float safe1, float safe2);

/* Double-precision sparse matrix-vector multiply (defined in psgsmv_d2.c) */
extern void psgsmv_d2(int abs, SuperMatrix *A_internal,
		      gridinfo_t *grid, psgsmv_comm_t *gsmv_comm,
		      double x[], double ax[]);
extern double *doubleMalloc_dist(int_t);

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   See psgsrfs_d2() for the meaning of the common arguments.  The additional
 *   3D arguments are:
 *
 * grid3d (input) gridinfo3d_t*
 *        The 3D process grid.  grid3d->grid2d is the 2D mesh used for the
 *        residual computation; grid3d->zscp is the Z (replication) communicator.
 *
 * trf3Dpartition (input) strf3Dpartition_t*
 *        The 3D factorization partition needed by psgstrs3d.
 *
 * err_bounds (output) float*, dimension (nrhs * 3) (global)
 *        err_bounds[j + 0*nrhs] : normwise forward error bound
 *        err_bounds[j + 1*nrhs] : componentwise forward error bound
 *        err_bounds[j + 2*nrhs] : componentwise backward error
 *
 * xtrue  (input) double* (local)
 *        The true solution (in the original, unscaled, unpermuted ordering).
 *        Used only for the optional PRNTlevel>=2 accuracy diagnostics; the
 *        refinement itself does not depend on it.
 * </pre>
 */
void
psgsrfs3d_d2(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
	     float anorm, sLUstruct_t *LUstruct,
	     sScalePermstruct_t *ScalePermstruct, gridinfo3d_t *grid3d,
	     strf3Dpartition_t *trf3Dpartition,
	     float *B, int_t ldb, float *X, int_t ldx, int nrhs,
	     sSOLVEstruct_t *SOLVEstruct, float *err_bounds,
	     SuperLUStat_t *stat, int *info,
	     double *xtrue /* xtrue[] is used only for checking purpose */)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int   on_layer0 = (grid3d->zscp.Iam == 0);

    float *resid, *dy, *temp, *Res, *B_col, *X_col, *C;
    int   *inv_perm_c = SOLVEstruct->inv_perm_c;
    double *ax, *y_col;
    int   iam, count, i, j, nz, m_loc, fst_row, colequ;
    float eps, safmin, safe1, safe2;       /* working (single) precision */

    double Cpi;
    double normy, normx, normdx, normdz, prev_normdx, prev_normdz;
    double yi, dyi, dx_x, final_dx_x, final_dz_z;
    double rho_x, rho_x_max, rho_z, rho_z_max, hugeval;
    double local_norms[3], global_norms[3];   /* for MPI reduction */
    double zero = 0.0;

    psgsmv_comm_t *gsmv_comm = SOLVEstruct->gsmv_comm;
    NRformat_loc *Astore;
    int_t *rowptr;

    typedef enum {UNSTABLE, WORKING, CONVERGED, NoPROGRESS} IRstate_t;
    int x_state, z_state;

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
	pxerr_dist("PSGSRFS3D_D2", grid, i);
	return;
    }

    /* Quick return if possible. */
    if ( n == 0 || nrhs == 0 ) return;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter psgsrfs3d_d2()");
#endif

    colequ = ( ScalePermstruct->DiagScale == COL ||
	       ScalePermstruct->DiagScale == BOTH );
    C      = ScalePermstruct->C;
    eps    = smach_dist("Epsilon");
    safmin = smach_dist("Safe minimum");
    hugeval= smach_dist("Overflow");

    /* NZ = maximum number of nonzero elements in each row of A */
    int nzloc = 0;
    for (i = 0; i < m_loc; ++i)
	nzloc = SUPERLU_MAX( nzloc, rowptr[i+1] - rowptr[i] );
    MPI_Allreduce( &nzloc, &nz, 1, MPI_INT, MPI_MAX, grid->comm );

    /* Set SAFE1 essentially to be the underflow threshold times the
       number of additions in each row. */
    safe1  = nz * safmin;
    safe2  = safe1 / eps;

    /* ax and y_col are DOUBLE (only needed on layer 0, but allocate anyway). */
    if ( !(ax = doubleMalloc_dist(2 * m_loc)) )
	ABORT("Malloc fails for ax[]");
    y_col = ax + m_loc;

    /* resid/dy/Res/temp are SINGLE; dy must exist on all layers for psgstrs3d. */
    if ( !(resid = floatMalloc_dist(3 * m_loc)) )
	ABORT("Malloc fails for resid[]");
    dy   = resid;             /* aliased: residual in, correction out */
    Res  = resid + m_loc;     /* keep a copy of the residual for BERR */
    temp = resid + 2 * m_loc;

    /* Do for each right-hand side ... */
    for (j = 0; j < nrhs; ++j) {
	B_col = &B[j*ldb];
	X_col = &X[j*ldx];

	if ( on_layer0 )
	    for (i = 0; i < m_loc; ++i) y_col[i] = (double) X_col[i];

	rho_x = rho_x_max = 0.0;
	rho_z = rho_z_max = 0.0;
	prev_normdx = prev_normdz = hugeval;
	final_dx_x = final_dz_z = hugeval;
	x_state = WORKING;
	z_state = UNSTABLE;
	dx_x = normdx = normdz = hugeval;

	/* Loop until stopping criterion is satisfied. */
	for (count = 0; count < ITMAX; ++count) {

	    int do_break = 0;

	    /* Compute residual R = diag(R)*B - op(A1)*Y in DOUBLE, on layer 0.
	       resid[] (single) = B_col - ax, where ax = A1*y_col (double). */
	    if ( on_layer0 ) {
		psgsmv_d2(0, A, grid, gsmv_comm, y_col, ax);
		for (i = 0; i < m_loc; ++i) resid[i] = B_col[i] - ax[i];
		for (i = 0; i < m_loc; ++i) Res[i] = resid[i];
	    }

	    /* Compute new dy via the 3D triangular solve (single precision).
	       All Z-layers must participate.  dy is aliased to resid; on
	       layer 0 it holds the residual on entry, the correction on exit. */
	    if ( get_new3dsolve() ) {
		psgstrs3d_newsolve(options, n, LUstruct, ScalePermstruct,
				   trf3Dpartition, grid3d, dy, m_loc, fst_row,
				   m_loc, 1, SOLVEstruct, stat, info);
	    } else {
		psgstrs3d(options, n, LUstruct, ScalePermstruct,
			  trf3Dpartition, grid3d, dy, m_loc, fst_row,
			  m_loc, 1, SOLVEstruct, stat, info);
	    }

	    /* Norm computation + state machine + solution update on layer 0. */
	    if ( on_layer0 ) {
		normx = normy = 0.0;
		normdx = normdz = 0.0;
		for (i = 0; i < m_loc; ++i) {
		    yi = fabs(y_col[i]);
		    dyi = fabs( (double) dy[i] );
		    if ( yi != zero ) normdz = SUPERLU_MAX( normdz, dyi / yi );
		    else rho_z = hugeval;

		    normy = SUPERLU_MAX( normy, yi );
		    if ( colequ ) { /* get unscaled norm */
			Cpi = C[inv_perm_c[i + fst_row]];
			normx = SUPERLU_MAX( normx, Cpi * yi );
			normdx = SUPERLU_MAX( normdx, Cpi * dyi );
		    } else {
			normx = normy;
			normdx = SUPERLU_MAX( normdx, dyi );
		    }
		}

		local_norms[0] = normx;
		local_norms[1] = normdx;
		local_norms[2] = normdz;
		MPI_Allreduce( local_norms, global_norms, 3,
			       MPI_DOUBLE, MPI_MAX, grid->comm );
		normx = global_norms[0];
		normdx = global_norms[1];
		normdz = global_norms[2];

		if ( normx != zero ) dx_x = normdx / normx;
		else if ( normdx == zero ) dx_x = zero;
		else dx_x = hugeval;

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
		    if ( z_state > WORKING ) final_dz_z = normdz;
		}

		/* Decide whether to stop. */
		if ( x_state != WORKING ) {
		    if ( z_state == NoPROGRESS || z_state == CONVERGED ) {
			if (stat->RefineSteps == -1) stat->RefineSteps = count;
			do_break = 1;
		    } else if ( z_state == UNSTABLE && count > 0 ) {
			if (stat->RefineSteps == -1) stat->RefineSteps = count;
			do_break = 1;
		    }
		}

		if ( !do_break ) {
		    /* Update solution in double. */
		    for (i = 0; i < m_loc; ++i)
			y_col[i] = y_col[i] + (double) dy[i];
		    prev_normdx = normdx;
		    prev_normdz = normdz;
		}
	    } /* end if on_layer0 */

	    /* All layers agree on whether to continue (keeps psgstrs3d in sync). */
	    MPI_Bcast( &do_break, 1, MPI_INT, 0, grid3d->zscp.comm );
	    if ( do_break ) break;

	} /* end for iteration count ... */

	if ( on_layer0 ) {
	    /* Copy the improved solution back to X (round to single). */
	    for (i = 0; i < m_loc; ++i) X_col[i] = (float) y_col[i];

	    /* Set final_* when count hits ITMAX without a state change. */
	    if ( x_state == WORKING ) final_dx_x = dx_x;
	    if ( z_state == WORKING ) final_dz_z = normdz;

	    /* Forward error bounds. */
	    float err_lowerbnd = SUPERLU_MAX(1.0, sqrt((double)nz)) * eps;
	    err_bounds[j       ] = SUPERLU_MAX( final_dx_x / (1 - rho_x_max),
						err_lowerbnd );
	    err_bounds[j + nrhs] = SUPERLU_MAX( final_dz_z / (1 - rho_z_max),
						err_lowerbnd );

	    /* Componentwise backward error. */
	    err_bounds[j + 2*nrhs] = compute_berr(m_loc, A, gsmv_comm, grid,
						  B_col, X_col, Res, temp,
						  safe1, safe2);
	}

    } /* for each RHS j ... */

    /* Make the refined solution and error bounds available on all Z-layers. */
    MPI_Bcast( X, ldx * nrhs, MPI_FLOAT, 0, grid3d->zscp.comm );
    MPI_Bcast( err_bounds, 3 * nrhs, MPI_FLOAT, 0, grid3d->zscp.comm );
    MPI_Bcast( &stat->RefineSteps, 1, MPI_INT, 0, grid3d->zscp.comm );

    /* Deallocate storage. */
    SUPERLU_FREE(ax);
    SUPERLU_FREE(resid);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit psgsrfs3d_d2()");
#endif

} /* PSGSRFS3D_D2 */
