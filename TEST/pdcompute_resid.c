/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Test for small residual.
 *
 * -- Distributed SuperLU routine (version 5.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 30, 2017
 *
 */
#include "superlu_ddefs.h"

int pdcompute_resid(int m, int n, int nrhs, SuperMatrix *A,
		    double *x, int ldx, double *b, int ldb,
		    gridinfo_t *grid, dSOLVEstruct_t *SOLVEstruct, double *resid)
{
/*  
    Purpose   
    =======   

    PDCOMPUTE_RESID computes the residual for a solution of a system of linear   
    equations  A*x = b  or  A'*x = b:   
       RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS ),   
    where EPS is the machine epsilon.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    NRHS    (input) INTEGER   
            The number of columns of B, the matrix of right hand sides.   
            NRHS >= 0.
	    
    A       (input/output) SuperMatrix*
            The original M x N sparse matrix A.   
	    On exit, the column indices are modified due to SPMV setup.

    X       (input) DOUBLE PRECISION array, dimension (LDX,NRHS)   
            The computed solution vectors for the system of linear   
            equations.   

    LDX     (input) INTEGER   
            The leading dimension of the array X.  If TRANS = NOTRANS,   
            LDX >= max(1,N); if TRANS = TRANS or CONJ, LDX >= max(1,M).   

    B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)   
            On entry, the right hand side vectors for the system of   
            linear equations.   
            On exit, B is overwritten with the difference B - A*X.   

    LDB     (input) INTEGER   
            The leading dimension of the array B.  IF TRANS = NOTRANS,
            LDB >= max(1,M); if TRANS = TRANS or CONJ, LDB >= max(1,N).

    SOLVEstruct (input) dSOLVEstruct_t*

    GRID    (input) gridinfo_t*
	    
    RESID   (output) double PRECISION   
            The maximum over the number of right-hand sides of
            norm(B - A*X) / ( norm(A) * norm(X) * EPS ).   

    =====================================================================
*/

    /* Table of constant values */
    int    inc  = 1;
    
    /* Local variables */
    int i, j;
    double anorm, rnorm, rnorm_g;
    double xnorm, xnorm_g;
    double eps;
    char transc[1];
    double *ax, *R;
    pdgsmv_comm_t gsmv_comm; 
    int m_loc = ((NRformat_loc*) A->Store)->m_loc;

    /* Function prototypes */
    extern double dasum_(int *, double *, int *);
    
    /* Function Body */
    if ( m <= 0 || n <= 0 || nrhs == 0) {
	*resid = 0.;
	return 0;
    }

    /* Exit with RESID = 1/EPS if ANORM = 0. */
    eps = dmach_dist("Epsilon");
    anorm = pdlangs("1", A, grid);
    if (anorm <= 0.) {
	*resid = 1. / eps;
	return 0;
    }

    if ( !(ax = doubleMalloc_dist(m_loc)) ) ABORT("Malloc fails for work[]");
    R = ax;

    /* A is modified with colind[] permuted to [internal, external]. */
    pdgsmv_init(A, SOLVEstruct->row_to_proc, grid, &gsmv_comm);

    /* Compute the maximum over the number of right-hand sides of   
       norm(B - A*X) / ( norm(A) * norm(X) * EPS ) . */
    *resid = 0.;
    for (j = 0; j < nrhs; ++j) {
	double *B_col = &b[j*ldb];
	double *X_col = &x[j*ldx];

	/* Compute residual R = B - op(A) * X,   
	   where op(A) = A, A**T, or A**H, depending on TRANS. */
	/* Matrix-vector multiply. */
	pdgsmv(0, A, grid, &gsmv_comm, X_col, ax);
	    
	/* Compute residual, stored in R[]. */
	for (i = 0; i < m_loc; ++i) R[i] = B_col[i] - ax[i];

	rnorm = dasum_(&m_loc, R, &inc);
	xnorm = dasum_(&m_loc, X_col, &inc);

	/* */
	MPI_Allreduce( &rnorm, &rnorm_g, 1, MPI_DOUBLE, MPI_SUM, grid->comm );
	MPI_Allreduce( &xnorm, &xnorm_g, 1, MPI_DOUBLE, MPI_SUM, grid->comm );
		
	if (xnorm_g <= 0.) {
	    *resid = 1. / eps;
	} else {
	    /* Computing MAX */
	    double d1, d2;
	    d1 = *resid;
	    d2 = rnorm_g / anorm / xnorm_g / eps;
	    *resid = SUPERLU_MAX(d1, d2);
	}
    } /* end for j ... */

    pdgsmv_finalize(&gsmv_comm);
    SUPERLU_FREE(ax);

    return 0;

} /* pdcompute_redid */
