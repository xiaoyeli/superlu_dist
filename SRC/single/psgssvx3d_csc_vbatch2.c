/*! @file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/



/*
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab
 * Last update:
 */
#include "superlu_sdefs.h"
#include "superlu_defs.h"
#include "superlu_upacked.h"
#include <stdbool.h>

/*! \brief Zero a batch context before its first use. */
void svbatch_ctx_init(svbatch_ctx_t *ctx)
{
    memset(ctx, 0, sizeof(svbatch_ctx_t));
} /* end svbatch_ctx_init */

/*! \brief Release everything psgssvx3d_csc_vbatch2() kept alive in the context.
 *
 * Safe to call on a context that was only initialized, or twice in a row.
 */
void svbatch_ctx_free(svbatch_ctx_t *ctx)
{
    if ( !ctx->initialized ) {
	if ( ctx->gridalloc ) {  /* grid outlived its factorization */
	    superlu_gridexit3d(&(ctx->grid));
	    ctx->gridalloc = 0;
	}
	return;
    }

    sDestroy_LU(ctx->n_big, &(ctx->grid.grid2d), &(ctx->LUstruct));
    sSolveFinalize(&(ctx->options_big), &(ctx->SOLVEstruct));
    sScalePermstructFree(&(ctx->ScalePermstruct));
    sLUstructFree(&(ctx->LUstruct));

    /* Destroy_CompRowLoc_Matrix_dist() frees a_big/colind/rowptr as well,
       since they are the store of A_big. */
    Destroy_CompRowLoc_Matrix_dist(&(ctx->A_big));
    ctx->a_big = NULL;
    ctx->colind = NULL;
    ctx->rowptr = NULL;

    SUPERLU_FREE(ctx->b);
    SUPERLU_FREE(ctx->berr);
    ctx->b = NULL;
    ctx->berr = NULL;

    superlu_gridexit3d(&(ctx->grid));
    ctx->gridalloc = 0;
    ctx->initialized = 0;
} /* end svbatch_ctx_free */


/*! \brief Solve a batch of linear systems Ai * Xi = Bi repeatedly, when every
 *    call shares one sparsity pattern; <br>
 * This is the variable-size interface: the matrices may have different
 * dimensions from each other, but matrix i must keep the same structure from
 * one call to the next.
 *
 * <pre>
 * This is the "same pattern" companion of psgssvx3d_csc_vbatch(), in the same
 * spirit as pddrive3d2 / PDGSSVX3D with options->Fact = SamePattern_SameRowPerm.
 * It is meant for time-stepping simulations, where at every step the batch has
 * the same structure and only the numerical values change.
 *
 * The caller drives it through options->Fact and a context that lives across
 * the whole time loop:
 *
 *   Fact = DOFACT
 *       First call.  Does the full preprocessing (equilibration, numerical
 *       pivoting, sparsity reordering), stacks the batch into one
 *       block-diagonal system, factors it, and stores in 'ctx' everything
 *       that only depends on the structure.
 *
 *   Fact = SamePattern_SameRowPerm
 *       Every later call.  Reuses from 'ctx' and from the caller's arrays:
 *           ReqPtr, CeqPtr    row/column equilibration of each matrix
 *           RpivPtr           row permutation of each matrix (MC64)
 *           CpivPtr           column permutation of each matrix
 *           DiagScale         how each matrix was equilibrated
 *           ctx->ScalePermstruct, ctx->LUstruct, ctx->SOLVEstruct
 *                             etree, perm_c and symbolic factorization of the
 *                             stacked system, plus its distributed L/U
 *       so the per-step work is the numerical factorization and the solve.
 *       The caller passes matrices holding fresh values in the ORIGINAL
 *       (unpermuted, unscaled) CSC structure; this routine applies the stored
 *       scalings and permutations to them.
 *
 * No other Fact value is supported; DOFACT must come first.
 *
 * @param[in]      options solver options; options->Fact selects the mode above
 * @param[in,out]  ctx persistent context, zeroed by svbatch_ctx_init() before
 *     the DOFACT call and released by svbatch_ctx_free() after the last call
 * @param[in]      batchCount number of matrices in the batch
 * @param[in]      m pointer to the row dimensions of the matrices in the batch
 * @param[in]      n pointer to the column dimensions of the matrices in the batch
 * @param[in]      nnz pointer to the number of non-zero entries of the matrices
 * @param[in]      nrhs number of right-hand-sides
 * @param[in,out]  SparseMatrix_handles array of sparse matrix handles, of size
 *     'batchCount', each pointing to the actual storage in CSC format
 *      Each A is overwritten by Pc*Pr*R*A*C
 * @param[in,out]  RHSptr array of pointers to dense storage of right-hand sides B
 *      Each B is overwritten by row scaling R*B
 * @param[in]      ldRHS array of leading dimensions of RHS
 * @param[in,out]  ReqPtr array of pointers to diagonal row scaling vectors R,
 *     of size 'batchCount', size of the kth one is m[k]
 *     Allocated internally on the DOFACT call, read on later calls
 * @param[in,out]  CeqPtr array of pointers to diagonal column scaling vectors C,
 *     of size 'batchCount', size of the kth one is n[k]
 *     Allocated internally on the DOFACT call, read on later calls
 * @param[in,out]  RpivPtr array of pointers to row permutation vectors,
 *     of size 'batchCount', size of the kth one is m[k]
 * @param[in,out]  CpivPtr array of pointers to column permutation vectors,
 *     of size 'batchCount', size of the kth one is n[k]
 * @param[in,out]  DiagScale array of indicators how equilibration is done
 * @param[out]     F array of handles pointing to the factored matrices (unused)
 * @param[out]     Xptr array of pointers to dense storage of solution
 * @param[in]      ldX array of leading dimensions of X
 * @param[out]     Berrs array of pointers to backward errors
 * @param[in]      grid3d contains MPI communicator
 * @param[out]     stat records algorithms statistics such as runtime, memory usage, etc.
 * @param[out]     info flags the errors on return
 *
 * </pre>
 */
int
psgssvx3d_csc_vbatch2(
		superlu_dist_options_t *options, /* options for algorithm choices and algorithm parameters */
		svbatch_ctx_t *ctx, /* persistent state across the calls sharing a pattern */
		int batchCount, /* number of matrices in the batch */
		int *m, /* array of matrix row dimensions, size batchCount */
		int *n, /* array of matrix column dimension, size batchCount */
		int *nnz, /* array of number of non-zero entries, size batchCount */
		int nrhs, /* number of right-hand-sides */
		handle_t  *SparseMatrix_handles, /* array of sparse matrix handles,
						  * of size 'batchCount',
						  * each pointing to the actual storage
						  */
		float **RHSptr, // array of pointers to dense RHS storage
		int *ldRHS, // array of leading dimensions of RHS
		float **ReqPtr, /* array of pointers to diagonal row scaling vectors, size batchCount,
				    size of the kth one is m[k]   */
		float **CeqPtr, /* array of pointers to diagonal column scaling vectors, size batchCount,
				    size of the kth one is n[k]    */
		int **RpivPtr, /* array of pointers to row permutation vectors , size batchCount,
				    size of the kth one is m[k] */
		int **CpivPtr, /* array of pointers to column permutation vectors , size batchCount,
				    size of the kth one is n[k] */
		DiagScale_t *DiagScale, /* array of indicators how equilibration is done for each matrix */
		handle_t *F, /* array of handles pointing to the factored matrices */
 		float **Xptr, // array of pointers to dense solution storage
		int *ldX, // array of leading dimensions of X
		float **Berrs, /* array of poiniters to backward errors */
		gridinfo3d_t *grid3d,
		SuperLUStat_t *stat,
		int *info
		)
{
    /* Test the options choices. */
    *info = 0;
    SuperMatrix *A0 = (SuperMatrix *) SparseMatrix_handles[0];
    fact_t Fact = options->Fact;
    int reuse = (Fact == SamePattern_SameRowPerm);

    if (Fact != DOFACT && Fact != SamePattern_SameRowPerm) {
	fprintf(stderr,
		"psgssvx3d_csc_vbatch2: only Fact = DOFACT (first call) and "
		"SamePattern_SameRowPerm (later calls) are supported.\n");
	*info = -1;
    }
    else if (options->RowPerm < 0 || options->RowPerm > MY_PERMR)
	*info = -1;
    else if (options->ColPerm < 0 || options->ColPerm > MY_PERMC)
	*info = -1;
    else if (options->IterRefine < 0 || options->IterRefine > SLU_EXTRA)
	*info = -1;
    else if (options->IterRefine == SLU_EXTRA)
	{
	    *info = -1;
	    fprintf(stderr,
		    "Extra precise iterative refinement yet to support.");
	}
    else if (batchCount < 0) *info = -3;
    else if (A0->nrow != A0->ncol || A0->nrow < 0 || A0->Stype != SLU_NC || A0->Dtype != SLU_S || A0->Mtype != SLU_GE)
	*info = -8;
    else if (nrhs < 0)
	{
	    *info = -7;
	}
    /* The context must match how it was built. */
    else if (reuse && !ctx->initialized) {
	fprintf(stderr, "psgssvx3d_csc_vbatch2: SamePattern_SameRowPerm asked "
		"for, but the context holds no factorization; "
		"call with Fact = DOFACT first.\n");
	*info = -2;
    }
    else if (reuse && (ctx->batchCount != batchCount || ctx->nrhs != nrhs)) {
	fprintf(stderr, "psgssvx3d_csc_vbatch2: batchCount/nrhs (%d/%d) differ "
		"from the ones the context was built with (%d/%d).\n",
		batchCount, nrhs, ctx->batchCount, ctx->nrhs);
	*info = -2;
    }
    else if (!reuse && ctx->initialized) {
	fprintf(stderr, "psgssvx3d_csc_vbatch2: Fact = DOFACT on a context that "
		"already holds a factorization; call svbatch_ctx_free() "
		"first.\n");
	*info = -2;
    }
    if (*info) {
	pxerr_dist("psgssvx3d_csc_vbatch2", &(grid3d->grid2d), -(*info));
	return -1;
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid3d->iam, "Enter psgssvx3d_csc_vbatch2()");
#endif

    int colequ, rowequ;
    int_t i, j, k;
    float *C, *R;
    int d; /* index into each matrix in the batch */

    double t = SuperLU_timer_();  /* the timer is double regardless of Ftype */

    if ( !reuse ) {

	/**** equilibration (LAPACK style) ****/
	/* ReqPtr[] and CeqPtr[] are allocated internally */
	/* Each A may be overwritten by R*A*C */
	sequil_vbatch(options, batchCount, m, n, SparseMatrix_handles,
		      ReqPtr, CeqPtr, DiagScale);

	stat->utime[EQUIL] = SuperLU_timer_() - t;
	t = SuperLU_timer_();

	/**** numerical pivoting (e.g., MC64) ****/
	/* If MC64(job=5 is invoked, further equilibration is done,
	 * DiagScale[] will be BOTH, and each A is modified,
	 * perm_r[]'s are applied to each matrix.
	 */
	/* no internal malloc */
	spivot_vbatch(options, batchCount, m, n, SparseMatrix_handles,
		      ReqPtr, CeqPtr, DiagScale, RpivPtr);

	stat->utime[ROWPERM] = SuperLU_timer_() - t;

	/**** sparsity reordering ****/
	/* col perms are computed for each matrix; may be different due to
	 * different row perm.  A may be overwritten as Pr*R*A*C from previous
	 * steps, but is not modified in this routine.
	 */
	t = SuperLU_timer_();

	get_perm_c_vbatch(options, batchCount, SparseMatrix_handles, CpivPtr);

	stat->utime[COLPERM] = SuperLU_timer_() - t;

    } else {

	/* Reuse path: the caller handed us matrices with fresh values in the
	 * original structure.  Redo by hand, from the stored data, exactly
	 * what sequil_vbatch() and spivot_vbatch() did on the DOFACT call:
	 * scale A by the stored R and C, then permute its rows by the stored
	 * perm_r.  Both are O(nnz); the reordering (get_perm_c_vbatch) is
	 * skipped altogether, since the pattern of Pr*A has not changed.
	 */
	for (d = 0; d < batchCount; ++d) {
	    SuperMatrix *Ad = (SuperMatrix *) SparseMatrix_handles[d];
	    NCformat *Astore = (NCformat *) Ad->Store;
	    float *a = (float *) Astore->nzval;
	    int_t *colptr = Astore->colptr;
	    int_t *rowind = Astore->rowind;
	    int *perm_r = RpivPtr[d];

	    rowequ = ( DiagScale[d] == ROW || DiagScale[d] == BOTH );
	    colequ = ( DiagScale[d] == COL || DiagScale[d] == BOTH );
	    R = ReqPtr[d];
	    C = CeqPtr[d];

	    /* A <- diag(R) * A * diag(C), using the untouched row indices */
	    if ( rowequ || colequ ) {
		for (j = 0; j < n[d]; ++j) {
		    float cj = colequ ? C[j] : 1.0;
		    for (i = colptr[j]; i < colptr[j+1]; ++i) {
			float ri = rowequ ? R[rowind[i]] : 1.0;
			a[i] *= ri * cj;
		    }
		}
	    }

	    /* A <- Pr * A */
	    for (i = 0; i < colptr[n[d]]; ++i)
		rowind[i] = perm_r[rowind[i]];
	}

	stat->utime[EQUIL] = SuperLU_timer_() - t;
	stat->utime[ROWPERM] = 0.0;
	stat->utime[COLPERM] = 0.0;
    }

#if (PRNTlevel >= 1)
    printf("<---- END PREPROCESSING ----\n");
#endif

    /*---------------------
     **** Stack the matrices into block diagonal form: A_big, and RHS B_big
     ----------------------*/

    /* Count total dimension and number of nonzeros. */
    SuperMatrix *A;
    int m_big = 0, n_big = 0, nnz_big = 0;
    for (d = 0; d < batchCount; ++d) {
	m_big += m[d];
	n_big += n[d];
	A = (SuperMatrix *) SparseMatrix_handles[d];
	NCformat *Astore = (NCformat *) A->Store;
	nnz_big += Astore->nnz;
    }

    if ( reuse &&
	 (m_big != ctx->m_big || n_big != ctx->n_big || nnz_big != ctx->nnz_big) ) {
	fprintf(stderr, "psgssvx3d_csc_vbatch2: the stacked system changed "
		"(m %d->%d, n %d->%d, nnz %d->%d); the pattern is not the "
		"same.\n", ctx->m_big, m_big, ctx->n_big, n_big,
		ctx->nnz_big, nnz_big);
	*info = -2;
	pxerr_dist("psgssvx3d_csc_vbatch2", &(grid3d->grid2d), -(*info));
	return -1;
    }

    float *a_big;
    int_t *colind;
    int_t *rowptr;
    float *b;

    if ( !reuse ) {
	/* Allocate the storage that the context will own from now on.  A_big
	   takes ownership of a_big/colind/rowptr below. */
	a_big = (float *) floatMalloc_dist(nnz_big);
	colind = (int_t *) intMalloc_dist(nnz_big);
	rowptr = (int_t *) intMalloc_dist(n_big + 1);
	if ( !(b = floatMalloc_dist(m_big * nrhs)) ) ABORT("Malloc fails for b[:,nrhs]");
    } else {
	a_big = ctx->a_big;
	colind = ctx->colind;
	rowptr = ctx->rowptr;
	b = ctx->b;
    }

    float *nzval_d; /* each diagonal block */
    int_t *colind_d;
    int_t *rowptr_d;
    int_t nnz_d, col, row, offset_m;
    int *perm_c, *perm_r;

    j = 0;   /* running sum of total nnz */
    row = 0;
    col = 0;
    float alpha = -1.0, beta = 1.0;
    offset_m = 0;

    for (d = 0; d < batchCount; ++d) {

	A = (SuperMatrix *) SparseMatrix_handles[d];
	NCformat *Astore = (NCformat *) A->Store;
	nnz_d = Astore->nnz;
	perm_r = RpivPtr[d];
	perm_c = CpivPtr[d];

	/* Apply perm_c[] to row of A to preserve diagonal: A <= Pc*A */
	for (i = 0; i < nnz_d; ++i)
	    Astore->rowind[i] = perm_c[Astore->rowind[i]];

	/* Convert to CSR format. */
	sCompCol_to_CompRow_dist(m[d], n[d], Astore->nnz, Astore->nzval, Astore->colptr,
				 Astore->rowind, &nzval_d, &rowptr_d, &colind_d);

	/* Copy this CSR matrix to a diagonal block of A_big.
	   Apply each perm_c[] to each matrix by column.
	   Now, diagonal block is permuted by Pc*A*Pc'
	*/

	/* Apply perm_c[] to columns of A (out-of-place) */
	for (i = 0; i < m[d]; ++i) {
	    rowptr[row++] = j;
	    for (k = rowptr_d[i]; k < rowptr_d[i+1]; ++k) {
		colind[j] = perm_c[colind_d[k]] + col;  // add the *col* shift
		a_big[j] = nzval_d[k];
		++j;
	    }
	}

	/* move to next block */
	col += n[d];

	SUPERLU_FREE(nzval_d);  /* TODO: remove repeated malloc/free */
	SUPERLU_FREE(colind_d);
	SUPERLU_FREE(rowptr_d);

	/* Transform the right-hand side: RHS overwritten by B <= R*B */
	float *rhs;

	rowequ = ( DiagScale[d] == ROW || DiagScale[d] == BOTH );
	if ( rowequ ) { /* Scale RHS by R[] */
	    R = ReqPtr[d];
	    rhs = RHSptr[d]; // first RHS
	    for (k = 0; k < nrhs; ++k) {
		for (i = 0; i < m[d]; ++i) rhs[i] *= R[i];
		rhs += ldRHS[d]; /* move to next RHS */
	    }
	}

	rhs = RHSptr[d]; // first RHS
	for (k = 0; k < nrhs; ++k) {
	    for (i = 0; i < m[d]; ++i) /* permute RHS by Pc*Pr (out-of-place) */
		b[k * m_big + offset_m + perm_c[perm_r[i]]] = rhs[i];
	    rhs += ldRHS[d]; /* move to next RHS */
	}
	offset_m += m[d];

    } /* end for d ... batchCount */

    rowptr[row] = nnz_big;  /* +1 as an end marker */

    /**** By now:  each A transformed to Pc*Pr*R*A*C
     ****          each B transformed to R*B
     **** Need to solve (Pc*Pr*R*A*C*Pc')*(Pc*C^{-1}*X) = (Pc*Pr*R)*B
     ****/

    if ( !reuse ) {
	/* Set up A_big in NR_loc format; it takes ownership of the arrays. */
	sCreate_CompRowLoc_Matrix_dist(&(ctx->A_big), m_big, n_big, nnz_big, m_big, 0,
				       a_big, colind, rowptr, SLU_NR_loc, SLU_S, SLU_GE);

	/* Modify the input options.
	 * Turn off preprocessing options for the big system.
	 */
	set_default_options_dist(&(ctx->options_big));
	ctx->options_big.Equil  = NO;
	ctx->options_big.ColPerm  = NATURAL;
	ctx->options_big.RowPerm  = NOROWPERM;
	ctx->options_big.ParSymbFact = NO;
	ctx->options_big.batchCount = batchCount;

	/* Need a grid of size 1; create it only on the DOFACT call. */
	if ( !ctx->gridalloc ) {
	    int nprow = 1, npcol = 1, npdep = 1;
	    superlu_gridinit3d (grid3d->comm, nprow, npcol, npdep, &(ctx->grid));
	    ctx->gridalloc = 1;
	}

	/* Initialize ScalePermstruct and LUstruct. */
	sScalePermstructInit (m_big, n_big, &(ctx->ScalePermstruct));
	sLUstructInit (n_big, &(ctx->LUstruct));

	if (!(ctx->berr = floatCalloc_dist (nrhs))) ABORT ("Malloc fails for berr[].");

	/* Seed these once.  psgssvx3d() flips them to YES as it initializes
	   SOLVEstruct and the refinement workspace; since both live in the
	   context, the later calls must see YES and skip re-initializing. */
	ctx->options_big.SolveInitialized = options->SolveInitialized;
	ctx->options_big.RefineInitialized = options->RefineInitialized;

	ctx->batchCount = batchCount;
	ctx->nrhs = nrhs;
	ctx->m_big = m_big;
	ctx->n_big = n_big;
	ctx->nnz_big = nnz_big;
	ctx->a_big = a_big;
	ctx->colind = colind;
	ctx->rowptr = rowptr;
	ctx->b = b;
	ctx->initialized = 1;
    }

    /* Copy the other options; these may legitimately change per call. */
    ctx->options_big.Fact = Fact;
    ctx->options_big.ReplaceTinyPivot = options->ReplaceTinyPivot;
    ctx->options_big.IterRefine = options->IterRefine;
    ctx->options_big.UseGMRES = options->UseGMRES;
    ctx->options_big.Trans = options->Trans;
    ctx->options_big.PrintStat = options->PrintStat;

    /*---------------------
     **** Call the linear equation solver
     ----------------------*/

    /* perm_c_big may not be Identity due to etree postordering, however,
     * since b[] is transormed back to the solution of the original BIG system,
     * we do not need to consider perm_c_big outside psgssvx3d().
     */
    if ( ctx->options_big.IterRefine >= SLU_DOUBLE ) {
	/* Double-precision iterative refinement on the stacked system, as in
	 * psgssvx3d_csc_vbatch().  err_bounds[] is scratch/output for the d2
	 * refine; xtrue_big is only touched under PRNTlevel>=2 diagnostics, so
	 * a zeroed dummy is sufficient.  Both are per-call scratch and are not
	 * worth keeping in the context. */
	extern double *doubleCalloc_dist(int_t);
	float *err_bounds;
	double *xtrue_big;
	if (!(err_bounds = floatCalloc_dist(3 * nrhs)))
	    ABORT("Malloc fails for err_bounds[].");
	if (!(xtrue_big = doubleCalloc_dist((int_t) m_big * nrhs)))
	    ABORT("Malloc fails for xtrue_big[].");

	psgssvx3d_d2 (&(ctx->options_big), &(ctx->A_big), &(ctx->ScalePermstruct),
		      b, m_big, nrhs, &(ctx->grid),
		      &(ctx->LUstruct), &(ctx->SOLVEstruct), ctx->berr,
		      err_bounds, stat, info, xtrue_big);

	SUPERLU_FREE(err_bounds);
	SUPERLU_FREE(xtrue_big);
    } else {
	psgssvx3d (&(ctx->options_big), &(ctx->A_big), &(ctx->ScalePermstruct),
		   b, m_big, nrhs, &(ctx->grid),
		   &(ctx->LUstruct), &(ctx->SOLVEstruct), ctx->berr, stat, info);
    }

#if (PRNTlevel >= 1)
    printf("\tBIG system: berr[0] %e\n", ctx->berr[0]);
#endif

    if ( *info ) {  /* Something is wrong */
        if ( grid3d->iam==0 ) {
	    printf("ERROR: INFO = %d returned from psgssvx3d()\n", *info);
	    fflush(stdout);
	}
    }

    if ( options->PrintStat == YES && ctx->grid.zscp.Iam == 0 ) { // process layer 0
	PStatPrint (options, stat, &(grid3d->grid2d)); /* Print 2D statistics.*/
    }

    /* NOTE: unlike psgssvx3d_csc_vbatch(), the L/U factors, the stacked matrix
       and the internal grid are deliberately NOT destroyed here -- they are
       what the next SamePattern_SameRowPerm call reuses.  svbatch_ctx_free()
       releases them. */

    /* Copy the big solution into individual ones, and compute B'errs */
    float bn, rn;  // inf-norm of B and R
    float *x;
    offset_m = 0;
    for (d = 0; d < batchCount; ++d) {

	A = (SuperMatrix *) SparseMatrix_handles[d];
	perm_c = CpivPtr[d];
        perm_r = RpivPtr[d];

	/* Permute the solution matrix z <= Pc'*y */
	x = Xptr[d];
	for (k = 0; k < nrhs; ++k) {
	    for (i = 0; i < n[d]; ++i)
		x[i] = b[k* m_big + offset_m + perm_c[i]];
	    x += ldX[d]; /* move to next x */
	}

	/* Compute residual: Pc*Pr*(R*b) - (Pc*Pr*R*A*C)*z
	 * Now x = Pc'*y, where y is computed from psgssvx3d()
	 */
	x = Xptr[d];
	for (k = 0; k < nrhs; ++k) {
	    bn = 0.; // norm of B
	    rn = 0.; // norm of R
	    for (i = 0; i < m[d]; ++i) {
		bn = SUPERLU_MAX( bn, fabs(RHSptr[d][k*m[d] + i]) );

		/* permute RHS by Pc*Pr, use b[] as temporary storage */
		b[k*m_big + offset_m + perm_c[perm_r[i]]] = RHSptr[d][k*ldRHS[d] + i];
	    }

	    sp_sgemv_dist("N", alpha, A, x, 1, beta, &b[k*m_big + offset_m], 1);

	    for (i = 0; i < m[d]; ++i) rn = SUPERLU_MAX( rn, fabs(b[k*m_big + offset_m + i]) );
	    Berrs[d][k] = rn / bn;
	    x += ldX[d]; /* move to next x */
	} /* end for k ... */
	offset_m += m[d];

	/* Transform the solution matrix X to the solution of the
	 * original system before equilibration: x <= C*z
	 */
	colequ = ( DiagScale[d] == COL || DiagScale[d] == BOTH );
	if ( colequ ) {
	    C = CeqPtr[d];
	    x = Xptr[d];
	    for (k = 0; k < nrhs; ++k) {
		for (i = 0; i < n[d]; ++i) x[i] *= C[i];
		x += ldX[d]; /* move to next x */
	    }
	}

    } /* end for d ... batchCount */

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid3d->iam, "Exit psgssvx3d_csc_vbatch2()");
#endif

    return 0;
} /* end psgssvx3d_csc_vbatch2 */
