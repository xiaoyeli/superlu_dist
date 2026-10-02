/*! @file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/



/*
 * -- Distributed SuperLU routine (version 9.3.0) --
 * Lawrence Berkeley National Lab
 *
 * Last update: September 23, 2026, v9.3.0
 */
 
#include "superlu_ddefs.h"
#include "superlu_defs.h"
#include "superlu_upacked.h"
#include <stdbool.h>

static int dvbatch_devres_on(void)
{
    static int on = -1;
    if (on < 0) { const char *e = getenv("SLU_BATCH_DEVRES"); on = e ? (atoi(e) != 0) : 1; }
    return on;
}

/*! \brief Drop the factorization but keep the internal process grid, so a new
 *  Fact = DOFACT call can rebuild on top of it.
 *
 * Tearing the grid down and recreating it per call is what this avoids: on a
 * GPU build that churns the CUDA device binding and fails with
 * "cudaErrorInvalidDevice: invalid device ordinal".
 */
static void dvbatch_ctx_release_factors(dvbatch_ctx_t *ctx)
{
    if ( !ctx->initialized ) return;

#ifdef HAVE_MAGMA
    if ( ctx->LUstruct.batch_dev ) {   /* unpins the host L/U/A buffers before they are freed */
	dbatchDevResFree((dBatchFactorize_Handle) ctx->LUstruct.batch_dev);
	ctx->LUstruct.batch_dev = NULL;
    }
    dvbatch_gpures_free(ctx);
#endif
    dDestroy_LU(ctx->n_big, &(ctx->grid.grid2d), &(ctx->LUstruct));
    dSolveFinalize(&(ctx->options_big), &(ctx->SOLVEstruct));
    dScalePermstructFree(&(ctx->ScalePermstruct));
    dLUstructFree(&(ctx->LUstruct));

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

    ctx->initialized = 0;  /* the grid stays; gridalloc is left set */
} /* end dvbatch_ctx_release_factors */

/*! \brief Release the contents of a context, grid included. */
static void dvbatch_ctx_destroy(dvbatch_ctx_t *ctx)
{
    dvbatch_ctx_release_factors(ctx);
    if ( ctx->gridalloc ) {
	superlu_gridexit3d(&(ctx->grid));
	ctx->gridalloc = 0;
    }
} /* end dvbatch_ctx_destroy */

/*! rief Release the batch state held in F[0] by pdgssvx3d_csc_vbatch().
 *
 * Call this after the last solve of a time-stepping loop.  Safe on F = NULL
 * and on an F[0] that was never populated; F[0] is reset to 0.
 */
void dvbatch_free(handle_t *F)
{
    if ( F == NULL || F[0] == 0 ) return;
    dvbatch_ctx_t *ctx = (dvbatch_ctx_t *) F[0];
    dvbatch_ctx_destroy(ctx);
    SUPERLU_FREE(ctx);
    F[0] = 0;
} /* end dvbatch_free */


/*! \brief Solve a batch of linear systems Ai * Xi = Bi repeatedly, when every
 *    call shares one sparsity pattern; <br>
 * This is the variable-size interface: the matrices may have different
 * dimensions from each other, but matrix i must keep the same structure from
 * one call to the next.
 *
 * <pre>
 * This is the "same pattern" companion of pdgssvx3d_csc_vbatch(), in the same
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
 *   Fact = SamePattern
 *       A later call on the same sparsity pattern whose values changed
 *       enough to pivot again.  Reuses only the column permutation of each
 *       matrix (CpivPtr).  Equilibration and the row permutation (MC64) are
 *       recomputed from the new values; since the row permutation may change,
 *       the stacked system, its symbolic factorization and its distributed
 *       L/U are rebuilt, as on a DOFACT call.
 *
 * No other Fact value is supported; DOFACT must come first.
 *
 * @param[in]      options solver options; options->Fact selects the mode above
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
 * @param[in,out]  F opaque handle to the batch state kept across calls that
 *     share a pattern.  F = NULL asks for a single-shot solve (everything is
 *     built and released inside this call).  Otherwise F[0] must be 0 on the
 *     Fact = DOFACT call; this routine stores the state there and every later
 *     SamePattern_SameRowPerm call reads it back.  Release it with
 *     dvbatch_free(F) after the last solve.
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
pdgssvx3d_csc_vbatch(
		superlu_dist_options_t *options, /* options for algorithm choices and algorithm parameters */
		int batchCount, /* number of matrices in the batch */
		int *m, /* array of matrix row dimensions, size batchCount */
		int *n, /* array of matrix column dimension, size batchCount */
		int *nnz, /* array of number of non-zero entries, size batchCount */
		int nrhs, /* number of right-hand-sides */
		handle_t  *SparseMatrix_handles, /* array of sparse matrix handles,
						  * of size 'batchCount',
						  * each pointing to the actual storage
						  */
		double **RHSptr, // array of pointers to dense RHS storage
		int *ldRHS, // array of leading dimensions of RHS
		double **ReqPtr, /* array of pointers to diagonal row scaling vectors, size batchCount,
				    size of the kth one is m[k]   */
		double **CeqPtr, /* array of pointers to diagonal column scaling vectors, size batchCount,
				    size of the kth one is n[k]    */
		int **RpivPtr, /* array of pointers to row permutation vectors , size batchCount,
				    size of the kth one is m[k] */
		int **CpivPtr, /* array of pointers to column permutation vectors , size batchCount,
				    size of the kth one is n[k] */
		DiagScale_t *DiagScale, /* array of indicators how equilibration is done for each matrix */
		handle_t *F, /* NULL, or F[0] = opaque handle to the batch state kept
			      * across calls sharing a pattern; see dvbatch_free() */
 		double **Xptr, // array of pointers to dense solution storage
		int *ldX, // array of leading dimensions of X
		double **Berrs, /* array of poiniters to backward errors */
		gridinfo3d_t *grid3d,
		SuperLUStat_t *stat,
		int *info
		)
{
    /* Test the options choices. */
    *info = 0;
    SuperMatrix *A0 = (SuperMatrix *) SparseMatrix_handles[0];
    fact_t Fact = options->Fact;

    /* F carries the state that repeated same-pattern solves reuse.
       F == NULL          : single-shot, everything is released before return.
       F != NULL, F[0]==0 : first call, the state is built and stored in F[0].
       F != NULL, F[0]!=0 : a previous call's state, reused or rebuilt. */
    int persist = (F != NULL);
    dvbatch_ctx_t *ctx = (persist && F[0]) ? (dvbatch_ctx_t *) F[0] : NULL;
    int reuse = (Fact == SamePattern_SameRowPerm && ctx != NULL);
    int keep_perm_c = (Fact == SamePattern);   /* CpivPtr[] of the previous call stay */

    if (Fact < 0 || Fact > FACTORED)
	*info = -1;
    else if (options->RowPerm < 0 || options->RowPerm > MY_PERMR)
	*info = -1;
    else if (options->ColPerm < 0 || options->ColPerm > MY_PERMC)
	*info = -1;
    else if (options->IterRefine < 0 || options->IterRefine > SLU_GMRES)
	*info = -1;
    else if (options->IterRefine == SLU_EXTRA)
	{
	    *info = -1;
	    fprintf(stderr,
		    "Extra precise iterative refinement yet to support.");
	}
    else if (batchCount < 0) *info = -2;
    else if (A0->nrow != A0->ncol || A0->nrow < 0 || A0->Stype != SLU_NC || A0->Dtype != SLU_D || A0->Mtype != SLU_GE)
	*info = -7;
    else if (nrhs < 0)
	{
	    *info = -6;
	}
    /* The stored state must match how it was built. */
    else if (Fact == SamePattern_SameRowPerm && ctx == NULL) {
	fprintf(stderr, "pdgssvx3d_csc_vbatch: SamePattern_SameRowPerm asked "
		"for, but F holds no state; call with Fact = DOFACT and a "
		"non-NULL F first.\n");
	*info = -16;
    }
    else if (Fact == SamePattern && (ctx == NULL || !ctx->initialized)) {
	fprintf(stderr, "pdgssvx3d_csc_vbatch: SamePattern asked for, but F "
		"holds no state; call with Fact = DOFACT and a non-NULL F "
		"first.\n");
	*info = -16;
    }
    else if (reuse && (ctx->batchCount != batchCount || ctx->nrhs != nrhs)) {
	fprintf(stderr, "pdgssvx3d_csc_vbatch: batchCount/nrhs (%d/%d) differ "
		"from the ones the state in F was built with (%d/%d).\n",
		batchCount, nrhs, ctx->batchCount, ctx->nrhs);
	*info = -16;
    }
    if (*info) {
	pxerr_dist("pdgssvx3d_csc_vbatch", &(grid3d->grid2d), -(*info));
	return -1;
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid3d->iam, "Enter pdgssvx3d_csc_vbatch()");
#endif

    /* Single-shot solves keep the state on the stack and release it before
       returning; persistent ones own a heap copy addressed by F[0]. */
    dvbatch_ctx_t local_ctx;
    if ( ctx == NULL ) {
	if ( persist ) {
	    ctx = (dvbatch_ctx_t *) SUPERLU_MALLOC( sizeof(dvbatch_ctx_t) );
	    if ( !ctx ) ABORT("Malloc fails for the batch state");
	    F[0] = (handle_t) ctx;
	} else {
	    ctx = &local_ctx;
	}
	memset(ctx, 0, sizeof(dvbatch_ctx_t));
    } else if ( !reuse ) {
	/* Fact = DOFACT on state that already holds a factorization: drop the
	   factors and rebuild, but keep the grid. */
	dvbatch_ctx_release_factors(ctx);
    }

    int colequ, rowequ;
    int_t i, j, k;
    double *C, *R;
    int d; /* index into each matrix in the batch */

    double t = SuperLU_timer_();
    double tv_entry = t, tv_stack0 = 0.0, tv_solver0 = 0.0, tv_phase[6] = {0,0,0,0,0,0};

    int gpures = (options->GPURES == YES);   /* RHSptr[], Xptr[] and the nzval of every handle are device pointers */
    /* fastA: pattern-reuse step with the A-side map ready -- the device L/U
       are filled straight from the caller's values; A is not stacked,
       scaled or permuted on the host, and the caller's matrices are left
       untouched. */
    int fastA = 0;
#ifdef HAVE_MAGMA
    dBatchFactorize_Handle devws = (reuse && ctx) ? (dBatchFactorize_Handle) ctx->LUstruct.batch_dev : NULL;
    fastA = (devws != NULL && dbatchDevResAReady(devws));
#endif
    int buildA = (!reuse && dvbatch_devres_on());   /* DOFACT call: record the entry -> stacked-position map */
    int64_t *posmap = NULL; double *scale2 = NULL; int *ent_sys = NULL, *ent_idx = NULL;
    double **user_nzval = NULL;   /* GPURES: the caller's device value pointers, swapped out for host copies on the DOFACT call */

#ifdef GPU_ACC
    if ( gpures && !reuse ) {
	user_nzval = (double **) SUPERLU_MALLOC(batchCount * sizeof(double *));
	for (d = 0; d < batchCount; ++d) {
	    NCformat *Astore = (NCformat *) ((SuperMatrix *) SparseMatrix_handles[d])->Store;
	    user_nzval[d] = (double *) Astore->nzval;
	    double *h = doubleMalloc_dist(Astore->nnz);
	    checkGPU(gpuMemcpy(h, user_nzval[d], Astore->nnz * sizeof(double), gpuMemcpyDeviceToHost));
	    Astore->nzval = h;
	}
    }
#endif

    if ( !reuse ) {
	int *ainfo = SUPERLU_MALLOC(batchCount * sizeof(int));

	/**** equilibration (LAPACK style) ****/
	/* ReqPtr[] and CeqPtr[] are allocated internally */
	/* Each A may be overwritten by R*A*C */
	dequil_vbatch(options, batchCount, m, n, SparseMatrix_handles,
		      ReqPtr, CeqPtr, DiagScale, ainfo);

	stat->utime[EQUIL] = SuperLU_timer_() - t;
	t = SuperLU_timer_();

	/**** numerical pivoting (e.g., MC64) ****/
	/* If MC64(job=5 is invoked, further equilibration is done,
	 * DiagScale[] will be BOTH, and each A is modified,
	 * perm_r[]'s are applied to each matrix.
	 */
	/* no internal malloc */
	dpivot_vbatch(options, batchCount, m, n, SparseMatrix_handles,
		      ReqPtr, CeqPtr, DiagScale, RpivPtr, ainfo);

	SUPERLU_FREE(ainfo);

	stat->utime[ROWPERM] = SuperLU_timer_() - t;

	/**** sparsity reordering ****/
	/* col perms are computed for each matrix; may be different due to
	 * different row perm.  A may be overwritten as Pr*R*A*C from previous
	 * steps, but is not modified in this routine.
	 */
	t = SuperLU_timer_();

	if ( !keep_perm_c )   /* SamePattern: the column permutations of the previous call are reused */
	    get_perm_c_vbatch(options, batchCount, SparseMatrix_handles, CpivPtr);

	stat->utime[COLPERM] = SuperLU_timer_() - t;

    } else {

	/* Reuse path: the caller handed us matrices with fresh values in the
	 * original structure.  Redo by hand, from the stored data, exactly
	 * what dequil_vbatch() and dpivot_vbatch() did on the DOFACT call:
	 * scale A by the stored R and C, then permute its rows by the stored
	 * perm_r.  Both are O(nnz); the reordering (get_perm_c_vbatch) is
	 * skipped altogether, since the pattern of Pr*A has not changed.
	 */
	for (d = 0; d < (fastA ? 0 : batchCount); ++d) {
	    SuperMatrix *Ad = (SuperMatrix *) SparseMatrix_handles[d];
	    NCformat *Astore = (NCformat *) Ad->Store;
	    double *a = (double *) Astore->nzval;
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
		    double cj = colequ ? C[j] : 1.0;
		    for (i = colptr[j]; i < colptr[j+1]; ++i) {
			double ri = rowequ ? R[rowind[i]] : 1.0;
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

    tv_stack0 = SuperLU_timer_();
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
	fprintf(stderr, "pdgssvx3d_csc_vbatch: the stacked system changed "
		"(m %d->%d, n %d->%d, nnz %d->%d); the pattern is not the "
		"same.\n", ctx->m_big, m_big, ctx->n_big, n_big,
		ctx->nnz_big, nnz_big);
	*info = -2;
	pxerr_dist("pdgssvx3d_csc_vbatch", &(grid3d->grid2d), -(*info));
	return -1;
    }

    double *a_big;
    int_t *colind;
    int_t *rowptr;
    double *b;

    if ( !reuse ) {
	/* Allocate the storage that the context will own from now on.  A_big
	   takes ownership of a_big/colind/rowptr below. */
	a_big = (double *) doubleMalloc_dist(nnz_big);
	colind = (int_t *) intMalloc_dist(nnz_big);
	rowptr = (int_t *) intMalloc_dist(n_big + 1);
	if ( !(b = doubleMalloc_dist(m_big * nrhs)) ) ABORT("Malloc fails for b[:,nrhs]");
    } else {
	a_big = ctx->a_big;
	colind = ctx->colind;
	rowptr = ctx->rowptr;
	b = ctx->b;
    }

    if ( buildA ) {
	posmap  = (int64_t *) SUPERLU_MALLOC(nnz_big * sizeof(int64_t));
	scale2  = (double *)  SUPERLU_MALLOC(nnz_big * sizeof(double));
	ent_sys = (int *)     SUPERLU_MALLOC(nnz_big * sizeof(int));
	ent_idx = (int *)     SUPERLU_MALLOC(nnz_big * sizeof(int));
	if ( !posmap || !scale2 || !ent_sys || !ent_idx ) ABORT("Malloc fails for the A-side map");
    }
    int_t q0 = 0;   /* running entry offset over the systems */
    double *nzval_d; /* each diagonal block */
    int_t *colind_d;
    int_t *rowptr_d;
    int_t nnz_d, col, row, offset_m;
    int *perm_c, *perm_r;

    j = 0;   /* running sum of total nnz */
    row = 0;
    col = 0;
    double alpha = -1.0, beta = 1.0;
    offset_m = 0;

    for (d = 0; d < batchCount; ++d) {

	A = (SuperMatrix *) SparseMatrix_handles[d];
	NCformat *Astore = (NCformat *) A->Store;
	nnz_d = Astore->nnz;
	perm_r = RpivPtr[d];
	perm_c = CpivPtr[d];

	double *idxval = NULL, *nzidx_d = NULL; int_t *rowptr_i = NULL, *colind_i = NULL;
	if ( buildA ) {
	    /* Original row of each entry (rows carry perm_r at this point) and
	       the equilibration factors that the reuse path would apply. */
	    int *inv_perm_r = int32Malloc_dist(m[d]);
	    for (i = 0; i < m[d]; ++i) inv_perm_r[perm_r[i]] = i;
	    rowequ = ( DiagScale[d] == ROW || DiagScale[d] == BOTH );
	    colequ = ( DiagScale[d] == COL || DiagScale[d] == BOTH );
	    R = ReqPtr[d]; C = CeqPtr[d];
	    for (int_t jc = 0; jc < n[d]; ++jc)
		for (int_t p = Astore->colptr[jc]; p < Astore->colptr[jc+1]; ++p) {
		    int io = inv_perm_r[Astore->rowind[p]];
		    scale2[q0 + p]  = (rowequ ? R[io] : 1.0) * (colequ ? C[jc] : 1.0);
		    ent_sys[q0 + p] = d;
		    ent_idx[q0 + p] = (int) p;
		}
	    SUPERLU_FREE(inv_perm_r);
	    idxval = doubleMalloc_dist(nnz_d);
	    for (i = 0; i < nnz_d; ++i) idxval[i] = (double) (i + 1);
	}

	if ( !fastA ) {
	/* Apply perm_c[] to row of A to preserve diagonal: A <= Pc*A */
	for (i = 0; i < nnz_d; ++i)
	    Astore->rowind[i] = perm_c[Astore->rowind[i]];

	/* Convert to CSR format. */
	dCompCol_to_CompRow_dist(m[d], n[d], Astore->nnz, Astore->nzval, Astore->colptr,
				 Astore->rowind, &nzval_d, &rowptr_d, &colind_d);
	if ( buildA )   /* same conversion on the entry indices: nzidx_d[k] = source entry + 1 */
	    dCompCol_to_CompRow_dist(m[d], n[d], Astore->nnz, idxval, Astore->colptr,
				     Astore->rowind, &nzidx_d, &rowptr_i, &colind_i);

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
		if ( buildA ) posmap[q0 + (int_t) (nzidx_d[k] + 0.5) - 1] = j;
		++j;
	    }
	}

	/* move to next block */
	col += n[d];

	SUPERLU_FREE(nzval_d);  /* TODO: remove repeated malloc/free */
	SUPERLU_FREE(colind_d);
	SUPERLU_FREE(rowptr_d);
	if ( buildA ) { SUPERLU_FREE(nzidx_d); SUPERLU_FREE(rowptr_i); SUPERLU_FREE(colind_i); SUPERLU_FREE(idxval); }
	} /* end if !fastA */
	q0 += nnz_d;

	/* Transform the right-hand side: RHS overwritten by B <= R*B */
	double *rhs;

	rowequ = ( DiagScale[d] == ROW || DiagScale[d] == BOTH );
	if ( gpures ) { offset_m += m[d]; continue; }   /* RHS is on the device: stacked below */
	R = ReqPtr[d];
	rhs = RHSptr[d]; // first RHS
	for (k = 0; k < nrhs; ++k) {
	    for (i = 0; i < m[d]; ++i) /* scale by R and permute by Pc*Pr (out-of-place) */
		b[k * m_big + offset_m + perm_c[perm_r[i]]] = (rowequ ? R[i] : 1.0) * rhs[i];
	    rhs += ldRHS[d]; /* move to next RHS */
	}
	offset_m += m[d];

    } /* end for d ... batchCount */

    if ( !fastA ) rowptr[row] = nnz_big;  /* +1 as an end marker */
#ifdef HAVE_MAGMA
    if ( fastA ) {
	/* Device L/U straight from the caller's values (device or host). */
	double **Aptrs = (double **) SUPERLU_MALLOC(batchCount * sizeof(double *));
	for (d = 0; d < batchCount; ++d)
	    Aptrs[d] = (double *) ((NCformat *) ((SuperMatrix *) SparseMatrix_handles[d])->Store)->nzval;
	dbatchDevResRefillA(devws, gpures, Aptrs, nnz);
	SUPERLU_FREE(Aptrs);
    }
#endif

    /**** By now:  each A transformed to Pc*Pr*R*A*C
     ****          each B transformed to R*B
     **** Need to solve (Pc*Pr*R*A*C*Pc')*(Pc*C^{-1}*X) = (Pc*Pr*R)*B
     ****/

    if ( !reuse ) {
	/* Set up A_big in NR_loc format; it takes ownership of the arrays. */
	dCreate_CompRowLoc_Matrix_dist(&(ctx->A_big), m_big, n_big, nnz_big, m_big, 0,
				       a_big, colind, rowptr, SLU_NR_loc, SLU_D, SLU_GE);

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
	dScalePermstructInit (m_big, n_big, &(ctx->ScalePermstruct));
	dLUstructInit (n_big, &(ctx->LUstruct));

	if (!(ctx->berr = doubleCalloc_dist (nrhs))) ABORT ("Malloc fails for berr[].");

	/* Seed these once.  pdgssvx3d() flips them to YES as it initializes
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
#ifdef HAVE_MAGMA
	if ( gpures ) dvbatch_gpures_setup(ctx, batchCount, m, RpivPtr, CpivPtr, ReqPtr, CeqPtr, DiagScale);
#endif
    }

    /* Copy the other options; these may legitimately change per call. */
    /* The stacked system is rebuilt from scratch unless this is a
       SamePattern_SameRowPerm call (its own ColPerm is NATURAL: the column
       permutations are applied per matrix above). */
    ctx->options_big.Fact = reuse ? Fact : DOFACT;
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
     * we do not need to consider perm_c_big outside pdgssvx3d().
     */
#ifdef HAVE_MAGMA
    if ( gpures ) dvbatch_gpures_stack(ctx, batchCount, RHSptr, ldRHS, nrhs);
#endif
    ctx->options_big.GPURES = gpures ? YES : NO;
    tv_solver0 = SuperLU_timer_(); tv_phase[1] = tv_solver0 - tv_stack0;
    pdgssvx3d (&(ctx->options_big), &(ctx->A_big), &(ctx->ScalePermstruct),
	       gpures ? ctx->d_b : b, m_big, nrhs, &(ctx->grid),
	       &(ctx->LUstruct), &(ctx->SOLVEstruct), ctx->berr, stat, info);
    tv_phase[2] = SuperLU_timer_() - tv_solver0;
#ifdef HAVE_MAGMA
    if ( buildA && ctx->LUstruct.batch_dev ) {
	dbatchDevResSetupA((dBatchFactorize_Handle) ctx->LUstruct.batch_dev, batchCount, nnz_big,
			   posmap, scale2, ent_sys, ent_idx, ctx->LUstruct.batch_anorm);
    }
#endif
    if ( buildA ) { SUPERLU_FREE(posmap); SUPERLU_FREE(scale2); SUPERLU_FREE(ent_sys); SUPERLU_FREE(ent_idx); }
#ifdef GPU_ACC
    if ( user_nzval ) {   /* hand the caller's device values back */
	for (d = 0; d < batchCount; ++d) {
	    NCformat *Astore = (NCformat *) ((SuperMatrix *) SparseMatrix_handles[d])->Store;
	    SUPERLU_FREE(Astore->nzval);
	    Astore->nzval = user_nzval[d];
	}
	SUPERLU_FREE(user_nzval);
    }
#endif

#if (PRNTlevel >= 1)
    printf("\tBIG system: berr[0] %e\n", ctx->berr[0]);
    printf("after pdgssvx3d: DiagScale %d\n", ctx->ScalePermstruct.DiagScale);
#endif

    if ( *info ) {  /* Something is wrong */
        if ( grid3d->iam==0 ) {
	    printf("ERROR: INFO = %d returned from pdgssvx3d()\n", *info);
	    fflush(stdout);
	}
    }

    if ( options->PrintStat == YES && ctx->grid.zscp.Iam == 0 ) { // process layer 0
	{ double tpp = SuperLU_timer_(); PStatPrint (options, stat, &(grid3d->grid2d)); tv_phase[3] = SuperLU_timer_() - tpp; }
    }

    /* NOTE: unlike pdgssvx3d_csc_vbatch(), the L/U factors, the stacked matrix
       and the internal grid are deliberately NOT destroyed here -- they are
       what the next SamePattern_SameRowPerm call reuses.  dvbatch_free(F)
       releases them. */

    double tv_post0 = SuperLU_timer_();
    /* Copy the big solution into individual ones, and compute B'errs */
    double bn, rn;  // inf-norm of B and R
    double *x;
    offset_m = 0;
#ifdef HAVE_MAGMA
    if ( gpures ) {
	/* Solution straight to the device Xptr[]; the residual check needs A
	   on the host and is not done in this mode. */
	dvbatch_gpures_unstack(ctx, batchCount, Xptr, ldX, nrhs);
	for (d = 0; d < batchCount; ++d)
	    for (k = 0; k < nrhs; ++k) Berrs[d][k] = -1.0;
    }
#endif
    for (d = 0; d < (gpures ? 0 : batchCount); ++d) {

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

	colequ = ( DiagScale[d] == COL || DiagScale[d] == BOTH );
	rowequ = ( DiagScale[d] == ROW || DiagScale[d] == BOTH );
	C = CeqPtr[d]; R = ReqPtr[d];

	if ( fastA ) {
	    /* The handle holds the caller's original A: finish x (x <= C*z)
	       and take the residual of the original system directly. */
	    x = Xptr[d];
	    for (k = 0; k < nrhs; ++k) {
		bn = 0.; rn = 0.;
		if ( colequ ) for (i = 0; i < n[d]; ++i) x[i] *= C[i];
		for (i = 0; i < m[d]; ++i) {
		    double bi = RHSptr[d][k*ldRHS[d] + i];
		    bn = SUPERLU_MAX( bn, fabs(bi) );
		    b[k*m_big + offset_m + i] = bi;
		}
		sp_dgemv_dist("N", alpha, A, x, 1, beta, &b[k*m_big + offset_m], 1);
		for (i = 0; i < m[d]; ++i) { double v = fabs(b[k*m_big + offset_m + i]); if ( !(v <= rn) ) rn = v; }   /* NaN propagates */
		Berrs[d][k] = rn / bn;
		x += ldX[d];
	    }
	    offset_m += m[d];
	    continue;
	}

	/* Compute residual: Pc*Pr*(R*b) - (Pc*Pr*R*A*C)*z
	 * Now x = Pc'*y, where y is computed from pdgssvx3d()
	 */
	x = Xptr[d];
	for (k = 0; k < nrhs; ++k) {
	    bn = 0.; // norm of B
	    rn = 0.; // norm of R
	    for (i = 0; i < m[d]; ++i) {
		double bi = (rowequ ? R[i] : 1.0) * RHSptr[d][k*ldRHS[d] + i];   /* R*b, the RHS the factors saw */
		bn = SUPERLU_MAX( bn, fabs(bi) );

		/* permute RHS by Pc*Pr, use b[] as temporary storage */
		b[k*m_big + offset_m + perm_c[perm_r[i]]] = bi;
	    }

	    sp_dgemv_dist("N", alpha, A, x, 1, beta, &b[k*m_big + offset_m], 1);

	    for (i = 0; i < m[d]; ++i) { double v = fabs(b[k*m_big + offset_m + i]); if ( !(v <= rn) ) rn = v; }   /* NaN propagates */
	    Berrs[d][k] = rn / bn;
	    x += ldX[d]; /* move to next x */
	} /* end for k ... */
	offset_m += m[d];

	/* Transform the solution matrix X to the solution of the
	 * original system before equilibration: x <= C*z
	 */
	if ( colequ ) {
	    x = Xptr[d];
	    for (k = 0; k < nrhs; ++k) {
		for (i = 0; i < n[d]; ++i) x[i] *= C[i];
		x += ldX[d]; /* move to next x */
	    }
	}

    } /* end for d ... batchCount */

    tv_phase[4] = SuperLU_timer_() - tv_post0;
    tv_phase[0] = tv_stack0 - tv_entry;
    printf("[vbatch] phases ms: scale/perm %.2f  stack %.2f  pdgssvx3d %.2f  PStatPrint %.2f  post(x,berr) %.2f  total %.2f\n",
           1e3*tv_phase[0], 1e3*tv_phase[1], 1e3*tv_phase[2], 1e3*tv_phase[3], 1e3*tv_phase[4], 1e3*(SuperLU_timer_() - tv_entry));
    if ( !persist ) dvbatch_ctx_destroy(ctx); /* single-shot: nothing survives */

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid3d->iam, "Exit pdgssvx3d_csc_vbatch()");
#endif

    return 0;
} /* end pdgssvx3d_csc_vbatch */
