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
 * January 13, 2024
 * Last update:
 */
#include "superlu_ddefs.h"
#include "superlu_defs.h"
#include "superlu_upacked.h"
#include <stdbool.h>

int file_sPrint_CompRowLoc_to_Triples(SuperMatrix *A)
{
    NRformat_loc *Astore = A->Store;
    int nnz, m, n, i, j;
    float  *dp;
    FILE *fp = fopen("CSR.txt", "w");

    m = A->nrow;
    n = A->ncol;
    nnz = Astore->nnz_loc;
    dp = Astore->nzval;

    printf("print to triples: m %d, n %d, nnz %d\n", m, n, nnz);
    for (i = 0; i < m; ++i) {
	for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
	    fprintf(fp, "%8d %8d %16.8e\n", i, (int) Astore->colind[j], dp[j]);
	}
    }
    fclose(fp);
    return 0;
}

/*! \brief Solve a batch of linear systems Ai * Xi = Bi with direct method,
 *    computing the LU factorization of each matrix Ai; <br>
 * This is the fixed-size interface: all the input matrices have the same sparsity structure
 *
 * <pre>
 * @param[in]      options solver options
 * @param[in]      batchCount number of matrices in the batch
 * @param[in]      m row dimension of the matrices
 * @param[in]      n column dimension of the matrices
 * @param[in]      nnz number of non-zero entries in each matrix
 * @param[in]      nrhs number of right-hand-sides
 * @param[in,out]  SparseMatrix_handles  array of sparse matrix handles, of size 'batchCount', each pointing to the actual storage in CSC format, see 'NCformat' in SuperMatix structure
 *      Each A is overwritten by row/col scaling R*A*C
 * @param[in,out]  RHSptr  array of pointers to dense storage of right-hand sides B
 *      Each B is overwritten by row/col scaling R*B*C
 * @param[in]      ldRHS array of leading dimensions of RHS
 * @param[in,out]  ReqPtr array of pointers to diagonal row scaling vectors R, each of size m
 *    ReqPtr[] are allocated internally if equilibration is asked for
 * @param[in,out]  CeqPtr array of pointers to diagonal colum scaling vectors C, each of size n
 *    CeqPtr[] are allocated internally if equilibration is asked for
 * @param[in,out]  RpivPtr array of pointers to row permutation vectors, each of size m
 * @param[in,out]  CpivPtr array of pointers to column permutation vectors, each of size n
 * @param[in,out]  DiagScale array of indicators how equilibration is done for each matrix
 * @param[out]     F array of handles pointing to the factored matrices
 * @param[out]     Xptr array of pointers to dense storage of solution
 * @param[in]      ldX array of leading dimensions of X
 * @param[out]     Berrs array of poiniters to backward errors
 * @param[in]]     grid3d contains MPI communicator
 * @param[out]     stat records algorithms statistics such as runtime, memory usage, etc.
 * @param[out]     info flags the errors on return
 *
 * </pre>
 */
int
psgssvx3d_csc_batch(
		superlu_dist_options_t *options, /* options for algorithm choices and algorithm parameters */
		int batchCount, /* number of matrices in the batch */
		int m, /* matrix row dimension */
		int n, /* matrix column dimension */
		int nnz, /* number of non-zero entries */
		int nrhs, /* number of right-hand-sides */
		handle_t  *SparseMatrix_handles, /* array of sparse matrix handles,
						  * of size 'batchCount',
						  * each pointing to the actual storage
						  */
		float **RHSptr, // array of pointers to dense RHS storage
		int *ldRHS, // array of leading dimensions of RHS
		float **ReqPtr, /* array of pointers to diagonal row scaling vectors,
				     each of size M   */
		float **CeqPtr, /* array of pointers to diagonal column scaling vectors,
				    each of size N    */
		int **RpivPtr, /* array of pointers to row permutation vectors , each of size M */
		int **CpivPtr, /* array of pointers to column permutation vectors , each of size N */
		DiagScale_t *DiagScale, /* indicate how equilibration is done for each matrix */
		handle_t *F, /* array of handles pointing to the factored matrices */
 		float **Xptr, // array of pointers to dense solution storage
		int *ldX, // array of leading dimensions of X
		float **Berrs, /* array of poiniters to backward errors */
		gridinfo3d_t *grid3d,
		SuperLUStat_t *stat,
		int *info
		//DeviceContext context /* device context including queues, events, dependencies */
		)
{
    /* Steps in this routine
     1. Loop through all matrices A_i, perform preprocessing (all in serial format)
            1.1 equilibration
	    1.2 numerical pivoting (e.g., MC64)
	    1.3 sparsity reordering

     2. Copy the matrices into block diagonal form: A_big
        (can be merged into Step 3, then no need to have a big copy)

     3. Factorize A_big -> LU_big
            3.1 symbolic factoization (not batched, can loop through each individual one serial)
	    3.2 numerical factorization

     4. Split LU_big into individual LU factors, store them in the handle array to return
        (this may be difficult, and the users may not need them.)

     5. Solve (2 designs)
             5.1 using LU_big -- requires RHS to be in contiguous memory
	         compute level set. Leverage B-to-X with an internal copy.
        (OR) 5.2 loop through individual LU -- may lose some data-parallel opportunity
    */

    /* Test the options choices. */
    *info = 0;
    SuperMatrix *A0 = (SuperMatrix *) SparseMatrix_handles[0];
    fact_t Fact = options->Fact;

    if (Fact < 0 || Fact > FACTORED)
	*info = -1;
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
    else if (batchCount < 0) *info = -2;
    /* Need to check M, N, NNZ */
    else if (A0->nrow != A0->ncol || A0->nrow < 0 || A0->Stype != SLU_NC || A0->Dtype != SLU_S || A0->Mtype != SLU_GE)
	*info = -7;
    else if (nrhs < 0)
	{
	    *info = -6;
	}
    if (*info) {
	pxerr_dist("psgssvx3d_csc_batch", &(grid3d->grid2d), -(*info));
	return -1;
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid3d->iam, "Enter psgssvx3d_csc_batch()");
#endif

    int colequ, Equil, factored, job, notran, rowequ, need_value;
    int_t i, iinfo, j, k, irow;
    int ldx; /* LDA for matrix X (local). */
    float *C, *R; //*C1, *R1, amax, anorm, colcnd, rowcnd;
    float GA_mem_use;	/* memory usage by global A */
    float dist_mem_use; /* memory usage during distribution */
    superlu_dist_mem_usage_t num_mem_usage, symb_mem_usage;
    int d; /* index into each matrix in the batch */

    double t = SuperLU_timer_();

    /**** equilibration (LAPACK style) ****/
    /* ReqPtr[] and CeqPtr[] are allocated internally */
    /* Each A may be overwritten by R*A*C */
    sequil_batch(options, batchCount, m, n, SparseMatrix_handles,
		 ReqPtr, CeqPtr, DiagScale);

    stat->utime[EQUIL] = SuperLU_timer_() - t;
    t = SuperLU_timer_();

    /**** numerical pivoting (e.g., MC64) ****/
    /* If MC64(job=5 is invoked, further equilibration is done,
     * DiagScale[] will be BOTH, and each A is modified,
     * perm_r[]'s are applied to each matrix.
     */
    /* no internal malloc */
    spivot_batch(options, batchCount, m, n, SparseMatrix_handles,
		 ReqPtr, CeqPtr, DiagScale, RpivPtr);

    stat->utime[ROWPERM] = SuperLU_timer_() - t;

#if 0
    for (d = 0; d < batchCount; ++d) {
	printf("DiagScale[%d] %d\n", d, DiagScale[d]);
	if ( DiagScale[d] ) {
	    Printfloat5("ReqPtr[d]", m, ReqPtr[d]);
	    Printfloat5("CeqPtr[d]", m, CeqPtr[d]);
	}
	PrintInt32("RpivPtr[d]", m, RpivPtr[d]);
    }
#endif

    /**** sparsity reordering ****/
    /* col perms are computed for each matrix; may be different due to different row perm.
     * A may be overwritten as Pr*R*A*C from previous steps, but is not modified in this routine.
     */
    t = SuperLU_timer_();

    get_perm_c_batch(options, batchCount, SparseMatrix_handles, CpivPtr);

    stat->utime[COLPERM] = SuperLU_timer_() - t;
#if 0
    for (d = 0; d < batchCount; ++d) {
	PrintInt32("CpivPtr[d]", m, CpivPtr[d]);
    }
#endif

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
	m_big += m;
	n_big += n;
	A = (SuperMatrix *) SparseMatrix_handles[d];
	NCformat *Astore = (NCformat *) A->Store;
	nnz_big += Astore->nnz;
    }

    /* Allocate storage in CSR containing all matrices in the batch */
    // TO-DELETE: dallocateA_dist(n, nnz, &nzval, &rowind, &colptr);
    float *a_big = (float *) floatMalloc_dist(nnz_big);
    int_t *colind = (int_t *) intMalloc_dist(nnz_big);
    int_t *rowptr = (int_t *) intMalloc_dist(n_big + 1);
    float *nzval_d; /* each diagonal block */
    int_t *colind_d;
    int_t *rowptr_d;
    int_t nnz_d, col, row;
    int *perm_c, *perm_r;

    /* B_big */
    float *b;
    if ( !(b = floatMalloc_dist(m_big * nrhs)) ) ABORT("Malloc fails for b[:,nrhs]");

    j = 0;   /* running sum of total nnz */
    row = 0;
    col = 0;
    float alpha = -1.0, beta = 1.0;

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
	sCompCol_to_CompRow_dist(m, n, Astore->nnz, Astore->nzval, Astore->colptr,
				 Astore->rowind, &nzval_d, &rowptr_d, &colind_d);

	//PrintInt32("rowptr_d", m+1, rowptr_d);

	/* Copy this CSR matrix to a diagonal block of A_big.
	   Apply each perm_c[] to each matrix by column.
	   Now, diagonal block is permuted by Pc*A*Pc'
	*/

	/* Apply perm_c[] to columns of A (out-of-place) */
	for (i = 0; i < m; ++i) {
	    rowptr[row++] = j;
	    //irow = iperm_c[i]; // old irow
	    //for (k = rowptr_d[irow]; k < rowptr_d[irow+1]; ++k) {
	    for (k = rowptr_d[i]; k < rowptr_d[i+1]; ++k) {
		colind[j] = perm_c[colind_d[k]] + col;  // add the *col* shift
		a_big[j] = nzval_d[k];
		++j;
	    }
	}

	/* move to next block */
	col += n;

	SUPERLU_FREE(nzval_d);  /* TODO: remove repeated malloc/free */
	SUPERLU_FREE(colind_d);
	SUPERLU_FREE(rowptr_d);

	/* Transform the right-hand side: RHS overwritten by B <= R*B */
	float *rhs;

	// NEED TO SAVE A COPY OF RHS ??

	rowequ = ( DiagScale[d] == ROW || DiagScale[d] == BOTH );
	//printf("  before transform RHS: rowequ %d\n", rowequ);
	if ( rowequ ) { /* Scale RHS by R[] */
	    R = ReqPtr[d];
	    rhs = RHSptr[d]; // first RHS
	    for (k = 0; k < nrhs; ++k) {
		for (i = 0; i < m; ++i) rhs[i] *= R[i];
		rhs += ldRHS[d]; /* move to next RHS */
	    }
	}

#if ( DEBUGlevel>=1 )
	printf("System %d, next row %d, next col %d, next j %d\n", d, row, col, j);
	//Printfloat5("big-RHS", m, RHSptr[d]);
#endif

	rhs = RHSptr[d]; // first RHS
	for (k = 0; k < nrhs; ++k) {
	    for (i = 0; i < m; ++i) /* permute RHS by Pc*Pr (out-of-place) */
		b[k * m_big + d * m + perm_c[perm_r[i]]] = rhs[i];
	    rhs += ldRHS[d]; /* move to next RHS */
	}

	//Printdouble5("big-RHS-permuted", m, &b[(k-1) * m_big + d * m]);

    } /* end for d ... batchCount */

    // assert(j == nnz_big);
    // assert(row == m_big);
    rowptr[row] = nnz_big;  /* +1 as an end marker */

    /**** By now:  each A transformed to Pc*Pr*R*A*C
     ****          each B transformed to R*B
     **** Need to solve (Pc*Pr*R*A*C*Pc')*(Pc*C^{-1}*X) = (Pc*Pr*R)*B
     ****/

    /* Set up A_big in NR_loc format */
    SuperMatrix A_big;
    sCreate_CompRowLoc_Matrix_dist(&A_big, m_big, n_big, nnz_big, m_big, 0,
				   a_big, colind, rowptr, SLU_NR_loc, SLU_S, SLU_GE);

    //file_dPrint_CompRowLoc_to_Triples(&A_big);

    superlu_dist_options_t options_big;
    set_default_options_dist(&options_big);
    options_big.Equil  = NO;
    options_big.ColPerm  = NATURAL;
    options_big.RowPerm  = NOROWPERM;
    options_big.ParSymbFact = NO;
    options_big.batchCount = batchCount;

    /* Copy most of the other options */
    options_big.Fact = options->Fact;
    options_big.ReplaceTinyPivot = options->ReplaceTinyPivot;
    options_big.IterRefine = options->IterRefine;
    options_big.Trans = options->Trans;
    options_big.SolveInitialized = options->SolveInitialized;
    options_big.RefineInitialized = options->RefineInitialized;
    options_big.PrintStat = options->PrintStat;

    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    sSOLVEstruct_t SOLVEstruct;
    gridinfo3d_t grid;
    float *berr;
    MPI_Comm comm = grid3d->comm;

    /* Need to create a grid of size 1 */
    int nprow = 1, npcol = 1, npdep = 1;
    superlu_gridinit3d (comm, nprow, npcol, npdep, &grid);

    /* Initialize ScalePermstruct and LUstruct. */
    sScalePermstructInit (m_big, n_big, &ScalePermstruct);
    sLUstructInit (n_big, &LUstruct);

    //printf("\tbefore pdgssvx3d: m_big %d, n_big %d, nrhs %d\n", m_big, n_big, nrhs);
    //dPrint_CompRowLoc_Matrix_dist(&A_big);

    if (!(berr = floatCalloc_dist (nrhs))) ABORT ("Malloc fails for berr[].");

    /*---------------------
     **** Call the linear equation solver
     ----------------------*/

    /*!!!! CHECK SETTING: TO BE SURE TO USE GPU VERSIONS !!!!
       gpu3dVersion
       superlu_acc_offload
    */
    /* perm_c_big may not be Identity due to etree postordering, however,
     * since b[] is transormed back to the solution of the original BIG system,
     * we do not need to consider perm_c_big outside psgssvx3d().
     */
    psgssvx3d (&options_big, &A_big, &ScalePermstruct, b, m_big, nrhs, &grid,
               &LUstruct, &SOLVEstruct, berr, stat, info);

#if (PRNTlevel >= 1)
    printf("\tBIG system: berr[0] %e\n", berr[0]);
    printf("after psgssvx3d: DiagScale %d\n", ScalePermstruct.DiagScale);
    //PrintInt10("after pdgssvx3d: ScalePermstruct.perm_c", (int_t) m_big, ScalePermstruct.perm_c);
    //Printdouble5("big-B-solution", m_big, b);
#endif

    if ( *info ) {  /* Something is wrong */
        if ( grid3d->iam==0 ) {
	    printf("ERROR: INFO = %d returned from psgssvx3d()\n", *info);
	    fflush(stdout);
	}
    }

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------ */

    sDestroy_LU (n_big, &(grid.grid2d), &LUstruct);
    if ( grid.zscp.Iam == 0 ) { // process layer 0
	    PStatPrint (options, stat, &(grid3d->grid2d)); /* Print 2D statistics.*/
    }

    sSolveFinalize (&options_big, &SOLVEstruct);

    Destroy_CompRowLoc_Matrix_dist (&A_big);
    sScalePermstructFree (&ScalePermstruct);
    sLUstructFree (&LUstruct);

    /* Copy the big solution into individual ones, and compute B'errs */
    float bn, rn;  // inf-norm of B and R
    float *x;
    for (d = 0; d < batchCount; ++d) {

	A = (SuperMatrix *) SparseMatrix_handles[d];
	perm_c = CpivPtr[d];
        perm_r = RpivPtr[d];

	/* Permute the solution matrix z <= Pc'*y */
	//PrintInt32("prepare Pc'*y: perm_c", n, perm_c);
	x = Xptr[d];
	for (k = 0; k < nrhs; ++k) {
	    for (i = 0; i < n; ++i)
		x[i] = b[k* m_big + d * m + perm_c[i]];
	    x += ldX[d]; /* move to next x */
	}

	//Printdouble5("Permuted-solution after iperm_c", n, Xptr[d]);

	/* Compute residual: Pc*Pr*(R*b) - (Pc*Pr*R*A*C)*z
	 * Now x = Pc'*y, where y is computed from pdgssvx3d()
	 */
	x = Xptr[d];
	for (k = 0; k < nrhs; ++k) {  // Sherry: can call sp_dgemm_dist() !!!!
	    bn = 0.; // norm of B
	    rn = 0.; // norm of R
	    for (i = 0; i < m; ++i) {
		bn = SUPERLU_MAX( bn, fabs(RHSptr[d][k*m + i]) );

		/* permute RHS by Pc*Pr, use b[] as temporary storage */
		b[k*m_big + d*m + perm_c[perm_r[i]]] = RHSptr[d][k*ldRHS[d] + i];
	    }

	    sp_sgemv_dist("N", alpha, A, x, 1, beta, &b[k*m_big + d*m], 1);

	    for (i = 0; i < m; ++i) rn = SUPERLU_MAX( rn, fabs(b[k*m_big + d*m + i]) );
	    Berrs[d][k] = rn / bn;
	    x += ldX[d]; /* move to next x */
	} /* end for k ... */

	/* Transform the solution matrix X to the solution of the
	 * original system before equilibration: x <= C*z
	 */
	colequ = ( DiagScale[d] == COL || DiagScale[d] == BOTH );
	if ( colequ ) {
	    C = CeqPtr[d];
	    x = Xptr[d];
	    for (k = 0; k < nrhs; ++k) {
		for (i = 0; i < n; ++i) x[i] *= C[i];
		x += ldX[d]; /* move to next x */
	    }
	}

    } /* end for d ... batchCount */

    SUPERLU_FREE (b);
    SUPERLU_FREE (berr);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(grid3d->iam, "Exit psgssvx3d_csc_batch()");
#endif

    return 0;
} /* end psgssvx3d_csc_batched */
