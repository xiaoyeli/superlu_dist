/*
 * -- Distributed SuperLU routine (version 2.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * July 10, 2003
 */

#include "superlu_ddefs.h"

#define HANDLE_SIZE  8

typedef struct {
    ScalePermstruct_t *ScalePermstruct;
    LUstruct_t *LUstruct;
} factors_dist_t;

int
c_fortran_pdgssvx_ABglobal_(int *iopt, int_t *n, int_t *nnz, int *nrhs,
			    double *values, int_t *rowind, int_t *colptr,
			    double *b, int *ldb, int grid_handle[HANDLE_SIZE],
			    double *berr, int factors[HANDLE_SIZE], int *info)

{
/* 
 * Purpose
 * =======
 *
 * This is a Fortran wrapper to use pdgssvx_ABglobal().
 *
 * Arguments
 * =========
 *
 * iopt (input) int
 *      Specifies the operation to be performed:
 *      = 1, performs LU decomposition for the first time
 *      = 2, performs a subsequent LU decomposition for a new matrix
 *           with the same sparsity pattern
 *      = 3, performs triangular solve
 *      = 4, frees all the storage in the end
 *
 * n    (input) int, order of the matrix A
 *
 * nnz  (input) int, number of nonzeros in matrix A
 *
 * nrhs (input) int, number of right-hand sides in the system AX = B
 *
 * values/rowind/colptr (input) column compressed data structure for A
 *
 * b    (input/output) double
 *      On input, the right-hand side matrix of dimension (ldb, nrhs)
 *      On output, the solution matrix
 * 
 * ldb  (input) int, leading dimension of the matrix B
 *
 * grid_handle (input) int array of size 8, holds a pointer to the process
 *      grid structure, which is created and freed separately.
 *
 * berr  (output) double, the backward error of each right-hand side
 *
 * factors (input/output) int array of size 8
 *      If iopt == 1, it is an output and contains the pointer pointing to
 *                    the structure of the factored matrices.
 *      Otherwise, it it an input.
 *
 * info (output) int
 *
 */
    superlu_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    ScalePermstruct_t *ScalePermstruct;
    LUstruct_t *LUstruct;
    int_t    nprow, npcol;
    int      iam;
    int      report;
    int      i;
    gridinfo_t *grid;
    factors_dist_t *LUfactors;

    /*
     * Set option for printing statistics.
     * report = 0: no reporting
     * report = 1: reporting
     */    	
    report = 0;

    /* Locate the process grid. */
    grid = (gridinfo_t *) grid_handle[0];
    iam = (*grid).iam;
    nprow = (int_t) grid->nprow;
    npcol = (int_t) grid->npcol;

    if ( *iopt == 1 ) { /* LU decomposition */

        if ( !iam ) printf(".. Process grid: %d X %d\n", nprow, npcol);

	/* Initialize the statistics variables. */
	PStatInit(&stat);

	dCreate_CompCol_Matrix_dist(&A, *n, *n, *nnz, values, rowind, colptr,
			            SLU_NC, SLU_D, SLU_GE);

	/* Set options. */
	set_default_options(&options);

	/* Initialize ScalePermstruct and LUstruct. */
        ScalePermstruct =
            (ScalePermstruct_t *) SUPERLU_MALLOC(sizeof(ScalePermstruct_t));
        ScalePermstructInit(*n, *n, ScalePermstruct);
        LUstruct = (LUstruct_t *) SUPERLU_MALLOC(sizeof(LUstruct_t));
        LUstructInit(*n, *n, LUstruct);

	/* Call global routine with nrhs=0 to perform the factorization. */
	pdgssvx_ABglobal(&options, &A, ScalePermstruct, NULL, *ldb, 0, 
	                 grid, LUstruct, berr, &stat, info);

	if ( *info == 0 ) {
          if ( report == 1 ) PStatPrint(&options, &stat, grid);
	} else {
	    printf("pdgssvx_ABglobal() error returns INFO= %d\n", *info);
	}
	
	/* Save the LU factors in the factors handle */
	LUfactors = (factors_dist_t*) SUPERLU_MALLOC(sizeof(factors_dist_t));
	LUfactors->ScalePermstruct = ScalePermstruct;
	LUfactors->LUstruct = LUstruct;
	factors[0] = (int) LUfactors;

	/* Free un-wanted storage */
	Destroy_SuperMatrix_Store_dist(&A);
        PStatFree(&stat);

    } else if ( *iopt == 2 ) {
        /* Factor a modified matrix with the same sparsity pattern using
	   existing permutations and L U storage */

	/* Extract the LU factors in the factors handle */
	LUfactors = (factors_dist_t*) factors[0];
	ScalePermstruct = LUfactors->ScalePermstruct;
	LUstruct = LUfactors->LUstruct;

	PStatInit(&stat);

	/* Reset SuperMatrix pointers. */
	dCreate_CompCol_Matrix_dist(&A, *n, *n, *nnz, values, rowind, colptr,
			            SLU_NC, SLU_D, SLU_GE);

	/* Set options. */
	set_default_options(&options);
        options.Fact = SamePattern_SameRowPerm;

	/* Call the routine with nrhs=0 to perform the factorization. */
	pdgssvx_ABglobal(&options, &A, ScalePermstruct, NULL, *ldb, 0, 
	                 grid, LUstruct, berr, &stat, info);

	if ( *info == 0 ) {
          if ( report == 1 ) PStatPrint(&options, &stat, grid);
	} else {
	    printf("pdgssvx_ABglobal() error returns INFO= %d\n", *info);
	}
	
	/* Free un-wanted storage */
	Destroy_SuperMatrix_Store_dist(&A);
        PStatFree(&stat);

    } else if ( *iopt == 3 ) { /* Triangular solve */

	/* Extract the LU factors in the factors handle */
	LUfactors = (factors_dist_t*) factors[0];
	ScalePermstruct = LUfactors->ScalePermstruct;
	LUstruct = LUfactors->LUstruct;

	PStatInit(&stat);

	/* Reset SuperMatrix pointers. */
	dCreate_CompCol_Matrix_dist(&A, *n, *n, *nnz, values, rowind, colptr,
			            SLU_NC, SLU_D, SLU_GE);

	/* Set options. */
	set_default_options(&options);
        options.Fact = FACTORED;

        /* Solve the system A*X=B, overwriting B with X. */
	pdgssvx_ABglobal(&options, &A, ScalePermstruct, b, *ldb, *nrhs, 
	                 grid, LUstruct, berr, &stat, info);

	/* Free un-wanted storage */
	Destroy_SuperMatrix_Store_dist(&A);
        PStatFree(&stat);

    } else if ( *iopt == 4 ) { /* Free storage */

	/* Free the LU factors in the factors handle */
	LUfactors = (factors_dist_t*) factors[0];
	Destroy_LU(*n, grid, LUfactors->LUstruct);
        LUstructFree(LUfactors->LUstruct);
	ScalePermstructFree(LUfactors->ScalePermstruct);
	SUPERLU_FREE(LUfactors->ScalePermstruct);
	SUPERLU_FREE(LUfactors->LUstruct);
        SUPERLU_FREE(LUfactors);

    } else {
	fprintf(stderr, "Invalid iopt=%d passed to c_fortran_pdgssvx_ABglobal()\n", *iopt);
	exit(-1);
    }
}
