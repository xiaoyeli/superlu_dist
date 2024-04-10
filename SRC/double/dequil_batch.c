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
 * November 5, 2023
 * Last update:
 */
#include "superlu_ddefs.h"

/*! \brief Equilibrate the systems using the LAPACK-style algorithm
 *
 * @param[in]      options solver options
 * @param[in]      batchCount number of matrices in the batch
 * @param[in]      m row dimension of the matrices
 * @param[in]      n column dimension of the matrices
 * @param[in, out] SparseMatrix_handles pointers to the matrices in the batch, each pointing to the actual stoage in CSC format
 *     On entry, the original matrices
 *     On exit, each matrix may be overwritten by diag(R)*A*diag(C)
 * @param[out]     ReqPtr pointers to row scaling vectors (allocated internally)
 * @param[out]     CeqPtr pointers to column scaling vectors (allocated internally)
 * @param[in, out] DiagScale arrays indicating how each system is equilibrated: {ROW, COL, BOTH}
 *
 * Return value i:
 *     = 0: successful exit
 *     > 0: indicates the first matrix in the batch has zero row or column
 *          if i <= m: the i-th row of A is exactly zero
 *          if i >  m: the (i-m)-th column of A is exactly zero
 * </pre>
 */
int
dequil_batch(
    superlu_dist_options_t *options, /* options for algorithm choices and algorithm parameters */
    int batchCount, /* number of matrices in the batch */
    int m, /* matrix row dimension */
    int n, /* matrix column dimension */
    handle_t  *SparseMatrix_handles, /* array of sparse matrix handles, of size 'batchCount',
				      * each pointing to the actual storage
				      */
    double **ReqPtr, /* array of pointers to diagonal row scaling vectors,
			each of size M   */
    double **CeqPtr, /* array of pointers to diagonal column scaling vectors,
			each of size N    */
    DiagScale_t *DiagScale /* How equilibration is done for each matrix. */
    //    DeviceContext context /* device context including queues, events, dependencies */
		  )
{
    int i, j, irow, icol, info = 0;
    fact_t Fact = options->Fact;
    int factored = (Fact == FACTORED);
    int Equil = (!factored && options->Equil == YES);
    int notran = (options->Trans == NOTRANS);
    
#if (DEBUGlevel >= 1)
    CHECK_MALLOC(0, "Enter dequil_batch()");
#endif
    /* Decipher the input matrices */
    SuperMatrix **A = SUPERLU_MALLOC(batchCount * sizeof(SuperMatrix *));
    for (i = 0; i < batchCount; ++i) {
	A[i] = (SuperMatrix *) SparseMatrix_handles[i];
    }

    /* Loop through each matrix in the batch */
    for (int k = 0; k < batchCount; ++k) {
	
	NCformat *Astore = (NCformat *) A[k]->Store;
	double *a = (double *) Astore->nzval;
	int_t *colptr = Astore->colptr;
	int_t *rowind = Astore->rowind;
	
	/* Assuming each matrix is in CSC format
	 * Otherwise, convert to CSC first -- use the code in dvpivot_batch.c
	 * ...
	 */

	/* The following arrays are replicated on all processes. */
	double *R = ReqPtr[k];
	double *C = CeqPtr[k];
	
	/* Allocate stoage if not factored & ask for equilibration */
	if (Equil && Fact != SamePattern_SameRowPerm) {
	    /* Allocate storage if not done so before. */
	    //switch (ScalePermstruct->DiagScale)
	    switch ( DiagScale[k] ) {
		case NOEQUIL:
		    if (!(R = (double *)doubleMalloc_dist(m))) ABORT("Malloc fails for R[].");
		    if (!(C = (double *)doubleMalloc_dist(n))) ABORT("Malloc fails for C[].");
		    ReqPtr[k] = R;
		    CeqPtr[k] = C;
		    break;
		case ROW: /* R[] was already allocated before */
		    if (!(C = (double *)doubleMalloc_dist(n))) ABORT("Malloc fails for C[].");
		    CeqPtr[k] = C;
		    break;
		case COL: /* C[] was already allocated before */
		    if (!(R = (double *)doubleMalloc_dist(m))) ABORT("Malloc fails for R[].");
		    ReqPtr[k] = R;
		    break;
		default:
		    break;
		}
	}

	/* ------------------------------------------------------------
	   Diagonal scaling to equilibrate the matrix.
	   ------------------------------------------------------------ */
	if ( Equil ) {
	    if (Fact == SamePattern_SameRowPerm) {
		/* Reuse R and C. */
		switch ( DiagScale[k] ) {
		    case NOEQUIL: break;
		    case ROW:
			for (j = 0; j < n; ++j) {
			    for (i = colptr[j]; i < colptr[j + 1]; ++i) {
				irow = rowind[i];
				a[i] *= R[irow]; /* Scale rows. */
			    }
			}
			break;
		    case COL:
			for (j = 0; j < n; ++j) {
			    double cj = C[j];
			    for (i = colptr[j]; i < colptr[j+1]; ++i) {
				a[i] *= cj; /* Scale columns. */
			    }
			}
			break;
		    case BOTH:
			for (j = 0; j < n; ++j) {
			    double cj = C[j];
			    for (i = colptr[j]; i < colptr[j + 1]; ++i) {
				irow = rowind[i];
				a[i] *= R[irow] * cj; /* Scale rows and cols. */

			    }
			}
			break;
		} /* end switch DiagScale[k] ... */
		
	    } else { /* Compute R[] & C[] from scratch */
		
		int iinfo;
		char equed[1];
		double amax, anorm, colcnd, rowcnd;
		
		/* Compute the row and column scalings. */

		dgsequ_dist(A[k], R, C, &rowcnd, &colcnd, &amax, &iinfo);
		
		if (iinfo > 0) {
		    if (iinfo <= m) {
#if (PRNTlevel >= 1)
			fprintf(stderr, "Matrix %d: the %d-th row of A is exactly zero\n", k, (int)iinfo);
#endif
		    } else {
#if (PRNTlevel >= 1)
			fprintf(stderr, "Matrix %d: the %d-th column of A is exactly zero\n", k, (int)iinfo - n);
#endif
		    }
		} else if (iinfo < 0) {
		    if ( info==0 ) info = iinfo;
		}

		/* Now iinfo == 0 */

		/* Equilibrate matrix A if it is badly-scaled.
		   A <-- diag(R)*A*diag(C)                     */
		dlaqgs_dist(A[k], R, C, rowcnd, colcnd, amax, equed);

		if (strncmp(equed, "R", 1) == 0) {
		    DiagScale[k] = ROW;
		} else if (strncmp(equed, "C", 1) == 0) {
		    DiagScale[k] = COL;
		} else if (strncmp(equed, "B", 1) == 0)	{
		    DiagScale[k] = BOTH;
		} else {
		    DiagScale[k] = NOEQUIL;
		}
#if (PRNTlevel >= 1)
		printf(".. equilibrated? *equed = %c, DiagScale[k] %d\n", *equed, DiagScale[k]);
		fflush(stdout);
#endif
	    } /* end if-else Fact ... */

	} /* end if Equil ... LAPACK style, not involving MC64 */
	
    } /* end for k ... batchCount */

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(0, "Exit dequil_batch()");
#endif
    return info;
} /* end dequil_batch */
