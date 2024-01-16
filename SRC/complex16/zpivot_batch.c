/*! \file
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
#include "superlu_zdefs.h"

/*! \brief Compute row pivotings for each matrix, for numerical stability
 * <pre>
 *
 * @param[in]      options solver options
 * @param[in]      batchCount number of matrices in the batch
 * @param[in]      m row dimension of the matrices
 * @param[in]      n column dimension of the matrices
 * @param[in, out] SparseMatrix_handles pointers to the matrices in the batch, each pointing to the actual stoage in CSC format
 *     On entry, the original matrices, may be overwritten by A1 <- diag(R)*A*diag(C) from dequil_batch()
 *     On exit, each matrix may be A2 <- Pr*A1
 * @param[in,out]  ReqPtr pointers to row scaling vectors, maybe overwritten by scaling from MC64
 * @param[in,out]  CeqPtr pointers to column scaling vectors, maybe overwritten by scaling from MC64
 * @param[in,out] DiagScale array indicating how each system is equilibrated: {ROW, COL, BOTH}
 * @param[in,out] RpivPtr pointers to row permutation vectors for each matrix, each of size m
 *     On exit, each RpivPtr[] is applied to each matrix
 *
 * Return value:
 *     0,  success
 *     -1, invalid RowPerm option; an Identity perm_r[] is returned
 *     d, indicates that the d-th matrix is the first one in the batch encountering error
 * </pre>
 */
int
zpivot_batch(
    superlu_dist_options_t *options, /* options for algorithm choices and algorithm parameters */
    int batchCount, /* number of matrices in the batch */
    int m, /* matrix row dimension */
    int n, /* matrix column dimension */
    handle_t  *SparseMatrix_handles, /* array of sparse matrix handles,
				      * of size 'batchCount', each pointing to the actual storage
				      */
    double **ReqPtr, /* array of pointers to diagonal row scaling vectors,
			each of size M   */
    double **CeqPtr, /* array of pointers to diagonal column scaling vectors,
			each of size N    */
    DiagScale_t *DiagScale, /* How equilibration is done for each matrix. */
    int **RpivPtr /* array of pointers to row permutation vectors , each of size M */
    //    DeviceContext context /* device context including queues, events, dependencies */
		  )
{
    int i, j, irow, iinfo, rowequ, colequ, info = 0;
    fact_t Fact = options->Fact;
    int factored = (Fact == FACTORED);
    int Equil = (!factored && options->Equil == YES);
    int notran = (options->Trans == NOTRANS);
    int job = 5;
    double *R1, *C1;

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(0, "Enter dpivot_batch()");
#endif
    
    /* Decipher the input matrices */
    SuperMatrix **A;
    A = SUPERLU_MALLOC(batchCount * sizeof(SuperMatrix *));
    for (i = 0; i < batchCount; ++i) {
	A[i] = (SuperMatrix *) SparseMatrix_handles[i];
    }

    if (job == 5) {
	/* Allocate storage for scaling factors. */
	if (!(R1 = doubleMalloc_dist(m)))
	    ABORT("SUPERLU_MALLOC fails for R1[]");
	if (!(C1 = doubleMalloc_dist(n)))
	    ABORT("SUPERLU_MALLOC fails for C1[]");
    }

    int_t *colptr;
    int_t *rowind;
    doublecomplex *a, *at;
    int_t nnz;
    
    /* Loop through each matrix in the batch */
    for (int d = 0; d < batchCount; ++d) {

	rowequ = ( DiagScale[d] == ROW || DiagScale[d] == BOTH );
	colequ = ( DiagScale[d] == COL || DiagScale[d] == BOTH );
	
	/* If the matrix type is SLU_NR (CSR), then need to convert to CSC first */
	if ( A[d]->Stype == SLU_NR ) { /* CSR format */
	    NRformat *Astore = (NRformat *) A[d]->Store;
	    a = (doublecomplex *)Astore->nzval;
	    
	    zCompRow_to_CompCol_dist(m, n, nnz, a,
				     Astore->colind, Astore->rowptr,
				     &at, &rowind, &colptr);
	    
	    a = at; // now a[] points to at[], stored in CSC format.
	    nnz = Astore->nnz;
	} else { /* CSC format */
	    NCformat *Astore = (NCformat *) A[d]->Store;
	    a = (doublecomplex *)Astore->nzval;
	    colptr = Astore->colptr;
	    rowind = Astore->rowind;
	    nnz = Astore->nnz;
	}

	/* Row and column scaling factors. */
	double *R = ReqPtr[d];
	double *C = CeqPtr[d];

	if ( !factored ) { /* Skip this if already factored. */
	    
	    int *perm_r = RpivPtr[d];

	    /* ------------------------------------------------------------
	       Find the row permutation for A.
	       ------------------------------------------------------------ */
	    if (options->RowPerm != NO)	{
		
		if (Fact != SamePattern_SameRowPerm) {
		    if (options->RowPerm == MY_PERMR) { /* Use user's perm_r. */
			/* Permute the matrix A for symbfact() */
			for (i = 0; i < colptr[n]; ++i) {
			    irow = rowind[i];
			    rowind[i] = perm_r[irow];
			}
		    } else if (options->RowPerm == LargeDiag_MC64) {
			/* Finds a row permutation (serial) */
			iinfo = zldperm_dist(job, m, nnz, colptr, rowind, a,
					     perm_r, R1, C1);

			if ( iinfo ) { /* Error */
			    printf(".. Matrix %d: LDPERM ERROR %d\n", d, iinfo);
			    if ( info==0 ) info = d+1 ;
			}
#if (PRNTlevel >= 2)
			dmin = damch_dist("Overflow");
			dsum = 0.0;
			dprod = 1.0;
#endif
			if (iinfo == 0)	{
			    if (job == 5) {
				if (Equil) {
				    for (i = 0; i < n; ++i) {
					R1[i] = exp(R1[i]);
					C1[i] = exp(C1[i]);
				    }

				    /* Scale the matrix further.
				       A <-- diag(R1)*A*diag(C1)            */
				    double cj;
				    for (j = 0; j < n; ++j) {
					cj = C1[j];
					for (i = colptr[j]; i < colptr[j + 1]; ++i) {
					    irow = rowind[i];
	                                    zd_mult(&a[i], &a[i], R1[irow]);
           				    zd_mult(&a[i], &a[i], cj);
					    
					}
				    }

				    /* Multiply together the scaling factors --
				       R/C from simple scheme, R1/C1 from MC64. */
				    if (rowequ)
					for (i = 0; i < m; ++i)	R[i] *= R1[i];
				    else
					for (i = 0; i < m; ++i) R[i] = R1[i];
				    if (colequ)
					for (i = 0; i < n; ++i)	C[i] *= C1[i];
				    else
					for (i = 0; i < n; ++i) C[i] = C1[i];

				    DiagScale[d] = BOTH;
				    rowequ = colequ = 1;

				} /* end if Equil */

				/* Now permute rows of A to prepare for symbfact() */
				for (j = 0; j < n; ++j)	{
				    for (i = colptr[j]; i < colptr[j + 1]; ++i) {
					irow = rowind[i];
					rowind[i] = perm_r[irow];
#if (PRNTlevel >= 2)
				        dprod *= slud_z_abs1(&a[i]);
#endif
				    }
				}
			    } else { /* job = 2,3,4 */
				for (j = 0; j < n; ++j)	{
				    for (i = colptr[j]; i < colptr[j + 1]; ++i)	{
					irow = rowind[i];
					rowind[i] = perm_r[irow];
					
#if (PRNTlevel >= 2)
					/* New diagonal */
					if (job == 2 || job == 3)
			                    dmin = SUPERLU_MIN(dmin, slud_z_abs1(&a[i]));
					else if (job == 4)
				            dsum += slud_z_abs1(&a[i]);
#endif					
				    } /* end for i ... */
				}  /* end for j ... */
			    }  /* end else job ... */
			    
			} else	{ /* if iinfo != 0 ... MC64 returns error */
			    for (i = 0; i < m; ++i) perm_r[i] = i;
			}
#if (PRNTlevel >= 2)
			if (job == 2 || job == 3) {
			    if (!iam) printf("\tsmallest diagonal %e\n", dmin);
			} else if (job == 4) {
			    if (!iam) printf("\tsum of diagonal %e\n", dsum);
			} else if (job == 5) {
n			    if (!iam) printf("\t product of diagonal %e\n", dprod);
			}
#endif
		    } else {
			printf(".. LDPERM invalid RowPerm option %d\n", options->RowPerm);
			info = -1;
			for (i = 0; i < m; ++i)	perm_r[i] = i;
		    } /* end if-else options->RowPerm ... */

#if (PRNTlevel >= 1)
		    printf(".. LDPERM job %d\n", (int) job);
		    fflush(stdout);
#endif
		} /* end if Fact not SamePattern_SameRowPerm ... */
		
	    } else { /* options->RowPerm == NOROWPERM / NATURAL */
		
		for (i = 0; i < m; ++i)	perm_r[i] = i;
	    }

#if ( DEBUGlevel>=1 )
	    check_perm_dist("perm_r", m, perm_r);
		PrintInt10("perm_r", m, perm_r);
#endif
	} /* end if (!factored) */

	if ( A[d]->Stype == SLU_NR ) {
	    SUPERLU_FREE(at);
	    SUPERLU_FREE(rowind);
	    SUPERLU_FREE(colptr);
	}
	
    } /* end for d ... batchCount */

    /* Deallocate storage */
    SUPERLU_FREE(A);
    if (job == 5) {
	SUPERLU_FREE(R1);
	SUPERLU_FREE(C1);
    }

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(0, "Exit zpivot_batch()");
#endif
    return info;
    
} /* end zpivot_batch */
