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

#include "superlu_dist_config.h"
#include "superlu_ddefs.h"

#ifdef HAVE_COLAMD
#include "colamd.h"
#endif

/*! \brief Gets sparsity permutations for a batch of matrices
 * <pre>
 *
 * @param[in]     options solver options
 * @param[in]     batchCount number of matrices in the batch
 * @param[in]     SparseMatrix_handles pointers to the matrices in the batch, each pointing to the actual stoage in CSC format
 *     On entry, the original matrices, may be overwritten by A1 <- Pr*diag(R)*A*diag(C) from dequil_batch() and dpivot_batch()
 * @param[out]    CpivPtr pointers to column permutation vectors for each matrix, each of size n
 *
 * </pre>
 */
void
get_perm_c_batch(
	superlu_dist_options_t *options, /* options for algorithm choices and algorithm parameters */
	int batchCount, /* number of matrices in the batch */
	handle_t  *SparseMatrix_handles, /* array of sparse matrix handles,
					  * of size 'batchCount',
					  * each pointing to the actual storage
					  */
	int **CpivPtr /* array of pointers to column permutation vectors , each of size N */
		 )
{
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Enter get_perm_c_batch()");
#endif

    /* Decipher the input matrices */
    SuperMatrix **A;
    A = SUPERLU_MALLOC(batchCount * sizeof(SuperMatrix *));
    for (int d = 0; d < batchCount; ++d) {
	A[d] = (SuperMatrix *) SparseMatrix_handles[d];
    }

    int_t delta, maxint;
    int_t *dhead, *qsize, *llist, *marker, *invp;
    int m = A[0]->nrow;
    int n = A[0]->ncol;

    if ( options->ColPerm == MMD_AT_PLUS_A || options->ColPerm == MMD_ATA ) {

	/* These arrays can be reused by multiple matrices */
	
	delta = 0; /* DELTA is a parameter to allow the choice of nodes
		      whose degree <= min-degree + DELTA. */
	maxint = 2147483647; /* 2**31 - 1 */
	invp = (int_t *) SUPERLU_MALLOC((n+delta)*sizeof(int_t));
	if ( !invp ) ABORT("SUPERLU_MALLOC fails for invp.");
	dhead = (int_t *) SUPERLU_MALLOC((n+delta)*sizeof(int_t));
	if ( !dhead ) ABORT("SUPERLU_MALLOC fails for dhead.");
	qsize = (int_t *) SUPERLU_MALLOC((n+delta)*sizeof(int_t));
	if ( !qsize ) ABORT("SUPERLU_MALLOC fails for qsize.");
	llist = (int_t *) SUPERLU_MALLOC(n*sizeof(int_t));
	if ( !llist ) ABORT("SUPERLU_MALLOC fails for llist.");
	marker = (int_t *) SUPERLU_MALLOC(n*sizeof(int_t));
	if ( !marker ) ABORT("SUPERLU_MALLOC fails for marker.");
    }
	
    int_t bnz;
    int_t *b_colptr, *b_rowind;  /* allocated in at_plus_a() or getata() */
    int_t nofsub;
    int i, j;
    double t;
	
    /* Loop through each matrix in the batch */
    for (int d = 0; d < batchCount; ++d) {
	
	NCformat *Astore = (NCformat *) A[d]->Store;
	int *perm_c = CpivPtr[d];

	t = SuperLU_timer_();
	bnz = 0;
	
	switch ( options->ColPerm ) {
	    
            case NATURAL: /* Natural ordering */
		for (i = 0; i < n; ++i) perm_c[i] = i;
		break;

            case MMD_AT_PLUS_A: /* Minimum degree ordering on A'+A */
		if ( m != n ) ABORT("Matrix is not square");
		at_plus_a_dist(n, Astore->nnz, Astore->colptr, Astore->rowind,
			       &bnz, &b_colptr, &b_rowind);
		t = SuperLU_timer_() - t;
		/*printf("Form A'+A time = %8.3f\n", t);*/
		
		break;

            case MMD_ATA: /* Minimum degree ordering on A'*A */
		getata_dist(m, n, Astore->nnz, Astore->colptr, Astore->rowind,
			    &bnz, &b_colptr, &b_rowind);
		t = SuperLU_timer_() - t;
		/*printf("Form A'*A time = %8.3f\n", t);*/
		
		break;

            case (COLAMD): /* Approximate minimum degree column ordering. */
		get_colamd_dist(m, n, Astore->nnz, Astore->colptr, Astore->rowind,
				perm_c);
		break;
#ifdef HAVE_PARMETIS
            case METIS_AT_PLUS_A: /* METIS ordering on A'+A */
		if ( m != n ) ABORT("Matrix is not square");
		at_plus_a_dist(n, Astore->nnz, Astore->colptr, Astore->rowind,
			       &bnz, &b_colptr, &b_rowind);

		if ( bnz ) { /* non-empty adjacency structure */
		    get_metis_dist(n, bnz, b_colptr, b_rowind, perm_c);
		} else { /* e.g., diagonal matrix */
		    for (i = 0; i < n; ++i) perm_c[i] = i;
		    SUPERLU_FREE(b_colptr);
		    /* b_rowind is not allocated in this case */
		}
		break;
#endif		
            default:
		ABORT("Invalid options->Colperm");
		
	} /* end switch (options->Colperm) */

	if ( options->ColPerm == MMD_AT_PLUS_A || options->ColPerm == MMD_ATA ) {
	    if ( bnz ) {
		t = SuperLU_timer_();

		/* Initialize and allocate storage for GENMMD. */
		delta = 0; /* DELTA is a parameter to allow the choice of nodes
			      whose degree <= min-degree + DELTA. */
		/* Transform adjacency list into 1-based indexing required by GENMMD.*/
		for (i = 0; i <= n; ++i) ++b_colptr[i];
		for (i = 0; i < bnz; ++i) ++b_rowind[i];

		int_t ln = n;
		genmmd_dist_(&ln, b_colptr, b_rowind, perm_c, invp, &delta, dhead, 
			     qsize, llist, marker, &maxint, &nofsub);

		/* Transform perm_c into 0-based indexing. */
		for (i = 0; i < n; ++i) --perm_c[i];

		t = SuperLU_timer_() - t;
		/*    printf("call GENMMD time = %8.3f\n", t);*/

		SUPERLU_FREE(b_colptr);  /* TODO: repeated malloc/free for the batch */
		SUPERLU_FREE(b_rowind);
		//printf("\tafter free b_rowind bnz %lld\n", (long long)bnz);
		
	    } else { /* Empty adjacency structure */
		for (i = 0; i < n; ++i) perm_c[i] = i;
	    }
	} /* end if MMD */

#if ( DEBUGlevel>=1 )
	check_perm_dist("perm_c", n, perm_c);
#endif
    } /* end for d = ... batchCount */
    
    if ( options->ColPerm == MMD_AT_PLUS_A || options->ColPerm == MMD_ATA ) {
	SUPERLU_FREE(invp);
	SUPERLU_FREE(dhead);
	SUPERLU_FREE(qsize);
	SUPERLU_FREE(llist);
	SUPERLU_FREE(marker);
    }

    SUPERLU_FREE(A);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Exit get_perm_c_batch()");
#endif

    return;
    
} /* end get_perm_c_batch */
