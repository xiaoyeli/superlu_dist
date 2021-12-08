/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Performs sparse matrix-vector multiplication
 *
 * <pre>
 * -- Distributed SuperLU routine (version 1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 * </pre>
 */

#include <math.h>
#include "superlu_sdefs.h"


static void screate_msr_matrix(SuperMatrix *, int_t [], int_t,
			      float **, int_t **);
static void sPrintMSRmatrix(int, float [], int_t [], gridinfo_t *);


int psgsmv_AXglobal_setup
(
 SuperMatrix *A,       /* Matrix A permuted by columns (input).
			  The type of A can be:
			  Stype = SLU_NCP; Dtype = SLU_S; Mtype = SLU_GE. */
 Glu_persist_t *Glu_persist, /* input */
 gridinfo_t *grid,     /* input */
 int_t *m,             /* output */
 int_t *update[],      /* output */
 float *val[],        /* output */
 int_t *bindx[],       /* output */
 int_t *mv_sup_to_proc /* output */
 )
{
    int n;
    int input_option;
    int N_update;    /* Number of variables updated on this process (output) */
    int iam = grid->iam;
    int nprocs = grid->nprow * grid->npcol;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    int_t nsupers;
    int i, nsup, p, t1, t2, t3;


    /* Initialize the list of global indices.
     * NOTE: the list of global indices must be in ascending order.
     */
    n = A->nrow;
    input_option = SUPER_LINEAR;
    nsupers = supno[n-1] + 1;

#if ( DEBUGlevel>=2 )
    if ( !iam ) {
	PrintInt10("xsup", supno[n-1]+1, xsup);
	PrintInt10("supno", n, supno);
    }
#endif

    if ( input_option == SUPER_LINEAR ) { /* Block partitioning based on
					     individual rows.  */
	/* Figure out mv_sup_to_proc[] on all processes. */
	for (p = 0; p < nprocs; ++p) {
	    t1 = n / nprocs;       /* Number of rows */
	    t2 = n - t1 * nprocs;  /* left-over, which will be assigned
				      to the first t2 processes.  */
	    if ( p >= t2 ) t2 += (p * t1); /* Starting row number */
	    else { /* First t2 processes will get one more row. */
 	        ++t1;              /* Number of rows. */
		t2 = p * t1;       /* Starting row. */
	    }
	    /* Make sure the starting and ending rows are at the
	       supernode boundaries. */
	    t3 = t2 + t1;      /* Ending row. */
	    nsup = supno[t2];
	    if ( t2 > xsup[nsup] ) { /* Round up the starting row. */
		t1 -= xsup[nsup+1] - t2;
		t2 = xsup[nsup+1];
	    }
	    nsup = supno[t3];
	    if ( t3 > xsup[nsup] ) /* Round up the ending row. */
		t1 += xsup[nsup+1] - t3;
	    t3 = t2 + t1 - 1;
	    if ( t1 ) {
		for (i = supno[t2]; i <= supno[t3]; ++i) {
		    mv_sup_to_proc[i] = p;
#if ( DEBUGlevel>=3 )
		    if ( mv_sup_to_proc[i] == p-1 ) {
			fprintf(stderr,
				"mv_sup_to_proc conflicts at supno %d\n", i);
			exit(-1);
		    }
#endif
		}
	    }

	    if ( iam == p ) {
		N_update = t1;
		if ( N_update ) {
		    if ( !(*update = intMalloc_dist(N_update)) )
			ABORT("Malloc fails for update[]");
		}
		for (i = 0; i < N_update; ++i) (*update)[i] = t2 + i;
#if ( DEBUGlevel>=3 )
		printf("(%2d) N_update = %4d\t"
		       "supers %4d to %4d\trows %4d to %4d\n",
		       iam, N_update, supno[t2], supno[t3], t2, t3);
#endif
	    }
	} /* for p ... */
    } else if ( input_option == SUPER_BLOCK ) { /* Block partitioning based on
						   individual supernodes.  */
	/* This may cause bad load balance, because the blocks are usually
	   small in the beginning and large toward the end.   */
	t1 = nsupers / nprocs;
	t2 = nsupers - t1 * nprocs; /* left-over */
	if ( iam >= t2 ) t2 += (iam * t1);
	else {
	    ++t1;          /* Number of blocks. */
	    t2 = iam * t1; /* Starting block. */
	}
	N_update = xsup[t2+t1] - xsup[t2];
	if ( !(*update = intMalloc_dist(N_update)) )
	    ABORT("Malloc fails for update[]");
	for (i = 0; i < N_update; ++i) (*update)[i] = xsup[t2] + i;
    }


    /* Create an MSR matrix in val/bindx to be used by pdgsmv(). */
    screate_msr_matrix(A, *update, N_update, val, bindx);

#if ( DEBUGlevel>=2 )
    PrintInt10("mv_sup_to_proc", nsupers, mv_sup_to_proc);
    sPrintMSRmatrix(N_update, *val, *bindx, grid);
#endif

    *m = N_update;
    return 0;
} /* PSGSMV_AXglobal_SETUP */


/*! \brief
 *
 * <pre>
 * Create the distributed modified sparse row (MSR) matrix: bindx/val.
 * For a submatrix of size m-by-n, the MSR arrays are as follows:
 *    bindx[0]      = m + 1
 *    bindx[0..m]   = pointer to start of each row
 *    bindx[ks..ke] = column indices of the off-diagonal nonzeros in row k,
 *                    where, ks = bindx[k], ke = bindx[k+1]-1
 *    val[k]        = A(k,k), k < m, diagonal elements
 *    val[m]        = not used
 *    val[ki]       = A(k, bindx[ki]), where ks <= ki <= ke
 * Both arrays are of length nnz + 1.
 * </pre>
*/
static void screate_msr_matrix
(
 SuperMatrix *A,       /* Matrix A permuted by columns (input).
			  The type of A can be:
			  Stype = SLU_NCP; Dtype = SLU_S; Mtype = SLU_GE. */
 int_t update[],       /* input (local) */
 int_t N_update,       /* input (local) */
 float **val,         /* output */
 int_t **bindx         /* output */
)
{
    int hi, i, irow, j, k, lo, n, nnz_local, nnz_diag;
    NCPformat *Astore;
    float *nzval;
    int_t *rowcnt;
    double zero = 0.0;

    if ( !N_update ) return;

    n = A->ncol;
    Astore = A->Store;
    nzval = Astore->nzval;

    /* One pass of original matrix A to count nonzeros of each row. */
    if ( !(rowcnt = (int_t *) intCalloc_dist(N_update)) )
	ABORT("Malloc fails for rowcnt[]");
    lo = update[0];
    hi = update[N_update-1];
    nnz_local = 0;
    nnz_diag = 0;
    for (j = 0; j < n; ++j) {
	for (i = Astore->colbeg[j]; i < Astore->colend[j]; ++i) {
	    irow = Astore->rowind[i];
	    if ( irow >= lo && irow <= hi ) {
		if ( irow != j ) /* Exclude diagonal */
		    ++rowcnt[irow - lo];
		else ++nnz_diag; /* Count nonzero diagonal entries */
		++nnz_local;
	    }
	}
    }

    /* Add room for the logical diagonal zeros which are not counted
       in nnz_local. */
    nnz_local += (N_update - nnz_diag);

    /* Allocate storage for bindx[] and val[]. */
    if ( !(*val = (float *) floatMalloc_dist(nnz_local+1)) )
	ABORT("Malloc fails for val[]");
    for (i = 0; i < N_update; ++i) (*val)[i] = zero; /* Initialize diagonal */
    if ( !(*bindx = (int_t *) SUPERLU_MALLOC((nnz_local+1) * sizeof(int_t))) )
	ABORT("Malloc fails for bindx[]");

    /* Set up row pointers. */
    (*bindx)[0] = N_update + 1;
    for (j = 1; j <= N_update; ++j) {
	(*bindx)[j] = (*bindx)[j-1] + rowcnt[j-1];
	rowcnt[j-1] = (*bindx)[j-1];
    }

    /* One pass of original matrix A to fill in matrix entries. */
    for (j = 0; j < n; ++j) {
	for (i = Astore->colbeg[j]; i < Astore->colend[j]; ++i) {
	    irow = Astore->rowind[i];
	    if ( irow >= lo && irow <= hi ) {
		if ( irow == j ) /* Diagonal */
		    (*val)[irow - lo] = nzval[i];
		else {
		    irow -= lo;
		    k = rowcnt[irow];
		    (*bindx)[k] = j;
		    (*val)[k] = nzval[i];
		    ++rowcnt[irow];
		}
	    }
	}
    }

    SUPERLU_FREE(rowcnt);
}

/*! \brief
 *
 * <pre>
 * Performs sparse matrix-vector multiplication.
 *   - val/bindx stores the distributed MSR matrix A
 *   - X is global
 *   - ax product is distributed the same way as A
 * </pre>
 */
int
psgsmv_AXglobal(int_t m, int_t update[], float val[], int_t bindx[],
                float X[], float ax[])
{
    int_t i, j, k;

    if ( m <= 0 ) return 0; /* number of rows (local) */

    for (i = 0; i < m; ++i) {
	ax[i] = 0.0;

	for (k = bindx[i]; k < bindx[i+1]; ++k) {
	    j = bindx[k];       /* column index */
	    ax[i] += val[k] * X[j];
	}
	ax[i] += val[i] * X[update[i]]; /* diagonal */
    }
    return 0;
} /* PSGSMV_AXglobal */

/*
 * Performs sparse matrix-vector multiplication.
 *   - val/bindx stores the distributed MSR matrix A
 *   - X is global
 *   - ax product is distributed the same way as A
 */
int
psgsmv_AXglobal_abs(int_t m, int_t update[], float val[], int_t bindx[],
	            float X[], float ax[])
{
    int_t i, j, k;

    if ( m <= 0 ) return 0; /* number of rows (local) */

    for (i = 0; i < m; ++i) {
	ax[i] = 0.0;
	for (k = bindx[i]; k < bindx[i+1]; ++k) {
	    j = bindx[k];       /* column index */
	    ax[i] += fabs(val[k]) * fabs(X[j]);
	}
	ax[i] += fabs(val[i]) * fabs(X[update[i]]); /* diagonal */
    }

    return 0;
} /* PSGSMV_AXglobal_ABS */

/*
 * Print the local MSR matrix
 */
static void sPrintMSRmatrix
(
 int m,       /* Number of rows of the submatrix. */
 float val[],
 int_t bindx[],
 gridinfo_t *grid
)
{
    int iam, nnzp1;

    if ( !m ) return;

    iam = grid->iam;
    nnzp1 = bindx[m];
    printf("(%2d) MSR submatrix has %d rows -->\n", iam, m);
    Printfloat5("val", nnzp1, val);
    PrintInt10("bindx", nnzp1, bindx);
}
