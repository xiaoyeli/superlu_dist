/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief
 *
 */
#include <stdio.h>
#include "superlu_sdefs.h"

#undef EXPAND_SYM

/*! brief
 *
 * <pre>
 * Output parameters
 * =================
 *   (nzval, rowind, colptr): (*rowind)[*] contains the row subscripts of
 *      nonzeros in columns of matrix A; (*nzval)[*] the numerical values;
 *	column i of A is given by (*nzval)[k], k = (*rowind)[i],...,
 *      (*rowind)[i+1]-1.
 * </pre>
 */

void
sreadtriple_noheader(FILE *fp, int_t *m, int_t *n, int_t *nonz,
	    float **nzval, int_t **rowind, int_t **colptr)
{
    int_t    i, j, k, jsize, lasta, nnz, nz, new_nonz, minn = 100;
    float *a, *val, vali;
    int_t    *asub, *xa, *row, *col;
    int      zero_base = 0, ret_val = 0;

    /* 	File format: Triplet in a line for each nonzero entry:
     *                 row    col    value
     *         or      row    col    real_part	imaginary_part
     */

    /* First pass: determine N and NNZ */
    nz = *n = 0;

#ifdef _LONGINT
    ret_val = fscanf(fp, "%lld%lld%f\n", &i, &j, &vali);
#else  // int
    ret_val = fscanf(fp, "%d%d%f\n", &i, &j, &vali);
#endif

    while (ret_val != EOF) {
	*n = SUPERLU_MAX(*n, i);
	*n = SUPERLU_MAX(*n, j);
	minn = SUPERLU_MIN(minn, i);
	minn = SUPERLU_MIN(minn, j);
	++nz;

#ifdef _LONGINT
    	ret_val = fscanf(fp, "%lld%lld%f\n", &i, &j, &vali);
#else  // int
        ret_val = fscanf(fp, "%d%d%f\n", &i, &j, &vali);
#endif
    }
    
    if ( minn == 0 ) { /* zero-based indexing */
	zero_base = 1;
	++(*n);
	printf("triplet file: row/col indices are zero-based.\n");
    } else {
	printf("triplet file: row/col indices are one-based.\n");
    }

    *m = *n;
    *nonz = nz;
    rewind(fp);

#ifdef EXPAND_SYM
    new_nonz = 2 * *nonz - *n;
#else
    new_nonz = *nonz;
#endif

    /* Second pass: read the actual matrix values */
    printf("m %ld, n %ld, nonz %ld\n", (long int) *m, (long int) *n, (long int) *nonz);
    sallocateA_dist(*n, new_nonz, nzval, rowind, colptr); /* Allocate storage */
    a    = *nzval;
    asub = *rowind;
    xa   = *colptr;

    if ( !(val = (float *) SUPERLU_MALLOC(new_nonz * sizeof(float))) )
        ABORT("Malloc fails for val[]");
    if ( !(row = (int_t *) SUPERLU_MALLOC(new_nonz * sizeof(int_t))) )
        ABORT("Malloc fails for row[]");
    if ( !(col = (int_t *) SUPERLU_MALLOC(new_nonz * sizeof(int_t))) )
        ABORT("Malloc fails for col[]");

    for (j = 0; j < *n; ++j) xa[j] = 0;

    /* Read into the triplet array from a file */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
#ifdef _LONGINT
	fscanf(fp, "%lld%lld%f\n", &row[nz], &col[nz], &val[nz]);
#else // int32
	fscanf(fp, "%d%d%f\n", &row[nz], &col[nz], &val[nz]);
#endif

	if ( !zero_base ) {
	    /* Change to 0-based indexing. */
	    --row[nz];
	    --col[nz];
	}

	if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
	    /*|| val[nz] == 0.*/) {
	    fprintf(stderr, "nz" IFMT ", (" IFMT ", " IFMT ") = %e out of bound, removed\n",
		    nz, row[nz], col[nz], val[nz]);
	    exit(-1);
	} else {
	    ++xa[col[nz]];
#ifdef EXPAND_SYM
	    if ( row[nz] != col[nz] ) { /* Excluding diagonal */
	      ++nz;
	      row[nz] = col[nz-1];
	      col[nz] = row[nz-1];
	      val[nz] = val[nz-1];
	      ++xa[col[nz]];
	    }
#endif
	    ++nz;
	}
    }

    *nonz = nz;
#ifdef EXPAND_SYM
    printf("new_nonz after symmetric expansion:\t%d\n", *nonz);
#endif


    /* Initialize the array of column pointers */
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
	k += jsize;
	jsize = xa[j];
	xa[j] = k;
    }

    /* Copy the triplets into the column oriented storage */
    for (nz = 0; nz < *nonz; ++nz) {
	j = col[nz];
	k = xa[j];
	asub[k] = row[nz];
	a[k] = val[nz];
	++xa[j];
    }

    /* Reset the column pointers to the beginning of each column */
    for (j = *n; j > 0; --j)
	xa[j] = xa[j-1];
    xa[0] = 0;

    SUPERLU_FREE(val);
    SUPERLU_FREE(row);
    SUPERLU_FREE(col);

#ifdef CHK_INPUT
    for (i = 0; i < *n; i++) {
	printf("Col %d, xa %d\n", i, xa[i]);
	for (k = xa[i]; k < xa[i+1]; k++)
	    printf("%d\t%16.10f\n", asub[k], a[k]);
    }
#endif

}

#if 0
void sreadrhs(int m, float *b)
{
    FILE *fp, *fopen();
    int i, j;

    if ( !(fp = fopen("b.dat", "r")) ) {
        fprintf(stderr, "zreadrhs: file does not exist\n");
	exit(-1);
    }
    for (i = 0; i < m; ++i)
      fscanf(fp, "%f\n", &b[i]);

    fclose(fp);
}
#endif

