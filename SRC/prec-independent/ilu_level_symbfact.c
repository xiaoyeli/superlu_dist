/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Performs a level-based ILU symbolic factorization
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * May 20, 2024
 * 
 * Modified:
 */

#include "superlu_defs.h"

/**
 * @brief This function performs the level-based ILU symbolic factorization on
 * matrix Pc*Pr*A*Pc' and sets up the nonzero data structures for L & U matrices.
 * In the process, the matrix is also ordered and its memory usage information is gathered.
 * 
 * @param options Input parameters to control how the ILU decomposition will be performed.
 * @param A Pointer to the global supermatrix A, permuted by columns in NCPfomat.
 * @param perm_c The column permutation vector.
 * @param Glu_persist Pointer to the structure which tracks the symbolic factorization information.
 * @param Glu_freeable Pointer to the structure which tracks the space used to store L/U data structures.
 * @param stat Information on program execution.
 */

/* 
 * Return value
 * ============
 *   < 0, number of bytes needed for LSUB and USUB.
 *   = 0, Error condition
 */
int_t ilu_level_symbfact
/************************************************************************/
(
 superlu_dist_options_t *options, /* input options */
 SuperMatrix *A,      /* original matrix A permuted by columns, GAC from caller (input) */
 int_t       *perm_c, /* column permutation vector (input) */
 int_t       *etree,  /* column elimination tree (input) */
 Glu_persist_t *Glu_persist,  /* output */
 Glu_freeable_t *Glu_freeable /* output */
 )
{
    
    if ( options->ILU_level != 0 ) {
	printf("ERROR: ILU(k>0) is not implemented yet\n");
	return (0);
    }

    /* Now, do ILU(0) */

    int_t iinfo;
    int i, n = A->ncol, m=A->nrow;
    double t;
    int min_mn = SUPERLU_MIN(m, n);

    /* Set up supernode partition */
    Glu_persist->supno = (int_t *) SUPERLU_MALLOC(n * sizeof(int_t));
    Glu_persist->xsup = (int_t *) SUPERLU_MALLOC((n+1) * sizeof(int_t));
    for (i = 0; i < n; i++) {
	Glu_persist->supno[i] = i;
	Glu_persist->xsup[i] = i;
    }
    Glu_persist->xsup[n] = n;


    NCPformat *GACstore = A->Store;
    int_t *colbeg, *colend, *rowind, irow;
    rowind = GACstore->rowind;
    colbeg = GACstore->colbeg;
    colend = GACstore->colend;

    Glu_freeable->xlsub = (int_t *) intCalloc_dist(n+1);
    Glu_freeable->xusub = (int_t *) intCalloc_dist(n+1);
    int_t nnzL = 0, nnzU = 0;
    
    /* Count nonzeros per column for L & U */
    for (int j = 0; j < n; ++j) {
#if 0	
	fscanf(fpU, "%d %d", &col_num, &Glu_freeable.usub[i]);
	Glu_freeable.xusub[col_num] += 1;
#endif
	for (i = colbeg[j]; i < colend[j]; ++i) { // (j,j) is diagonal
	    irow = rowind[i];
	    if ( irow < j ) { // in U
		nnzU++;
		Glu_freeable->xusub[j+1] += 1;
	    } else { // in L, including diagonal of U
		nnzL++;
		Glu_freeable->xlsub[j+1] += 1;
	    }
	}
    }

    Glu_freeable->nnzLU = nnzU + nnzL;
    Glu_freeable->lsub = (int_t *) SUPERLU_MALLOC(nnzL * sizeof(int_t));
    Glu_freeable->usub = (int_t *) SUPERLU_MALLOC(nnzU * sizeof(int_t));
    Glu_freeable->nzlmax = nnzL;
    Glu_freeable->nzumax = nnzU;
    Glu_freeable->nnzLU = nnzL + nnzU - min_mn;	
    
    /* YL: Assign lsub & usub */
    nnzL=0;
    nnzU=0;
    for (int j = 0; j < n; ++j) {
	for (i = colbeg[j]; i < colend[j]; ++i) { // (j,j) is diagonal
	    irow = rowind[i];
	    //if(j==0){
	    //printf("irow %5d \n",irow);
	    //}
	    if ( irow < j ) { // in U
		Glu_freeable->usub[nnzU] = irow;
		nnzU++;
	    } else { // in L, including diagonal of U
		// printf("%5d %5d\n",j,irow);
		Glu_freeable->lsub[nnzL] = irow;
		nnzL++;
	    }
	}
    }

    /* Do prefix sum to set up column pointers */
    for(i = 1; i <= n; i++) {
	Glu_freeable->xlsub[i] += Glu_freeable->xlsub[i-1];	
	Glu_freeable->xusub[i] += Glu_freeable->xusub[i-1];
    }

    return ( -(Glu_freeable->xlsub[n] + Glu_freeable->xusub[n]) );
}

