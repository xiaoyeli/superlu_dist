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
n * @param stat Information on program execution.
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
    int iam = 0;
    
#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Enter ilu_level_symbfact()");
#endif
    
    /* Now, do ILU(0) */

    int_t iinfo;
    int n = A->ncol, m = A->nrow;
    int nsuper, i, j, fsupc;
    double t;
    int min_mn = SUPERLU_MIN(m, n), irow;


    NCPformat *GACstore = A->Store;
    int_t *colbeg, *colend, *rowind;
    rowind = GACstore->rowind;
    colbeg = GACstore->colbeg;
    colend = GACstore->colend;

    Glu_freeable->xlsub = (int_t *) intCalloc_dist(n+1);
    Glu_freeable->xusub = (int_t *) intCalloc_dist(n+1);
    int_t nnzL = 0, nnzU = 0;
    
    /* Set up supernode partition */
    if ( options->UserDefineSupernode == NO ) {
	Glu_persist->supno = (int_t *) SUPERLU_MALLOC(n * sizeof(int_t));
	Glu_persist->xsup = (int_t *) SUPERLU_MALLOC((n+1) * sizeof(int_t));
	for (i = 0; i < n; i++) { /* set up trivial supernode for now. */
	    Glu_persist->supno[i] = i;
	    Glu_persist->xsup[i] = i;
	}
	Glu_persist->xsup[n] = n;
    } else {
	/* User alreay allocated and defined the above supernode partition */
    }

    /* Sherry: fix to accommodate supernodes (11/23/24)
       Need to add an outer loop for supernodes 
    */
    int_t *xusub = Glu_freeable->xusub;
    int_t *xlsub = Glu_freeable->xlsub;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;
    int_t k, nextl = 0;
    nsuper = (Glu_persist->supno)[n-1];
    
    /* Count nonzeros per column for L & U;
       Diagonal block of a supernode is stored in L
    */
    for (i = 0; i <= nsuper; ++i) { /* loop through each supernode */
	fsupc = xsup[i];
	xlsub[fsupc] = nextl;
	for (j = fsupc; j < xsup[i+1]; ++j) { // loop through columns in supernode i 
	    for (k = colbeg[j]; k < colend[j]; ++k) {
		irow = rowind[k];
		if ( irow < fsupc ) { // in U
		    nnzU++;
		    xusub[j+1] += 1;
		}
	    }
	}
	
	/* only check first column of supernode in L;
	   similar to fixupL_dist()
	*/
	for (k = colbeg[fsupc]; k < colend[fsupc]; ++k) {
	    irow = rowind[k];
	    if ( irow >= fsupc ) ++nextl; // in L, including diagonal block
	}
	for (j = fsupc+1; j < xsup[i+1]; ++j) // loop through columns in supernode i 
	    xlsub[j] = nextl;
	nnzL += (nextl - xlsub[fsupc]) * (xsup[i+1]-fsupc);
    }
    xlsub[n] = nextl;

#if ( PRNTlevel>=1 )    
    printf(".... nnzL %d, nnzU %d, nextl %d, nsuper %d\n", nnzL, nnzU, nextl, nsuper);
    fflush(stdout);
#endif    
    
    /* Do prefix sum to set up column pointers */
    for(i = 1; i <= n; i++) {
	//Glu_freeable->xlsub[i] += Glu_freeable->xlsub[i-1];	
	Glu_freeable->xusub[i] += Glu_freeable->xusub[i-1];
    }
    
    Glu_freeable->nnzLU = nnzU + nnzL;
    Glu_freeable->nzlmax = nnzL;
    Glu_freeable->nzumax = nnzU;
    Glu_freeable->nnzLU = nnzL + nnzU - min_mn;	
    Glu_freeable->lsub = (int_t *) intMalloc_dist(nnzL);
    Glu_freeable->usub = (int_t *) intMalloc_dist(nnzU); 

    /* YL: Assign usub & lsub */
    nnzU = 0;
    for (j = 0; j < n; ++j) { // loop through each column 
	fsupc = xsup[supno[j]];
	for (i = colbeg[j]; i < colend[j]; ++i) {
	    irow = rowind[i];
	    //if(j==0){
	    //printf("irow %5d \n",irow);
	    //}
	    if ( irow < fsupc ) { // in U
		Glu_freeable->usub[nnzU++] = irow;
	    }
	}
    }
    
    /* only check first column of supernode in L;
       similar to fixupL_dist()
    */
    nnzL = 0;
    for (i = 0; i <= nsuper; ++i) { // loop through each supernode
	fsupc = xsup[i];
	for (k = colbeg[fsupc]; k < colend[fsupc]; ++k) {
	    irow = rowind[k];
	    if ( irow >= fsupc ) { // in L, including diagonal block
		Glu_freeable->lsub[nnzL++] = irow;
	    }
	}
    }

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Exit ilu_level_symbfact()");
#endif

    return ( -(Glu_freeable->xlsub[n] + Glu_freeable->xusub[n]) );
}

