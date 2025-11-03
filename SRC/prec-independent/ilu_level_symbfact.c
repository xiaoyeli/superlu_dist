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
 int       *perm_c, /* column permutation vector (input) */
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
    
    /* ---- Set up supernode partition ---- */
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
    int_t k;
    int_t nextl = 0, nextu = 0; // counts of size of skeleton graphs of L & U
    int_t nnzL = 0, nnzU = 0;   // counts of actual nonzeros in L & U
    
    nsuper = (Glu_persist->supno)[n-1];


    /* ------ First pass through A: Count nonzero indices in L & U ------ */
    /* Diagonal block of a supernode is stored in L.
       U count is an upper bound of the skeleton graph.
       L count is the exact supernodal skeleton graph.
     */
    for (i = 0; i <= nsuper; ++i) { /* loop through each supernode */
	fsupc = xsup[i];
	xlsub[fsupc] = nextl;
	for (j = fsupc; j < xsup[i+1]; ++j) { // loop through columns in supernode i
	    for (k = colbeg[j]; k < colend[j]; ++k) { // loop through column j
		irow = rowind[k];
		if ( irow < fsupc ) ++nextu; // in U
	    }
	}
	
	/* For L: only check first column of a supernode;
	   similar to fixupL_dist()
	*/
	for (k = colbeg[fsupc]; k < colend[fsupc]; ++k) {
	    irow = rowind[k];
	    if ( irow >= fsupc ) ++nextl; // in L, including diagonal block
	}
	for (j = fsupc+1; j < xsup[i+1]; ++j) // remaining columns in supernode i 
	    xlsub[j] = nextl;
	nnzL += (nextl-xlsub[fsupc]) * (xsup[i+1]-fsupc); // exact 
    }

    Glu_freeable->lsub = (int_t *) intMalloc_dist(nextl); //intMalloc_dist(nnzL);
    Glu_freeable->usub = (int_t *) intMalloc_dist(nextu); //intMalloc_dist(nnzU);
    int_t *lsub = Glu_freeable->lsub;
    int_t *usub = Glu_freeable->usub;

    /* ------ Second pass through A: populate indices ------ */
    /* data structures for skeleton graph of U */
    int krep, nseg, *segrep, *repfnz;
    segrep = (int *) int32Malloc_dist(2*m);
    repfnz = segrep + m;
    ifill_dist(repfnz, m, SLU_EMPTY);
    xusub[0] = 0;
    nextl = nextu = 0;

    for (i = 0; i <= nsuper; ++i) { /* loop through each supernode */
	fsupc = xsup[i];
	xlsub[fsupc] = nextl;

	/* Sherry fix (11/1/2025)
	   U segments must be stored in topological order, and only store
	   the index of the first nonzero in the segment; 
	   similar to set_usub() in symbact()
	*/
	for (j = fsupc; j < xsup[i+1]; ++j) { // loop through columns in supernode i
	    nseg = 0;
	    
	    for (k = colbeg[j]; k < colend[j]; ++k) { // loop through column j
		irow = rowind[k];
		if ( irow < fsupc ) { // in U
		    krep = xsup[supno[irow]+1] - 1; // last row index of supernode containing irow
		    if ( repfnz[krep] != SLU_EMPTY ) { /* krep was visited before */
			if ( irow < repfnz[krep] ) repfnz[krep] = irow;
		    } else {
			repfnz[krep] = irow;
			segrep[nseg] = krep;
			++nseg;
		    }
		}
	    }

	    /* Set up the first nonzero index of each segment.
	       Reset repfnz[*] to prepare for the next column.
	    */
	    for (k = 0; k < nseg; k++) {
		krep = segrep[k];
		usub[nextu + k] = repfnz[krep]; // first nonzero
		nnzU += (krep - repfnz[krep] + 1);
		repfnz[krep] = SLU_EMPTY;
	    }
	    
	    nextu += nseg;
	    xusub[j+1] = nextu;
	    
	    /* The U-segments needs to be in topological order. Sorting is the simplest */
	    isort1( xusub[j+1] - xusub[j], &usub[xusub[j]] );
	    
	} /* end for j ... columns in supernode i */
	
	/* YL: Assign usub & lsub */
	/* Only check first column of supernode in L;
	   similar to fixupL_dist()
	*/
	for (k = colbeg[fsupc]; k < colend[fsupc]; ++k) {
	    irow = rowind[k];
	    if ( irow >= fsupc ) { // in L, including diagonal block
		lsub[nextl] = irow;
		++nextl;
	    }
	}

	/* The indices of the Diagonal block needs to be at the beginning
	   and sorted. Sorting the whole list is simpler, although not necessary.
	*/
	isort1( xlsub[fsupc+1] - xlsub[fsupc], &lsub[xlsub[fsupc]] );
	
    } /* end for i ... supernodes */
    
    xlsub[n] = nextl;

    /* -------------------------------- */
    
#if ( PRNTlevel>=1 )
    printf(".... nnzL %lld, nnzU %lld, nextl %lld, nextu %lld, nsuper %d\n",
	   (long long) nnzL, (long long) nnzU, (long long) nextl, (long long) nextu, nsuper);
    fflush(stdout);
#endif    

    assert (nnzU >= nextu && nnzL >= nextl );

    Glu_freeable->nnzLU = nnzU + nnzL;
    Glu_freeable->nzlmax = nnzL;
    Glu_freeable->nzumax = nnzU;
    Glu_freeable->nnzLU = nnzL + nnzU - min_mn;	

#if ( DEBUGlevel>=1 )
    PrintInt32("lsub", Glu_freeable->xlsub[n], Glu_freeable->lsub);
    PrintInt32("xlsub", n+1, Glu_freeable->xlsub);
    PrintInt32("usub", Glu_freeable->xusub[n], Glu_freeable->usub);
    PrintInt32("xusub", n+1, Glu_freeable->xusub);
    PrintInt32("supno", n, Glu_persist->supno);
    PrintInt32("xsup", n+1, Glu_persist->xsup);
#endif

    SUPERLU_FREE(segrep);
    
#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Exit ilu_level_symbfact()");
#endif

    return ( -(Glu_freeable->xlsub[n] + Glu_freeable->xusub[n]) );
}

