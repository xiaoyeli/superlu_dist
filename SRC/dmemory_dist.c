/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Memory utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 * </pre>
 */

#include "superlu_ddefs.h"


/* Variables external to this file */
extern SuperLU_LU_stack_t stack;


void *duser_malloc_dist(int_t bytes, int_t which_end)
{
    void *buf;

    if ( SuperLU_StackFull(bytes) ) return (NULL);

    if ( which_end == HEAD ) {
	buf = (char*) stack.array + stack.top1;
	stack.top1 += bytes;
    } else {
	stack.top2 -= bytes;
	buf = (char*) stack.array + stack.top2;
    }

    stack.used += bytes;
    return buf;
}


void duser_free_dist(int_t bytes, int_t which_end)
{
    if ( which_end == HEAD ) {
	stack.top1 -= bytes;
    } else {
	stack.top2 += bytes;
    }
    stack.used -= bytes;
}



/*! \brief
 *
 * <pre>
 * mem_usage consists of the following fields:
 *    - for_lu (float)
 *      The amount of space used in bytes for the L\U data structures.
 *    - total (float)
 *      The amount of space needed in bytes to perform factorization.
 *    - expansions (int)
 *      Number of memory expansions during the LU factorization.
 * </pre>
 */
int_t dQuerySpace_dist(int_t n, LUstruct_t *LUstruct, gridinfo_t *grid,
		       SuperLUStat_t *stat, superlu_dist_mem_usage_t *mem_usage)
{
    register int_t dword, gb, iword, k, nb, nsupers;
    int_t *index, *xsup;
    int iam, mycol, myrow;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    LocalLU_t *Llu = LUstruct->Llu;

    iam = grid->iam;
    myrow = MYROW( iam, grid );
    mycol = MYCOL( iam, grid );
    iword = sizeof(int_t);
    dword = sizeof(double);
    nsupers = Glu_persist->supno[n-1] + 1;
    xsup = Glu_persist->xsup;
    mem_usage->for_lu = 0.;

    /* For L factor */
    nb = CEILING( nsupers, grid->npcol ); /* Number of local column blocks */
    for (k = 0; k < nb; ++k) {
	gb = k * grid->npcol + mycol; /* Global block number. */
	if ( gb < nsupers ) {
	    index = Llu->Lrowind_bc_ptr[k];
	    if ( index ) {
		mem_usage->for_lu += (float)
		    ((BC_HEADER + index[0]*LB_DESCRIPTOR + index[1]) * iword);
		mem_usage->for_lu += (float)(index[1]*SuperSize( gb )*dword);
	    }
	}
    }

    /* For U factor */
    nb = CEILING( nsupers, grid->nprow ); /* Number of local row blocks */
    for (k = 0; k < nb; ++k) {
	gb = k * grid->nprow + myrow; /* Global block number. */
	if ( gb < nsupers ) {
	    index = Llu->Ufstnz_br_ptr[k];
	    if ( index ) {
		mem_usage->for_lu += (float)(index[2] * iword);
		mem_usage->for_lu += (float)(index[1] * dword);
	    }
	}
    }

    /* Working storage to support factorization */
    mem_usage->total = mem_usage->for_lu;
#if 0
    mem_usage->total +=
	(float)(( Llu->bufmax[0] + Llu->bufmax[2] ) * iword +
		( Llu->bufmax[1] + Llu->bufmax[3] + maxsup ) * dword );
    /**** another buffer to use mpi_irecv in pdgstrf_irecv.c ****/
    mem_usage->total +=
	(float)( Llu->bufmax[0] * iword +  Llu->bufmax[1] * dword );
    mem_usage->total += (float)( maxsup * maxsup + maxsup) * iword;
    k = CEILING( nsupers, grid->nprow );
    mem_usage->total += (float)(2 * k * iword);
#else
    /*mem_usage->total += stat->current_buffer;*/
    mem_usage->total += stat->peak_buffer;

#if ( PRNTlevel>=1 )
    if (iam==0) printf(".. dQuerySpace: peak_buffer %.2f (MB)\n",
                       stat->peak_buffer * 1.0e-6);
#endif
#endif
    return 0;
} /* dQuerySpace_dist */


/*
 * Allocate storage for original matrix A
 */
void
dallocateA_dist(int_t n, int_t nnz, double **a, int_t **asub, int_t **xa)
{
    *a    = (double *) doubleMalloc_dist(nnz);
    *asub = (int_t *) intMalloc_dist(nnz);
    *xa   = (int_t *) intMalloc_dist(n+1);
}


double *doubleMalloc_dist(int_t n)
{
    double *buf;
    buf = (double *) SUPERLU_MALLOC( SUPERLU_MAX(1, n) * sizeof(double) );
    return (buf);
}

double *doubleCalloc_dist(int_t n)
{
    double *buf;
    register int_t i;
    double zero = 0.0;
    buf = (double *) SUPERLU_MALLOC( SUPERLU_MAX(1, n) * sizeof(double));
    if ( !buf ) return (buf);
    for (i = 0; i < n; ++i) buf[i] = zero;
    return (buf);
}

