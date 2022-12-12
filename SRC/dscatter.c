/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Scatter the computed blocks into LU destination.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 *
 * Modified:
 *   September 18, 2017, enable SIMD vectorized scatter operation.
 *
 */
#include <math.h>
#include "superlu_ddefs.h"

void
dscatter_l_1 (int ib,
           int ljb,
           int nsupc,
           int_t iukp,
           int_t* xsup,
           int klst,
           int nbrow,
           int_t lptr,
           int temp_nbrow,
           int * usub,
           int * lsub,
           double *tempv,
           int * indirect_thread,
           int_t ** Lrowind_bc_ptr, double **Lnzval_bc_ptr,
	   gridinfo_t * grid)
{
    // TAU_STATIC_TIMER_START("SCATTER_LB");
    // printf("hello\n");
    int_t rel, i, segsize, jj;
    double *nzval;
    int_t *index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the dest lusup. */
    int_t lptrj = BC_HEADER;
    int_t luptrj = 0;
    int_t ijb = index[lptrj];
    while (ijb != ib)
    {
        /* Search for dest block --
           blocks are not ordered! */
        luptrj += index[lptrj + 1];
        lptrj += LB_DESCRIPTOR + index[lptrj + 1];

        ijb = index[lptrj];
    }
    /*
     * Build indirect table. This is needed because the
     * indices are not sorted for the L blocks.
     */
    int_t fnz = FstBlockC (ib);
    lptrj += LB_DESCRIPTOR;
    for (i = 0; i < index[lptrj - 1]; ++i)
    {
        rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;

    }

    nzval = Lnzval_bc_ptr[ljb] + luptrj;
    // tempv =bigV + (cum_nrow + cum_ncol*nbrow);
    for (jj = 0; jj < nsupc; ++jj)
    {
        segsize = klst - usub[iukp + jj];
        // printf("segsize %d \n",segsize);
        if (segsize) {
            /*#pragma _CRI cache_bypass nzval,tempv */
            for (i = 0; i < temp_nbrow; ++i) {
                rel = lsub[lptr + i] - fnz;
                nzval[indirect_thread[rel]] -= tempv[i];
                // printf("i (src) %d, perm (dest) %d  \n",i,indirect_thread[rel]);
#ifdef PI_DEBUG
                double zz = 0.0;
                // if(!(*(long*)&zz == *(long*)&tempv[i]) )
                printf ("(%d %d, %0.3e, %0.3e, %3e ) ", ljb,
                        nzval - Lnzval_bc_ptr[ljb] + indirect_thread[rel],
                        nzval[indirect_thread[rel]] + tempv[i],
                        nzval[indirect_thread[rel]],tempv[i]);
                //printing triplets (location??, old value, new value ) if none of them is zero
#endif
            }
            // printf("\n");
            tempv += nbrow;
#ifdef PI_DEBUG
            // printf("\n");
#endif
        }
        nzval += ldv;
        // printf("%d\n",nzval );
    }
    // TAU_STATIC_TIMER_STOP("SCATTER_LB");
} /* dscatter_l_1 */

void
dscatter_l (
           int ib,    /* row block number of source block L(i,k) */
           int ljb,   /* local column block number of dest. block L(i,j) */
           int nsupc, /* number of columns in destination supernode */
           int_t iukp, /* point to destination supernode's index[] */
           int_t* xsup,
           int klst,
           int nbrow,  /* LDA of the block in tempv[] */
           int_t lptr, /* Input, point to index[] location of block L(i,k) */
	   int temp_nbrow, /* number of rows of source block L(i,k) */
           int_t* usub,
           int_t* lsub,
           double *tempv,
           int* indirect_thread,int* indirect2,
           int_t ** Lrowind_bc_ptr, double **Lnzval_bc_ptr,
           gridinfo_t * grid)
{

    int_t rel, i, segsize, jj;
    double *nzval;
    int_t *index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the destination lusup. */
    int_t lptrj = BC_HEADER;
    int_t luptrj = 0;
    int_t ijb = index[lptrj];

    while (ijb != ib)  /* Search for destination block L(i,j) */
    {
        luptrj += index[lptrj + 1];
        lptrj += LB_DESCRIPTOR + index[lptrj + 1];
        ijb = index[lptrj];
    }

    /*
     * Build indirect table. This is needed because the indices are not sorted
     * in the L blocks.
     */
    int_t fnz = FstBlockC (ib);
    int_t dest_nbrow;
    lptrj += LB_DESCRIPTOR;
    dest_nbrow=index[lptrj - 1];

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    for (i = 0; i < dest_nbrow; ++i) {
        rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;

    }

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    /* can be precalculated? */
    for (i = 0; i < temp_nbrow; ++i) { /* Source index is a subset of dest. */
        rel = lsub[lptr + i] - fnz;
        indirect2[i] =indirect_thread[rel];
    }

    nzval = Lnzval_bc_ptr[ljb] + luptrj; /* Destination block L(i,j) */
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for (jj = 0; jj < nsupc; ++jj) {
        segsize = klst - usub[iukp + jj];
        if (segsize) {
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (i = 0; i < temp_nbrow; ++i) {
                nzval[indirect2[i]] -= tempv[i];
            }
            tempv += nbrow;
        }
        nzval += ldv;
    }

} /* dscatter_l */


void
dscatter_u (int ib,
           int jb,
           int nsupc,
           int_t iukp,
           int_t * xsup,
           int klst,
 	   int nbrow,      /* LDA of the block in tempv[] */
           int_t lptr,     /* point to index location of block L(i,k) */
	   int temp_nbrow, /* number of rows of source block L(i,k) */
           int_t* lsub,
           int_t* usub,
           double* tempv,
           int_t ** Ufstnz_br_ptr, double **Unzval_br_ptr,
           gridinfo_t * grid)
{
#ifdef PI_DEBUG
    printf ("A(%d,%d) goes to U block \n", ib, jb);
#endif
    // TAU_STATIC_TIMER_START("SCATTER_U");
    // TAU_STATIC_TIMER_START("SCATTER_UB");

    int_t jj, i, fnz, rel;
    int segsize;
    double *ucol;
    int_t ilst = FstBlockC (ib + 1);
    int_t lib = LBi (ib, grid);
    int_t *index = Ufstnz_br_ptr[lib];

    /* Reinitilize the pointers to the beginning of the k-th column/row of
     * L/U factors.
     * usub[] - index array for panel U(k,:)
     */
    int_t iuip_lib, ruip_lib;
    iuip_lib = BR_HEADER;
    ruip_lib = 0;

    int_t ijb = index[iuip_lib];
    while (ijb < jb) {   /* Search for destination block. */
        ruip_lib += index[iuip_lib + 1];
        // printf("supersize[%ld] \t:%ld \n",ijb,SuperSize( ijb ) );
        iuip_lib += UB_DESCRIPTOR + SuperSize (ijb);
        ijb = index[iuip_lib];
    }
    /* Skip descriptor. Now point to fstnz index of block U(i,j). */
    iuip_lib += UB_DESCRIPTOR;

    // tempv = bigV + (cum_nrow + cum_ncol*nbrow);
    for (jj = 0; jj < nsupc; ++jj) {
        segsize = klst - usub[iukp + jj];
        fnz = index[iuip_lib++];
        if (segsize) {          /* Nonzero segment in U(k,j). */
            ucol = &Unzval_br_ptr[lib][ruip_lib];

            // printf("========Entering loop=========\n");
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (i = 0; i < temp_nbrow; ++i) {
                rel = lsub[lptr + i] - fnz;
                // printf("%d %d %d %d %d \n",lptr,i,fnz,temp_nbrow,nbrow );
                // printf("hello   ucol[%d] %d %d : \n",rel,lsub[lptr + i],fnz);
                ucol[rel] -= tempv[i];

#ifdef PI_DEBUG
                double zz = 0.0;
                if (!(*(long *) &zz == *(long *) &tempv[i]))
                    printf ("(%d, %0.3e, %0.3e ) ", rel, ucol[rel] + tempv[i],
                            ucol[rel]);
                //printing triplets (location??, old value, new value ) if none of them is zero
#endif
            } /* for i = 0:temp_nbropw */
            tempv += nbrow; /* Jump LDA to next column */
#ifdef PI_DEBUG
            // printf("\n");
#endif
        }  /* if segsize */

        ruip_lib += ilst - fnz;

    }  /* for jj = 0:nsupc */
#ifdef PI_DEBUG
    // printf("\n");
#endif
    // TAU_STATIC_TIMER_STOP("SCATTER_UB");
} /* dscatter_u */




