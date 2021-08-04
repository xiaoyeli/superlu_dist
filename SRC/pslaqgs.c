/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Equilibrates a general sparse M by N matrix
 *
 * <pre>
 * File name:	pslaqgs.c
 * History:     Modified from LAPACK routine SLAQGE
 * </pre>
 */
#include <math.h>
#include "superlu_sdefs.h"

/*! \brief

<pre>
    Purpose
    =======

    PSLAQGS equilibrates a general sparse M by N matrix A using the row
    and column scaling factors in the vectors R and C.

    See supermatrix.h for the definition of 'SuperMatrix' structure.

    Arguments
    =========

    A       (input/output) SuperMatrix*
            On exit, the equilibrated matrix.  See EQUED for the form of
            the equilibrated matrix. The type of A can be:
	    Stype = SLU_NR_loc; Dtype = SLU_S; Mtype = SLU_GE.

    R       (input) float*, dimension (A->nrow)
            The row scale factors for A.

    C       (input) float*, dimension (A->ncol)
            The column scale factors for A.

    ROWCND  (input) float
            Ratio of the smallest R(i) to the largest R(i).

    COLCND  (input) float
            Ratio of the smallest C(i) to the largest C(i).

    AMAX    (input) float
            Absolute value of largest matrix entry.

    EQUED   (output) char*
            Specifies the form of equilibration that was done.
            = 'N':  No equilibration
            = 'R':  Row equilibration, i.e., A has been premultiplied by
                    diag(R).
            = 'C':  Column equilibration, i.e., A has been postmultiplied
                    by diag(C).
            = 'B':  Both row and column equilibration, i.e., A has been
                    replaced by diag(R) * A * diag(C).

    Internal Parameters
    ===================

    THRESH is a threshold value used to decide if row or column scaling
    should be done based on the ratio of the row or column scaling
    factors.  If ROWCND < THRESH, row scaling is done, and if
    COLCND < THRESH, column scaling is done.

    LARGE and SMALL are threshold values used to decide if row scaling
    should be done based on the absolute size of the largest matrix
    element.  If AMAX > LARGE or AMAX < SMALL, row scaling is done.

    =====================================================================
</pre>
*/

void
pslaqgs(SuperMatrix *A, float *r, float *c,
       float rowcnd, float colcnd, float amax, char *equed)
{

#define THRESH    (0.1)

    /* Local variables */
    NRformat_loc *Astore;
    float *Aval;
    int_t i, j, irow, jcol, m_loc;
    float large, small;

    /* Quick return if possible */
    if (A->nrow <= 0 || A->ncol <= 0) {
	*(unsigned char *)equed = 'N';
	return;
    }

    Astore = A->Store;
    Aval = Astore->nzval;
    m_loc = Astore->m_loc;

    /* Initialize LARGE and SMALL. */
    small = smach_dist("Safe minimum") / smach_dist("Precision");
    large = 1. / small;

    if (rowcnd >= THRESH && amax >= small && amax <= large) {
	if (colcnd >= THRESH)
	    *(unsigned char *)equed = 'N';
	else {
	    /* Column scaling */
	    irow = Astore->fst_row;
	    for (i = 0; i < m_loc; ++i) {
	        for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
		    jcol = Astore->colind[j];
		    Aval[j] *= c[jcol];
	      }
	      ++irow;
	    }
	    *(unsigned char *)equed = 'C';
	}
    } else if (colcnd >= THRESH) {
	/* Row scaling, no column scaling */
	irow = Astore->fst_row;
	for (i = 0; i < m_loc; ++i) {
	    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j)
	        Aval[j] *= r[irow];
	    ++irow;
	}
	*(unsigned char *)equed = 'R';
    } else {
	/* Both row and column scaling */
	irow = Astore->fst_row;
	for (i = 0; i < m_loc; ++i) {
	    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
	        jcol = Astore->colind[j];
	        Aval[j] = Aval[j] * r[irow] * c[jcol];
	    }
	    ++irow;
	}
	*(unsigned char *)equed = 'B';
    }

    return;

} /* pslaqgs */

