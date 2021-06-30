/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Returns the value of the one norm, or the Frobenius norm, or the infinity norm, or the element of largest value
 *
 * <pre>
 * File name:	pslangs.c
 * History:     Modified from lapack routine SLANGE
 * </pre>
 */
#include <math.h>
#include "superlu_sdefs.h"

/*! \brief

<pre>
    Purpose
    =======

    PSLANGS returns the value of the one norm, or the Frobenius norm, or
    the infinity norm, or the element of largest absolute value of a
    real matrix A.

    Description
    ===========

    PSLANGE returns the value

       PSLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                 (
                 ( norm1(A),         NORM = '1', 'O' or 'o'
                 (
                 ( normI(A),         NORM = 'I' or 'i'
                 (
                 ( normF(A),         NORM = 'F', 'f', 'E' or 'e'

    where  norm1  denotes the  one norm of a matrix (maximum column sum),
    normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
    normF  denotes the  Frobenius norm of a matrix (square root of sum of
    squares).  Note that  max(abs(A(i,j)))  is not a  matrix norm.

    Arguments
    =========

    NORM    (input) CHARACTER*1
            Specifies the value to be returned in DLANGE as described above.
    A       (input) SuperMatrix*
            The M by N sparse matrix A.
    GRID    (input) gridinof_t*
            The 2D process mesh.
   =====================================================================
</pre>
*/

float pslangs(char *norm, SuperMatrix *A, gridinfo_t *grid)
{
    /* Local variables */
    NRformat_loc *Astore;
    int_t    m_loc;
    float   *Aval;
    int_t    i, j, jcol;
    float   value=0., sum;
    float   *rwork;
    float   tempvalue;
    float   *temprwork;

    Astore = (NRformat_loc *) A->Store;
    m_loc = Astore->m_loc;
    Aval   = (float *) Astore->nzval;

    if ( SUPERLU_MIN(A->nrow, A->ncol) == 0) {
	value = 0.;
    } else if ( strncmp(norm, "M", 1)==0 ) {
	/* Find max(abs(A(i,j))). */
	value = 0.;
	for (i = 0; i < m_loc; ++i) {
	    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j)
		value = SUPERLU_MAX( value, fabs(Aval[j]) );
	}

	MPI_Allreduce(&value, &tempvalue, 1, MPI_FLOAT, MPI_MAX, grid->comm);
	value = tempvalue;

    } else if ( strncmp(norm, "O", 1)==0 || *(unsigned char *)norm == '1') {
	/* Find norm1(A). */
	value = 0.;
#if 0
	for (j = 0; j < A->ncol; ++j) {
	    sum = 0.;
	    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++)
		sum += fabs(Aval[i]);
	    value = SUPERLU_MAX(value,sum);
	}
#else /* Sherry ==> */
	if ( !(rwork = floatCalloc_dist(A->ncol)) )
	    ABORT("floatCalloc_dist fails for rwork.");
	for (i = 0; i < m_loc; ++i) {
	    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
	        jcol = Astore->colind[j];
		rwork[jcol] += fabs(Aval[j]);
	    }
	}

	if ( !(temprwork = floatCalloc_dist(A->ncol)) )
	    ABORT("floatCalloc_dist fails for temprwork.");
	MPI_Allreduce(rwork, temprwork, A->ncol, MPI_FLOAT, MPI_SUM, grid->comm);
	value = 0.;
	for (j = 0; j < A->ncol; ++j) {
	    value = SUPERLU_MAX(value, temprwork[j]);
	}
	SUPERLU_FREE (temprwork);
	SUPERLU_FREE (rwork);
#endif
    } else if ( strncmp(norm, "I", 1)==0 ) {
	/* Find normI(A). */
	value = 0.;
	sum = 0.;
	for (i = 0; i < m_loc; ++i) {
	    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j)
	        sum += fabs(Aval[j]);
	    value = SUPERLU_MAX(value, sum);
	}
	MPI_Allreduce(&value, &tempvalue, 1, MPI_FLOAT, MPI_MAX, grid->comm);
	value = tempvalue;

    } else if ( strncmp(norm, "F", 1)==0 || strncmp(norm, "E", 1)==0 ) {
	/* Find normF(A). */
	ABORT("Not implemented.");
    } else {
	ABORT("Illegal norm specified.");
    }

    return (value);

} /* pslangs */
