/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file 
 * \brief Get heavy-weight perfect matching (HWPM).
 *
 * <pre>
 * -- Distributed SuperLU routine (version 8.1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * July 5, 2022
 * Modified: 
 * </pre>
 */
#include <math.h>
#include "superlu_sdefs.h"
#include "superlu_ddefs.h"
//#include "dHWPM_CombBLAS.hpp"   -- multiple definition

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * Get heavy-weight perfect matching (HWPM).
 * 
 * Reference:
 * 
 *
 * Arguments
 * =========
 *
 * A      (input) SuperMatrix*
 *        The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        The type of A can be: Stype = SLU_NR_loc; Dtype = SLU_S; Mtype = SLU_GE.
 *
 * grid   (input) gridinfo_t*
 *        SuperLU's 2D process mesh.
 *
 * ScalePermstruct (output) sScalePermstruct_t*
 *        ScalePermstruct->perm_r stores the permutation obtained from HWPM.
 *
 * </pre>
 */
int
s_c2cpp_GetHWPM(SuperMatrix *A, gridinfo_t *grid, sScalePermstruct_t *ScalePermstruct)
{
    extern void dGetHWPM(SuperMatrix *, gridinfo_t *, dScalePermstruct_t *);

    /* copy to double, then use double-prec version */
    NRformat_loc *Astore = (NRformat_loc *) A->Store;
    int nnz_loc = Astore->nnz_loc;
    float *f_nzval = (float *) Astore->nzval;
    double *d_nzval = (double *) doubleMalloc_dist(nnz_loc);
    for (int i = 0; i < nnz_loc; ++i) d_nzval[i] = f_nzval[i];

    /* This up-casting is okay, since R[] and C[] are not referenced in dGetHWPM */
    dScalePermstruct_t *d_ScalePermstruct = (dScalePermstruct_t*) ScalePermstruct;

    SuperMatrix dA;
    dCreate_CompRowLoc_Matrix_dist(&dA, A->nrow, A->ncol, nnz_loc,
				   Astore-> m_loc, Astore->fst_row,
				   d_nzval, Astore->colind, Astore->rowptr,
				   SLU_NR_loc, SLU_D, SLU_GE);
 
    dGetHWPM(&dA, grid, d_ScalePermstruct);

    SUPERLU_FREE(d_nzval);
    return 0;
}
