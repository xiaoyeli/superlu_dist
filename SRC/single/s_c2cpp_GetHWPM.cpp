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
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * April 1, 2018
 * Modified: April 2, 2020
 * </pre>
 */
#include <math.h>
#include "sHWPM_CombBLAS.hpp"
#include "superlu_sdefs.h"

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
 *        The type of A can be: Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.
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
s_c2cpp_GetHWPM(SuperMatrix *A, gridinfo_t *grid,
		sScalePermstruct_t *ScalePermstruct)
{
    sGetHWPM(A, grid, ScalePermstruct);
    return 0;
}
