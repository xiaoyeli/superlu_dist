/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief  Header file for the Python bridge routines
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.1.0) --
 * Lawrence Berkeley National Lab
 * November 10, 2024
 * </pre>
 */


#ifndef __SUPERLU_DIST_PDBRIDGE /* allow multiple inclusions */
#define __SUPERLU_DIST_PDBRIDGE
#include "superlu_ddefs.h"

typedef struct {
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dSOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
} slu_handle;

#ifdef __cplusplus
extern "C" {
#endif

/*== APIs for python caller =======*/
extern void pdbridge_init(int_t m, int_t n, int_t nnz, int_t *rowind, int_t *colptr, double *nzval, void ** pyobj, int argc, char *argv[]);
extern void pdbridge_solve(void ** pyobj, int nrhs, double   *b_global);
extern void pdbridge_free(void ** pyobj);
extern void pdbridge_factor(void ** pyobj);
extern void pdbridge_logdet(void ** pyobj, int * sign, double * logdet);

#ifdef __cplusplus
  }
#endif
#endif 

int dcreate_matrix_from_csc(SuperMatrix *A, int_t m, int_t n, int_t nnz, int_t *rowind0, int_t *colptr0, double *nzval0, gridinfo_t *grid);
int count_swaps(int perm[], int n);
