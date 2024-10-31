/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief  Distributed SuperLU data types and function prototypes
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology
 * November 1, 2007
 * April 5, 2015
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
 * May 10, 2019 version 7.0.0
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

