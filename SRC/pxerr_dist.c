/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief
 *
 * <pre>
 * -- Distributed SuperLU routine (version 4.3) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 *
 * Modified: November 21, 1999
 *
 * </pre>
 */

#include "superlu_ddefs.h"

/* pxerbla */
void pxerr_dist(char *srname, gridinfo_t *grid, int_t info)
{
    printf("{" IFMT "," IFMT "}: On entry to %6s, parameter number " IFMT " had an illegal value\n",
	   MYROW(grid->iam, grid), MYCOL(grid->iam, grid), srname, info);

}
