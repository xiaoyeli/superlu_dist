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

<pre>
 * -- Distributed SuperLU routine (version 4.3) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 *
 * Modified: November 21, 1999
 *
</pre> 
*/
#include <stdio.h>

/* xerbla */
int xerr_dist(char *srname, int *info)
{
    printf("** On entry to %6s, parameter number %2d had an illegal value\n",
		srname, *info);
    return 0;
} /* xerr_dist */

