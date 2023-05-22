/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/** @file superlu_dist_version.h
 * \brief Gets the SuperLU_DIST's version information from the library.
 *
 * -- Distributed SuperLU routine (version 5.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley, 
 * October 13, 2017
 *
 */

#include "superlu_defs.h"

int superlu_dist_GetVersionNumber(int *major, int *minor, int *bugfix)
{
  if (major) *major = SUPERLU_DIST_MAJOR_VERSION;
  if (minor) *minor = SUPERLU_DIST_MINOR_VERSION;
  if (bugfix) *bugfix = SUPERLU_DIST_PATCH_VERSION;
  return 0;
}


