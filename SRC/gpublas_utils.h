/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * <pre>
 * -- Distributed SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 * </pre>
 */

#ifndef GPUBLAS_UTILS_H
#define GPUBLAS_UTILS_H

#ifdef GPU_ACC

#include "gpu_wrapper.h"

extern void DisplayHeader();
extern const char* gpublasGetErrorString(gpublasStatus_t status);
extern gpuError_t checkGPU(gpuError_t);
extern gpublasStatus_t checkGPUblas(gpublasStatus_t);
extern gpublasHandle_t create_handle ();
extern void destroy_handle (gpublasHandle_t handle);

#endif 
#endif 
