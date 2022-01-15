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

#ifndef gpu_api_utils_H
#define gpu_api_utils_H

#ifdef GPU_ACC

#include "gpu_wrapper.h"
typedef struct LUstruct_gpu_  LUstruct_gpu;  // Sherry - not in this distribution

#ifdef __cplusplus
extern "C" {
#endif
extern void DisplayHeader();
extern const char* cublasGetErrorString(cublasStatus_t status);
extern cudaError_t checkCuda(cudaError_t);
extern cublasStatus_t checkCublas(cublasStatus_t);
extern cublasHandle_t create_handle ();
extern void destroy_handle (cublasHandle_t handle);
#ifdef __cplusplus
  }
#endif

#endif 
