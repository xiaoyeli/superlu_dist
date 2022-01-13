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

#if defined(HAVE_CUDA)
#include <cublas_v2.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"

extern const char* cublasGetErrorString(cublasStatus_t status);
extern cudaError_t checkCuda(cudaError_t);
extern cublasStatus_t checkCublas(cublasStatus_t);
extern cublasHandle_t create_handle ();
extern void destroy_handle (cublasHandle_t handle);
#endif // HAVE_CUDA

#if defined(HAVE_SYCL)
#include <CL/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#endif // HAVE_SYCL

#ifdef GPU_ACC
typedef struct LUstruct_gpu_  LUstruct_gpu;  // Sherry - not in this distribution

extern void DisplayHeader();
#endif // GPU_ACC

#endif // GPUBLAS_UTILS_H
