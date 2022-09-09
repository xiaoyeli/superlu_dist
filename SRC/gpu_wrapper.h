/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Wrappers for multiple types of GPUs
 *
 * <pre>
 * -- Distributed SuperLU routine (version 8.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * May 22, 2022
 * </pre>
 */

#ifndef __SUPERLU_GPUWRAPPER /* allow multiple inclusions */
#define __SUPERLU_GPUWRAPPER

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include <cusparse.h>				 
#include <cuda_profiler_api.h>

#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuHostMalloc cudaHostAlloc
#define gpuHostMallocDefault cudaHostAllocDefault
#define gpuMallocManaged cudaMallocManaged
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpy2DAsync cudaMemcpy2DAsync
#define gpuFreeHost cudaFreeHost
#define gpuFree cudaFree
#define gpuMemPrefetchAsync cudaMemPrefetchAsync
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMemcpy cudaMemcpy
#define gpuMemAttachGlobal cudaMemAttachGlobal
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroyWithFlags cudaStreamDestroyWithFlags
#define gpuStreamDefault cudaStreamDefault
#define gpublasStatus_t cublasStatus_t
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuMemGetInfo cudaMemGetInfo
#define gpuOccupancyMaxPotentialBlockSize cudaOccupancyMaxPotentialBlockSize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuDeviceReset cudaDeviceReset
#define gpuMallocHost cudaMallocHost
#define gpuEvent_t cudaEvent_t
#define gpuMemset cudaMemset
#define  GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS 
#define  GPUBLAS_STATUS_NOT_INITIALIZED CUBLAS_STATUS_NOT_INITIALIZED 
#define  GPUBLAS_STATUS_ALLOC_FAILED CUBLAS_STATUS_ALLOC_FAILED 
#define  GPUBLAS_STATUS_INVALID_VALUE CUBLAS_STATUS_INVALID_VALUE 
#define  GPUBLAS_STATUS_ARCH_MISMATCH CUBLAS_STATUS_ARCH_MISMATCH 
#define  GPUBLAS_STATUS_MAPPING_ERROR CUBLAS_STATUS_MAPPING_ERROR 
#define  GPUBLAS_STATUS_EXECUTION_FAILED CUBLAS_STATUS_EXECUTION_FAILED 
#define  GPUBLAS_STATUS_INTERNAL_ERROR CUBLAS_STATUS_INTERNAL_ERROR 
#define  GPUBLAS_STATUS_LICENSE_ERROR CUBLAS_STATUS_LICENSE_ERROR 
#define  GPUBLAS_STATUS_NOT_SUPPORTED CUBLAS_STATUS_NOT_SUPPORTED 
#define  gpublasCreate cublasCreate 
#define  gpublasDestroy cublasDestroy
#define  gpublasHandle_t cublasHandle_t
#define  gpublasSetStream cublasSetStream
#define  gpublasDgemm cublasDgemm
#define  gpublasSgemm cublasSgemm
#define  gpublasZgemm cublasZgemm
#define  gpublasCgemm cublasCgemm
#define  GPUBLAS_OP_N CUBLAS_OP_N
#define  gpuDoubleComplex cuDoubleComplex
#define  gpuRuntimeGetVersion cudaRuntimeGetVersion
#define  threadIdx_x threadIdx.x
#define  threadIdx_y threadIdx.y
#define  blockIdx_x blockIdx.x
#define  blockIdx_y blockIdx.y
#define  blockDim_x blockDim.x
#define  blockDim_y blockDim.y
#define  gridDim_x gridDim.x
#define  gridDim_y gridDim.y




#elif defined(HAVE_HIP)

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hipblas.h"

// #include "roctracer_ext.h"    // need to pass the include dir directly to HIP_HIPCC_FLAGS
// // roctx header file
// #include <roctx.h>

#define gpuDeviceProp hipDeviceProp_t
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuHostMalloc hipHostMalloc
#define gpuHostMallocDefault hipHostMallocDefault  
#define gpuMallocManaged hipMallocManaged
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpy2DAsync hipMemcpy2DAsync
#define gpuFreeHost hipHostFree
#define gpuFree hipFree
#define gpuMemPrefetchAsync hipMemPrefetchAsync   // not sure about this
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMemcpy hipMemcpy
#define gpuMemAttachGlobal hipMemAttachGlobal
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamDestroyWithFlags hipStreamDestroyWithFlags
#define gpuStreamDefault hipStreamDefault
#define gpublasStatus_t hipblasStatus_t
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuMemGetInfo hipMemGetInfo
#define gpuOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuDeviceReset hipDeviceReset
#define gpuMallocHost hipHostMalloc
#define gpuEvent_t hipEvent_t
#define gpuMemset hipMemset
#define  GPUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS 
#define  GPUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED 
#define  GPUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED 
#define  GPUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE 
#define  GPUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH 
#define  GPUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR 
#define  GPUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED 
#define  GPUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR 
#define  GPUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_LICENSE_ERROR 
#define  GPUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED 
#define  gpublasCreate hipblasCreate 
#define  gpublasDestroy hipblasDestroy
#define  gpublasHandle_t hipblasHandle_t
#define  gpublasSetStream hipblasSetStream
#define  gpublasDgemm hipblasDgemm
#define  gpublasSgemm hipblasSgemm
#define  gpublasZgemm hipblasZgemm
#define  gpublasCgemm hipblasCgemm
#define  GPUBLAS_OP_N HIPBLAS_OP_N
#define  gpuDoubleComplex hipblasDoubleComplex
#define  gpuRuntimeGetVersion hipRuntimeGetVersion
#define  threadIdx_x hipThreadIdx_x
#define  threadIdx_y hipThreadIdx_y
#define  blockIdx_x hipBlockIdx_x
#define  blockIdx_y hipBlockIdx_y
#define  blockDim_x hipBlockDim_x
#define  blockDim_y hipBlockDim_y
#define  gridDim_x hipGridDim_x
#define  gridDim_y hipGridDim_y


#endif


 #define gpublasCheckErrors(fn) \
	 do { \
		 gpublasStatus_t __err = fn; \
		 if (__err != GPUBLAS_STATUS_SUCCESS) { \
			 fprintf(stderr, "Fatal gpublas error: %d (at %s:%d)\n", \
				 (int)(__err), \
				 __FILE__, __LINE__); \
			 fprintf(stderr, "*** FAILED - ABORTING\n"); \
			 exit(1); \
		 } \
	 } while(0);


#endif /* __SUPERLU_GPUWRAPPER */
