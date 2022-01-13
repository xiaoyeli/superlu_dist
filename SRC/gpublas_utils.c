/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
#include "superlu_defs.h"

#ifdef HAVE_CUDA  //////////////////////////////  enable CUDA

 void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    // cout << "NBody.GPU" << endl << "=========" << endl << endl;

    printf("CUDA version:   v %d\n",CUDART_VERSION);
    //cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl; 

    int devCount;
    cudaGetDeviceCount(&devCount);
    printf( "CUDA Devices: \n \n"); 

    for(int i = 0; i < devCount; ++i)
    {
        struct cudaDeviceProp props;       
        cudaGetDeviceProperties(&props, i);
        printf("%d : %s %d %d\n",i, props.name,props.major,props.minor );
        // cout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        printf("  Global memory:   %ld mb \n", props.totalGlobalMem / mb);
        // cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
        printf("  Shared memory:   %ld kb \n", props.sharedMemPerBlock / kb ); //<<  << "kb" << endl;
        printf("  Constant memory: %ld kb \n", props.totalConstMem / kb );
        printf("  Block registers: %d \n\n", props.regsPerBlock );

        // to do these later
        // printf("  Warp size:         %d" << props.warpSize << endl;
        // printf("  Threads per block: %d" << props.maxThreadsPerBlock << endl;
        // printf("  Max block dimensions: [ %d" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        // printf("  Max grid dimensions:  [ %d" << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;

        // cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
        // cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
        // cout << "  Block registers: " << props.regsPerBlock << endl << endl;

        // cout << "  Warp size:         " << props.warpSize << endl;
        // cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        // cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        // cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
        // cout << endl;
    }
}


const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR"; 
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED"; 
    }
    return "unknown error";
}

/*error reporting functions */
//inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

cublasStatus_t checkCublas(cublasStatus_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Blas Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
#endif
  return result;
}


cublasHandle_t create_handle ()
{
       cublasHandle_t handle;
       checkCublas(cublasCreate(&handle));
       return handle;
 }

 void destroy_handle (cublasHandle_t handle)
 {
      checkCublas(cublasDestroy(handle));
 }

#endif  // HAVE_CUDA


#ifdef HAVE_SYCL  // enable SYCL

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    // cout << "NBody.GPU" << endl << "=========" << endl << endl;

    int devCount=0;
    char *sycl_explicit_scale = nullptr;
    sycl_explicit_scale = getenv ("SUPERLU_SYCL_EXPLICIT_SCALE");

    sycl::platform platform(sycl::gpu_selector{});
    auto const& gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
    std::vector<sycl::device> sycl_devices{}; // might include explicit or implicit devices
    for (int i = 0; i < gpu_devices.size(); i++) {
      if (sycl_explicit_scale != nullptr) {
	if (gpu_devices[i].get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
	  auto subDevices = gpu_devices[i].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
	  sycl_devices.insert( std::end(sycl_devices), std::begin(subDevices), std::end(subDevices) );
	  devCount += subDevices.size();
	}
      } // explicit scaling
      else {
	sycl_devices.push_back( gpu_devices[i] );	
	devCount++;
      }
    }
    std::cout << "SYCL version:    v" << sycl_devices[0].get_info<sycl::info::device::version>() << std::endl;
    std::cout << "SYCL Devices:     ";

    for(int i = 0; i < devCount; ++i)
    {
	std::cout << "[" << i << "] : " << sycl_devices[i].get_info<sycl::info::device::name>() << std::endl;
	std::cout << "  Global memory (mb):   " << sycl_devices[i].get_info<sycl::info::device::global_mem_size>() / mb << std::endl;
	std::cout << "  Shared memory (kb):   " << sycl_devices[i].get_info<sycl::info::device::local_mem_size>() / kb << std::endl;
	std::cout << "  Constant memory (kb): " << sycl_devices[i].get_info<sycl::info::device::max_constant_buffer_size>() / kb << std::endl;
    }
    std::cout << std::endl;
}

#endif  // enable SYCL
