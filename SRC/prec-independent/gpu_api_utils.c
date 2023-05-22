/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
#include "superlu_defs.h"

#ifdef GPU_ACC  // enable CUDA

#include <stdio.h>
#include "gpu_api_utils.h"
 void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    int version;
    // cout << "NBody.GPU" << endl << "=========" << endl << endl;
    
    gpuRuntimeGetVersion( &version ); 
    printf("GPU Driver version:   v %d\n",version);
    //cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl; 

    int devCount;
    gpuGetDeviceCount(&devCount);
    printf( "GPU Devices: \n \n"); 

    for(int i = 0; i < devCount; ++i)
    {
        struct gpuDeviceProp props;       
        gpuGetDeviceProperties(&props, i);
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


const char* gpublasGetErrorString(gpublasStatus_t status)
{
    switch(status)
    {
        case GPUBLAS_STATUS_SUCCESS: return "GPUBLAS_STATUS_SUCCESS";
        case GPUBLAS_STATUS_NOT_INITIALIZED: return "GPUBLAS_STATUS_NOT_INITIALIZED";
        case GPUBLAS_STATUS_ALLOC_FAILED: return "GPUBLAS_STATUS_ALLOC_FAILED";
        case GPUBLAS_STATUS_INVALID_VALUE: return "GPUBLAS_STATUS_INVALID_VALUE"; 
        case GPUBLAS_STATUS_ARCH_MISMATCH: return "GPUBLAS_STATUS_ARCH_MISMATCH"; 
        case GPUBLAS_STATUS_MAPPING_ERROR: return "GPUBLAS_STATUS_MAPPING_ERROR";
        case GPUBLAS_STATUS_EXECUTION_FAILED: return "GPUBLAS_STATUS_EXECUTION_FAILED"; 
        case GPUBLAS_STATUS_INTERNAL_ERROR: return "GPUBLAS_STATUS_INTERNAL_ERROR"; 
#ifdef HAVE_CUDA        
        case GPUBLAS_STATUS_LICENSE_ERROR: return "GPUBLAS_STATUS_LICENSE_ERROR"; //HIPBLAS_STATUS_LICENSE_ERROR is not a valid enum type in rocm yet
#endif        
        case GPUBLAS_STATUS_NOT_SUPPORTED: return "GPUBLAS_STATUS_NOT_SUPPORTED"; 
    }
    return "unknown error";
}

/*error reporting functions */
//inline
gpuError_t checkGPU(gpuError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != gpuSuccess) {
        fprintf(stderr, "GPU Runtime Error: %s\n", gpuGetErrorString(result));
        assert(result == gpuSuccess);
    }
#endif
    return result;
}

gpublasStatus_t checkGPUblas(gpublasStatus_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != GPUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "GPU Blas Runtime Error: %s\n", gpublasGetErrorString(result));
    assert(result == GPUBLAS_STATUS_SUCCESS);
  }
#endif
  return result;
}


gpublasHandle_t create_handle ()
{
       gpublasHandle_t handle;
       checkGPUblas(gpublasCreate(&handle));
       return handle;
 }

 void destroy_handle (gpublasHandle_t handle)
 {
      checkGPUblas(gpublasDestroy(handle));
 }

void printGPUStats(int nsupers, SuperLUStat_t *stat, gridinfo3d_t *grid3d )
{
    int iam = grid3d->iam;
    double tGemm = 0;
    double tScatter = 0;
    double tPCIeH2D = 0;
    double tPCIeD2H = 0;
    double flopcnt;

    // Agregate the local sum from all supernodes
    for (int_t i = 0; i < nsupers; ++i)
    {
        float milliseconds = 0;

	if (stat->isOffloaded[i])
	{
	    gpuEventElapsedTime(&milliseconds, stat->ePCIeH2D[i], stat->GemmStart[i]);
	    tPCIeH2D += 1e-3 * (double) milliseconds;
	    milliseconds = 0;
	    gpuEventElapsedTime(&milliseconds, stat->GemmStart[i], stat->GemmEnd[i]);
	    tGemm += 1e-3 * (double) milliseconds;
    	    milliseconds = 0;
	    gpuEventElapsedTime(&milliseconds, stat->GemmEnd[i], stat->ScatterEnd[i]);
	    tScatter += 1e-3 * (double) milliseconds;
	}

	milliseconds = 0;
	gpuEventElapsedTime(&milliseconds, stat->ePCIeD2H_Start[i], stat->ePCIeD2H_End[i]);
	tPCIeD2H += 1e-3 * (double) milliseconds;
    }

    MPI_Reduce(&(stat->GemmFLOPCounter), &flopcnt, 1, MPI_DOUBLE, MPI_SUM, 0, grid3d->comm);

    if (iam == 0) {
      printf("*-- GPU flops: \n\tFlops offloaded %.3e, Time %lf, Flop rate %lf GF/sec \n",
	     flopcnt, tGemm, 1e-9 * flopcnt / tGemm  );
      printf("*-- GPU memory: \n\tMop offloaded %.3e, Time %lf, Bandwidth %lf GByte/sec \n",
	     stat->ScatterMOPCounter, tScatter, 8e-9 * stat->ScatterMOPCounter / tScatter  );
      printf("*-- PCIe Data Transfer H2D:\n\tData Sent %.3e(GB)\n\tTime observed from CPU %lf\n\tActual time spent %lf\n\tBandwidth %lf GByte/sec \n",
	     1e-9 * stat->cPCIeH2D, stat->tHost_PCIeH2D, tPCIeH2D, 1e-9 * stat->cPCIeH2D / tPCIeH2D  );
      printf("*-- PCIe Data Transfer D2H:\n\tData Sent %.3e(GB)\n\tTime observed from CPU %lf\n\tActual time spent %lf\n\tBandwidth %lf GByte/sec \n",
	     1e-9 * stat->cPCIeD2H, stat->tHost_PCIeD2H, tPCIeD2H, 1e-9 * stat->cPCIeD2H / tPCIeD2H  );
      fflush(stdout);
    }
    
} /* end printGPUStats */


#endif  // enable GPU
