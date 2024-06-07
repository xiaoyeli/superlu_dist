#ifndef __SUPERLU_BATCH_FACTORIZE_H__
#define __SUPERLU_BATCH_FACTORIZE_H__

#include "batch_wrappers.h"
#include "gpuCommon.hpp"
#include "batch_factorize_marshall.h"
#include "luAuxStructTemplated.hpp"

// Device memory used to store marshalled batch data for LU and TRSM
template<class T>
struct TBatchLUMarshallData 
{
    TBatchLUMarshallData()
    {
        dev_diag_ptrs = dev_panel_ptrs = NULL;
        dev_diag_ld_array = dev_diag_dim_array = dev_info_array = NULL;
        dev_panel_ld_array = dev_panel_dim_array = NULL;
    }

    ~TBatchLUMarshallData()
    {
        gpuErrchk(cudaFree(dev_diag_ptrs));
        gpuErrchk(cudaFree(dev_panel_ptrs));
        gpuErrchk(cudaFree(dev_diag_ld_array));
        gpuErrchk(cudaFree(dev_diag_dim_array));
        gpuErrchk(cudaFree(dev_info_array));
        gpuErrchk(cudaFree(dev_panel_ld_array));
        gpuErrchk(cudaFree(dev_panel_dim_array));
    }

    void setBatchSize(BatchDim_t batch_size)
    {
        gpuErrchk(cudaMalloc(&dev_diag_ptrs, batch_size * sizeof(T*)));
        gpuErrchk(cudaMalloc(&dev_panel_ptrs, batch_size * sizeof(T*)));

        gpuErrchk(cudaMalloc(&dev_diag_ld_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_diag_dim_array, (batch_size + 1) * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_info_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_panel_ld_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_panel_dim_array, (batch_size + 1) * sizeof(BatchDim_t)));
    }

    // Diagonal device pointer data 
    T **dev_diag_ptrs;
    BatchDim_t *dev_diag_ld_array, *dev_diag_dim_array, *dev_info_array;
    
    // TRSM panel device pointer data 
    T **dev_panel_ptrs;
    BatchDim_t *dev_panel_ld_array, *dev_panel_dim_array;

    // Max of marshalled device data 
    BatchDim_t max_panel, max_diag;

    // Number of marshalled operations
    BatchDim_t batchsize;
};

// Device memory used to store marshalled batch data for Schur complement update 
template<class T>
struct TBatchSCUMarshallData 
{
    TBatchSCUMarshallData()
    {
        dev_A_ptrs = dev_B_ptrs = dev_C_ptrs = NULL;
        dev_lda_array = dev_ldb_array = dev_ldc_array = NULL;
        dev_m_array = dev_n_array = dev_k_array = NULL;
        dev_ist = dev_iend = dev_jst = dev_jend = NULL;
    }

    ~TBatchSCUMarshallData()
    {
        gpuErrchk(cudaFree(dev_A_ptrs));
        gpuErrchk(cudaFree(dev_B_ptrs));
        gpuErrchk(cudaFree(dev_C_ptrs));
        gpuErrchk(cudaFree(dev_lda_array));
        gpuErrchk(cudaFree(dev_ldb_array));
        gpuErrchk(cudaFree(dev_ldc_array));
        gpuErrchk(cudaFree(dev_m_array));
        gpuErrchk(cudaFree(dev_n_array));
        gpuErrchk(cudaFree(dev_k_array));
        gpuErrchk(cudaFree(dev_ist));
        gpuErrchk(cudaFree(dev_iend));
        gpuErrchk(cudaFree(dev_jst));
        gpuErrchk(cudaFree(dev_jend));
    }

    void setBatchSize(BatchDim_t batch_size)
    {
        gpuErrchk(cudaMalloc(&dev_A_ptrs, batch_size * sizeof(T*)));
        gpuErrchk(cudaMalloc(&dev_B_ptrs, batch_size * sizeof(T*)));
        gpuErrchk(cudaMalloc(&dev_C_ptrs, batch_size * sizeof(T*)));

        gpuErrchk(cudaMalloc(&dev_lda_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_ldb_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_ldc_array, batch_size * sizeof(BatchDim_t)));

        gpuErrchk(cudaMalloc(&dev_m_array, (batch_size + 1) * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_n_array, (batch_size + 1) * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_k_array, (batch_size + 1) * sizeof(BatchDim_t)));

        gpuErrchk(cudaMalloc(&dev_ist, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_iend, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_jst, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(cudaMalloc(&dev_jend, batch_size * sizeof(BatchDim_t)));
    }

    // GEMM device pointer data 
    T **dev_A_ptrs, **dev_B_ptrs, **dev_C_ptrs;
    BatchDim_t *dev_lda_array, *dev_ldb_array, *dev_ldc_array;
    BatchDim_t *dev_m_array, *dev_n_array, *dev_k_array;

    // Panel device pointer data and scu loop limits 
    BatchDim_t* dev_ist, *dev_iend, *dev_jst, *dev_jend;
    
    // Max of marshalled gemm device data 
    BatchDim_t max_m, max_n, max_k;    
    
    // Max of marshalled loop limits  
    BatchDim_t max_ilen, max_jlen;

    // Number of marshalled operations
    BatchDim_t batchsize;
};

template<class T>
struct TBatchFactorizeWorkspace {
    // Library handles 
#ifdef HAVE_MAGMA    
    magma_queue_t magma_queue;
#endif
    cudaStream_t stream;
    cublasHandle_t cuhandle;
    
    // Marshall data 
    TBatchLUMarshallData<T> marshall_data;
    TBatchSCUMarshallData<T> sc_marshall_data;

    // GPU copy of the supernode data
    int_t* perm_c_supno, *xsup;
    int_t maxSuperSize, ldt, nsupers;

    // GPU copy of the local LU data 
    LocalLU_type<T> d_localLU;

    // GPU buffers for the SCU gemms 
    T** gemm_buff_ptrs, *gemm_buff_base;
    int64_t *gemm_buff_offsets;

    // Copy of the lower panel index data in a more parallel friendly format 
    int_t *d_lblock_gid_dat, **d_lblock_gid_ptrs;
    int_t *d_lblock_start_dat, **d_lblock_start_ptrs;
    int64_t *d_lblock_gid_offsets, *d_lblock_start_offsets;
    int64_t total_l_blocks, total_start_size;
};

#endif 
