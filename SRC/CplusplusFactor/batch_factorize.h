#ifndef __SUPERLU_BATCH_FACTORIZE_H__
#define __SUPERLU_BATCH_FACTORIZE_H__

#include "superlu_ddefs.h"
//#include "superlu_summit.h"

#ifdef HAVE_MAGMA
#include "magma.h"
#define BatchDim_t magma_int_t
#else 
#define BatchDim_t int_t
#endif

// Device memory used to store marshalled batch data for LU and TRSM
struct BatchLUMarshallData 
{
    BatchLUMarshallData();
    ~BatchLUMarshallData();

    // Diagonal device pointer data 
    double **dev_diag_ptrs;
    BatchDim_t *dev_diag_ld_array, *dev_diag_dim_array, *dev_info_array;
    
    // TRSM panel device pointer data 
    double **dev_panel_ptrs;
    BatchDim_t *dev_panel_ld_array, *dev_panel_dim_array;

    // Max of marshalled device data 
    BatchDim_t max_panel, max_diag;

    // Number of marshalled operations
    BatchDim_t batchsize;

    void setBatchSize(BatchDim_t batch_size);
};

// Device memory used to store marshalled batch data for Schur complement update 
struct BatchSCUMarshallData 
{
    BatchSCUMarshallData();
    ~BatchSCUMarshallData();

    // GEMM device pointer data 
    double **dev_A_ptrs, **dev_B_ptrs, **dev_C_ptrs;
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

    void setBatchSize(BatchDim_t batch_size);
};

struct BatchFactorizeWorkspace {
    // Library handles 
#ifdef HAVE_MAGMA    
    magma_queue_t magma_queue;
#endif
    cudaStream_t stream;
    cublasHandle_t cuhandle;
    
    // Marshall data 
    BatchLUMarshallData marshall_data;
    BatchSCUMarshallData sc_marshall_data;

    // GPU copy of the supernode data
    int_t* perm_c_supno, *xsup;
    int_t maxSuperSize, ldt, nsupers;

    // GPU copy of the local LU data 
    dLocalLU_t d_localLU;

    // GPU buffers for the SCU gemms 
    double** gemm_buff_ptrs, *gemm_buff_base;
    int64_t *gemm_buff_offsets;

    // Copy of the lower panel index data in a more parallel friendly format 
    int_t *d_lblock_gid_dat, **d_lblock_gid_ptrs;
    int_t *d_lblock_start_dat, **d_lblock_start_ptrs;
    int64_t *d_lblock_gid_offsets, *d_lblock_start_offsets;
    int64_t total_l_blocks, total_start_size;
};

#endif 
