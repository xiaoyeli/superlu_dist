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

    void DeleteTBatchLUMarshallData()
    {
        gpuErrchk(gpuFree(dev_diag_ptrs));
        gpuErrchk(gpuFree(dev_panel_ptrs));
        gpuErrchk(gpuFree(dev_diag_ld_array));
        gpuErrchk(gpuFree(dev_diag_dim_array));
        gpuErrchk(gpuFree(dev_info_array));
        gpuErrchk(gpuFree(dev_panel_ld_array));
        gpuErrchk(gpuFree(dev_panel_dim_array));
    }

    void setBatchSize(BatchDim_t batch_size)
    {
        gpuErrchk(gpuMalloc(&dev_diag_ptrs, batch_size * sizeof(T*)));
        gpuErrchk(gpuMalloc(&dev_panel_ptrs, batch_size * sizeof(T*)));

        gpuErrchk(gpuMalloc(&dev_diag_ld_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_diag_dim_array, (batch_size + 1) * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_info_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_panel_ld_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_panel_dim_array, (batch_size + 1) * sizeof(BatchDim_t)));
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

    void DeleteTBatchSCUMarshallData()
    {
        gpuErrchk(gpuFree(dev_A_ptrs));
        gpuErrchk(gpuFree(dev_B_ptrs));
        gpuErrchk(gpuFree(dev_C_ptrs));
        gpuErrchk(gpuFree(dev_lda_array));
        gpuErrchk(gpuFree(dev_ldb_array));
        gpuErrchk(gpuFree(dev_ldc_array));
        gpuErrchk(gpuFree(dev_m_array));
        gpuErrchk(gpuFree(dev_n_array));
        gpuErrchk(gpuFree(dev_k_array));
        gpuErrchk(gpuFree(dev_ist));
        gpuErrchk(gpuFree(dev_iend));
        gpuErrchk(gpuFree(dev_jst));
        gpuErrchk(gpuFree(dev_jend));
    }

    void setBatchSize(BatchDim_t batch_size)
    {
        gpuErrchk(gpuMalloc(&dev_A_ptrs, batch_size * sizeof(T*)));
        gpuErrchk(gpuMalloc(&dev_B_ptrs, batch_size * sizeof(T*)));
        gpuErrchk(gpuMalloc(&dev_C_ptrs, batch_size * sizeof(T*)));

        gpuErrchk(gpuMalloc(&dev_lda_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_ldb_array, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_ldc_array, batch_size * sizeof(BatchDim_t)));

        gpuErrchk(gpuMalloc(&dev_m_array, (batch_size + 1) * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_n_array, (batch_size + 1) * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_k_array, (batch_size + 1) * sizeof(BatchDim_t)));

        gpuErrchk(gpuMalloc(&dev_ist, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_iend, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_jst, batch_size * sizeof(BatchDim_t)));
        gpuErrchk(gpuMalloc(&dev_jend, batch_size * sizeof(BatchDim_t)));
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
    gpuStream_t stream;
    gpublasHandle_t cuhandle;
    
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

    // Device-resident reuse: values of A on the device, map from each nonzero
    // of A to its slot in the flat L (offset < Lnzval_bc_cnt) or U
    // (Lnzval_bc_cnt + offset) value arrays, and the host buffers pinned for
    // the per-step transfers.
    T *d_avals = nullptr;
    int64_t *d_map = nullptr;
    int_t nnz_a = 0;
    void *pinned[3] = {nullptr, nullptr, nullptr};

    // Stage 3: map from each entry of the solve's column-wise U value array
    // (Unzval_bc_dat) to its source in the factorization's row-wise device U
    // (Unzval_br_new_dat); -1 for entries with no source.
    int64_t *d_umap = nullptr;
    int64_t ucnt = 0;
    int solve_ready = 0;             // stage 3 set up (map or row-data copy)

    // Stage 4 (A side): map from each entry of the caller's per-system CSC
    // values, concatenated in system order, to the device L/U slot, with the
    // fixed equilibration R[i]*C[j] folded in.  Built once on the DOFACT call;
    // afterwards a reuse step never touches A on the host.
    int64_t *h_map = nullptr;        // host copy of d_map (stacked position -> slot)
    int64_t *d_map2 = nullptr;       // entry q -> slot, -1 if none
    double  *d_scale2 = nullptr;     // entry q -> R[i]*C[j]
    int     *d_ent_sys = nullptr;    // entry q -> system d
    int     *d_ent_idx = nullptr;    // entry q -> position in that system's nzval
    double **d_Aptrs = nullptr;      // device copy of the per-system value pointers
    double  *h_avals_cat = nullptr;  // pinned staging for host-side values
    int_t    nnz2 = 0;
    int      nsys = 0;
    int      a_prefilled = 0;        // 1: the wrapper already refilled the device L/U
    double   anorm_cache = 0.0;
};

#endif 
