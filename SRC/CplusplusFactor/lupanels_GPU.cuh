
#pragma once
#include <vector>
#include <iostream>
#include "superlu_ddefs.h"

#ifdef HAVE_CUDA
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
#ifdef HAVE_MAGMA
  #include "magma.h"
#endif 
#endif

#include "lu_common.hpp"
// #include "lupanels.hpp" 

#ifdef __CUDACC__
#define DEVICE_CALLABLE __device__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#define DEVICE_CALLABLE
#endif

class lpanel_t;
class upanel_t;

class lpanelGPU_t 
{
    
    public:
        int_t *index;
        double *val;
        // bool isDiagIncluded;
        CUDA_CALLABLE
        lpanelGPU_t(int_t k, int_t *lsub, double *nzval, int_t *xsup, int_t isDiagIncluded);
        // default constuctor
        
        CUDA_CALLABLE
        lpanelGPU_t()
        {
            index = NULL;
            val = NULL;
        }
        CUDA_CALLABLE
        lpanelGPU_t(int_t *index_, double *val_): index(index_), val(val_) {return;};
        
    
        // index[0] is number of blocks
        CUDA_CALLABLE
        int_t nblocks()
        {
            return index[0];
        }
        // number of rows
        CUDA_CALLABLE
        int_t nzrows() { return index[1]; }
        CUDA_CALLABLE
        int_t haveDiag() { return index[2]; }
        CUDA_CALLABLE
        int_t ncols() { return index[3]; }
    
        // global block id of k-th block in the panel
        CUDA_CALLABLE
        int_t gid(int_t k)
        {
            return index[LPANEL_HEADER_SIZE + k];
        }
    
        // number of rows in the k-th block
        CUDA_CALLABLE
        int_t nbrow(int_t k)
        {
            return index[LPANEL_HEADER_SIZE + nblocks() + k + 1] - index[LPANEL_HEADER_SIZE + nblocks() + k];
        }
    
        // 
        CUDA_CALLABLE
        int_t stRow(int k)
        {
            return index[LPANEL_HEADER_SIZE + nblocks() + k]; 
        } 
        // row
        CUDA_CALLABLE
        int_t *rowList(int_t k)
        {
            // LPANEL_HEADER
            // nblocks() : blocks list
            // nblocks()+1 : blocks st_points
            // index[LPANEL_HEADER_SIZE + nblocks() + k] statrting of the block
            return &index[LPANEL_HEADER_SIZE +
                          2 * nblocks() + 1 + index[LPANEL_HEADER_SIZE + nblocks() + k]];
        }
    
        CUDA_CALLABLE
        double *blkPtr(int_t k)
        {
            return &val[index[LPANEL_HEADER_SIZE + nblocks() + k]];
        }
    
        CUDA_CALLABLE
        int_t LDA() { return index[1]; }

        DEVICE_CALLABLE
        int_t find(int_t k);
        // // for L panel I don't need any special transformation function
        // int_t panelSolve(int_t ksupsz, double *DiagBlk, int_t LDD);
        // int_t diagFactor(int_t k, double *UBlk, int_t LDU, double thresh, int_t *xsup,
        //                  superlu_dist_options_t *options, SuperLUStat_t *stat, int *info);
        // int_t packDiagBlock(double *DiagLBlk, int_t LDD);

        CUDA_CALLABLE
        int_t isEmpty() { return index == NULL; }

        CUDA_CALLABLE
        int_t nzvalSize()
        {
            if (index == NULL)
                return 0;
            return ncols() * nzrows();
        }
        
        CUDA_CALLABLE
        int_t indexSize()
        {
            if (index == NULL)
                return 0;
            return LPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzrows();
        }
    
        // return the maximal iEnd such that stRow(iEnd)-stRow(iSt) < maxRow;
        // Wajih: copied from the cpu panel code 
        CUDA_CALLABLE
        int getEndBlock(int iSt, int maxRows)
        {
            int nlb = nblocks();
            if(iSt >= nlb )
                return nlb; 
            int iEnd = iSt; 
            int ii = iSt +1;

            while (
                stRow(ii) - stRow(iSt) <= maxRows &&
                ii < nlb)
                ii++;

        #if 1
            if (stRow(ii) - stRow(iSt) > maxRows)
                iEnd = ii-1;
            else 
                iEnd =ii; 
        #else 
            if (ii == nlb)
            {
                if (stRow(ii) - stRow(iSt) <= maxRows)
                    iEnd = nlb;
                else
                    iEnd = nlb - 1;
            }
            else
                iEnd = ii - 1;
        #endif 
            return iEnd; 
        }
        // lpanelGPU_t::lpanelGPU_t(lpanel_t& lpanel);
        // int check(lpanel_t& lpanel);
    
};

class upanelGPU_t 
{
public:
    int_t *index;
    double *val;
    // upanelGPU_t* upanelGPU;

    // upanelGPU_t(int_t *usub, double *uval);
    CUDA_CALLABLE
    upanelGPU_t(int_t k, int_t *usub, double *uval, int_t *xsup);

    CUDA_CALLABLE
    upanelGPU_t()
    {
        index = NULL;
        val = NULL;
    }
    // classstructing from recevied index and val 
    CUDA_CALLABLE
    upanelGPU_t(int_t *index_, double *val_): index(index_), val(val_) {return;};
    // index[0] is number of blocks
    CUDA_CALLABLE
    int_t nblocks()
    {
        return index[0];
    }
    // number of rows
    CUDA_CALLABLE
    int_t nzcols() { return index[1]; }

    CUDA_CALLABLE
    int_t LDA() { return index[2]; } // is also supersize of that coloumn

    // global block id of k-th block in the panel
    CUDA_CALLABLE
    int_t gid(int_t k)
    {
        return index[UPANEL_HEADER_SIZE + k];
    }

    // number of rows in the k-th block
    CUDA_CALLABLE
    int_t nbcol(int_t k)
    {
        return index[UPANEL_HEADER_SIZE + nblocks() + k + 1] - index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    // row
    CUDA_CALLABLE
    int_t *colList(int_t k)
    {
        // UPANEL_HEADER
        // nblocks() : blocks list
        // nblocks()+1 : blocks st_points
        // index[UPANEL_HEADER_SIZE + nblocks() + k] statrting of the block
        return &index[UPANEL_HEADER_SIZE +
                      2 * nblocks() + 1 + index[UPANEL_HEADER_SIZE + nblocks() + k]];
    }

    CUDA_CALLABLE
    double *blkPtr(int_t k)
    {
        return &val[LDA() * index[UPANEL_HEADER_SIZE + nblocks() + k]];
    }

    CUDA_CALLABLE
    size_t blkPtrOffset(int_t k)
    {
        return LDA() * index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    // for U panel
    // int_t packed2skyline(int_t* usub, double* uval );
    // int_t packed2skyline(int_t k, int_t *usub, double *uval, int_t *xsup);
    // int_t panelSolve(int_t ksupsz, double *DiagBlk, int_t LDD);
    // int_t diagFactor(int_t k, double *UBlk, int_t LDU, double thresh, int_t *xsup,
    //                  superlu_dist_options_t *options,
    //                  SuperLUStat_t *stat, int *info);

    // double* blkPtr(int_t k);
    // int_t LDA();
    DEVICE_CALLABLE
    int_t find(int_t k);
    CUDA_CALLABLE
    int_t isEmpty() { return index == NULL; }
    CUDA_CALLABLE
    int_t nzvalSize()
    {
        if (index == NULL)
            return 0;
        return LDA() * nzcols();
    }

    CUDA_CALLABLE
    int_t indexSize()
    {
        if (index == NULL)
            return 0;
        return UPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzcols();
    }

    CUDA_CALLABLE
    int_t stCol(int k)
    {
        return index[UPANEL_HEADER_SIZE + nblocks() + k];
    } 

    // Taken from the upanel
    CUDA_CALLABLE
    int getEndBlock(int iSt, int maxCols)
    {
        int nlb = nblocks();
        if(iSt >= nlb )
            return nlb; 
        int iEnd = iSt; 
        int ii = iSt +1;

        while (
            stCol(ii) - stCol(iSt) <= maxCols &&
            ii < nlb)
            ii++;

    #if 1
        if (stCol(ii) - stCol(iSt) > maxCols)
            iEnd = ii-1;
        else 
            iEnd =ii; 
    #else 
        if (ii == nlb)
        {
            if (stCol(ii) - stCol(iSt) <= maxCols)
                iEnd = nlb;
            else
                iEnd = nlb - 1;
        }
        else
            iEnd = ii - 1;
    #endif 
        return iEnd; 
    }
};

#if 0
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Wajih: Device and host memory used to store marshalled batch data for LU and TRSM
struct LUMarshallData 
{
    LUMarshallData();
    ~LUMarshallData();

    // Diagonal device pointer data 
    double **dev_diag_ptrs;
    int *dev_diag_ld_array, *dev_diag_dim_array, *dev_info_array;
    
    // TRSM panel device pointer data 
    double **dev_panel_ptrs;
    int *dev_panel_ld_array, *dev_panel_dim_array;

    // Max of marshalled device data 
    int max_panel, max_diag;

    // Number of marshalled operations
    int batchsize;

    // Data accumulated on the host
    std::vector<double*> host_diag_ptrs;
    std::vector<int> host_diag_ld_array, host_diag_dim_array;
    
    std::vector<double*> host_panel_ptrs;
    std::vector<int> host_panel_ld_array, host_panel_dim_array;
    
    void setBatchSize(int batch_size);
    void setMaxDiag();
    void setMaxPanel();
};

// Wajih: Device and host memory used to store marshalled batch data for Schur complement update 
struct SCUMarshallData 
{
    SCUMarshallData();
    ~SCUMarshallData();

    // GEMM device pointer data 
    double **dev_A_ptrs, **dev_B_ptrs, **dev_C_ptrs;
    int *dev_lda_array, *dev_ldb_array, *dev_ldc_array;
    int *dev_m_array, *dev_n_array, *dev_k_array;

    // Panel device pointer data and scu loop limits 
    lpanelGPU_t* dev_gpu_lpanels;
    upanelGPU_t* dev_gpu_upanels;
    int* dev_ist, *dev_iend, *dev_jst, *dev_jend;
    int *dev_maxGemmRows, *dev_maxGemmCols;
    
    // Max of marshalled gemm device data 
    int max_m, max_n, max_k;    
    
    // Max of marshalled loop limits  
    int max_ilen, max_jlen;

    // Number of marshalled operations
    int batchsize;

    // Data accumulated on the host
    std::vector<double*> host_A_ptrs, host_B_ptrs, host_C_ptrs;
    std::vector<int> host_lda_array, host_ldb_array, host_ldc_array;
    std::vector<int> host_m_array, host_n_array, host_k_array;

    // Host data initialized once per level 
    std::vector<upanel_t> upanels;
    std::vector<lpanel_t> lpanels;
    std::vector<upanelGPU_t> host_gpu_upanels;
    std::vector<lpanelGPU_t> host_gpu_lpanels;
    std::vector<int> ist, iend, jst, jend, maxGemmRows, maxGemmCols;

    void setBatchSize(int batch_size);
    void setMaxDims();
    void copyPanelDataToGPU();
    void copyToGPU();
};

#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_CUDA_STREAMS 64 
struct LUstructGPU_t
{
    // all pointers are device pointers 

    upanelGPU_t* uPanelVec;
    lpanelGPU_t* lPanelVec; 
    int_t* xsup; 
    int Pr, Pc, Pd;
    
    size_t gemmBufferSize; 
    int numCudaStreams;     
    int maxSuperSize;
    // double arrays are problematic 
    cudaStream_t cuStreams[MAX_CUDA_STREAMS];
    cublasHandle_t cuHandles[MAX_CUDA_STREAMS];
    
    // Magma is needed for non-uniform batched execution 
#ifdef HAVE_MAGMA
    magma_queue_t magma_queue;
#endif

#if 0
    LUMarshallData marshall_data;
    SCUMarshallData sc_marshall_data;
#endif

    int* dperm_c_supno;

    /* Sherry: Allocate an array of buffers for the diagonal blocks
       on the leaf level.
       The sizes are uniform: ldt is the maximum among all the nodes.    */
    //    double* dFBufs[MAX_CUDA_STREAMS];
    // double* gpuGemmBuffs[MAX_CUDA_STREAMS];
    double **dFBufs;       
    double ** gpuGemmBuffs;

    // GPU accessible array of gemm buffers 
    double** dgpuGemmBuffs;
    
    double* LvalRecvBufs[MAX_CUDA_STREAMS];
    double* UvalRecvBufs[MAX_CUDA_STREAMS];
    int_t* LidxRecvBufs[MAX_CUDA_STREAMS];
    int_t* UidxRecvBufs[MAX_CUDA_STREAMS];

    cusolverDnHandle_t cuSolveHandles[MAX_CUDA_STREAMS];
    double* diagFactWork[MAX_CUDA_STREAMS];
    int* diagFactInfo[MAX_CUDA_STREAMS]; // CPU pointers
    /*data structure for lookahead Update */
    cublasHandle_t lookAheadLHandle[MAX_CUDA_STREAMS];
    cudaStream_t lookAheadLStream[MAX_CUDA_STREAMS];

    double *lookAheadLGemmBuffer[MAX_CUDA_STREAMS];

    cublasHandle_t lookAheadUHandle[MAX_CUDA_STREAMS];
    cudaStream_t lookAheadUStream[MAX_CUDA_STREAMS];

    double *lookAheadUGemmBuffer[MAX_CUDA_STREAMS];
    
    __device__
    int_t supersize(int_t k) { return xsup[k + 1] - xsup[k]; }
    __device__
    int_t g2lRow(int_t k) { return k / Pr; }
    __device__
    int_t g2lCol(int_t k) { return k / Pc; }
    
};/* LUstructGPU_t{} */

void scatterGPU_driver(
    int iSt, int iEnd, int jSt, int jEnd, double *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt, lpanelGPU_t lpanel, upanelGPU_t upanel, 
    LUstructGPU_t *dA, cudaStream_t cuStream
);

void scatterGPU_batchDriver(
    int* iSt_batch, int *iEnd_batch, int *jSt_batch, int *jEnd_batch, 
    int max_ilen, int max_jlen, double **gemmBuff_ptrs, int *LDgemmBuff_batch, 
    int maxSuperSize, int ldt, lpanelGPU_t *lpanels, upanelGPU_t *upanels, 
    LUstructGPU_t *dA, int batchCount, cudaStream_t cuStream
);