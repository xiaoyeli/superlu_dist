
#pragma once
#include <vector>
#include <iostream>

#include "superlu_ddefs.h"

#ifdef HAVE_CUDA
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
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
// class lpanel_t;
// class upanel_t;

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
        CUDA_CALLABLE
        int getEndBlock(int iSt, int maxRows);
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
};


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
    double* gpuGemmBuffs[MAX_CUDA_STREAMS];
    double* dFBufs[MAX_CUDA_STREAMS];  
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


