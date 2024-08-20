
#pragma once
#include <vector>
#include <iostream>
#include "superlu_ddefs.h"

#ifdef HAVE_CUDA
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
#endif

#include "lu_common.hpp"
// #include "lupanels.hpp"  //unneeded?
#include "lupanels_GPU.cuh" 

#ifdef __CUDACC__
#define DEVICE_CALLABLE __device__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#define DEVICE_CALLABLE
#endif

template <typename Ftype>
class xlpanel_t;
template <typename Ftype>
class xupanel_t;

template <typename Ftype>
class xlpanelGPU_t 
{
    
    public:
        int_t *index;
        Ftype *val;
        // bool isDiagIncluded;
        CUDA_CALLABLE
        xlpanelGPU_t(int_t k, int_t *lsub, Ftype *nzval, int_t *xsup, int_t isDiagIncluded);
        // default constuctor
        
        CUDA_CALLABLE
        xlpanelGPU_t()
        {
            index = NULL;
            val = NULL;
        }
        CUDA_CALLABLE
        xlpanelGPU_t(int_t *index_, Ftype *val_): index(index_), val(val_) {return;};
        
    
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
        Ftype *blkPtr(int_t k)
        {
            return &val[index[LPANEL_HEADER_SIZE + nblocks() + k]];
        }
    
        CUDA_CALLABLE
        int_t LDA() { return index[1]; }

        DEVICE_CALLABLE
        int_t find(int_t k);
        // // for L panel I don't need any special transformation function
        // int_t panelSolve(int_t ksupsz, Ftype *DiagBlk, int_t LDD);
        // int_t diagFactor(int_t k, Ftype *UBlk, int_t LDU, Ftype thresh, int_t *xsup,
        //                  superlu_dist_options_t *options, SuperLUStat_t *stat, int *info);
        // int_t packDiagBlock(Ftype *DiagLBlk, int_t LDD);

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
        // xlpanelGPU_t::lpanelGPU_t(lpanel_t& lpanel);
        // int check(lpanel_t& lpanel);
    
};

template <typename Ftype>
class xupanelGPU_t 
{
public:
    int_t *index;
    Ftype *val;
    // xupanelGPU_t* upanelGPU;

    // xupanelGPU_t(int_t *usub, Ftype *uval);
    CUDA_CALLABLE
    xupanelGPU_t(int_t k, int_t *usub, Ftype *uval, int_t *xsup);

    CUDA_CALLABLE
    xupanelGPU_t()
    {
        index = NULL;
        val = NULL;
    }
    // classstructing from recevied index and val 
    CUDA_CALLABLE
    xupanelGPU_t(int_t *index_, Ftype *val_): index(index_), val(val_) {return;};
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
    Ftype *blkPtr(int_t k)
    {
        return &val[LDA() * index[UPANEL_HEADER_SIZE + nblocks() + k]];
    }

    CUDA_CALLABLE
    size_t blkPtrOffset(int_t k)
    {
        return LDA() * index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    // for U panel
    // int_t packed2skyline(int_t* usub, Ftype* uval );
    // int_t packed2skyline(int_t k, int_t *usub, Ftype *uval, int_t *xsup);
    // int_t panelSolve(int_t ksupsz, Ftype *DiagBlk, int_t LDD);
    // int_t diagFactor(int_t k, Ftype *UBlk, int_t LDU, Ftype thresh, int_t *xsup,
    //                  superlu_dist_options_t *options,
    //                  SuperLUStat_t *stat, int *info);

    // Ftype* blkPtr(int_t k);
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

#define MAX_CUDA_STREAMS 64
template <typename Ftype> 
struct xLUstructGPU_t
{
    // all pointers are device pointers 

    xupanelGPU_t<Ftype>* uPanelVec;
    xlpanelGPU_t<Ftype>* lPanelVec; 
    int_t* xsup; 
    int Pr, Pc, Pd;
    
    size_t gemmBufferSize; 
    int numCudaStreams;     
    int maxSuperSize;
    // Ftype arrays are problematic 
    cudaStream_t cuStreams[MAX_CUDA_STREAMS];
    cublasHandle_t cuHandles[MAX_CUDA_STREAMS];
    
    int* dperm_c_supno;

    /* Sherry: Allocate an array of buffers for the diagonal blocks
       on the leaf level.
       The sizes are uniform: ldt is the maximum among all the nodes.    */
    //    Ftype* dFBufs[MAX_CUDA_STREAMS];
    // Ftype* gpuGemmBuffs[MAX_CUDA_STREAMS];
    Ftype **dFBufs;       
    Ftype ** gpuGemmBuffs;

    // GPU accessible array of gemm buffers 
    Ftype** dgpuGemmBuffs;
    
    Ftype* LvalRecvBufs[MAX_CUDA_STREAMS];
    Ftype* UvalRecvBufs[MAX_CUDA_STREAMS];
    int_t* LidxRecvBufs[MAX_CUDA_STREAMS];
    int_t* UidxRecvBufs[MAX_CUDA_STREAMS];

    cusolverDnHandle_t cuSolveHandles[MAX_CUDA_STREAMS];
    Ftype* diagFactWork[MAX_CUDA_STREAMS];
    int* diagFactInfo[MAX_CUDA_STREAMS]; // CPU pointers
    /*data structure for lookahead Update */
    cublasHandle_t lookAheadLHandle[MAX_CUDA_STREAMS];
    cudaStream_t lookAheadLStream[MAX_CUDA_STREAMS];

    Ftype *lookAheadLGemmBuffer[MAX_CUDA_STREAMS];

    cublasHandle_t lookAheadUHandle[MAX_CUDA_STREAMS];
    cudaStream_t lookAheadUStream[MAX_CUDA_STREAMS];

    Ftype *lookAheadUGemmBuffer[MAX_CUDA_STREAMS];
    
    __device__
    int_t supersize(int_t k) { return xsup[k + 1] - xsup[k]; }
    __device__
    int_t g2lRow(int_t k) { return k / Pr; }
    __device__
    int_t g2lCol(int_t k) { return k / Pc; }
    
};/* xLUstructGPU_t{} */

template <typename Ftype>
void scatterGPU_driver(
    int iSt, int iEnd, int jSt, int jEnd, Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt, xlpanelGPU_t<Ftype> lpanel, xupanelGPU_t<Ftype> upanel, 
    xLUstructGPU_t<Ftype> *dA, cudaStream_t cuStream
);

template <typename Ftype>
void scatterGPU_batchDriver(
    int* iSt_batch, int *iEnd_batch, int *jSt_batch, int *jEnd_batch, 
    int max_ilen, int max_jlen, Ftype **gemmBuff_ptrs, int *LDgemmBuff_batch, 
    int maxSuperSize, int ldt, xlpanelGPU_t<Ftype> *lpanels, xupanelGPU_t<Ftype> *upanels, 
    xLUstructGPU_t<Ftype> *dA, int batchCount, cudaStream_t cuStream
);
