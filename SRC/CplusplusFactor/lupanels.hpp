#pragma once
#include <vector>
#include <iostream>
#include "superlu_ddefs.h"
#include "lu_common.hpp"

#ifdef HAVE_CUDA
#include "lupanels_GPU.cuh"
#include "gpuCommon.hpp"
#endif

#include "commWrapper.hpp"
#include "anc25d.hpp"
//class lpanelGPU_t;
//class upanelGPU_t;
#define GLOBAL_BLOCK_NOT_FOUND -1
// it can be templatized for double and complex double
class lpanel_t
{
public:
    int_t *index;
    double *val;
    // ifdef GPU acceraleration

#ifdef HAVE_CUDA
    lpanelGPU_t gpuPanel;
#endif
    // bool isDiagIncluded;

    lpanel_t(int_t k, int_t *lsub, double *nzval, int_t *xsup, int_t isDiagIncluded);
  
    // default constuctor
#ifdef HAVE_CUDA
    lpanel_t() : gpuPanel(NULL, NULL)
    {
        index = NULL;
        val = NULL;
    }
#else
    lpanel_t()
    {
        index = NULL;
        val = NULL;
    }
#endif

    lpanel_t(int_t *index_, double *val_) : index(index_), val(val_) { return; };

    // index[0] is number of blocks
    int_t nblocks()
    {
        return index[0];
    }
    // number of rows
    int_t nzrows() { return index[1]; }
    int_t haveDiag() { return index[2]; }
    int_t ncols() { return index[3]; }

    // global block id of k-th block in the panel
    int_t gid(int_t k)
    {
        return index[LPANEL_HEADER_SIZE + k];
    }

    // number of rows in the k-th block
    int_t nbrow(int_t k)
    {
        return index[LPANEL_HEADER_SIZE + nblocks() + k + 1] - index[LPANEL_HEADER_SIZE + nblocks() + k];
    }

    //
    int_t stRow(int k)
    {
        return index[LPANEL_HEADER_SIZE + nblocks() + k];
    }
    // row
    int_t *rowList(int_t k)
    {
        // LPANEL_HEADER
        // nblocks() : blocks list
        // nblocks()+1 : blocks st_points
        // index[LPANEL_HEADER_SIZE + nblocks() + k] statrting of the block
        return &index[LPANEL_HEADER_SIZE +
                      2 * nblocks() + 1 + index[LPANEL_HEADER_SIZE + nblocks() + k]];
    }

    double *blkPtr(int_t k)
    {
        return &val[index[LPANEL_HEADER_SIZE + nblocks() + k]];
    }

    size_t blkPtrOffset(int_t k)
    {
        return index[LPANEL_HEADER_SIZE + nblocks() + k];
    }

    int_t LDA() { return index[1]; }
    int_t find(int_t k);
    // for L panel I don't need any special transformation function
    int_t panelSolve(int_t ksupsz, double *DiagBlk, int_t LDD);
    int_t diagFactor(int_t k, double *UBlk, int_t LDU, double thresh, int_t *xsup,
                     superlu_dist_options_t *options, SuperLUStat_t *stat, int *info);
    int_t packDiagBlock(double *DiagLBlk, int_t LDD);
    int_t isEmpty() { return index == NULL; }
    int_t nzvalSize()
    {
        if (index == NULL)
            return 0;
        return ncols() * nzrows();
    }

    int_t indexSize()
    {
        if (index == NULL)
            return 0;
        return LPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzrows();
    }

    size_t totalSize()
    {
        return sizeof(int_t)*indexSize() + sizeof(double)*nzvalSize();
    }

    // return the maximal iEnd such that stRow(iEnd)-stRow(iSt) < maxRow;
    int getEndBlock(int iSt, int maxRows);

    // ~lpanel_t()
    // {
    //     SUPERLU_FREE(index);    
    //     // SUPERLU_FREE(val);    
    // }

#ifdef HAVE_CUDA
    lpanelGPU_t copyToGPU();
    lpanelGPU_t copyToGPU(void *basePtr); // when we are doing a single allocation
    int checkGPU();
    int copyBackToGPU();

    int_t panelSolveGPU(cublasHandle_t handle, cudaStream_t cuStream,
                        int_t ksupsz,
                        double *DiagBlk, // device pointer
                        int_t LDD);

    int_t diagFactorPackDiagBlockGPU(int_t k,
                                     double *UBlk, int_t LDU,     // CPU pointers
                                     double *DiagLBlk, int_t LDD, // CPU pointers
                                     double thresh, int_t *xsup,
                                     superlu_dist_options_t *options,
                                     SuperLUStat_t *stat, int *info);
    int_t diagFactorCuSolver(int_t k,
                                     cusolverDnHandle_t cusolverH, cudaStream_t cuStream,
                                    double *dWork, int* dInfo,  // GPU pointers
                                    double *dDiagBuf, int_t LDD, // GPU pointers
                                    double thresh, int_t *xsup,
                                    superlu_dist_options_t *options,
                                    SuperLUStat_t *stat, int *info);

    double *blkPtrGPU(int k)
    {
        return &gpuPanel.val[blkPtrOffset(k)];
    }

    lpanel_t(int_t *index_, double *val_, int_t *indexGPU, double *valGPU) : index(index_), val(val_), gpuPanel(indexGPU, valGPU)
    {
        return;
    };
    int_t copyFromGPU();
#endif
};

class upanel_t
{
public:
    int_t *index;
    double *val;
#ifdef HAVE_CUDA
    // upanelGPU_t* upanelGPU;
    upanelGPU_t gpuPanel;
#endif

    // upanel_t(int_t *usub, double *uval);
    upanel_t(int_t k, int_t *usub, double *uval, int_t *xsup);
#ifdef HAVE_CUDA
    upanel_t() : gpuPanel(NULL, NULL)
    {
        index = NULL;
        val = NULL;
    }
#else
    upanel_t()
    {
        index = NULL;
        val = NULL;
    }
#endif
    // constructing from recevied index and val
    upanel_t(int_t *index_, double *val_) : index(index_), val(val_) { return; };
    // index[0] is number of blocks
    int_t nblocks()
    {
        return index[0];
    }
    // number of rows
    int_t nzcols() {
      if (index == NULL) return 0; /* Sherry added this check. 2/22/2023 */
      return index[1];
    }
    int_t LDA() { return index[2]; } // is also supersize of that coloumn

    // global block id of k-th block in the panel
    int_t gid(int_t k)
    {
        return index[UPANEL_HEADER_SIZE + k];
    }

    // number of rows in the k-th block
    int_t nbcol(int_t k)
    {
        return index[UPANEL_HEADER_SIZE + nblocks() + k + 1] - index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    // row
    int_t *colList(int_t k)
    {
        // UPANEL_HEADER
        // nblocks() : blocks list
        // nblocks()+1 : blocks st_points
        // index[UPANEL_HEADER_SIZE + nblocks() + k] statrting of the block
        return &index[UPANEL_HEADER_SIZE +
                      2 * nblocks() + 1 + index[UPANEL_HEADER_SIZE + nblocks() + k]];
    }

    double *blkPtr(int_t k)
    {
        return &val[LDA() * index[UPANEL_HEADER_SIZE + nblocks() + k]];
    }

    size_t blkPtrOffset(int_t k)
    {
        return LDA() * index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    // for U panel
    // int_t packed2skyline(int_t* usub, double* uval );
    int_t packed2skyline(int_t k, int_t *usub, double *uval, int_t *xsup);
    int_t panelSolve(int_t ksupsz, double *DiagBlk, int_t LDD);
    int_t diagFactor(int_t k, double *UBlk, int_t LDU, double thresh, int_t *xsup,
                     superlu_dist_options_t *options,
                     SuperLUStat_t *stat, int *info);

    // double* blkPtr(int_t k);
    // int_t LDA();
    int_t find(int_t k);
    int_t isEmpty() { return index == NULL; }
    int_t nzvalSize()
    {
        if (index == NULL)
            return 0;
        return LDA() * nzcols();
    }

    int_t indexSize()
    {
        if (index == NULL)
            return 0;
        return UPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzcols();
    }
    size_t totalSize()
    {
        return sizeof(int_t)*indexSize() + sizeof(double)*nzvalSize();
    }
    int_t checkCorrectness()
    {
        if (index == NULL)
        {
            std::cout << "## Warning: Empty Panel"
                      << "\n";
            return 0;
        }
        int_t alternateNzcols = index[UPANEL_HEADER_SIZE + 2 * nblocks()];
        // std::cout<<nblocks()<<"  nzcols "<<nzcols()<<" alternate nzcols "<< alternateNzcols << "\n";
        if (nzcols() != alternateNzcols)
        {
            printf("Error 175\n");
            exit(-1);
        }

        return UPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzcols();
    }

    int_t stCol(int k)
    {
        return index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    int getEndBlock(int jSt, int maxCols);

    // ~upanel_t()
    // {
    //     SUPERLU_FREE(index);    
    //     SUPERLU_FREE(val);    
    // }


#ifdef HAVE_CUDA
    upanelGPU_t copyToGPU();
    //TODO: implement with baseptr
    upanelGPU_t copyToGPU(void *basePtr);
    int copyBackToGPU();

    int_t panelSolveGPU(cublasHandle_t handle, cudaStream_t cuStream,
                        int_t ksupsz, double *DiagBlk, int_t LDD);
    int checkGPU();

    double *blkPtrGPU(int k)
    {
        return &gpuPanel.val[blkPtrOffset(k)];
    }

    upanel_t(int_t *index_, double *val_, int_t *indexGPU, double *valGPU) : index(index_), val(val_), gpuPanel(indexGPU, valGPU)
    {
        return;
    };
    int_t copyFromGPU();
#endif
};

// Defineing GPU data types
//lapenGPU_t has exact same structure has lapanel_t but
// the pointers are on GPU

struct LUstruct_v100
{
    lpanel_t *lPanelVec;
    upanel_t *uPanelVec;
    gridinfo3d_t *grid3d;
    gridinfo_t *grid;
    int_t iam, Pc, Pr, myrow, mycol, ldt;
    int_t *xsup;
    int_t nsupers;
    // variables for scattering ldt*THREAD_Size
    int nThreads;
    int_t *indirect, *indirectRow, *indirectCol;
    double *bigV; // size = THREAD_Size*ldt*ldt
    int *isNodeInMyGrid;
    double thresh;
    int *info;
    //TODO: get it from environment
    int numDiagBufs = 32;  /* Sherry: not fixed yet */

    // Add SCT_t here
    SCT_t *SCT;
    superlu_dist_options_t *options;
    SuperLUStat_t *stat;


    // Adding more variables for factorization 
    dtrf3Dpartition_t *trf3Dpartition;
    int_t maxLvl;
    int maxLeafNodes; /* Sherry added 12/31/22. Computed in LUstruct_v100 constructor */
    
    ddiagFactBufs_t **dFBufs; /* stores L and U diagonal blocks */
    int superlu_acc_offload;
    // myNodeCount,
    // treePerm
    // myZeroTrIdxs
    // sForests
    // myTreeIdxs
    // gEtreeInfo

    // buffers for communication
    int_t maxLvalCount = 0;
    int_t maxLidxCount = 0;
    int_t maxUvalCount = 0;
    int_t maxUidxCount = 0;
    std::vector<double *> diagFactBufs; /* stores diagonal blocks, 
					   each one is a normal dense matrix.
					Sherry: where are they free'd ?? */
    std::vector<double *> LvalRecvBufs;
    std::vector<double *> UvalRecvBufs;
    std::vector<int_t *> LidxRecvBufs;
    std::vector<int_t *> UidxRecvBufs;

    // send and recv count for 2d comm
    std::vector<int_t> LvalSendCounts;
    std::vector<int_t> UvalSendCounts;
    std::vector<int_t> LidxSendCounts;
    std::vector<int_t> UidxSendCounts;

    //
    std::vector<bcastStruct> bcastDiagRow;
    std::vector<bcastStruct> bcastDiagCol;
    std::vector<bcastStruct> bcastLval;
    std::vector<bcastStruct> bcastUval;
    std::vector<bcastStruct> bcastLidx;
    std::vector<bcastStruct> bcastUidx;

    int_t krow(int_t k) { return k % Pr; }
    int_t kcol(int_t k) { return k % Pc; }
    int_t procIJ(int_t i, int_t j) { return PNUM(krow(i), kcol(j), grid); }
    int_t supersize(int_t k) { return xsup[k + 1] - xsup[k]; }
    int_t g2lRow(int_t k) { return k / Pr; }
    int_t g2lCol(int_t k) { return k / Pc; }

    anc25d_t anc25d;
    
    // For GPU acceleration
    LUstructGPU_t *dA_gpu; // pointing to memory on GPU
    LUstructGPU_t A_gpu;   // pointing to memory accessible on CPU

    /////////////////////////////////////////////////////////////////
    // Intermediate for flat batched 
    /////////////////////////////////////////////////////////////////
    dLocalLU_t* host_Llu;
    dLocalLU_t d_localLU;

    int *d_lblock_gid_dat, **d_lblock_gid_ptrs;
    int *d_lblock_start_dat, **d_lblock_start_ptrs;
    int64_t *d_lblock_gid_offsets, *d_lblock_start_offsets;
    int64_t total_l_blocks, total_start_size;

    void computeLBlockData();
    /////////////////////////////////////////////////////////////////

    enum indirectMapType
    {
        ROW_MAP,
        COL_MAP
    };

    /**
    *          C O N / D E S - T R U C T O R S
    */
    LUstruct_v100(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition, 
                  dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                  SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                  double thresh_, int *info_);

    ~LUstruct_v100()
    {

    /* Yang: Deallocate the lPanelVec[i] and uPanelVec[i] here instead of using destructors ~lpanel_t or ~upanel_t, 
    as lpanel_t/upanel_t is used for holding temporary communication buffers as well. Note that lPanelVec[i].val is not deallocated here as it's pointing to the L data in the C code*/
        
        for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
            if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1){
                if(lPanelVec[i].index)
                SUPERLU_FREE(lPanelVec[i].index);
                // SUPERLU_FREE(lPanelVec[i].val);
            }

        for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
            if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1){
                if(uPanelVec[i].index)
                SUPERLU_FREE(uPanelVec[i].index);
                if(uPanelVec[i].val)
                SUPERLU_FREE(uPanelVec[i].val);
            }
    
        delete[] lPanelVec;
        delete[] uPanelVec;


	/* free diagonal L and U blocks */
	dfreeDiagFactBufsArr(maxLeafNodes, dFBufs);

	SUPERLU_FREE(bigV);
	SUPERLU_FREE(indirect);
	SUPERLU_FREE(indirectRow);
	SUPERLU_FREE(indirectCol);

	int i;
	for (i = 0; i < options->num_lookaheads; i++) {
	    SUPERLU_FREE(LvalRecvBufs[i]);
	    SUPERLU_FREE(UvalRecvBufs[i]);
	    SUPERLU_FREE(LidxRecvBufs[i]);
	    SUPERLU_FREE(UidxRecvBufs[i]);
	}
	
	for (i = 0; i < numDiagBufs; i++) SUPERLU_FREE(diagFactBufs[i]);

	/* Sherry added the following, which comes from batch setup */
	superlu_acc_offload = sp_ienv_dist(10, options); //get_acc_offload();    
    if (superlu_acc_offload){
	//printf(".. free batch buffers\n"); fflush(stdout);
	SUPERLU_FREE(A_gpu.dFBufs);
	SUPERLU_FREE(A_gpu.gpuGemmBuffs);

    for (int stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {
        cusolverDnDestroy(A_gpu.cuSolveHandles[stream]);
        cublasDestroy(A_gpu.cuHandles[stream]);
        cublasDestroy(A_gpu.lookAheadLHandle[stream]);
        cublasDestroy(A_gpu.lookAheadUHandle[stream]);
    }


    }

    SUPERLU_FREE(isNodeInMyGrid);

    } /* end destructor LUstruct_v100 */

  
    /**
    *           Compute Functions
    */
   int_t pdgstrf3d();
    int_t dSchurComplementUpdate(int_t k, lpanel_t &lpanel, upanel_t &upanel);
    int_t *computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                              int_t dstLen, int_t *dstVec);

    int_t dScatter(int_t m, int_t n,
                   int_t gi, int_t gj,
                   double *V, int_t ldv,
                   int_t *srcRowList, int_t *srcColList);

    int_t lookAheadUpdate(
        int_t k, int_t laIdx, lpanel_t &lpanel, upanel_t &upanel);
    int_t blockUpdate(int_t k,
                      int_t ii, int_t jj, lpanel_t &lpanel, upanel_t &upanel);
    int_t dSchurCompUpdateExcludeOne(
        int_t k, int_t ex, // suypernodes to be excluded
        lpanel_t &lpanel, upanel_t &upanel);

    int_t dsparseTreeFactor(
        sForest_t *sforest,
        ddiagFactBufs_t **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);
  
    int dsparseTreeFactorBatchGPU(
				  sForest_t *sforest,
				  ddiagFactBufs_t **dFBufs, // size maxEtree level
				  gEtreeInfo_t *gEtreeInfo, // global etree info
				  int tag_ub);

    // Helper routine to marshall batch LU data into the device data in A_gpu 
    void marshallBatchedLUData(int k_st, int k_end, int_t *perm_c_supno);
    void marshallBatchedBufferCopyData(int k_st, int k_end, int_t *perm_c_supno);
    void marshallBatchedTRSMUData(int k_st, int k_end, int_t *perm_c_supno);
    void marshallBatchedTRSMLData(int k_st, int k_end, int_t *perm_c_supno);
    void marshallBatchedSCUData(int k_st, int k_end, int_t *perm_c_supno);
    void initSCUMarshallData(int k_st, int k_end, int_t *perm_c_supno);
    int marshallSCUBatchedDataInner(int k_st, int k_end, int_t *perm_c_supno);
    int marshallSCUBatchedDataOuter(int k_st, int k_end, int_t *perm_c_supno);
    void dFactBatchSolve(int k_st, int k_end, int_t *perm_c_supno);

    //
    int_t dDiagFactorPanelSolve(int_t k, int_t offset, ddiagFactBufs_t **dFBufs);
    int_t dPanelBcast(int_t k, int_t offset);
    int_t dsparseTreeFactorBaseline(
        sForest_t *sforest,
        ddiagFactBufs_t **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    int_t packedU2skyline(dLUstruct_t *LUstruct);

    int_t ancestorReduction3d(int_t ilvl, int_t *myNodeCount,
                              int_t **treePerm);

    int_t zSendLPanel(int_t k0, int_t receiverGrid);
    int_t zRecvLPanel(int_t k0, int_t senderGrid, double alpha, double beta);
    int_t zSendUPanel(int_t k0, int_t receiverGrid);
    int_t zRecvUPanel(int_t k0, int_t senderGrid, double alpha, double beta);

    int_t dAncestorFactorBaseline(
        int_t alvl,
        sForest_t *sforest,
        ddiagFactBufs_t **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);



    // GPU related functions
#ifdef HAVE_CUDA
    int_t setLUstruct_GPU();
    int_t dsparseTreeFactorGPU(
        sForest_t *sforest,
        ddiagFactBufs_t **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);
    int_t dsparseTreeFactorGPUBaseline(
        sForest_t *sforest,
        ddiagFactBufs_t **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    int_t dAncestorFactorBaselineGPU(
        int_t alvl,
        sForest_t *sforest,
        ddiagFactBufs_t **dFBufs, // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        int tag_ub);

    int_t dSchurComplementUpdateGPU(
        int streamId,
        int_t k, lpanel_t &lpanel, upanel_t &upanel);
    int_t dSchurCompUpdatePartGPU(
        int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
        int_t k, lpanel_t &lpanel, upanel_t &upanel,
        cublasHandle_t handle, cudaStream_t cuStream,
        double *gemmBuff);
    int_t lookAheadUpdateGPU(
        int streamId,
        int_t k, int_t laIdx, lpanel_t &lpanel, upanel_t &upanel);
    int_t dSchurCompUpLimitedMem(
        int streamId,
        int_t lStart, int_t lEnd,
        int_t uStart, int_t uEnd,
        int_t k, lpanel_t &lpanel, upanel_t &upanel);
    int_t dSchurCompUpdateExcludeOneGPU(
        int streamId,
        int_t k, int_t ex, // suypernodes to be excluded
        lpanel_t &lpanel, upanel_t &upanel);

    int_t dDiagFactorPanelSolveGPU(int_t k, int_t offset, ddiagFactBufs_t **dFBufs);
    int_t dPanelBcastGPU(int_t k, int_t offset);

    int_t ancestorReduction3dGPU(int_t ilvl, int_t *myNodeCount,
                                 int_t **treePerm);
    int_t zSendLPanelGPU(int_t k0, int_t receiverGrid);
    int_t zRecvLPanelGPU(int_t k0, int_t senderGrid, double alpha, double beta);
    int_t zSendUPanelGPU(int_t k0, int_t receiverGrid);
    int_t zRecvUPanelGPU(int_t k0, int_t senderGrid, double alpha, double beta);
    int_t copyLUGPUtoHost();
    int_t copyLUHosttoGPU();
    int_t checkGPU();

    // some more helper functions
    upanel_t getKUpanel(int_t k, int_t offset);
    lpanel_t getKLpanel(int_t k, int_t offset);
    int_t SyncLookAheadUpdate(int streamId);

    double *gpuLvalBasePtr, *gpuUvalBasePtr;
    int_t *gpuLidxBasePtr, *gpuUidxBasePtr;
    size_t gpuLvalSize, gpuUvalSize, gpuLidxSize, gpuUidxSize;

    lpanelGPU_t* copyLpanelsToGPU();
    upanelGPU_t* copyUpanelsToGPU();

    // to perform diagFactOn GPU
    int_t dDFactPSolveGPU(int_t k, int_t offset, ddiagFactBufs_t **dFBufs);
    int_t dDFactPSolveGPU(int_t k, int_t handle_offset, int buffer_offset, ddiagFactBufs_t **dFBufs);
#endif
};

