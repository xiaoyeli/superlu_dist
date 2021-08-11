#pragma once
#include <vector>
#include <iostream>
#include "superlu_ddefs.h"

class lpanelGPU_t;
class upanelGPU_t;

#define LPANEL_HEADER_SIZE 4

// it can be templatized for double and complex double
class lpanel_t
{
public:
    int_t *index;
    double *val;
    // ifdef GPU acceraleration 
    lpanelGPU_t* lpanelGPU;
    // bool isDiagIncluded;

    lpanel_t(int_t k, int_t *lsub, double *nzval, int_t *xsup, int_t isDiagIncluded);
    // default constuctor
    lpanel_t()
    {
        index = NULL;
        val = NULL;
    }

    lpanel_t(int_t *index_, double *val_): index(index_), val(val_) {return;};
    

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

    // return the maximal iEnd such that stRow(iEnd)-stRow(iSt) < maxRow;
    int getEndBlock(int iSt, int maxRows);
};

#define UPANEL_HEADER_SIZE 3
class upanel_t
{
public:
    int_t *index;
    double *val;
    upanelGPU_t* upanelGPU;

    // upanel_t(int_t *usub, double *uval);
    upanel_t(int_t k, int_t *usub, double *uval, int_t *xsup);
    upanel_t()
    {
        index = NULL;
        val = NULL;
    }
    // constructing from recevied index and val 
    upanel_t(int_t *index_, double *val_): index(index_), val(val_) {return;};
    // index[0] is number of blocks
    int_t nblocks()
    {
        return index[0];
    }
    // number of rows
    int_t nzcols() { return index[1]; }
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

    int_t checkCorrectness()
    {
        if (index == NULL)
        {
            std::cout<<"## Warning: Empty Panel" << "\n";
            return 0;
        }
        int_t alternateNzcols = index[UPANEL_HEADER_SIZE + 2 * nblocks()] ;
        // std::cout<<nblocks()<<"  nzcols "<<nzcols()<<" alternate nzcols "<< alternateNzcols << "\n";
        if(nzcols()!= alternateNzcols)
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
    int_t *isNodeInMyGrid;

    // Add SCT_t here 
    SCT_t* SCT;
    superlu_dist_options_t *options;
    SuperLUStat_t *stat;

    // buffers for communication 
    int_t maxLvalCount =0;
    int_t maxLidxCount =0;
    int_t maxUvalCount =0;
    int_t maxUidxCount =0;
    std::vector<double*> LvalRecvBufs;
    std::vector<double*> UvalRecvBufs;
    std::vector<int_t*> LidxRecvBufs;
    std::vector<int_t*> UidxRecvBufs;

    // send and recv count for 2d comm 
    std::vector<int_t> LvalSendCounts;
    std::vector<int_t> UvalSendCounts;
    std::vector<int_t> LidxSendCounts;
    std::vector<int_t> UidxSendCounts;

    int_t krow(int_t k) { return k % Pr; }
    int_t kcol(int_t k) { return k % Pc; }
    int_t procIJ(int_t i, int_t j) { return PNUM(krow(i), kcol(j), grid); }
    int_t supersize(int_t k) { return xsup[k + 1] - xsup[k]; }
    int_t g2lRow(int_t k) { return k / Pr; }
    int_t g2lCol(int_t k) { return k / Pc; }

    enum indirectMapType
    {
        ROW_MAP,
        COL_MAP
    };

    /**
    *          C O N / D E S - T R U C T O R S
    */
    LUstruct_v100(int_t nsupers, int_t ldt_, int_t *isNodeInMyGrid,
                  dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                  SCT_t* SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat
                  );

    ~LUstruct_v100()
    {
        delete lPanelVec;
        delete uPanelVec;
    }

    /**
    *           Compute Functions 
    */
    int_t dSchurComplementUpdate(int_t k, lpanel_t &lpanel, upanel_t &upanel);
    int_t *computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                              int_t dstLen, int_t *dstVec);

    int_t dScatter(int_t m, int_t n,
                   int_t gi, int_t gj,
                   double *V, int_t ldv,
                   int_t *srcRowList, int_t *srcColList);

    int_t dsparseTreeFactor(
        sForest_t *sforest,
        commRequests_t **comReqss, // lists of communication requests // size maxEtree level
        dscuBufs_t *scuBufs,        // contains buffers for schur complement update
        packLUInfo_t *packLUInfo,
        msgs_t **msgss,           // size=num Look ahead
        dLUValSubBuf_t **LUvsbs,  // size=num Look ahead
        ddiagFactBufs_t **dFBufs,  // size maxEtree level
        gEtreeInfo_t *gEtreeInfo, // global etree info
        
        int_t *gIperm_c_supno,
        
        double thresh,  int tag_ub,
        int *info);

    int_t packedU2skyline(dLUstruct_t *LUstruct);

    int_t ancestorReduction3d(int_t ilvl, int_t *myNodeCount,
                              int_t **treePerm);

    int_t zSendLPanel(int_t k0, int_t receiverGrid);
    int_t zRecvLPanel(int_t k0, int_t senderGrid, double alpha, double beta);
    int_t zSendUPanel(int_t k0, int_t receiverGrid);
    int_t zRecvUPanel(int_t k0, int_t senderGrid, double alpha, double beta);
};

