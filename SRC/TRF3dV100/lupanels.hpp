#pragma once 
#include "superlu_ddefs.h"

#define LPANEL_HEADER_SIZE 3

// it can be templatized for double and complex double 
class lpanel_t
{
public: 
    int_t* index;
    double* val;
    // bool isDiagIncluded;


    lpanel_t(int_t* lsub, double* nzval, int_t*xsup, int_t isDiagIncluded);
    // default constuctor
    lpanel_t() 
    {
        index = NULL;
        val=NULL;
    }

    // index[0] is number of blocks
    int_t nblocks()
    {
        return index[0];
    }
    // number of rows
    int_t nzrows() { return index[1];}
    int_t haveDiag() {return index[2];}
    
    // global block id of k-th block in the panel
    int_t gid(int_t k) {
        return index[LPANEL_HEADER_SIZE + k]; 
    }

    // number of rows in the k-th block
    int_t nbrow(int_t k)
    {
        return index[LPANEL_HEADER_SIZE + nblocks() + k+1]
            -index[LPANEL_HEADER_SIZE + nblocks() + k];
    }
    // row 
    int_t* rowList(int_t k)
    {
        // LPANEL_HEADER 
        // nblocks() : blocks list
        // nblocks()+1 : blocks st_points 
        // index[LPANEL_HEADER_SIZE + nblocks() + k] statrting of the block 
        return &index[LPANEL_HEADER_SIZE + 
                        2*nblocks() + 1 
                        + index[LPANEL_HEADER_SIZE + nblocks() + k]];
    } 

    double* blkPtr(int_t k)
    {
        return &val[index[LPANEL_HEADER_SIZE + nblocks() + k]];
    }

    int_t LDA() { return index[1];}
    int_t find(int_t k);
    // for L panel I don't need any special transformation function 
    int_t panelSolve(int_t ksupsz, double* DiagBlk, int_t LDD);
};

#define UPANEL_HEADER_SIZE 3
class upanel_t
{
public: 
    int_t* index;
    double* val;


    upanel_t(int_t* usub, double* uval);
    upanel_t(int_t k, int_t *usub, double *uval, int_t *xsup);
    upanel_t() 
    {
        index = NULL;
        val=NULL;
    }

    // index[0] is number of blocks
    int_t nblocks()
    {
        return index[0];
    }
    // number of rows
    int_t nzcols() { return index[1];}
    int_t LDA() { return index[2];}     // is also supersize of that coloumn
    
    // global block id of k-th block in the panel
    int_t gid(int_t k) {
        return index[UPANEL_HEADER_SIZE + k]; 
    }

    // number of rows in the k-th block
    int_t nbcol(int_t k)
    {
        return index[UPANEL_HEADER_SIZE + nblocks() + k+1]
            -index[UPANEL_HEADER_SIZE + nblocks() + k];
    }
    // row 
    int_t* colList(int_t k)
    {
        // UPANEL_HEADER 
        // nblocks() : blocks list
        // nblocks()+1 : blocks st_points 
        // index[UPANEL_HEADER_SIZE + nblocks() + k] statrting of the block 
        return &index[UPANEL_HEADER_SIZE + 
                        2*nblocks() + 1 
                        + index[UPANEL_HEADER_SIZE + nblocks() + k]];
    } 

    double* blkPtr(int_t k)
    {
        return &val[LDA()*index[UPANEL_HEADER_SIZE + nblocks() + k]];
    }
    // for U panel 
    // int_t packed2skyline(int_t* usub, double* uval );
    int_t packed2skyline(int_t k, int_t *usub, double *uval, int_t*xsup);
    int_t panelSolve(int_t ksupsz, double* DiagBlk, int_t LDD);
    // double* blkPtr(int_t k);
    // int_t LDA();
    int_t find(int_t k);
};

struct LUstruct_v100
{

    lpanel_t* lPanelVec;
    upanel_t* uPanelVec;
    gridinfo3d_t* grid3d; 
    gridinfo_t *grid;
    int_t iam, Pc, Pr, myrow, mycol, ldt;
    int_t* xsup; 
    // variables for scattering ldt*THREAD_Size 
    int_t *indirect, *indirectRow, *indirectCol; 
    double* bigV;       // size = THREAD_Size*ldt*ldt 

    int_t krow(int_t k) {return k%Pr;}
    int_t kcol(int_t k) {return k%Pc;}
    int_t supersize(int_t k)  {return xsup[k+1]-xsup[k];}
    int_t g2lRow(int_t k) {return k/Pr;}
    int_t g2lCol(int_t k) {return k/Pc;}

    enum indirectMapType { ROW_MAP, COL_MAP };
    
    /**
    *          C O N / D E S - T R U C T O R S
    */
    LUstruct_v100(int_t nsupers, int_t *isNodeInMyGrid,
                LUstruct_t *LUstruct,gridinfo3d_t *grid3d);


    ~LUstruct_v100()
    {
        delete lPanelVec;
        delete uPanelVec;
    }

    /**
    *           Compute Functions 
    */
    int_t dSchurComplementUpdate(int_t k, lpanel_t& lpanel, upanel_t& upanel);
    int_t* computeIndirectMap(indirectMapType direction, int_t srcLen, int_t *srcVec,
                                         int_t dstLen, int_t *dstVec);

    int_t dScatter(int_t m , int_t n,
            int_t gi, int_t gj, 
            double* V, int_t ldv,
            int_t* srcRowList,int_t* srcColList);

    
};