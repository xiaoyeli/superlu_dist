#pragma once 
#include "superlu_ddefs.h"

#define LPANEL_HEADER_SIZE 2
#define UPANEL_HEADER_SIZE 2
// it can be templatized for double and complex double 
class lpanel_t
{
public: 
    int_t* index;
    double* val;


    lpanel_t(int_t* lsub, double* nzval);
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
        return &index[LPANEL_HEADER_SIZE + nblocks() + k];
    } 

    // for L panel I don't need any special transformation function 
};


class upanel_t
{
public: 
    int_t* index;
    double* val;


    upanel_t(int_t* usub, double* uval);
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
    int_t* rowList(int_t k)
    {
        return &index[UPANEL_HEADER_SIZE + nblocks() + k];
    } 

    // for U panel 
    int_t packed2skyline(int_t* usub, double* uval );
};

struct LUstruct_v100
{

    lpanel_t* lPanelVec;
    upanel_t* uPanelVec;

    // constructor
    LUstruct_v100(int_t nsupers, int_t *isNodeInMyGrid,
                LUstruct_t *LUstruct,gridinfo3d_t *grid3d);
    int_t dSchurComplementUpdate()

    ~LUstruct_v100()
    {
        delete lPanelVec;
        delete uPanelVec;
    }
};