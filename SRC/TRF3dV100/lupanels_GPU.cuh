

struct lpanelGPU_t 
{
    
    public:
        int_t *index;
        double *val;
        // bool isDiagIncluded;
        __host__
        lpanelGPU_t(int_t k, int_t *lsub, double *nzval, int_t *xsup, int_t isDiagIncluded);
        // default constuctor
        
        __host__
        lpanelGPU_t()
        {
            index = NULL;
            val = NULL;
        }
        __host__
        lpanelGPU_t(int_t *index_, double *val_): index(index_), val(val_) {return;};
        
    
        // index[0] is number of blocks
        __device__ 
        int_t nblocks()
        {
            return index[0];
        }
        // number of rows
        __device__
        int_t nzrows() { return index[1]; }
        __device__
        int_t haveDiag() { return index[2]; }
        __device__
        int_t ncols() { return index[3]; }
    
        // global block id of k-th block in the panel
        __device__
        int_t gid(int_t k)
        {
            return index[LPANEL_HEADER_SIZE + k];
        }
    
        // number of rows in the k-th block
        __device__
        int_t nbrow(int_t k)
        {
            return index[LPANEL_HEADER_SIZE + nblocks() + k + 1] - index[LPANEL_HEADER_SIZE + nblocks() + k];
        }
    
        // 
        __device__
        int_t stRow(int k)
        {
            return index[LPANEL_HEADER_SIZE + nblocks() + k]; 
        } 
        // row
        __device__
        int_t *rowList(int_t k)
        {
            // LPANEL_HEADER
            // nblocks() : blocks list
            // nblocks()+1 : blocks st_points
            // index[LPANEL_HEADER_SIZE + nblocks() + k] statrting of the block
            return &index[LPANEL_HEADER_SIZE +
                          2 * nblocks() + 1 + index[LPANEL_HEADER_SIZE + nblocks() + k]];
        }
    
        __device__
        double *blkPtr(int_t k)
        {
            return &val[index[LPANEL_HEADER_SIZE + nblocks() + k]];
        }
    
        __device__
        int_t LDA() { return index[1]; }

        __device__
        int_t find(int_t k);
        // // for L panel I don't need any special transformation function
        // int_t panelSolve(int_t ksupsz, double *DiagBlk, int_t LDD);
        // int_t diagFactor(int_t k, double *UBlk, int_t LDU, double thresh, int_t *xsup,
        //                  superlu_dist_options_t *options, SuperLUStat_t *stat, int *info);
        // int_t packDiagBlock(double *DiagLBlk, int_t LDD);

        __device__
        int_t isEmpty() { return index == NULL; }

        __device__
        int_t nzvalSize()
        {
            if (index == NULL)
                return 0;
            return ncols() * nzrows();
        }
        
        __device__
        int_t indexSize()
        {
            if (index == NULL)
                return 0;
            return LPANEL_HEADER_SIZE + 2 * nblocks() + 1 + nzrows();
        }
    
        // return the maximal iEnd such that stRow(iEnd)-stRow(iSt) < maxRow;
        __device__
        int getEndBlock(int iSt, int maxRows);
        lpanelGPU_t::lpanelGPU_t(lpanel_t& lpanel);
        int check(lpanel_t& lpanel);
    private: 
        lpanel_t& lpanel_CPU;
};

class upanelGPU_t : public upanel_t
{
    public: 
    upanelGPU_t::upanelGPU_t(upanel_t& upanel);
        int check(upanel_t& upanel);
    private: 
        upanel_t& upanel_CPU;
};


#define MAX_CUDA_STREAMS 64 
struct LUstructGPU_t
{
    // all pointers are device pointers 

    upanelGPU_t* uPanelVec;
    lpanelGPU_t* lPanelVec; 
    int_t* xsup; 
    int Pr, Pc, Pd;
    // TODO: get num cuda streams
    size_t gemmBufferSize; 
    int numCudaStreams;     
    // double arrays are problematic 
    double* gpuGemmBuffs[MAX_CUDA_STREAMS];  
    double* LvalRecvBufs[MAX_CUDA_STREAMS;
    double* UvalRecvBufs[MAX_CUDA_STREAMS;
    int_t* LidxRecvBufs[MAX_CUDA_STREAMS;
    int_t* UidxRecvBufs[MAX_CUDA_STREAMS;
    
    
    __device__
    int_t supersize(int_t k) { return xsup[k + 1] - xsup[k]; }
    __device__
    int_t g2lRow(int_t k) { return k / Pr; }
    __device__
    int_t g2lCol(int_t k) { return k / Pc; }
    
};