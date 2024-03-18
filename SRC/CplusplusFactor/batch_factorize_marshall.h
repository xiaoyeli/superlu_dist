#ifndef __SUPERLU_BATCH_FACTORIZE_MARSHALL_H__
#define __SUPERLU_BATCH_FACTORIZE_MARSHALL_H__

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Marshall Functors for batched execution 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "superlu_defs.h"

struct MarshallLUFunc_flat {
    BatchDim_t *ld_batch, *dim_batch;
    double** diag_ptrs, **Lnzval_bc_ptr;
    int_t k_st, **Lrowind_bc_ptr, *xsup, *dperm_c_supno;
    
    MarshallLUFunc_flat(
        int_t k_st, double** diag_ptrs, BatchDim_t *ld_batch, BatchDim_t *dim_batch, 
        double** Lnzval_bc_ptr, int_t** Lrowind_bc_ptr, int_t *dperm_c_supno, int_t *xsup
    )
    {
        this->k_st = k_st;
        this->ld_batch = ld_batch;
        this->dim_batch = dim_batch;
        this->diag_ptrs = diag_ptrs;

        this->Lnzval_bc_ptr = Lnzval_bc_ptr;
        this->Lrowind_bc_ptr = Lrowind_bc_ptr;
        this->dperm_c_supno = dperm_c_supno;
        this->xsup = xsup;
    }
    
    __device__ void operator()(const int_t &i) const
    {   
        int_t k = dperm_c_supno[k_st + i];
        int_t *Lrowind_bc = Lrowind_bc_ptr[k];
        double* Lnzval = Lnzval_bc_ptr[k];

        if(Lnzval && Lrowind_bc)
        {
            diag_ptrs[i] = Lnzval;
            ld_batch[i] = Lrowind_bc[1];
            dim_batch[i] = SuperSize(k);
        }
        else
        {
            diag_ptrs[i] = NULL;
            ld_batch[i] = 1;
            dim_batch[i] = 0;
        }
    }
};

struct MarshallTRSMUFunc_flat {
    BatchDim_t *diag_ld_batch, *diag_dim_batch, *panel_ld_batch, *panel_dim_batch;
    double** diag_ptrs, **panel_ptrs, **Unzval_br_new_ptr, **Lnzval_bc_ptr;
    int_t k_st, **Ucolind_br_ptr, **Lrowind_bc_ptr, *xsup, *dperm_c_supno;

    MarshallTRSMUFunc_flat(
        int_t k_st, double** diag_ptrs, BatchDim_t *diag_ld_batch, BatchDim_t *diag_dim_batch, double** panel_ptrs,
        BatchDim_t *panel_ld_batch, BatchDim_t *panel_dim_batch, double **Unzval_br_new_ptr, int_t** Ucolind_br_ptr, 
        double** Lnzval_bc_ptr, int_t** Lrowind_bc_ptr, int_t *dperm_c_supno, int_t *xsup
    )
    {
        this->k_st = k_st;
        this->diag_ptrs = diag_ptrs;
        this->diag_ld_batch = diag_ld_batch;
        this->diag_dim_batch = diag_dim_batch;
        this->panel_ptrs = panel_ptrs;
        this->panel_ld_batch = panel_ld_batch;
        this->panel_dim_batch = panel_dim_batch;
        this->Unzval_br_new_ptr = Unzval_br_new_ptr;
        this->Ucolind_br_ptr = Ucolind_br_ptr; 
        this->Lnzval_bc_ptr = Lnzval_bc_ptr;
        this->Lrowind_bc_ptr = Lrowind_bc_ptr;
        this->dperm_c_supno = dperm_c_supno;
        this->xsup = xsup;
    }
    
    __device__ void operator()(const int_t &i) const
    {   
        int_t k = dperm_c_supno[k_st + i];
        int_t ksupc = SuperSize(k);

        int_t *Ucolind_br = Ucolind_br_ptr[k];
        double* Unzval = Unzval_br_new_ptr[k];
        int_t *Lrowind_bc = Lrowind_bc_ptr[k];
        double* Lnzval = Lnzval_bc_ptr[k];

        if(Ucolind_br && Unzval && Lrowind_bc && Lnzval)
        {
            int upanel_rows = Ucolind_br[2];
            int sup_offset = ksupc - upanel_rows;

            panel_ptrs[i] = Unzval;
            panel_ld_batch[i] = upanel_rows;
            panel_dim_batch[i] = Ucolind_br[1];

            diag_ptrs[i] = Lnzval + sup_offset + sup_offset * Lrowind_bc[1];
            diag_ld_batch[i] = Lrowind_bc[1];
            diag_dim_batch[i] = upanel_rows;
        }
        else
        {
            panel_ptrs[i] = diag_ptrs[i] = NULL;
            panel_ld_batch[i] = diag_ld_batch[i] = 1;
            panel_dim_batch[i] = diag_dim_batch[i] = 0;
        }    
    }
};

struct MarshallTRSMLFunc_flat {
    BatchDim_t *diag_ld_batch, *diag_dim_batch, *panel_ld_batch, *panel_dim_batch;
    double** diag_ptrs, **panel_ptrs, **Lnzval_bc_ptr;
    int_t k_st, **Lrowind_bc_ptr, *xsup,  *dperm_c_supno;

    MarshallTRSMLFunc_flat(
        int_t k_st, double** diag_ptrs, BatchDim_t *diag_ld_batch, BatchDim_t *diag_dim_batch, double** panel_ptrs,
        BatchDim_t *panel_ld_batch, BatchDim_t *panel_dim_batch, double** Lnzval_bc_ptr, int_t** Lrowind_bc_ptr, 
        int_t *dperm_c_supno, int_t *xsup
    )
    {
        this->k_st = k_st;
        this->diag_ptrs = diag_ptrs;
        this->diag_ld_batch = diag_ld_batch;
        this->diag_dim_batch = diag_dim_batch;
        this->panel_ptrs = panel_ptrs;
        this->panel_ld_batch = panel_ld_batch;
        this->panel_dim_batch = panel_dim_batch;

        this->Lnzval_bc_ptr = Lnzval_bc_ptr;
        this->Lrowind_bc_ptr = Lrowind_bc_ptr;
        this->dperm_c_supno = dperm_c_supno;
        this->xsup = xsup;
    }
    
    __device__ void operator()(const int_t &i) const
    {
        int_t k = dperm_c_supno[k_st + i];
        int_t ksupc = SuperSize(k);
        int_t *Lrowind_bc = Lrowind_bc_ptr[k];
        double* Lnzval = Lnzval_bc_ptr[k];

        if(Lnzval && Lrowind_bc)
        {
            int_t diag_block_offset = Lrowind_bc[BC_HEADER + 1];
            int_t nzrows = Lrowind_bc[1];
            int_t len = nzrows - diag_block_offset;

            panel_ptrs[i] = Lnzval + diag_block_offset;
            panel_ld_batch[i] = nzrows;
            panel_dim_batch[i] = len;
            diag_ptrs[i] = Lnzval;
            diag_ld_batch[i] = nzrows;
            diag_dim_batch[i] = ksupc;
        }
        else
        {
            panel_ptrs[i] = diag_ptrs[i] = NULL;
            panel_ld_batch[i] = diag_ld_batch[i] = 1;
            panel_dim_batch[i] = diag_dim_batch[i] = 0;
        }    
    }
};

struct MarshallSCUFunc_flat {
    double** A_ptrs, **B_ptrs, **C_ptrs;
    BatchDim_t* lda_array, *ldb_array, *ldc_array, *m_array, *n_array, *k_array;
    double **Unzval_br_new_ptr, **Lnzval_bc_ptr, **dgpuGemmBuffs;
    int_t** Ucolind_br_ptr, **Lrowind_bc_ptr, *xsup, *dperm_c_supno, k_st;
    BatchDim_t *ist, *iend, *jst, *jend;

    MarshallSCUFunc_flat(
        int_t k_st, double** A_ptrs, BatchDim_t* lda_array, double** B_ptrs, BatchDim_t* ldb_array, 
        double **C_ptrs, BatchDim_t *ldc_array, BatchDim_t *m_array, BatchDim_t *n_array, BatchDim_t *k_array, 
        BatchDim_t *ist, BatchDim_t *iend, BatchDim_t *jst, BatchDim_t *jend, double **Unzval_br_new_ptr, 
        int_t** Ucolind_br_ptr, double** Lnzval_bc_ptr, int_t** Lrowind_bc_ptr, int_t *dperm_c_supno, 
        int_t *xsup, double** dgpuGemmBuffs
    )
    {
        this->k_st = k_st;
        this->A_ptrs = A_ptrs;
        this->B_ptrs = B_ptrs;
        this->C_ptrs = C_ptrs;
        this->lda_array = lda_array;
        this->ldb_array = ldb_array;
        this->ldc_array = ldc_array;
        this->m_array = m_array;
        this->n_array = n_array;
        this->k_array = k_array;
        this->ist = ist;
        this->iend = iend;
        this->jst = jst;
        this->jend = jend;
        this->Unzval_br_new_ptr = Unzval_br_new_ptr;
        this->Ucolind_br_ptr = Ucolind_br_ptr; 
        this->Lnzval_bc_ptr = Lnzval_bc_ptr;
        this->Lrowind_bc_ptr = Lrowind_bc_ptr;
        this->dperm_c_supno = dperm_c_supno;
        this->xsup = xsup;
        this->dgpuGemmBuffs = dgpuGemmBuffs;
    }

    __device__ void operator()(const int_t &i) const
    {
        int_t k = dperm_c_supno[k_st + i];
        
        int_t ksupc = SuperSize(k);
        int_t *Ucolind_br = Ucolind_br_ptr[k];
        double* Unzval = Unzval_br_new_ptr[k];
        int_t *Lrowind_bc = Lrowind_bc_ptr[k];
        double* Lnzval = Lnzval_bc_ptr[k];

        if(Ucolind_br && Unzval && Lrowind_bc && Lnzval)
        {
            int upanel_rows = Ucolind_br[2];
            int sup_offset = ksupc - upanel_rows;

            int_t diag_block_offset = Lrowind_bc[BC_HEADER + 1];
            int_t L_nzrows = Lrowind_bc[1];
            int_t L_len = L_nzrows - diag_block_offset;

            A_ptrs[i] = Lnzval + diag_block_offset + sup_offset * L_nzrows;
            B_ptrs[i] = Unzval;
            C_ptrs[i] = dgpuGemmBuffs[i];

            lda_array[i] = L_nzrows;
            ldb_array[i] = upanel_rows;
            ldc_array[i] = L_len;
                        
            m_array[i] = L_len;
            n_array[i] = Ucolind_br[1];
            k_array[i] = upanel_rows;

            ist[i] = 1;
            jst[i] = 0;
            iend[i] = Lrowind_bc[0];
            jend[i] = Ucolind_br[0];
        }
        else
        {
            A_ptrs[i] = B_ptrs[i] = C_ptrs[i] = NULL;
            lda_array[i] = ldb_array[i] = ldc_array[i] = 1;
            m_array[i] = n_array[i] = k_array[i] = 0;
        }
    }
};

template <class T, class offT>
struct UnaryOffsetPtrAssign
{
	T *base_mem, **ptrs;
	offT* offsets;
	
    UnaryOffsetPtrAssign(T *base_mem, offT* offsets, T **ptrs)
	{
		this->base_mem = base_mem;
		this->offsets = offsets;
		this->ptrs = ptrs;
	}
	
    inline __host__ __device__ void operator()(const offT &index) const
    {
        ptrs[index] = (offsets[index] < 0 ? NULL : base_mem + offsets[index]);
    }
};

template<class T, class offT>
inline void generateOffsetPointers(T *base_mem, offT *offsets, T **ptrs, size_t num_arrays)
{
    UnaryOffsetPtrAssign<T, offT> offset_ptr_functor(base_mem, offsets, ptrs);

    thrust::for_each(
        thrust::system::cuda::par, thrust::counting_iterator<offT>(0),
        thrust::counting_iterator<offT>(num_arrays), offset_ptr_functor
    );
}

template<typename T>
struct element_diff : public thrust::unary_function<T,T>
{
    T* st, *end;
    element_diff(T* st, T *end) 
    {
        this->st = st;
        this->end = end;
    }
    
    __device__ T operator()(const T &x) const
    {
        return end[x] - st[x];
    }
};

#endif 
