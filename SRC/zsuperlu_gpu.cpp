
/*! @file
 * \brief Descriptions and declarations for structures used in GPU
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology, Oak Ridge National Laboratory
 * March 14, 2021 version 7.0.0
 * </pre>
 */

//#define GPU_DEBUG

#include <CL/sycl.hpp>
#include "mpi.h"
// #include "sec_structs.h"
#include <ctime>
#include <oneapi/mkl/blas.hpp>

// requires to set environment variables: ZES_ENABLE_SYSMAN=1, SYCL_DEVICE_FILTER=level_zero (only for get_acc_memory())
#include "level_zero/ze_api.h"
#include "level_zero/zes_api.h"
#include "CL/sycl/backend/level_zero.hpp"

#undef Reduce

#include "zlustruct_gpu_sycl.hpp"

#include "dcomplex.h"
#include <complex>

using localAcc = sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local>;

// #define UNIT_STRIDE

#define THREAD_BLOCK_SIZE  256  /* Sherry: was 192. should be <= MAX_SUPER_SIZE */

inline
void zdevice_scatter_l_2D (int thread_id,
			   int nsupc, int temp_nbrow,
			   int_t *usub, int iukp, int_t klst,
			   doublecomplex *nzval, int ldv,
			   const doublecomplex *tempv, int nbrow,
			   localAcc indirect2_thread,
			   int nnz_cols, int ColPerBlock,
			   localAcc IndirectJ3
    )
{
    int i;
    if ( thread_id < temp_nbrow * ColPerBlock )	{
	int thread_id_x  = thread_id % temp_nbrow;
	int thread_id_y  = thread_id / temp_nbrow;

#define UNROLL_ITER 8

#pragma unroll 4
	for (int col = thread_id_y; col < nnz_cols ; col += ColPerBlock)
	{
   	    i = ldv * IndirectJ3[col] + indirect2_thread[thread_id_x];
	    z_sub(&nzval[i], &nzval[i], &tempv[nbrow * col + thread_id_x]);
	}
    }
}

inline
void device_scatter_u_2D (int thread_id,
                          int temp_nbrow,  int nsupc,
                          doublecomplex * ucol,
                          int_t * usub, int iukp,
                          int_t ilst, int_t klst,
                          int_t * index, int iuip_lib,
                          doublecomplex * tempv, int nbrow,
                          localAcc indirect,
                          int nnz_cols, int ColPerBlock,
                          localAcc IndirectJ1,
                          localAcc IndirectJ3
    )
{
    int i;

    if ( thread_id < temp_nbrow * ColPerBlock )
    {    
	/* 1D threads are logically arranged in 2D shape. */
	int thread_id_x  = thread_id % temp_nbrow;
	int thread_id_y  = thread_id / temp_nbrow;

#pragma unroll 4
	for (int col = thread_id_y; col < nnz_cols ; col += ColPerBlock)
	{
	    i = IndirectJ1[IndirectJ3[col]] + indirect[thread_id_x];
	    z_sub(&ucol[i], &ucol[i], &tempv[nbrow * col + thread_id_x]);
	}
    }
}


inline
void device_scatter_u (int_t thread_id,
                       int_t temp_nbrow,  int_t nsupc,
                       doublecomplex * ucol,
                       int_t * usub, int_t iukp,
                       int_t ilst, int_t klst,
                       int_t * index, int_t iuip_lib,
                       doublecomplex * tempv, int_t nbrow,
                       // int_t *indirect
                       int *indirect
    )
{
    int_t segsize, fnz, jj;
    for (jj = 0; jj < nsupc; ++jj)
    {
	segsize = klst - usub[iukp + jj];
	fnz = index[iuip_lib++];
	ucol -= fnz;
	if (segsize) {            /* Nonzero segment in U(k.j). */
	    if (thread_id < temp_nbrow)
	    {
#ifndef UNIT_STRIDE
		z_sub(&ucol[indirect[thread_id]], &ucol[indirect[thread_id]],
		      &tempv[thread_id]);
#else
		/* making access unit strided;
		   it doesn't work; it is for measurements */
		z_sub(&ucol[thread_id], &ucol[thread_id],
		      &tempv[thread_id]);
#endif
	    }
	    tempv += nbrow;
	}
	ucol += ilst ;
    }
}



void Scatter_GPU_kernel(
    int_t streamId,
    int_t ii_st, int_t ii_end,
    int_t jj_st, int_t jj_end, /* defines rectangular Schur block to be scatter */
    int_t klst,
    int_t jj0,   /* 0 on entry */
    int_t nrows, int_t ldt, int_t npcol, int_t nprow,
    zLUstruct_gpu_t * A_gpu, sycl::nd_item<3>& item,
    localAcc indirect_thread,   /* local, row-wise */
    localAcc indirect2_thread,  /* local, row-wise */
    localAcc IndirectJ1,        /* local, column-wise */
    localAcc IndirectJ3,        /* local, column-wise */
    localAcc ljb_ind, localAcc lib_ind) /* local */
{

    /* initializing pointers */
    int_t *xsup = A_gpu->xsup;
    int_t *UrowindPtr = A_gpu->UrowindPtr;
    int_t *UrowindVec = A_gpu->UrowindVec;
    int_t *UnzvalPtr = A_gpu->UnzvalPtr;
    doublecomplex *UnzvalVec = A_gpu->UnzvalVec;
    int_t *LrowindPtr = A_gpu->LrowindPtr;
    int_t *LrowindVec = A_gpu->LrowindVec;
    int_t *LnzvalPtr = A_gpu->LnzvalPtr;
    doublecomplex *LnzvalVec = A_gpu->LnzvalVec;
    doublecomplex *bigV = A_gpu->scubufs[streamId].bigV;
    local_l_blk_info_t *local_l_blk_infoVec = A_gpu->local_l_blk_infoVec;
    local_u_blk_info_t *local_u_blk_infoVec = A_gpu->local_u_blk_infoVec;
    int_t *local_l_blk_infoPtr = A_gpu->local_l_blk_infoPtr;
    int_t *local_u_blk_infoPtr = A_gpu->local_u_blk_infoPtr;
    Remain_info_t *Remain_info = A_gpu->scubufs[streamId].Remain_info;
    Ublock_info_t *Ublock_info = A_gpu->scubufs[streamId].Ublock_info;
    int_t *lsub  = A_gpu->scubufs[streamId].lsub;
    int_t *usub  = A_gpu->scubufs[streamId].usub;

    /* thread block assignment: this thread block is
       assigned to block (lb, j) in 2D grid */
    int lb = item.get_group(2) + ii_st;
    int j = item.get_group(1) + jj_st;

    int thread_id = item.get_local_id(2);

    int iukp = Ublock_info[j].iukp;
    int jb = Ublock_info[j].jb;
    int nsupc = SuperSize (jb);
    int ljb = jb / npcol;

    doublecomplex *tempv1;
    if (jj_st == jj0)
    {
	tempv1 = (j == jj_st) ? bigV
	    : bigV + Ublock_info[j - 1].full_u_cols * nrows;
    }
    else
    {
	tempv1 = (j == jj_st) ? bigV
	    : bigV + (Ublock_info[j - 1].full_u_cols -
		      Ublock_info[jj_st - 1].full_u_cols) * nrows;
    }

    /* # of nonzero columns in block j  */
    int nnz_cols = (j == 0) ? Ublock_info[j].full_u_cols
	: (Ublock_info[j].full_u_cols - Ublock_info[j - 1].full_u_cols);
    int cum_ncol = (j == 0) ? 0	: Ublock_info[j - 1].full_u_cols;

    int lptr = Remain_info[lb].lptr;
    int ib   = Remain_info[lb].ib;
    int temp_nbrow = lsub[lptr + 1]; /* number of rows in the current L block */
    lptr += LB_DESCRIPTOR;

    int_t cum_nrow;
    if (ii_st == 0)
    {
	cum_nrow = (lb == 0 ? 0 : Remain_info[lb - 1].FullRow);
    }
    else
    {
	cum_nrow = (lb == 0 ? 0 : Remain_info[lb - 1].FullRow - Remain_info[ii_st - 1].FullRow);
    }

    tempv1 += cum_nrow;

    if (ib < jb)  /*scatter U code */
    {
	int ilst = FstBlockC (ib + 1);
	int lib =  ib / nprow;   /* local index of row block ib */
	int_t *index = &UrowindVec[UrowindPtr[lib]];

	int num_u_blocks = index[0];

	int ljb = (jb) / npcol; /* local index of column block jb */

	/* Each thread is responsible for one block column */

	/*do a search ljb_ind at local row lib*/
	int blks_per_threads = CEILING(num_u_blocks, THREAD_BLOCK_SIZE);
	for (int i = 0; i < blks_per_threads; ++i)
	    /* each thread is assigned a chunk of consecutive U blocks to search */
	{
	    /* only one thread finds the block index matching ljb */
	    if (thread_id * blks_per_threads + i < num_u_blocks &&
		local_u_blk_infoVec[ local_u_blk_infoPtr[lib] + thread_id * blks_per_threads + i ].ljb == ljb)
	    {
		ljb_ind[0] = thread_id * blks_per_threads + i;
	    }
	}
	item.barrier();

	int iuip_lib = local_u_blk_infoVec[local_u_blk_infoPtr[lib] + ljb_ind[0]].iuip;
	int ruip_lib = local_u_blk_infoVec[local_u_blk_infoPtr[lib] + ljb_ind[0]].ruip;
	iuip_lib += UB_DESCRIPTOR;
	doublecomplex *Unzval_lib = &UnzvalVec[UnzvalPtr[lib]];
	doublecomplex *ucol = &Unzval_lib[ruip_lib];

	if (thread_id < temp_nbrow) /* row-wise */
	{
	    /* cyclically map each thread to a row */
	    indirect_thread[thread_id] = (int) lsub[lptr + thread_id];
	}

	/* column-wise: each thread is assigned one column */
	if (thread_id < nnz_cols)
	    IndirectJ3[thread_id] = A_gpu->scubufs[streamId].usub_IndirectJ3[cum_ncol + thread_id];
	/* indirectJ3[j] == kk means the j-th nonzero segment
	   points to column kk in this supernode */

	item.barrier();

	/* threads are divided into multiple columns */
	int ColPerBlock = THREAD_BLOCK_SIZE / temp_nbrow;

	if (thread_id < THREAD_BLOCK_SIZE)
	    IndirectJ1[thread_id] = 0;

	if (thread_id < THREAD_BLOCK_SIZE)
	{
	    if (thread_id < nsupc)
	    {
		/* fstnz subscript of each column in the block */
		IndirectJ1[thread_id] = index[iuip_lib + thread_id];
	    }
	}

	/* perform an inclusive block-wide prefix sum among all threads */
	if (thread_id < THREAD_BLOCK_SIZE)
	    inclusive_scan_over_group(item.get_group(), IndirectJ1[thread_id], sycl::plus<>());

	if (thread_id < THREAD_BLOCK_SIZE)
	    IndirectJ1[thread_id] = -IndirectJ1[thread_id] + ilst * thread_id;

	item.barrier();

	device_scatter_u_2D (
	    thread_id,
	    temp_nbrow,  nsupc,
	    ucol,
	    usub, iukp,
	    ilst, klst,
	    index, iuip_lib,
	    tempv1, nrows,
	    indirect_thread,
	    nnz_cols, ColPerBlock,
	    IndirectJ1,
	    IndirectJ3 );

    }
    else     /* ib >= jb, scatter L code */
    {

	int rel;
	doublecomplex *nzval;
	int_t *index = &LrowindVec[LrowindPtr[ljb]];
	int num_l_blocks = index[0];
	int ldv = index[1];

	int fnz = FstBlockC (ib);
	int lib = ib / nprow;

	/*do a search lib_ind for lib*/
	int blks_per_threads = CEILING(num_l_blocks, THREAD_BLOCK_SIZE);
	for (int i = 0; i < blks_per_threads; ++i)
	{
	    if (thread_id * blks_per_threads + i < num_l_blocks &&
		local_l_blk_infoVec[ local_l_blk_infoPtr[ljb] + thread_id * blks_per_threads + i ].lib == lib)
	    {
		lib_ind[0] = thread_id * blks_per_threads + i;
	    }
	}
	item.barrier();

	int lptrj = local_l_blk_infoVec[local_l_blk_infoPtr[ljb] + lib_ind[0]].lptrj;
	int luptrj = local_l_blk_infoVec[local_l_blk_infoPtr[ljb] + lib_ind[0]].luptrj;
	lptrj += LB_DESCRIPTOR;
	int dest_nbrow = index[lptrj - 1];

	if (thread_id < dest_nbrow)
	{
	    rel = index[lptrj + thread_id] - fnz;
	    indirect_thread[rel] = thread_id;
	}
	item.barrier();

	/* can be precalculated */
	if (thread_id < temp_nbrow)
	{
	    rel = lsub[lptr + thread_id] - fnz;
	    indirect2_thread[thread_id] = indirect_thread[rel];
	}
	if (thread_id < nnz_cols)
	    IndirectJ3[thread_id] = (int) A_gpu->scubufs[streamId].usub_IndirectJ3[cum_ncol + thread_id];
	item.barrier();

	int ColPerBlock = THREAD_BLOCK_SIZE / temp_nbrow;

	nzval = &LnzvalVec[LnzvalPtr[ljb]] + luptrj;
	zdevice_scatter_l_2D(
	    thread_id,
	    nsupc, temp_nbrow,
	    usub, iukp, klst,
	    nzval, ldv,
	    tempv1, nrows, indirect2_thread,
	    nnz_cols, ColPerBlock,
	    IndirectJ3);
    } /* end else ib >= jb */

} /* end Scatter_GPU_kernel */


#define GPU_2D_SCHUDT  /* Not used */

int zSchurCompUpdate_GPU(
    int_t streamId,
    int_t jj_cpu, /* 0 on entry, pointing to the start of Phi part */
    int_t nub,    /* jj_cpu on entry, pointing to the end of the Phi part */
    int_t klst, int_t knsupc,
    int_t Rnbrow, int_t RemainBlk,
    int_t Remain_lbuf_send_size,
    int_t bigu_send_size, int_t ldu,
    int_t mcb,    /* num_u_blks_hi */
    int_t buffer_size, int_t lsub_len, int_t usub_len,
    int_t ldt, int_t k0,
    zsluGPU_t *sluGPU, gridinfo_t *grid
    )
{

    zLUstruct_gpu_t * A_gpu = sluGPU->A_gpu;
    zLUstruct_gpu_t * dA_gpu = sluGPU->dA_gpu;
    int_t nprow = grid->nprow;
    int_t npcol = grid->npcol;

    sycl::queue *FunCallStream = sluGPU->funCallStreams[streamId];
    int_t * lsub = A_gpu->scubufs[streamId].lsub_buf;
    int_t * usub = A_gpu->scubufs[streamId].usub_buf;
    Remain_info_t *Remain_info = A_gpu->scubufs[streamId].Remain_info_host;
    doublecomplex * Remain_L_buff = A_gpu->scubufs[streamId].Remain_L_buff_host;
    Ublock_info_t *Ublock_info = A_gpu->scubufs[streamId].Ublock_info_host;
    doublecomplex * bigU = A_gpu->scubufs[streamId].bigU_host;

    A_gpu->isOffloaded[k0] = 1;
    /* start by sending data to  */
    int_t *xsup = A_gpu->xsup_host;
    int_t col_back = (jj_cpu == 0) ? 0 : Ublock_info[jj_cpu - 1].full_u_cols;
    // if(nub<1) return;
    int_t ncols  = Ublock_info[nub - 1].full_u_cols - col_back;

    /* Sherry: can get max_super_size from sp_ienv(3) */
    int_t indirectJ1[MAX_SUPER_SIZE]; // 0 indicates an empry segment
    int_t indirectJ2[MAX_SUPER_SIZE]; // # of nonzero segments so far
    int_t indirectJ3[MAX_SUPER_SIZE]; /* indirectJ3[j] == k means the
					 j-th nonzero segment points
					 to column k in this supernode */
    /* calculate usub_indirect */
    for (int jj = jj_cpu; jj < nub; ++jj)
    {
	int_t iukp = Ublock_info[jj].iukp;
	int_t jb = Ublock_info[jj].jb;
	int_t nsupc = SuperSize (jb);
	int_t addr = (jj == 0) ? 0
	    : Ublock_info[jj - 1].full_u_cols - col_back;

	for (int_t kk = 0; kk < nsupc; ++kk) // old: MAX_SUPER_SIZE
	{
	    indirectJ1[kk] = 0;
	}

	for (int_t kk = 0; kk < nsupc; ++kk)
	{
	    indirectJ1[kk] = ((klst - usub[iukp + kk]) == 0) ? 0 : 1;
	}

	/*prefix sum - indicates # of nonzero segments up to column kk */
	indirectJ2[0] = indirectJ1[0];
	for (int_t kk = 1; kk < nsupc; ++kk) // old: MAX_SUPER_SIZE
	{
	    indirectJ2[kk] = indirectJ2[kk - 1] + indirectJ1[kk];
	}

	/* total number of nonzero segments in this supernode */
	int nnz_col = indirectJ2[nsupc - 1]; // old: MAX_SUPER_SIZE

	/* compactation */
	for (int_t kk = 0; kk < nsupc; ++kk) // old: MAX_SUPER_SIZE
	{
	    if (indirectJ1[kk]) /* kk is a nonzero segment */
	    {
		/* indirectJ3[j] == kk means the j-th nonzero segment
		   points to column kk in this supernode */
		indirectJ3[indirectJ2[kk] - 1] = kk;
	    }
	}

	for (int i = 0; i < nnz_col; ++i)
	{
	    /* addr == total # of full columns before current block jj */
	    A_gpu->scubufs[streamId].usub_IndirectJ3_host[addr + i] = indirectJ3[i];
	}
    } /* end for jj ... calculate usub_indirect */

    //printf("zSchurCompUpdate_GPU[3]: jj_cpu %d, nub %d\n", jj_cpu, nub); fflush(stdout);

    /*sizeof RemainLbuf = Rnbuf*knsupc */
    double tTmp = SuperLU_timer_();
    //cudaEventRecord(A_gpu->ePCIeH2D[k0], FunCallStream);

    FunCallStream->memcpy(A_gpu->scubufs[streamId].usub_IndirectJ3,
			  A_gpu->scubufs[streamId].usub_IndirectJ3_host,
			  ncols * sizeof(int_t));
			  
    FunCallStream->memcpy(A_gpu->scubufs[streamId].Remain_L_buff, Remain_L_buff,
			  Remain_lbuf_send_size * sizeof(doublecomplex));
		       
    FunCallStream->memcpy(A_gpu->scubufs[streamId].bigU, bigU,
			  bigu_send_size * sizeof(doublecomplex));

    FunCallStream->memcpy(A_gpu->scubufs[streamId].Remain_info, Remain_info,
			  RemainBlk * sizeof(Remain_info_t));

    FunCallStream->memcpy(A_gpu->scubufs[streamId].Ublock_info, Ublock_info,
			  mcb * sizeof(Ublock_info_t));

    FunCallStream->memcpy(A_gpu->scubufs[streamId].lsub, lsub,
			  lsub_len * sizeof(int_t));

    FunCallStream->memcpy(A_gpu->scubufs[streamId].usub, usub,
			  usub_len * sizeof(int_t));

    A_gpu->tHost_PCIeH2D += SuperLU_timer_() - tTmp;
    A_gpu->cPCIeH2D += Remain_lbuf_send_size * sizeof(doublecomplex)
	+ bigu_send_size * sizeof(doublecomplex)
	+ RemainBlk * sizeof(Remain_info_t)
	+ mcb * sizeof(Ublock_info_t)
	+ lsub_len * sizeof(int_t)
	+ usub_len * sizeof(int_t);

    std::complex<double> alpha_onemkl = {1.0, 0.0}, beta_onemkl = {0.0, 0.0};

    /* The following are used in gemm<std::complex> call */
    std::complex<double> *cu_A, *cu_B, *cu_C; /* C <- A*B */

    int_t ii_st  = 0;
    int_t ii_end = 0;
    int_t maxGemmBlockDim = (int) sqrt(buffer_size);
    // int_t maxGemmBlockDim = 8000;

    /* Organize GEMM by blocks of [ii_st : ii_end, jj_st : jj_end] that
       fits in the buffer_size  */
    while (ii_end < RemainBlk) {
	ii_st = ii_end;
	ii_end = RemainBlk;
	int_t nrow_max = maxGemmBlockDim;
// nrow_max = Rnbrow;
	int_t remaining_rows = (ii_st == 0) ? Rnbrow : Rnbrow - Remain_info[ii_st - 1].FullRow;
	nrow_max = (remaining_rows / nrow_max) > 0 ? remaining_rows / CEILING(remaining_rows,  nrow_max) : nrow_max;

	int_t ResRow = (ii_st == 0) ? 0 : Remain_info[ii_st - 1].FullRow;
	for (int_t i = ii_st; i < RemainBlk - 1; ++i)
	{
	    if ( Remain_info[i + 1].FullRow > ResRow + nrow_max)
	    {
		ii_end = i;
		break;  /* row dimension reaches nrow_max */
	    }
	}

	int_t nrows;   /* actual row dimension for GEMM */
	int_t st_row;
	if (ii_st > 0)
	{
	    nrows = Remain_info[ii_end - 1].FullRow - Remain_info[ii_st - 1].FullRow;
	    st_row = Remain_info[ii_st - 1].FullRow;
	}
	else
	{
	    nrows = Remain_info[ii_end - 1].FullRow;
	    st_row = 0;
	}

	int jj_st = jj_cpu;
	int jj_end = jj_cpu;

	while (jj_end < nub && nrows > 0 )
	{
	    int_t remaining_cols = (jj_st == jj_cpu) ? ncols : ncols - Ublock_info[jj_st - 1].full_u_cols;
	    if ( remaining_cols * nrows < buffer_size)
	    {
		jj_st = jj_end;
		jj_end = nub;
	    }
	    else  /* C matrix cannot fit in buffer, need to break into pieces */
	    {
		int_t ncol_max = buffer_size / nrows;
		/** Must revisit **/
		ncol_max = SUPERLU_MIN(ncol_max, maxGemmBlockDim);
		ncol_max = (remaining_cols / ncol_max) > 0 ?
		    remaining_cols / CEILING(remaining_cols,  ncol_max)
		    : ncol_max;

		jj_st = jj_end;
		jj_end = nub;

		int_t ResCol = (jj_st == 0) ? 0 : Ublock_info[jj_st - 1].full_u_cols;
		for (int_t j = jj_st; j < nub - 1; ++j)
		{
		    if (Ublock_info[j + 1].full_u_cols > ResCol + ncol_max)
		    {
			jj_end = j;
			break;
		    }
		}
	    } /* end-if-else */

	    int ncols;
	    int st_col;
	    if (jj_st > 0)
	    {
		ncols = Ublock_info[jj_end - 1].full_u_cols - Ublock_info[jj_st - 1].full_u_cols;
		st_col = Ublock_info[jj_st - 1].full_u_cols;
		if (ncols == 0) exit(0);
	    }
	    else
	    {
		ncols = Ublock_info[jj_end - 1].full_u_cols;
		st_col = 0;
	    }

	    /* none of the matrix dimension is zero. */
	    if (nrows > 0 && ldu > 0 && ncols > 0)
	    {
		if (nrows * ncols > buffer_size) {
		    std::cout << "!! Matrix size (" << nrows << " X " << ncols
			      << ") exceeds buffer_size : " << buffer_size << std::endl;
		    fflush(stdout);
		}
		assert(nrows * ncols <= buffer_size);
		//cudaEventRecord(A_gpu->GemmStart[k0], FunCallStream);

		cu_A = (std::complex<double> *) &A_gpu->scubufs[streamId].Remain_L_buff[(knsupc - ldu) * Rnbrow + st_row];
		cu_B = (std::complex<double> *) &A_gpu->scubufs[streamId].bigU[st_col * ldu];
		cu_C = (std::complex<double> *) A_gpu->scubufs[streamId].bigV;
		
		oneapi::mkl::blas::gemm(*FunCallStream,
					oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
					nrows, ncols, ldu,
					alpha_onemkl,
					cu_A, Rnbrow,
					cu_B, ldu,
					beta_onemkl,
					cu_C, nrows);

// #define SCATTER_OPT
#ifdef SCATTER_OPT
		FunCallStream->wait();
#warning this function is synchronous
#endif
		//cudaEventRecord(A_gpu->GemmEnd[k0], FunCallStream);

		A_gpu->GemmFLOPCounter += 8.0 * (double) nrows * ncols * ldu;

		/*
		 * Scattering the output
		 */
		sycl::range<3> dimBlock(1, 1, THREAD_BLOCK_SIZE); // 1d thread
		sycl::range<3> dimGrid(1, jj_end - jj_st, ii_end - ii_st);

		FunCallStream->submit([&](sycl::handler& cgh) {

			localAcc indirect_thread_acc(MAX_SUPER_SIZE, cgh);   /* row-wise */
			localAcc indirect2_thread_acc(MAX_SUPER_SIZE, cgh);  /* row-wise */
			localAcc IndirectJ1_acc(THREAD_BLOCK_SIZE, cgh);     /* column-wise */
			localAcc IndirectJ3_acc(THREAD_BLOCK_SIZE, cgh);     /* column-wise */
			localAcc ljb_ind_acc(0, cgh);
			localAcc lib_ind_acc(0, cgh);

			auto global_range = dimBlock * dimGrid;

			cgh.parallel_for(sycl::nd_range<3>(global_range, dimBlock),
					 [=](sycl::nd_item<3> item) {
					     Scatter_GPU_kernel(streamId, ii_st, ii_end, jj_st, jj_end, klst,
								0, nrows, ldt, npcol, nprow, dA_gpu, item,
								indirect_thread_acc, indirect2_thread_acc,
								IndirectJ1_acc, IndirectJ3_acc,
								ljb_ind_acc, lib_ind_acc);
					 });
		    });
		
#ifdef SCATTER_OPT
		FunCallStream->wait();
#warning this function is synchrnous
#endif

		//cudaEventRecord(A_gpu->ScatterEnd[k0], FunCallStream);

		A_gpu->ScatterMOPCounter +=  3.0 * (double) nrows * ncols;
	    } /* endif ... none of the matrix dimension is zero. */

	} /* end while jj_end < nub */

    } /* end while (ii_end < RemainBlk) */

    return 0;
} /* end zSchurCompUpdate_GPU */

static size_t get_acc_memory ()
{
    size_t mfree=0, mtotal=0;

    sycl::platform platform(sycl::gpu_selector{});
    std::vector<sycl::device> gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
    sycl::context super_ctxt = sycl::context(gpu_devices);

    ze_context_handle_t ze_super_ctxt = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(super_ctxt);
    ze_device_handle_t ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpu_devices[0]);
    zes_device_handle_t zes_device = (zes_device_handle_t)ze_device;
    ze_result_t status = ZE_RESULT_SUCCESS;

    uint32_t module_count = 0;
    status = zesDeviceEnumMemoryModules(zes_device, &module_count, nullptr);
    if( module_count > 0 ) {
	std::vector<zes_mem_handle_t> module_list(module_count);
	std::vector<zes_mem_state_t> state_list(module_count);
	
	status = zesDeviceEnumMemoryModules(zes_device, &module_count, module_list.data());

	for (uint32_t i = 0; i < module_count; ++i) {
	    status = zesMemoryGetState(module_list[i], &(state_list[i]));
	    mfree += state_list[i].free;
	    mtotal += state_list[i].size;
	}
    }
	
#if 1
    printf("Total memory %zu & free memory %zu\n", mtotal, mfree);
#endif

    return (size_t) (0.9 * (double) mfree) / get_mpi_process_per_gpu ();
}

int zfree_LUstruct_gpu (zLUstruct_gpu_t * A_gpu, zsluGPU_t *sluGPU)
{
    sycl::queue* q = sluGPU->funCallStreams[0];
    
    sycl::free(A_gpu->LrowindVec, *q);
    sycl::free(A_gpu->LrowindPtr, *q);

    sycl::free(A_gpu->LnzvalVec, *q);
    sycl::free(A_gpu->LnzvalPtr, *q);
    free(A_gpu->LnzvalPtr_host);
    /*freeing the pinned memory*/
    int_t streamId = 0;
    sycl::free(A_gpu->scubufs[streamId].Remain_info_host, *q);
    sycl::free(A_gpu->scubufs[streamId].Ublock_info_host, *q);
    sycl::free(A_gpu->scubufs[streamId].Remain_L_buff_host, *q);
    sycl::free(A_gpu->scubufs[streamId].bigU_host, *q);

    sycl::free(A_gpu->acc_L_buff, *q);
    sycl::free(A_gpu->acc_U_buff, *q);
    sycl::free(A_gpu->scubufs[streamId].lsub_buf, *q);
    sycl::free(A_gpu->scubufs[streamId].usub_buf, *q);


    free(A_gpu->isOffloaded);
    free(A_gpu->GemmStart);
    free(A_gpu->GemmEnd);
    free(A_gpu->ScatterEnd);
    free(A_gpu->ePCIeH2D);

    free(A_gpu->ePCIeD2H_Start);
    free(A_gpu->ePCIeD2H_End);

    sycl::free(A_gpu->UrowindVec, *q);
    sycl::free(A_gpu->UrowindPtr, *q);

    free(A_gpu->UrowindPtr_host);

    sycl::free(A_gpu->UnzvalVec, *q);
    sycl::free(A_gpu->UnzvalPtr, *q);

    sycl::free(A_gpu->grid, *q);



    sycl::free(A_gpu->scubufs[streamId].bigV, *q);
    sycl::free(A_gpu->scubufs[streamId].bigU, *q);

    sycl::free(A_gpu->scubufs[streamId].Remain_L_buff, *q);
    sycl::free(A_gpu->scubufs[streamId].Ublock_info, *q);
    sycl::free(A_gpu->scubufs[streamId].Remain_info, *q);

    // sycl::free(A_gpu->indirect, *q);
    // sycl::free(A_gpu->indirect2, *q);
    sycl::free(A_gpu->xsup, *q);

    sycl::free(A_gpu->scubufs[streamId].lsub, *q);
    sycl::free(A_gpu->scubufs[streamId].usub, *q);


    sycl::free(A_gpu->local_l_blk_infoVec, *q);
    sycl::free(A_gpu->local_l_blk_infoPtr, *q);
    sycl::free(A_gpu->jib_lookupVec, *q);
    sycl::free(A_gpu->jib_lookupPtr, *q);
    sycl::free(A_gpu->local_u_blk_infoVec, *q);
    sycl::free(A_gpu->local_u_blk_infoPtr, *q);
    sycl::free(A_gpu->ijb_lookupVec, *q);
    sycl::free(A_gpu->ijb_lookupPtr, *q);

    return 0;
}



// void zPrint_matrix( char *desc, int_t m, int_t n, doublecomplex * dA, int_t lda )
// {
//     doublecomplex *cPtr = new doublecomplex[lda * n];
//     checkCuda((dpct::get_default_queue()
// 	       .memcpy(cPtr, dA, lda * n * sizeof(doublecomplex))
// 	       .wait(),
// 	       0));

//     int_t i, j;
//     printf( "\n %s\n", desc );
//     for ( i = 0; i < m; i++ )
//     {
// 	for ( j = 0; j < n; j++ ) printf( " %.3e", cPtr[i + j * lda] );
// 	printf( "\n" );
//     }
//     delete[] cPtr;
// }

// void zprintGPUStats(zLUstruct_gpu_t * A_gpu)
// {
//     double tGemm = 0;
//     double tScatter = 0;
//     double tPCIeH2D = 0;
//     double tPCIeD2H = 0;

//     for (int_t i = 0; i < A_gpu->nsupers; ++i)
//     {
// 	float milliseconds = 0;

// 	if (A_gpu->isOffloaded[i])
// 	{
// 	    cudaEventElapsedTime(&milliseconds, A_gpu->ePCIeH2D[i], A_gpu->GemmStart[i]);
// 	    tPCIeH2D += 1e-3 * (double) milliseconds;
// 	    milliseconds = 0;
// 	    cudaEventElapsedTime(&milliseconds, A_gpu->GemmStart[i], A_gpu->GemmEnd[i]);
// 	    tGemm += 1e-3 * (double) milliseconds;
// 	    milliseconds = 0;
// 	    cudaEventElapsedTime(&milliseconds, A_gpu->GemmEnd[i], A_gpu->ScatterEnd[i]);
// 	    tScatter += 1e-3 * (double) milliseconds;
// 	}

// 	milliseconds = 0;
// 	cudaEventElapsedTime(&milliseconds, A_gpu->ePCIeD2H_Start[i], A_gpu->ePCIeD2H_End[i]);
// 	tPCIeD2H += 1e-3 * (double) milliseconds;
//     }

//     printf("GPU: Flops offloaded %.3e Time spent %lf Flop rate %lf GF/sec \n",
// 	   A_gpu->GemmFLOPCounter, tGemm, 1e-9 * A_gpu->GemmFLOPCounter / tGemm  );
//     printf("GPU: Mop offloaded %.3e Time spent %lf Bandwidth %lf GByte/sec \n",
// 	   A_gpu->ScatterMOPCounter, tScatter, 8e-9 * A_gpu->ScatterMOPCounter / tScatter  );
//     printf("PCIe Data Transfer H2D:\n\tData Sent %.3e(GB)\n\tTime observed from CPU %lf\n\tActual time spent %lf\n\tBandwidth %lf GByte/sec \n",
// 	   1e-9 * A_gpu->cPCIeH2D, A_gpu->tHost_PCIeH2D, tPCIeH2D, 1e-9 * A_gpu->cPCIeH2D / tPCIeH2D  );
//     printf("PCIe Data Transfer D2H:\n\tData Sent %.3e(GB)\n\tTime observed from CPU %lf\n\tActual time spent %lf\n\tBandwidth %lf GByte/sec \n",
// 	   1e-9 * A_gpu->cPCIeD2H, A_gpu->tHost_PCIeD2H, tPCIeD2H, 1e-9 * A_gpu->cPCIeD2H / tPCIeD2H  );
//     fflush(stdout);

// } /* end printGPUStats */


int zinitSluGPU3D_t(
    zsluGPU_t *sluGPU,
    zLUstruct_t *LUstruct,
    gridinfo3d_t * grid3d,
    int_t* perm_c_supno,
    int_t n,
    int_t buffer_size,    /* read from env variable MAX_BUFFER_SIZE */
    int_t bigu_size,
    int_t ldt             /* NSUP read from sp_ienv(3) */
    )
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t *Llu = LUstruct->Llu;
    int_t* isNodeInMyGrid = sluGPU->isNodeInMyGrid;

    sluGPU->nCudaStreams = getnCudaStreams();
    if (grid3d->iam == 0)
    {
	printf("zinitSluGPU3D_t: Using hardware acceleration, with %ld cuda streams \n", sluGPU->nCudaStreams);
	fflush(stdout);
	if ( MAX_SUPER_SIZE < ldt )
	{
	    ABORT("MAX_SUPER_SIZE smaller than requested NSUP");
	}
    }

    // create a super context from platform
    sycl::platform platform(sycl::gpu_selector{});
    std::vector<sycl::device> gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
    sycl::context super_ctxt = sycl::context(gpu_devices);

    auto asyncHandler = [&](sycl::exception_list eL) {
	for (auto& e : eL) {
	    try {
		std::rethrow_exception(e);
	    } catch (sycl::exception& e) {
		std::cout << e.what() << std::endl;
		std::cout << "SYCL fail" << std::endl;
		std::terminate();
	    }
	}
    };

    sluGPU->CopyStream = new sycl::queue(super_ctxt,
					 sycl::gpu_selector{},
					 asyncHandler);

    for (int streamId = 0; streamId < sluGPU->nCudaStreams; streamId++)
    {
	sluGPU->funCallStreams[streamId] = new sycl::queue(super_ctxt,
							   sycl::gpu_selector{},
							   asyncHandler);
	sluGPU->lastOffloadStream[streamId] = -1;
    }

    sluGPU->A_gpu = (zLUstruct_gpu_t *) malloc (sizeof(zLUstruct_gpu_t));
    sluGPU->A_gpu->perm_c_supno = perm_c_supno;
    zCopyLUToGPU3D ( isNodeInMyGrid,
		     Llu,             /* referred to as A_host */
		     sluGPU, Glu_persist, n, grid3d, buffer_size, bigu_size, ldt
	);

    return 0;
} /* end zinitSluGPU3D_t */

int zinitD2Hreduce(
    int next_k,  d2Hreduce_t* d2Hred, int last_flag, HyP_t* HyP,
    zsluGPU_t *sluGPU, gridinfo_t *grid, zLUstruct_t *LUstruct, SCT_t* SCT
    )
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;


    // int_t next_col = SUPERLU_MIN (k0 + num_look_aheads + 1, nsupers - 1);
    // int_t next_k = perm_c_supno[next_col];  /* global block number for next colum*/
    int_t mkcol, mkrow;
    
    int_t kljb = LBj( next_k, grid );   /*local block number for next block*/
    int_t kijb = LBi( next_k, grid );   /*local block number for next block*/
    
    int_t *kindexL ;                     /*for storing index vectors*/
    int_t *kindexU ;
    mkrow = PROW (next_k, grid);
    mkcol = PCOL (next_k, grid);
    int_t ksup_size = SuperSize(next_k);
    
    int_t copyL_kljb = 0;
    int_t copyU_kljb = 0;
    int_t l_copy_len = 0;
    int_t u_copy_len = 0;
    
    if (mkcol == mycol &&  Lrowind_bc_ptr[kljb] != nullptr  && last_flag)
    {
	if (HyP->Lblock_dirty_bit[kljb] > -1)
	{
	    copyL_kljb = 1;
	    int_t lastk0 = HyP->Lblock_dirty_bit[kljb];
	    int_t streamIdk0Offload =  lastk0 % sluGPU->nCudaStreams;
	    if (sluGPU->lastOffloadStream[streamIdk0Offload] == lastk0 && lastk0 != -1)
	    {
		// printf("Waiting for Offload =%d to finish StreamId=%d\n", lastk0, streamIdk0Offload);
		double ttx = SuperLU_timer_();
		(sluGPU->funCallStreams[streamIdk0Offload])->wait();		
		SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
		sluGPU->lastOffloadStream[streamIdk0Offload] = -1;
	    }
	}

	kindexL = Lrowind_bc_ptr[kljb];
	l_copy_len = kindexL[1] * ksup_size;
    }

    if ( mkrow == myrow && Ufstnz_br_ptr[kijb] != nullptr    && last_flag )
    {
	if (HyP->Ublock_dirty_bit[kijb] > -1)
	{
	    copyU_kljb = 1;
	    int_t lastk0 = HyP->Ublock_dirty_bit[kijb];
	    int_t streamIdk0Offload =  lastk0 % sluGPU->nCudaStreams;
	    if (sluGPU->lastOffloadStream[streamIdk0Offload] == lastk0 && lastk0 != -1)
	    {
		// printf("Waiting for Offload =%d to finish StreamId=%d\n", lastk0, streamIdk0Offload);
		double ttx = SuperLU_timer_();
		(sluGPU->funCallStreams[streamIdk0Offload])->wait();		
		SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
		sluGPU->lastOffloadStream[streamIdk0Offload] = -1;
	    }
	}
	// copyU_kljb = HyP->Ublock_dirty_bit[kijb]>-1? 1: 0;
	kindexU = Ufstnz_br_ptr[kijb];
	u_copy_len = kindexU[1];
    }

    // wait for streams if they have not been finished
    
    // d2Hred->next_col = next_col;
    d2Hred->next_k = next_k;
    d2Hred->kljb = kljb;
    d2Hred->kijb = kijb;
    d2Hred->copyL_kljb = copyL_kljb;
    d2Hred->copyU_kljb = copyU_kljb;
    d2Hred->l_copy_len = l_copy_len;
    d2Hred->u_copy_len = u_copy_len;
    d2Hred->kindexU = kindexU;
    d2Hred->kindexL = kindexL;
    d2Hred->mkrow = mkrow;
    d2Hred->mkcol = mkcol;
    d2Hred->ksup_size = ksup_size;
    return 0;
} /* zinitD2Hreduce */

int zreduceGPUlu(
    int last_flag,
    d2Hreduce_t* d2Hred,
    zsluGPU_t *sluGPU,
    SCT_t *SCT,
    gridinfo_t *grid,
    zLUstruct_t *LUstruct
    )
{
    zLocalLU_t *Llu = LUstruct->Llu;
    int iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    doublecomplex** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    doublecomplex** Unzval_br_ptr = Llu->Unzval_br_ptr;

    sycl::queue *CopyStream;
    zLUstruct_gpu_t *A_gpu;
    A_gpu = sluGPU->A_gpu;
    CopyStream = sluGPU->CopyStream;

    int_t kljb = d2Hred->kljb;
    int_t kijb = d2Hred->kijb;
    int_t copyL_kljb = d2Hred->copyL_kljb;
    int_t copyU_kljb = d2Hred->copyU_kljb;
    int_t mkrow = d2Hred->mkrow;
    int_t mkcol = d2Hred->mkcol;
    int_t ksup_size = d2Hred->ksup_size;
    int_t *kindex;
    if ((copyL_kljb || copyU_kljb) && last_flag )
    {
	double ttx = SuperLU_timer_();
	CopyStream->wait();
	SCT->PhiWaitTimer_2 += SuperLU_timer_() - ttx;
    }

    double tt_start = SuperLU_timer_();

    if (last_flag) {
	if (mkcol == mycol && Lrowind_bc_ptr[kljb] != nullptr )
	{
	    kindex = Lrowind_bc_ptr[kljb];
	    int_t len = kindex[1];

	    if (copyL_kljb)
	    {
		doublecomplex *nzval_host;
		nzval_host = Lnzval_bc_ptr[kljb];
		int_t llen = ksup_size * len;
		doublecomplex alpha = {1.0, 0.0};
		superlu_zaxpy (llen, alpha, A_gpu->acc_L_buff, 1, nzval_host, 1);
	    }

	}
    }
    if (last_flag) {
	if (mkrow == myrow && Ufstnz_br_ptr[kijb] != nullptr )
	{
	    kindex = Ufstnz_br_ptr[kijb];
	    int_t len = kindex[1];

	    if (copyU_kljb)
	    {
		doublecomplex *nzval_host;
		nzval_host = Unzval_br_ptr[kijb];

		doublecomplex alpha = {1.0, 0.0};
		superlu_zaxpy (len, alpha, A_gpu->acc_U_buff, 1, nzval_host, 1);
	    }
	}
    }

    double tt_end = SuperLU_timer_();
    SCT->AssemblyTimer += tt_end - tt_start;
    return 0;
} /* zreduceGPUlu */


int zwaitGPUscu(int streamId, zsluGPU_t *sluGPU, SCT_t *SCT)
{
    double ttx = SuperLU_timer_();
    (sluGPU->funCallStreams[streamId])->wait();
    SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
    return 0;
}

int zsendLUpanelGPU2HOST(
    int_t k0,
    d2Hreduce_t* d2Hred,
    zsluGPU_t *sluGPU
    )
{
    int_t kljb = d2Hred->kljb;
    int_t kijb = d2Hred->kijb;
    int_t copyL_kljb = d2Hred->copyL_kljb;
    int_t copyU_kljb = d2Hred->copyU_kljb;
    int_t l_copy_len = d2Hred->l_copy_len;
    int_t u_copy_len = d2Hred->u_copy_len;
    sycl::queue *CopyStream = sluGPU->CopyStream;;
    zLUstruct_gpu_t *A_gpu = sluGPU->A_gpu;
    double tty = SuperLU_timer_();

    //cudaEventRecord(A_gpu->ePCIeD2H_Start[k0], CopyStream);

    if (copyL_kljb)
	CopyStream->memcpy(A_gpu->acc_L_buff,
			   &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[kljb]],
			   l_copy_len * sizeof(doublecomplex));

    if (copyU_kljb)
	CopyStream->memcpy(A_gpu->acc_U_buff,
			   &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[kijb]],
			   u_copy_len * sizeof(doublecomplex));

    //cudaEventRecord(A_gpu->ePCIeD2H_End[k0], CopyStream);
    A_gpu->tHost_PCIeD2H += SuperLU_timer_() - tty;
    A_gpu->cPCIeD2H += u_copy_len * sizeof(doublecomplex) + l_copy_len * sizeof(doublecomplex);

    return 0;
}

/* Copy L and U panel data structures from host to the host part of the
   data structures in A_gpu.
   GPU is not involved in this routine. */
int zsendSCUdataHost2GPU(
    int_t streamId,
    int_t* lsub,
    int_t* usub,
    doublecomplex* bigU,
    int_t bigu_send_size,
    int_t Remain_lbuf_send_size,
    zsluGPU_t *sluGPU,
    HyP_t* HyP
    )
{
    //{printf("....[enter] zsendSCUdataHost2GPU, bigu_send_size %d\n", bigu_send_size); fflush(stdout);}

    int_t usub_len = usub[2];
    int_t lsub_len = lsub[1] + BC_HEADER + lsub[0] * LB_DESCRIPTOR;
    //{printf("....[2] in zsendSCUdataHost2GPU, lsub_len %d\n", lsub_len); fflush(stdout);}
    zLUstruct_gpu_t *A_gpu = sluGPU->A_gpu;
    memcpy(A_gpu->scubufs[streamId].lsub_buf, lsub, sizeof(int_t)*lsub_len);
    memcpy(A_gpu->scubufs[streamId].usub_buf, usub, sizeof(int_t)*usub_len);
    memcpy(A_gpu->scubufs[streamId].Remain_info_host, HyP->Remain_info,
	   sizeof(Remain_info_t)*HyP->RemainBlk);
    memcpy(A_gpu->scubufs[streamId].Ublock_info_host, HyP->Ublock_info_Phi,
	   sizeof(Ublock_info_t)*HyP->num_u_blks_Phi);
    memcpy(A_gpu->scubufs[streamId].Remain_L_buff_host, HyP->Remain_L_buff,
	   sizeof(doublecomplex)*Remain_lbuf_send_size);
    memcpy(A_gpu->scubufs[streamId].bigU_host, bigU,
	   sizeof(doublecomplex)*bigu_send_size);

    return 0;
}

/* Sherry: not used ?*/
#if 0
int freeSluGPU(zsluGPU_t *sluGPU)
{
    return 0;
}
#endif

void zCopyLUToGPU3D (
    int_t* isNodeInMyGrid,
    zLocalLU_t *A_host, /* distributed LU structure on host */
    zsluGPU_t *sluGPU,
    Glu_persist_t *Glu_persist, int_t n,
    gridinfo3d_t *grid3d,
    int_t buffer_size, /* bigV size on GPU for Schur complement update */
    int_t bigu_size,
    int_t ldt
    )
{
    sycl::queue* q = sluGPU->funCallStreams[0];

    gridinfo_t* grid = &(grid3d->grid2d);
    zLUstruct_gpu_t * A_gpu =  sluGPU->A_gpu;
    zLUstruct_gpu_t **dA_gpu =  &(sluGPU->dA_gpu);

    int_t *xsup ;
    xsup = Glu_persist->xsup;
    int iam = grid->iam;
    int nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t mrb =    (nsupers + Pr - 1) / Pr;
    int_t mcb =    (nsupers + Pc - 1) / Pc;
    int_t remain_l_max = A_host->bufmax[1];

    /*copies of scalars for easy access*/
    A_gpu->nsupers = nsupers;
    A_gpu->ScatterMOPCounter = 0;
    A_gpu->GemmFLOPCounter = 0;
    A_gpu->cPCIeH2D = 0;
    A_gpu->cPCIeD2H = 0;
    A_gpu->tHost_PCIeH2D = 0;
    A_gpu->tHost_PCIeD2H = 0;

    /*initializing memory*/
    size_t max_gpu_memory = get_acc_memory ();
    size_t gpu_mem_used = 0;

    void *tmp_ptr;

    A_gpu->xsup_host = xsup;

    int_t nCudaStreams = sluGPU->nCudaStreams;
    /*pinned memory allocations.
      Paged-locked memory by cudaMallocHost is accessible to the device.*/
    for (int streamId = 0; streamId < nCudaStreams; streamId++ ) {
	void *tmp_ptr;
	tmp_ptr = (void *)sycl::malloc_host((n) * sizeof(int_t), *q);
        A_gpu->scubufs[streamId].usub_IndirectJ3_host = (int_t*) tmp_ptr;

	tmp_ptr = (void *)sycl::malloc_device((n) * sizeof(int_t), *q);
        A_gpu->scubufs[streamId].usub_IndirectJ3 =  (int_t*) tmp_ptr;
	gpu_mem_used += ( n) * sizeof(int_t);
        tmp_ptr = (void *)sycl::malloc_host(mrb * sizeof(Remain_info_t), *q);
        A_gpu->scubufs[streamId].Remain_info_host = (Remain_info_t*)tmp_ptr;
        tmp_ptr = (void *)sycl::malloc_host(mcb * sizeof(Ublock_info_t), *q);
        A_gpu->scubufs[streamId].Ublock_info_host = (Ublock_info_t*)tmp_ptr;
        tmp_ptr = (void *)sycl::malloc_host(remain_l_max * sizeof(doublecomplex), *q);
        A_gpu->scubufs[streamId].Remain_L_buff_host = (doublecomplex *) tmp_ptr;
        tmp_ptr = (void *)sycl::malloc_host(bigu_size * sizeof(doublecomplex), *q);
        A_gpu->scubufs[streamId].bigU_host = (doublecomplex *) tmp_ptr;

	tmp_ptr = (void *)sycl::malloc_host(sizeof(doublecomplex) * (A_host->bufmax[1]), *q);
        A_gpu->acc_L_buff = (doublecomplex *) tmp_ptr;
	tmp_ptr = (void *)sycl::malloc_host(sizeof(doublecomplex) * (A_host->bufmax[3]), *q);
        A_gpu->acc_U_buff = (doublecomplex *) tmp_ptr;
        tmp_ptr = (void *)sycl::malloc_host(sizeof(int_t) * (A_host->bufmax[0]), *q);
        A_gpu->scubufs[streamId].lsub_buf =  (int_t *) tmp_ptr;
        tmp_ptr = (void *)sycl::malloc_host(sizeof(int_t) * (A_host->bufmax[2]), *q);
        A_gpu->scubufs[streamId].usub_buf = (int_t *) tmp_ptr;

        tmp_ptr = (void *)sycl::malloc_device(remain_l_max * sizeof(doublecomplex), *q);
        A_gpu->scubufs[streamId].Remain_L_buff = (doublecomplex *) tmp_ptr;
	gpu_mem_used += remain_l_max * sizeof(doublecomplex);
        tmp_ptr = (void *)sycl::malloc_device(bigu_size * sizeof(doublecomplex), *q);
        A_gpu->scubufs[streamId].bigU = (doublecomplex *) tmp_ptr;
	gpu_mem_used += bigu_size * sizeof(doublecomplex);
        tmp_ptr = (void *)sycl::malloc_device(mcb * sizeof(Ublock_info_t), *q);
        A_gpu->scubufs[streamId].Ublock_info = (Ublock_info_t *) tmp_ptr;
	gpu_mem_used += mcb * sizeof(Ublock_info_t);
        tmp_ptr = (void *)sycl::malloc_device(mrb * sizeof(Remain_info_t), *q);
        A_gpu->scubufs[streamId].Remain_info = (Remain_info_t *) tmp_ptr;
	gpu_mem_used += mrb * sizeof(Remain_info_t);
        tmp_ptr = (void *)sycl::malloc_device(buffer_size * sizeof(doublecomplex), *q);
        A_gpu->scubufs[streamId].bigV = (doublecomplex *) tmp_ptr;
	gpu_mem_used += buffer_size * sizeof(doublecomplex);
        tmp_ptr = (void *)sycl::malloc_device(A_host->bufmax[0] * sizeof(int_t), *q);
        A_gpu->scubufs[streamId].lsub = (int_t *) tmp_ptr;
	gpu_mem_used += A_host->bufmax[0] * sizeof(int_t);
        tmp_ptr = (void *)sycl::malloc_device(A_host->bufmax[2] * sizeof(int_t), *q);
        A_gpu->scubufs[streamId].usub = (int_t *) tmp_ptr;
	gpu_mem_used += A_host->bufmax[2] * sizeof(int_t);
	
    } /* endfor streamID ... allocate paged-locked memory */

    A_gpu->isOffloaded = (int *) SUPERLU_MALLOC (sizeof(int) * nsupers);
    // A_gpu->GemmStart      = new sycl::event[nsupers];
    // A_gpu->GemmEnd        = new sycl::event[nsupers];
    // A_gpu->ScatterEnd     = new sycl::event[nsupers];
    // A_gpu->ePCIeH2D       = new sycl::event[nsupers];
    // A_gpu->ePCIeD2H_Start = new sycl::event[nsupers];
    // A_gpu->ePCIeD2H_End   = new sycl::event[nsupers];

    // for (int i = 0; i < nsupers; ++i)
    // {
    // 	A_gpu->isOffloaded[i] = 0;
    // 	A_gpu->GemmStart[i]      = sycl::event();
    // 	A_gpu->GemmEnd[i]        = sycl::event();
    // 	A_gpu->ScatterEnd[i]     = sycl::event();
    // 	A_gpu->ePCIeH2D[i]       = sycl::event();
    // 	A_gpu->ePCIeD2H_Start[i] = sycl::event();
    // 	A_gpu->ePCIeD2H_End[i]   = sycl::event();
    // }

    /*---- Copy L data structure to GPU ----*/

    /*pointers and address of local blocks for easy accessibility */
    local_l_blk_info_t  *local_l_blk_infoVec;
    int_t  * local_l_blk_infoPtr;
    local_l_blk_infoPtr =  (int_t *) malloc( CEILING(nsupers, Pc) * sizeof(int_t ) );

    /* First pass: count total L blocks */
    int_t cum_num_l_blocks = 0;  /* total number of L blocks I own */
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
	/* going through each block column I own */

	if (A_host->Lrowind_bc_ptr[i] != nullptr && isNodeInMyGrid[i * Pc + mycol] == 1)
	{
	    int_t *index = A_host->Lrowind_bc_ptr[i];
	    int_t num_l_blocks = index[0];
	    cum_num_l_blocks += num_l_blocks;
	}
    }

    /*allocating memory*/
    local_l_blk_infoVec =  (local_l_blk_info_t *) malloc(cum_num_l_blocks * sizeof(local_l_blk_info_t));

    /* Second pass: set up the meta-data for the L structure */
    cum_num_l_blocks = 0;

    /*initialzing vectors */
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
	if (A_host->Lrowind_bc_ptr[i] != nullptr && isNodeInMyGrid[i * Pc + mycol] == 1)
	{
	    int_t *index = A_host->Lrowind_bc_ptr[i];
	    int_t num_l_blocks = index[0]; /* # L blocks in this column */

	    if (num_l_blocks > 0)
	    {

		local_l_blk_info_t *local_l_blk_info_i = local_l_blk_infoVec + cum_num_l_blocks;
		local_l_blk_infoPtr[i] = cum_num_l_blocks;

		int_t lptrj = BC_HEADER;
		int_t luptrj = 0;

		for (int_t j = 0; j < num_l_blocks ; ++j)
		{

		    int_t ijb = index[lptrj];

		    local_l_blk_info_i[j].lib = ijb / Pr;
		    local_l_blk_info_i[j].lptrj = lptrj;
		    local_l_blk_info_i[j].luptrj = luptrj;
		    luptrj += index[lptrj + 1];
		    lptrj += LB_DESCRIPTOR + index[lptrj + 1];
					
		}
	    }
	    cum_num_l_blocks += num_l_blocks;
	}

    } /* endfor all block columns */

    /* Allocate L memory on GPU, and copy the values from CPU to GPU */
    tmp_ptr = (void *)sycl::malloc_device(cum_num_l_blocks * sizeof(local_l_blk_info_t), *q);
    A_gpu->local_l_blk_infoVec = (local_l_blk_info_t *) tmp_ptr;
    gpu_mem_used += cum_num_l_blocks * sizeof(local_l_blk_info_t);
    q->memcpy( (A_gpu->local_l_blk_infoVec), local_l_blk_infoVec, cum_num_l_blocks * sizeof(local_l_blk_info_t) );

    tmp_ptr = (void *)sycl::malloc_device(CEILING(nsupers, Pc) * sizeof(int_t), *q);
    A_gpu->local_l_blk_infoPtr = (int_t *) tmp_ptr;
    gpu_mem_used += CEILING(nsupers, Pc) * sizeof(int_t);
    q->memcpy( (A_gpu->local_l_blk_infoPtr), local_l_blk_infoPtr, CEILING(nsupers, Pc)*sizeof(int_t) );

    /*---- Copy U data structure to GPU ----*/

    local_u_blk_info_t  *local_u_blk_infoVec;
    int_t  * local_u_blk_infoPtr;
    local_u_blk_infoPtr =  (int_t *) malloc( CEILING(nsupers, Pr) * sizeof(int_t ) );

    /* First pass: count total U blocks */
    int_t cum_num_u_blocks = 0;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {

	if (A_host->Ufstnz_br_ptr[i] != nullptr && isNodeInMyGrid[i * Pr + myrow] == 1)
	{
	    int_t *index = A_host->Ufstnz_br_ptr[i];
	    int_t num_u_blocks = index[0];
	    cum_num_u_blocks += num_u_blocks;

	}
    }

    local_u_blk_infoVec =  (local_u_blk_info_t *) malloc(cum_num_u_blocks * sizeof(local_u_blk_info_t));

    /* Second pass: set up the meta-data for the U structure */
    cum_num_u_blocks = 0;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
	if (A_host->Ufstnz_br_ptr[i] != nullptr && isNodeInMyGrid[i * Pr + myrow] == 1)
	{
	    int_t *index = A_host->Ufstnz_br_ptr[i];
	    int_t num_u_blocks = index[0];

	    if (num_u_blocks > 0)
	    {
		local_u_blk_info_t  *local_u_blk_info_i = local_u_blk_infoVec + cum_num_u_blocks;
		local_u_blk_infoPtr[i] = cum_num_u_blocks;

		int_t iuip_lib, ruip_lib;
		iuip_lib = BR_HEADER;
		ruip_lib = 0;

		for (int_t j = 0; j < num_u_blocks ; ++j)
		{

		    int_t ijb = index[iuip_lib];
		    local_u_blk_info_i[j].ljb = ijb / Pc;
		    local_u_blk_info_i[j].iuip = iuip_lib;
		    local_u_blk_info_i[j].ruip = ruip_lib;

		    ruip_lib += index[iuip_lib + 1];
		    iuip_lib += UB_DESCRIPTOR + SuperSize (ijb);

		}
	    }
	    cum_num_u_blocks +=  num_u_blocks;
	}
    }

    tmp_ptr = (void *)sycl::malloc_device(cum_num_u_blocks * sizeof(local_u_blk_info_t), *q);
    A_gpu->local_u_blk_infoVec = (local_u_blk_info_t *) tmp_ptr;
    gpu_mem_used += cum_num_u_blocks * sizeof(local_u_blk_info_t);
    q->memcpy( (A_gpu->local_u_blk_infoVec), local_u_blk_infoVec, cum_num_u_blocks * sizeof(local_u_blk_info_t) );

    tmp_ptr = (void *)sycl::malloc_device(CEILING(nsupers, Pr) * sizeof(int_t), *q);
    A_gpu->local_u_blk_infoPtr = (int_t *) tmp_ptr;
    gpu_mem_used += CEILING(nsupers, Pr) * sizeof(int_t);
    q->memcpy( (A_gpu->local_u_blk_infoPtr), local_u_blk_infoPtr, CEILING(nsupers, Pr)*sizeof(int_t) );

    /* Copy the actual L indices and values */
    int_t l_k = CEILING( nsupers, grid->npcol ); /* # of local block columns */
    int_t *temp_LrowindPtr    = (int_t *) malloc(sizeof(int_t) * l_k);
    int_t *temp_LnzvalPtr     = (int_t *) malloc(sizeof(int_t) * l_k);
    int_t *Lnzval_size = (int_t *) malloc(sizeof(int_t) * l_k);
    int_t l_ind_len = 0;
    int_t l_val_len = 0;
    for (int_t jb = 0; jb < nsupers; ++jb) /* for each block column ... */
    {
	int_t pc = PCOL( jb, grid );
	if (mycol == pc && isNodeInMyGrid[jb] == 1)
	{
	    int_t ljb = LBj( jb, grid ); /* Local block number */
	    int_t  *index_host;
	    index_host = A_host->Lrowind_bc_ptr[ljb];

	    temp_LrowindPtr[ljb] = l_ind_len;
	    temp_LnzvalPtr[ljb] = l_val_len;        // ###
	    Lnzval_size[ljb] = 0;       //###
	    if (index_host != nullptr)
	    {
		int_t nrbl  = index_host[0];   /* number of L blocks */
		int_t len   = index_host[1];   /* LDA of the nzval[] */
		int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;

		/* Global block number is mycol +  ljb*Pc */
		int_t nsupc = SuperSize(jb);

		l_ind_len += len1;
		l_val_len += len * nsupc;
		Lnzval_size[ljb] = len * nsupc ; // ###
	    }
	    else
	    {
		Lnzval_size[ljb] = 0 ; // ###
	    }
	}
    } /* endfor jb = 0 ... */

    /* Copy the actual U indices and values */
    int_t u_k = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
    int_t *temp_UrowindPtr    = (int_t *) malloc(sizeof(int_t) * u_k);
    int_t *temp_UnzvalPtr     = (int_t *) malloc(sizeof(int_t) * u_k);
    int_t *Unzval_size = (int_t *) malloc(sizeof(int_t) * u_k);
    int_t u_ind_len = 0;
    int_t u_val_len = 0;
    for ( int_t lb = 0; lb < u_k; ++lb)
    {
	int_t *index_host;
	index_host =  A_host->Ufstnz_br_ptr[lb];
	temp_UrowindPtr[lb] = u_ind_len;
	temp_UnzvalPtr[lb] = u_val_len;
	Unzval_size[lb] = 0;
	if (index_host != nullptr && isNodeInMyGrid[lb * Pr + myrow] == 1)
	{
	    int_t len = index_host[1];
	    int_t len1 = index_host[2];
		    
	    u_ind_len += len1;
	    u_val_len += len;
	    Unzval_size[lb] = len;
	}
	else
	{
	    Unzval_size[lb] = 0;
	}
    }

    gpu_mem_used += l_ind_len * sizeof(int_t);
    gpu_mem_used += 2 * l_k * sizeof(int_t);
    gpu_mem_used += u_ind_len * sizeof(int_t);
    gpu_mem_used += 2 * u_k * sizeof(int_t);

    /*left memory shall be divided among the two */

    for (int_t i = 0;  i < l_k; ++i)
    {
	temp_LnzvalPtr[i] = -1;
    }

    for (int_t i = 0; i < u_k; ++i)
    {
	temp_UnzvalPtr[i] = -1;
    }

    /*setting these pointers back */
    l_val_len = 0;
    u_val_len = 0;

    int_t num_gpu_l_blocks = 0;
    int_t num_gpu_u_blocks = 0;
    size_t mem_l_block, mem_u_block;

    /* Find the trailing matrix size that can fit into GPU memory */
    for (int_t i = nsupers - 1; i > -1; --i)
    {
	/* ulte se chalte hai eleimination tree  */
	/* bottom up ordering  */
	int_t i_sup = A_gpu->perm_c_supno[i];

	int_t pc = PCOL( i_sup, grid );
	if (isNodeInMyGrid[i_sup] == 1)
	{
	    if (mycol == pc )
	    {
		int_t ljb  = LBj(i_sup, grid);
		mem_l_block = sizeof(doublecomplex) * Lnzval_size[ljb];
		if (gpu_mem_used + mem_l_block > max_gpu_memory)
		{
		    break;
		}
		else
		{
		    gpu_mem_used += mem_l_block;
		    temp_LnzvalPtr[ljb] = l_val_len;
		    l_val_len += Lnzval_size[ljb];
		    num_gpu_l_blocks++;
		    A_gpu->first_l_block_gpu = i;
		}
	    }

	    int_t pr = PROW( i_sup, grid );
	    if (myrow == pr)
	    {
		int_t lib  = LBi(i_sup, grid);
		mem_u_block = sizeof(doublecomplex) * Unzval_size[lib];
		if (gpu_mem_used + mem_u_block > max_gpu_memory)
		{
		    break;
		}
		else
		{
		    gpu_mem_used += mem_u_block;
		    temp_UnzvalPtr[lib] = u_val_len;
		    u_val_len += Unzval_size[lib];
		    num_gpu_u_blocks++;
		    A_gpu->first_u_block_gpu = i;
		}
	    }
	} /* endif */

    } /* endfor i .... nsupers */

// #if (PRNTlevel>=2)
//     printf("(%d) Number of L blocks in GPU %d, U blocks %d\n",
// 	   grid3d->iam, num_gpu_l_blocks, num_gpu_u_blocks );
//     printf("(%d) elimination order of first block in GPU: L block %d, U block %d\n",
// 	   grid3d->iam, A_gpu->first_l_block_gpu, A_gpu->first_u_block_gpu);
//     printf("(%d) Memory of L %.1f GB, memory for U %.1f GB, Total device memory used %.1f GB, Memory allowed %.1f GB \n", grid3d->iam,
// 	   l_val_len * sizeof(doublecomplex) * 1e-9,
// 	   u_val_len * sizeof(doublecomplex) * 1e-9,
// 	   gpu_mem_used * 1e-9, max_gpu_memory * 1e-9);
//     fflush(stdout);
// #endif

    /* Assemble index vector on temp */
    int_t *indtemp = (int_t *) malloc(sizeof(int_t) * l_ind_len);
    for (int_t jb = 0; jb < nsupers; ++jb)   /* for each block column ... */
    {
	int_t pc = PCOL( jb, grid );
	if (mycol == pc && isNodeInMyGrid[jb] == 1)
	{
	    int_t ljb = LBj( jb, grid ); /* Local block number */
	    int_t  *index_host;
	    index_host = A_host->Lrowind_bc_ptr[ljb];

	    if (index_host != nullptr)
	    {
		int_t nrbl  =   index_host[0]; /* number of L blocks */
		int_t len   = index_host[1];   /* LDA of the nzval[] */
		int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
			    
		memcpy(&indtemp[temp_LrowindPtr[ljb]] , index_host, len1 * sizeof(int_t)) ;
	    }
	}
    }

    tmp_ptr = (void *)sycl::malloc_device(l_ind_len * sizeof(int_t), *q);
    A_gpu->LrowindVec = (int_t *) tmp_ptr;
    q->memcpy( (A_gpu->LrowindVec), indtemp, l_ind_len * sizeof(int_t) );

    tmp_ptr = (void *)sycl::malloc_device(l_val_len * sizeof(doublecomplex), *q);
    A_gpu->LnzvalVec = (doublecomplex *) tmp_ptr;
    q->memset( (A_gpu->LnzvalVec), 0, l_val_len * sizeof(doublecomplex) );

    tmp_ptr = (void *)sycl::malloc_device(l_k * sizeof(int_t), *q);
    A_gpu->LrowindPtr = (int_t *) tmp_ptr;
    q->memcpy( (A_gpu->LrowindPtr), temp_LrowindPtr, l_k * sizeof(int_t) ) ;

    tmp_ptr = (void *)sycl::malloc_device(l_k * sizeof(int_t), *q);
    A_gpu->LnzvalPtr = (int_t *) tmp_ptr;
    q->memcpy( (A_gpu->LnzvalPtr), temp_LnzvalPtr, l_k * sizeof(int_t) );

    A_gpu->LnzvalPtr_host = temp_LnzvalPtr;

    int_t *indtemp1 = (int_t *) malloc(sizeof(int_t) * u_ind_len);
    for ( int_t lb = 0; lb < u_k; ++lb)
    {
	int_t *index_host;
	index_host =  A_host->Ufstnz_br_ptr[lb];

	if (index_host != nullptr && isNodeInMyGrid[lb * Pr + myrow] == 1)
	{
	    int_t len1 = index_host[2];
	    memcpy(&indtemp1[temp_UrowindPtr[lb]] , index_host, sizeof(int_t)*len1);
	}
    }

    tmp_ptr = (void *)sycl::malloc_device(u_ind_len * sizeof(int_t), *q);
    A_gpu->UrowindVec = (int_t *) tmp_ptr;
    q->memcpy( (A_gpu->UrowindVec), indtemp1, u_ind_len * sizeof(int_t) ) ;

    tmp_ptr = (void *)sycl::malloc_device(u_val_len * sizeof(doublecomplex), *q);
    A_gpu->UnzvalVec = (doublecomplex *) tmp_ptr;
    q->memset( (A_gpu->UnzvalVec), 0, u_val_len * sizeof(doublecomplex));

    tmp_ptr = (void *)sycl::malloc_device(u_k * sizeof(int_t), *q);
    A_gpu->UrowindPtr = (int_t *) tmp_ptr;
    q->memcpy( (A_gpu->UrowindPtr), temp_UrowindPtr, u_k * sizeof(int_t) );

    A_gpu->UnzvalPtr_host = temp_UnzvalPtr;

    tmp_ptr = (void *)sycl::malloc_device(u_k * sizeof(int_t), *q);
    A_gpu->UnzvalPtr = (int_t *) tmp_ptr;
    q->memcpy( (A_gpu->UnzvalPtr), temp_UnzvalPtr, u_k * sizeof(int_t) );

    tmp_ptr = (void *)sycl::malloc_device((nsupers + 1) * sizeof(int_t), *q);
    A_gpu->xsup = (int_t *) tmp_ptr;
    q->memcpy( (A_gpu->xsup), xsup, (nsupers + 1)*sizeof(int_t) );

    tmp_ptr = (void *)sycl::malloc_device(sizeof(zLUstruct_gpu_t), *q);
    *dA_gpu = (zLUstruct_gpu_t *) tmp_ptr;
    q->memcpy( *dA_gpu, A_gpu, sizeof(zLUstruct_gpu_t) ) ;

    free (temp_LrowindPtr);
    free (temp_UrowindPtr);
    free (indtemp1);
    free (indtemp);

} /* end zCopyLUToGPU3D */



int zreduceAllAncestors3d_GPU(int_t ilvl, int_t* myNodeCount,
			      int_t** treePerm,
			      zLUValSubBuf_t*LUvsb,
			      zLUstruct_t* LUstruct,
			      gridinfo3d_t* grid3d,
			      zsluGPU_t *sluGPU,
			      d2Hreduce_t* d2Hred,
			      factStat_t *factStat,
			      HyP_t* HyP, SCT_t* SCT )
{
    // first synchronize all cuda streams
    int superlu_acc_offload =   HyP->superlu_acc_offload;

    int_t maxLvl = log2i( (int_t) grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;
    gridinfo_t* grid = &(grid3d->grid2d);
    int_t* gpuLUreduced = factStat->gpuLUreduced;

    int_t sender;
    if ((myGrid % (1 << (ilvl + 1))) == 0)
    {
	sender = myGrid + (1 << ilvl);
	    
    }
    else
    {
	sender = myGrid;
    }

    /*Reduce all the ancestors from the GPU*/
    if (myGrid == sender && superlu_acc_offload)
    {
        for (int_t streamId = 0; streamId < sluGPU->nCudaStreams; streamId++)
	{
	    double ttx = SuperLU_timer_();
	    (sluGPU->funCallStreams[streamId])->wait();
	    SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
	    sluGPU->lastOffloadStream[streamId] = -1;
	}

	for (int_t alvl = ilvl + 1; alvl < maxLvl; ++alvl)
	{
	    /* code */
	    // int_t atree = myTreeIdxs[alvl];
	    int_t nsAncestor = myNodeCount[alvl];
	    int_t* cAncestorList = treePerm[alvl];

	    for (int_t node = 0; node < nsAncestor; node++ )
	    {
	        int_t k = cAncestorList[node];
	        if (!gpuLUreduced[k])
		{
		    zinitD2Hreduce(k, d2Hred, 1,
				   HyP, sluGPU, grid, LUstruct, SCT);
		    int_t copyL_kljb = d2Hred->copyL_kljb;
		    int_t copyU_kljb = d2Hred->copyU_kljb;

		    double tt_start1 = SuperLU_timer_();
		    SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
		    if (copyL_kljb || copyU_kljb) SCT->PhiMemCpyCounter++;
		    zsendLUpanelGPU2HOST(k, d2Hred, sluGPU);
		    /*
		      Reduce the LU panels from GPU
		    */
		    zreduceGPUlu(1, d2Hred, sluGPU, SCT, grid, LUstruct);
		    gpuLUreduced[k] = 1;
		}
	    }
	}
    } /*if (myGrid == sender)*/

    zreduceAllAncestors3d(ilvl, myNodeCount, treePerm,
			  LUvsb, LUstruct, grid3d, SCT );
    return 0;
} /* zreduceAllAncestors3d_GPU */


void zsyncAllfunCallStreams(zsluGPU_t* sluGPU, SCT_t* SCT)
{
    for (int streamId = 0; streamId < sluGPU->nCudaStreams; streamId++)
    {
        double ttx = SuperLU_timer_();
        (sluGPU->funCallStreams[streamId])->wait();
	SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
        sluGPU->lastOffloadStream[streamId] = -1;
    }
}
