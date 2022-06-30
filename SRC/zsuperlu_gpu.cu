
/*! @file
 * \brief Descriptions and declarations for structures used in GPU
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology, Oak Ridge National Laboratory
 *
 * Last update: November 14, 2021  remove dependence on CUB/scan
 * </pre>
 */

//#define GPU_DEBUG

#include "superlu_defs.h"

#undef Reduce

//#include <thrust/system/cuda/detail/cub/cub.cuh>

#include "zlustruct_gpu.h"

#include "dcomplex.h"

//extern "C" {
//	void cblas_daxpy(const int N, const double alpha, const double *X,
//	                 const int incX, double *Y, const int incY);
//}

// gpublasStatus_t checkGPUblas(gpublasStatus_t result)
// {
// #if defined(DEBUG) || defined(_DEBUG)
// 	if (result != GPUBLAS_STATUS_SUCCESS)
// 	{
// 		fprintf(stderr, "GPU BLAS Runtime Error: %s\n", gpublasGetErrorString(result));
// 		assert(result == GPUBLAS_STATUS_SUCCESS);
// 	}
// #endif
// 	return result;
// }


// #define UNIT_STRIDE

#if 0  ////////// this routine is not used anymore
__device__ inline
void device_scatter_l (int_t thread_id,
                       int_t nsupc, int_t temp_nbrow,
                       int_t *usub, int_t iukp, int_t klst,
                       doublecomplex *nzval, int_t ldv,
                       doublecomplex *tempv, int_t nbrow,
                       // int_t *indirect2_thread
                       int *indirect2_thread
                      )
{


	int_t segsize, jj;

	for (jj = 0; jj < nsupc; ++jj)
	{
		segsize = klst - usub[iukp + jj];
		if (segsize)
		{
			if (thread_id < temp_nbrow)
			{

#ifndef UNIT_STRIDE
				nzval[indirect2_thread[thread_id]] -= tempv[thread_id];
#else
				nzval[thread_id] -= tempv[thread_id]; /*making access unit strided*/
#endif
			}
			tempv += nbrow;
		}
		nzval += ldv;
	}
}
#endif ///////////// not used

//#define THREAD_BLOCK_SIZE  256  /* Sherry: was 192. should be <= MAX_SUPER_SIZE */

__device__ inline
void zdevice_scatter_l_2D (int thread_id,
                          int nsupc, int temp_nbrow,
                          int_t *usub, int iukp, int_t klst,
                          doublecomplex *nzval, int ldv,
                          const doublecomplex *tempv, int nbrow,
                          int *indirect2_thread,
                          int nnz_cols, int ColPerBlock,
                          int *IndirectJ3
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

/* Sherry: this routine is not used */
#if 0 //////////////////////////////////////////////
__global__
void cub_scan_test(void)
{
	int thread_id = threadIdx.x;
	typedef cub::BlockScan<int, MAX_SUPER_SIZE > BlockScan; /*1D int data type*/

	__shared__ typename BlockScan::TempStorage temp_storage; /*storage temp*/

	__shared__ int IndirectJ1[MAX_SUPER_SIZE];
	__shared__ int IndirectJ2[MAX_SUPER_SIZE];

	if (thread_id < MAX_SUPER_SIZE)
	{
		IndirectJ1[thread_id] = (thread_id + 1) % 2;
	}

	__syncthreads();
	if (thread_id < MAX_SUPER_SIZE)
		BlockScan(temp_storage).InclusiveSum (IndirectJ1[thread_id], IndirectJ2[thread_id]);


	if (thread_id < MAX_SUPER_SIZE)
		printf("%d %d\n", thread_id, IndirectJ2[thread_id]);

}
#endif  /////////////////////////////////// not used


__device__ inline
void device_scatter_u_2D (int thread_id,
                          int temp_nbrow,  int nsupc,
                          doublecomplex * ucol,
                          int_t * usub, int iukp,
                          int_t ilst, int_t klst,
                          int_t * index, int iuip_lib,
                          doublecomplex * tempv, int nbrow,
                          int *indirect,
                          int nnz_cols, int ColPerBlock,
                          int *IndirectJ1,
                          int *IndirectJ3
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
           i = IndirectJ1[IndirectJ3[col]]-ilst + indirect[thread_id_x];
	   z_sub(&ucol[i], &ucol[i], &tempv[nbrow * col + thread_id_x]);
	}
    }
}

__global__
void Scatter_GPU_kernel(
    int_t streamId,
    int_t ii_st, int_t ii_end,
    int_t jj_st, int_t jj_end, /* defines rectangular Schur block to be scatter */
    int_t klst,
    int_t jj0,   /* 0 on entry */
    int_t nrows, int_t ldt, int_t npcol, int_t nprow,
    zLUstruct_gpu_t * A_gpu)
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
	int lb = blockIdx.x + ii_st;
	int j  = blockIdx.y + jj_st;
	
	extern __shared__ int s[];
	int* indirect_lptr = s;  /* row-wise */
	int* indirect2_thread= (int*) &indirect_lptr[ldt]; /* row-wise */
	int* IndirectJ1= (int*) &indirect2_thread[ldt];    /* column-wise */
	int* IndirectJ3= (int*) &IndirectJ1[ldt];    /* column-wise */
	//int THREAD_BLOCK_SIZE =ldt; 
	
	int* pfxStorage = (int*) &IndirectJ3[ldt];
	
	int thread_id = threadIdx.x;

	int iukp = Ublock_info[j].iukp;
	int jb = Ublock_info[j].jb;
	int nsupc = SuperSize (jb);
	int ljb = jb / npcol;

	typedef int pfx_dtype ;
        extern  __device__ void incScan(pfx_dtype *inOutArr, pfx_dtype *temp, int n);

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
	int cum_ncol = (j == 0) ? 0	
					: Ublock_info[j - 1].full_u_cols;

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
		__shared__ int ljb_ind;
		/*do a search ljb_ind at local row lib*/
		int blks_per_threads = CEILING(num_u_blocks, blockDim.x);
		// printf("blockDim.x =%d \n", blockDim.x);
		
		for (int i = 0; i < blks_per_threads; ++i)
			/* each thread is assigned a chunk of consecutive U blocks to search */
		{
			/* only one thread finds the block index matching ljb */
			if (thread_id * blks_per_threads + i < num_u_blocks &&
			        local_u_blk_infoVec[ local_u_blk_infoPtr[lib] + thread_id * blks_per_threads + i ].ljb == ljb)
			{
				ljb_ind = thread_id * blks_per_threads + i;
			}
		}
		__syncthreads();

		int iuip_lib = local_u_blk_infoVec[ local_u_blk_infoPtr[lib] + ljb_ind].iuip;
		int ruip_lib = local_u_blk_infoVec[ local_u_blk_infoPtr[lib] + ljb_ind].ruip;
		iuip_lib += UB_DESCRIPTOR;
		doublecomplex *Unzval_lib = &UnzvalVec[UnzvalPtr[lib]];
		doublecomplex *ucol = &Unzval_lib[ruip_lib];

		if (thread_id < temp_nbrow) /* row-wise */
		{
		    /* cyclically map each thread to a row */
		    indirect_lptr[thread_id] = (int) lsub[lptr + thread_id];
		}

		/* column-wise: each thread is assigned one column */
		if (thread_id < nnz_cols)
			IndirectJ3[thread_id] = A_gpu->scubufs[streamId].usub_IndirectJ3[cum_ncol + thread_id];
		/* indirectJ3[j] == kk means the j-th nonzero segment
		   points to column kk in this supernode */

		__syncthreads();

		/* threads are divided into multiple columns */
		int ColPerBlock = blockDim.x / temp_nbrow;

		// if (thread_id < blockDim.x)
		// 	IndirectJ1[thread_id] = 0;
		if (thread_id < ldt)
			IndirectJ1[thread_id] = 0;

		if (thread_id < blockDim.x)
		{
		    if (thread_id < nsupc)
		    {
			/* fstnz subscript of each column in the block */
			IndirectJ1[thread_id] = -index[iuip_lib + thread_id] + ilst;
		    }
		}

		/* perform an inclusive block-wide prefix sum among all threads */
		__syncthreads();
		
		incScan(IndirectJ1, pfxStorage, nsupc);
		
		__syncthreads();

		device_scatter_u_2D (
		    thread_id,
		    temp_nbrow,  nsupc,
		    ucol,
		    usub, iukp,
		    ilst, klst,
		    index, iuip_lib,
		    tempv1, nrows,
		    indirect_lptr,
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

		__shared__ int lib_ind;
		/*do a search lib_ind for lib*/
		int blks_per_threads = CEILING(num_l_blocks, blockDim.x);
		for (int i = 0; i < blks_per_threads; ++i)
		{
			if (thread_id * blks_per_threads + i < num_l_blocks &&
			        local_l_blk_infoVec[ local_l_blk_infoPtr[ljb] + thread_id * blks_per_threads + i ].lib == lib)
			{
				lib_ind = thread_id * blks_per_threads + i;
			}
		}
		__syncthreads();

		int lptrj = local_l_blk_infoVec[ local_l_blk_infoPtr[ljb] + lib_ind].lptrj;
		int luptrj = local_l_blk_infoVec[ local_l_blk_infoPtr[ljb] + lib_ind].luptrj;
		lptrj += LB_DESCRIPTOR;
		int dest_nbrow = index[lptrj - 1];

		if (thread_id < dest_nbrow)
		{
		    rel = index[lptrj + thread_id] - fnz;
		    indirect_lptr[rel] = thread_id;
		}
		__syncthreads();

		/* can be precalculated */
		if (thread_id < temp_nbrow)
		{
			rel = lsub[lptr + thread_id] - fnz;
			indirect2_thread[thread_id] = indirect_lptr[rel];
		}
		if (thread_id < nnz_cols)
			IndirectJ3[thread_id] = (int) A_gpu->scubufs[streamId].usub_IndirectJ3[cum_ncol + thread_id];
		__syncthreads();

		int ColPerBlock = blockDim.x / temp_nbrow;

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
    zsluGPU_t *sluGPU, gridinfo_t *grid,
    SuperLUStat_t *stat    
)
{
    int SCATTER_THREAD_BLOCK_SIZE=512;

	zLUstruct_gpu_t * A_gpu = sluGPU->A_gpu;
	zLUstruct_gpu_t * dA_gpu = sluGPU->dA_gpu;
	int_t nprow = grid->nprow;
	int_t npcol = grid->npcol;

	gpuStream_t FunCallStream = sluGPU->funCallStreams[streamId];
	gpublasHandle_t gpublas_handle0 = sluGPU->gpublasHandles[streamId];
	int_t * lsub = A_gpu->scubufs[streamId].lsub_buf;
	int_t * usub = A_gpu->scubufs[streamId].usub_buf;
	Remain_info_t *Remain_info = A_gpu->scubufs[streamId].Remain_info_host;
	doublecomplex * Remain_L_buff = A_gpu->scubufs[streamId].Remain_L_buff_host;
	Ublock_info_t *Ublock_info = A_gpu->scubufs[streamId].Ublock_info_host;
	doublecomplex * bigU = A_gpu->scubufs[streamId].bigU_host;

	stat->isOffloaded[k0] = 1;
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
	gpuEventRecord(stat->ePCIeH2D[k0], FunCallStream);

	checkGPU(gpuMemcpyAsync(A_gpu->scubufs[streamId].usub_IndirectJ3,
	                          A_gpu->scubufs[streamId].usub_IndirectJ3_host,
	                          ncols * sizeof(int_t), gpuMemcpyHostToDevice,
	                          FunCallStream)) ;

	checkGPU(gpuMemcpyAsync(A_gpu->scubufs[streamId].Remain_L_buff, Remain_L_buff,
	                          Remain_lbuf_send_size * sizeof(doublecomplex),
	                          gpuMemcpyHostToDevice, FunCallStream)) ;

	checkGPU(gpuMemcpyAsync(A_gpu->scubufs[streamId].bigU, bigU,
	                          bigu_send_size * sizeof(doublecomplex),
	                          gpuMemcpyHostToDevice, FunCallStream) );

	checkGPU(gpuMemcpyAsync(A_gpu->scubufs[streamId].Remain_info, Remain_info,
	                          RemainBlk * sizeof(Remain_info_t),
	                          gpuMemcpyHostToDevice, FunCallStream) );

	checkGPU(gpuMemcpyAsync(A_gpu->scubufs[streamId].Ublock_info, Ublock_info,
	                          mcb * sizeof(Ublock_info_t), gpuMemcpyHostToDevice,
	                          FunCallStream) );

	checkGPU(gpuMemcpyAsync(A_gpu->scubufs[streamId].lsub, lsub,
	                          lsub_len * sizeof(int_t), gpuMemcpyHostToDevice,
	                          FunCallStream) );

	checkGPU(gpuMemcpyAsync(A_gpu->scubufs[streamId].usub, usub,
	                          usub_len * sizeof(int_t), gpuMemcpyHostToDevice,
	                          FunCallStream) );

	stat->tHost_PCIeH2D += SuperLU_timer_() - tTmp;
	stat->cPCIeH2D += Remain_lbuf_send_size * sizeof(doublecomplex)
	                   + bigu_send_size * sizeof(doublecomplex)
	                   + RemainBlk * sizeof(Remain_info_t)
	                   + mcb * sizeof(Ublock_info_t)
	                   + lsub_len * sizeof(int_t)
	                   + usub_len * sizeof(int_t);

        doublecomplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};

        /* The following are used in gpublasZgemm() call */
        gpuDoubleComplex *cu_alpha = (gpuDoubleComplex *) &alpha;
        gpuDoubleComplex *cu_beta = (gpuDoubleComplex  *) &beta;
        gpuDoubleComplex *cu_A, *cu_B, *cu_C; /* C <- A*B */

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
			printf("!! Matrix size %lld x %lld exceeds buffer_size %lld\n",
			       nrows, ncols, buffer_size);
			fflush(stdout);
		    }
		    assert(nrows * ncols <= buffer_size);
		    gpublasSetStream(gpublas_handle0, FunCallStream);
		    gpuEventRecord(stat->GemmStart[k0], FunCallStream);
		    cu_A = (gpuDoubleComplex*) &A_gpu->scubufs[streamId].Remain_L_buff[(knsupc - ldu) * Rnbrow + st_row];
		    cu_B = (gpuDoubleComplex*) &A_gpu->scubufs[streamId].bigU[st_col * ldu];
		    cu_C = (gpuDoubleComplex*) A_gpu->scubufs[streamId].bigV;
		    gpublasZgemm(gpublas_handle0, GPUBLAS_OP_N, GPUBLAS_OP_N,
			            nrows, ncols, ldu, cu_alpha,
			            cu_A, Rnbrow, cu_B, ldu, cu_beta,
				    cu_C, nrows);

// #define SCATTER_OPT
#ifdef SCATTER_OPT
		    gpuStreamSynchronize(FunCallStream);
#warning this function is synchronous
#endif
		    gpuEventRecord(stat->GemmEnd[k0], FunCallStream);

		    stat->GemmFLOPCounter += 8.0 * (double) nrows * ncols * ldu;

		    /*
		     * Scattering the output
		     */
		     // dim3 dimBlock(THREAD_BLOCK_SIZE);   // 1d thread
		    dim3 dimBlock(ldt);   // 1d thread

		    dim3 dimGrid(ii_end - ii_st, jj_end - jj_st);

		    Scatter_GPU_kernel <<< dimGrid, dimBlock, (4*ldt + 2*SCATTER_THREAD_BLOCK_SIZE)*sizeof(int), FunCallStream>>>
			(streamId, ii_st, ii_end,  jj_st, jj_end, klst,
			 0, nrows, ldt, npcol, nprow, dA_gpu);
#ifdef SCATTER_OPT
		    gpuStreamSynchronize(FunCallStream);
#warning this function is synchrnous
#endif

		    gpuEventRecord(stat->ScatterEnd[k0], FunCallStream);

		    stat->ScatterMOPCounter +=  3.0 * (double) nrows * ncols;
		} /* endif ... none of the matrix dimension is zero. */

	    } /* end while jj_end < nub */

	} /* end while (ii_end < RemainBlk) */

	return 0;
} /* end zSchurCompUpdate_GPU */


static void print_occupancy()
{
    int blockSize;   // The launch configurator returned block size
    int minGridSize; /* The minimum grid size needed to achieve the
    		        best potential occupancy  */

    gpuOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                        Scatter_GPU_kernel, 0, 0);
    printf("Occupancy: MinGridSize %d blocksize %d \n", minGridSize, blockSize);
}

static void printDevProp(gpuDeviceProp devProp)
{
	size_t mfree, mtotal;
	gpuMemGetInfo	(&mfree, &mtotal);
	
	printf("pciBusID:                      %d\n",  devProp.pciBusID);
	printf("pciDeviceID:                   %d\n",  devProp.pciDeviceID);
	printf("GPU Name:                      %s\n",  devProp.name);
	printf("Total global memory:           %zu\n",  devProp.totalGlobalMem);
	printf("Total free memory:             %zu\n",  mfree);
	printf("Clock rate:                    %d\n",  devProp.clockRate);

	return;
}


static size_t get_acc_memory ()
{

	size_t mfree, mtotal;
	gpuMemGetInfo	(&mfree, &mtotal);
#if 0
	printf("Total memory %zu & free memory %zu\n", mtotal, mfree);
#endif
	return (size_t) (0.9 * (double) mfree) / get_mpi_process_per_gpu ();


}

/* Free all the data structures allocated on GPU.
   This routine is called from Host                 */
int zfree_LUstruct_gpu (
    zsluGPU_t * sluGPU,
    SuperLUStat_t* stat )
{
	zLUstruct_gpu_t * A_gpu = sluGPU->A_gpu;
	int streamId = 0;
    
	/* Free the L data structure on GPU */
	checkGPU(gpuFree(A_gpu->LrowindVec));
	checkGPU(gpuFree(A_gpu->LrowindPtr));

	checkGPU(gpuFree(A_gpu->LnzvalVec));
	checkGPU(gpuFree(A_gpu->LnzvalPtr));
	free(A_gpu->LnzvalPtr_host);
	
	/*freeing the pinned memory*/
	checkGPU (gpuFreeHost (A_gpu->scubufs[streamId].Remain_info_host));
	checkGPU (gpuFreeHost (A_gpu->scubufs[streamId].Ublock_info_host));
	checkGPU (gpuFreeHost (A_gpu->scubufs[streamId].Remain_L_buff_host));
	checkGPU (gpuFreeHost (A_gpu->scubufs[streamId].bigU_host));

	checkGPU(gpuFreeHost(A_gpu->acc_L_buff));
	checkGPU(gpuFreeHost(A_gpu->acc_U_buff));
	checkGPU(gpuFreeHost(A_gpu->scubufs[streamId].lsub_buf));
	checkGPU(gpuFreeHost(A_gpu->scubufs[streamId].usub_buf));


	SUPERLU_FREE(stat->isOffloaded); // changed to SUPERLU_MALLOC/SUPERLU_FREE
	SUPERLU_FREE(stat->GemmStart);
	SUPERLU_FREE(stat->GemmEnd);
	SUPERLU_FREE(stat->ScatterEnd);
	SUPERLU_FREE(stat->ePCIeH2D);
	SUPERLU_FREE(stat->ePCIeD2H_Start);
	SUPERLU_FREE(stat->ePCIeD2H_End);

	/* Free the U data structure on GPU */
	checkGPU(gpuFree(A_gpu->UrowindVec));
	checkGPU(gpuFree(A_gpu->UrowindPtr));

	//free(A_gpu->UrowindPtr_host); // Sherry: this is NOT allocated

	checkGPU(gpuFree(A_gpu->UnzvalVec));
	checkGPU(gpuFree(A_gpu->UnzvalPtr));

	checkGPU(gpuFree(A_gpu->grid));

	/* Free the Schur complement structure on GPU */
	checkGPU(gpuFree(A_gpu->scubufs[streamId].bigV));
	checkGPU(gpuFree(A_gpu->scubufs[streamId].bigU));

	checkGPU(gpuFree(A_gpu->scubufs[streamId].Remain_L_buff));
	checkGPU(gpuFree(A_gpu->scubufs[streamId].Ublock_info));
	checkGPU(gpuFree(A_gpu->scubufs[streamId].Remain_info));

	// checkGPU(gpuFree(A_gpu->indirect));
	// checkGPU(gpuFree(A_gpu->indirect2));
	checkGPU(gpuFree(A_gpu->xsup));

	checkGPU(gpuFree(A_gpu->scubufs[streamId].lsub));
	checkGPU(gpuFree(A_gpu->scubufs[streamId].usub));

	checkGPU(gpuFree(A_gpu->local_l_blk_infoVec));
	checkGPU(gpuFree(A_gpu->local_l_blk_infoPtr));
	checkGPU(gpuFree(A_gpu->jib_lookupVec));
	checkGPU(gpuFree(A_gpu->jib_lookupPtr));
	checkGPU(gpuFree(A_gpu->local_u_blk_infoVec));
	checkGPU(gpuFree(A_gpu->local_u_blk_infoPtr));
	checkGPU(gpuFree(A_gpu->ijb_lookupVec));
	checkGPU(gpuFree(A_gpu->ijb_lookupPtr));

	/* Destroy all the meta-structures associated with the streams. */
    	gpuStreamDestroy(sluGPU->CopyStream);
	for (streamId = 0; streamId < sluGPU->nGPUStreams; streamId++) {
	    gpuStreamDestroy(sluGPU->funCallStreams[streamId]);
	    gpublasDestroy(sluGPU->gpublasHandles[streamId]);
    	}
    
	return 0;
} /* end zfree_LUstruct_gpu */



void zPrint_matrix( char *desc, int_t m, int_t n, doublecomplex * dA, int_t lda )
{
	doublecomplex *cPtr = (doublecomplex *) malloc(sizeof(doublecomplex) * lda * n);
	checkGPU(gpuMemcpy( cPtr, dA,
	                      lda * n * sizeof(doublecomplex), gpuMemcpyDeviceToHost)) ;

	int_t i, j;
	printf( "\n %s\n", desc );
	for ( i = 0; i < m; i++ )
	{
		for ( j = 0; j < n; j++ ) printf( " %.3e", cPtr[i + j * lda] );
		printf( "\n" );
	}
	free(cPtr);
}


/* Initialize the GPU side of the data structure. */
int zinitSluGPU3D_t(
    zsluGPU_t *sluGPU, // LU structures on GPU, see zlustruct_gpu.h 
    zLUstruct_t *LUstruct,
    gridinfo3d_t * grid3d,
    int_t* perm_c_supno,
    int_t n,
    int_t buffer_size,    /* read from env variable SUPERLU_MAX_BUFFER_SIZE */
    int_t bigu_size,
    int_t ldt,             /* SUPERLU_MAXSUP read from sp_ienv(3) */
    SuperLUStat_t *stat
)
{
    checkGPUErrors(gpuDeviceReset ());
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t *Llu = LUstruct->Llu;
    int* isNodeInMyGrid = sluGPU->isNodeInMyGrid;

    sluGPU->nGPUStreams = getnGPUStreams();
    
    int SCATTER_THREAD_BLOCK_SIZE = ldt; 
    if(getenv("SCATTER_THREAD_BLOCK_SIZE"))
    {
	int stbs = atoi(getenv("SCATTER_THREAD_BLOCK_SIZE"));
	if(stbs>=ldt)
	{
	    SCATTER_THREAD_BLOCK_SIZE = stbs; 
	}
	
    }
    
    if (grid3d->iam == 0)
    {
	printf("dinitSluGPU3D_t: Using hardware acceleration, with %d gpu streams \n", sluGPU->nGPUStreams);
	fflush(stdout);
	printf("dinitSluGPU3D_t: Using %d threads per block for scatter \n", SCATTER_THREAD_BLOCK_SIZE);
	
	if ( MAX_SUPER_SIZE < ldt )
	{
		ABORT("MAX_SUPER_SIZE smaller than requested NSUP");
	}
    }

    gpuStreamCreate(&(sluGPU->CopyStream));

    for (int streamId = 0; streamId < sluGPU->nGPUStreams; streamId++)
    {
	gpuStreamCreate(&(sluGPU->funCallStreams[streamId]));
	gpublasCreate(&(sluGPU->gpublasHandles[streamId]));
	sluGPU->lastOffloadStream[streamId] = -1;
    }

    sluGPU->A_gpu = (zLUstruct_gpu_t *) malloc (sizeof(zLUstruct_gpu_t));
    sluGPU->A_gpu->perm_c_supno = perm_c_supno;

    /* Allocate GPU memory for the LU data structures, and copy
       the host LU structure to GPU side.  */
    zCopyLUToGPU3D ( isNodeInMyGrid,
	        Llu,             /* referred to as A_host */
	        sluGPU, Glu_persist, n, grid3d, buffer_size, bigu_size, ldt, stat
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
    
    if (mkcol == mycol &&  Lrowind_bc_ptr[kljb] != NULL  && last_flag)
    {
	if (HyP->Lblock_dirty_bit[kljb] > -1)
	    {
		copyL_kljb = 1;
		int_t lastk0 = HyP->Lblock_dirty_bit[kljb];
		int_t streamIdk0Offload =  lastk0 % sluGPU->nGPUStreams;
		if (sluGPU->lastOffloadStream[streamIdk0Offload] == lastk0 && lastk0 != -1)
		    {
			// printf("Waiting for Offload =%d to finish StreamId=%d\n", lastk0, streamIdk0Offload);
			double ttx = SuperLU_timer_();
			gpuStreamSynchronize(sluGPU->funCallStreams[streamIdk0Offload]);
			SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
			sluGPU->lastOffloadStream[streamIdk0Offload] = -1;
		    }
	    }

	kindexL = Lrowind_bc_ptr[kljb];
	l_copy_len = kindexL[1] * ksup_size;
    }

    if ( mkrow == myrow && Ufstnz_br_ptr[kijb] != NULL    && last_flag )
    {
	if (HyP->Ublock_dirty_bit[kijb] > -1)
	    {
		copyU_kljb = 1;
		int_t lastk0 = HyP->Ublock_dirty_bit[kijb];
		int_t streamIdk0Offload =  lastk0 % sluGPU->nGPUStreams;
		if (sluGPU->lastOffloadStream[streamIdk0Offload] == lastk0 && lastk0 != -1)
		    {
			// printf("Waiting for Offload =%d to finish StreamId=%d\n", lastk0, streamIdk0Offload);
			double ttx = SuperLU_timer_();
			gpuStreamSynchronize(sluGPU->funCallStreams[streamIdk0Offload]);
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
    
    gpuStream_t CopyStream;
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
	    gpuStreamSynchronize(CopyStream);
	    SCT->PhiWaitTimer_2 += SuperLU_timer_() - ttx;
	}

    double tt_start = SuperLU_timer_();

    if (last_flag) {
	if (mkcol == mycol && Lrowind_bc_ptr[kljb] != NULL )
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
	if (mkrow == myrow && Ufstnz_br_ptr[kijb] != NULL )
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
    gpuStreamSynchronize(sluGPU->funCallStreams[streamId]);
    SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
    return 0;
}

int zsendLUpanelGPU2HOST(
    int_t k0,
    d2Hreduce_t* d2Hred,
    zsluGPU_t *sluGPU,
    SuperLUStat_t *stat     
)
{
    int_t kljb = d2Hred->kljb;
    int_t kijb = d2Hred->kijb;
    int_t copyL_kljb = d2Hred->copyL_kljb;
    int_t copyU_kljb = d2Hred->copyU_kljb;
    int_t l_copy_len = d2Hred->l_copy_len;
    int_t u_copy_len = d2Hred->u_copy_len;
    gpuStream_t CopyStream = sluGPU->CopyStream;;
    zLUstruct_gpu_t *A_gpu = sluGPU->A_gpu;
    double tty = SuperLU_timer_();
    gpuEventRecord(stat->ePCIeD2H_Start[k0], CopyStream);
    if (copyL_kljb)
	checkGPU(gpuMemcpyAsync(A_gpu->acc_L_buff, &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[kljb]],
				  l_copy_len * sizeof(doublecomplex), gpuMemcpyDeviceToHost, CopyStream ) );

    if (copyU_kljb)
	checkGPU(gpuMemcpyAsync(A_gpu->acc_U_buff, &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[kijb]],
				  u_copy_len * sizeof(doublecomplex), gpuMemcpyDeviceToHost, CopyStream ) );
    gpuEventRecord(stat->ePCIeD2H_End[k0], CopyStream);
    stat->tHost_PCIeD2H += SuperLU_timer_() - tty;
    stat->cPCIeD2H += u_copy_len * sizeof(doublecomplex) + l_copy_len * sizeof(doublecomplex);

    return 0;
} /* end zsendLUpanelGPU2HOST */

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

/* Allocate GPU memory for the LU data structures, and copy
   the host LU structure to GPU side.
   After factorization, the GPU LU structure should be freed by
   calling zfree_LUstruct_gpu().    */
void zCopyLUToGPU3D (
    int* isNodeInMyGrid,
    zLocalLU_t *A_host, /* distributed LU structure on host */
    zsluGPU_t *sluGPU,  /* hold LU structure on GPU */
    Glu_persist_t *Glu_persist, int_t n,
    gridinfo3d_t *grid3d,
    int_t buffer_size, /* bigV size on GPU for Schur complement update */
    int_t bigu_size,
    int_t ldt,
    SuperLUStat_t *stat
)
{
    gridinfo_t* grid = &(grid3d->grid2d);
    zLUstruct_gpu_t * A_gpu =  sluGPU->A_gpu;
    zLUstruct_gpu_t **dA_gpu =  &(sluGPU->dA_gpu);

#if ( PRNTlevel>=1 )
    if ( grid3d->iam == 0 ) print_occupancy();
#endif

#ifdef GPU_DEBUG
    // if ( grid3d->iam == 0 )
    {
	gpuDeviceProp devProp;
	gpuGetDeviceProperties(&devProp, 0);
	printDevProp(devProp);
    }
#endif
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
    stat->ScatterMOPCounter = 0;
    stat->GemmFLOPCounter = 0;
    stat->cPCIeH2D = 0;
    stat->cPCIeD2H = 0;
    stat->tHost_PCIeH2D = 0;
    stat->tHost_PCIeD2H = 0;

    /*initializing memory*/
    size_t max_gpu_memory = get_acc_memory ();
    size_t gpu_mem_used = 0;

    void *tmp_ptr;

    A_gpu->xsup_host = xsup;

    int_t nGPUStreams = sluGPU->nGPUStreams;
    /*pinned memory allocations.
      Paged-locked memory by gpuMallocHost is accessible to the device.*/
    for (int streamId = 0; streamId < nGPUStreams; streamId++ ) {
	void *tmp_ptr;
	checkGPUErrors(gpuMallocHost(  &tmp_ptr, (n) * sizeof(int_t) )) ;
	A_gpu->scubufs[streamId].usub_IndirectJ3_host = (int_t*) tmp_ptr;

	checkGPUErrors(gpuMalloc( &tmp_ptr,  ( n) * sizeof(int_t) ));
	A_gpu->scubufs[streamId].usub_IndirectJ3 =  (int_t*) tmp_ptr;
	gpu_mem_used += ( n) * sizeof(int_t);
	checkGPUErrors(gpuMallocHost(  &tmp_ptr, mrb * sizeof(Remain_info_t) )) ;
	A_gpu->scubufs[streamId].Remain_info_host = (Remain_info_t*)tmp_ptr;
	checkGPUErrors(gpuMallocHost(  &tmp_ptr, mcb * sizeof(Ublock_info_t) )) ;
	A_gpu->scubufs[streamId].Ublock_info_host = (Ublock_info_t*)tmp_ptr;
	checkGPUErrors(gpuMallocHost(  &tmp_ptr,  remain_l_max * sizeof(doublecomplex) )) ;
	A_gpu->scubufs[streamId].Remain_L_buff_host = (doublecomplex *) tmp_ptr;
	checkGPUErrors(gpuMallocHost(  &tmp_ptr,  bigu_size * sizeof(doublecomplex) )) ;
	A_gpu->scubufs[streamId].bigU_host = (doublecomplex *) tmp_ptr;

	checkGPUErrors(gpuMallocHost ( &tmp_ptr, sizeof(doublecomplex) * (A_host->bufmax[1])));
	A_gpu->acc_L_buff = (doublecomplex *) tmp_ptr;
	checkGPUErrors(gpuMallocHost ( &tmp_ptr, sizeof(doublecomplex) * (A_host->bufmax[3])));
	A_gpu->acc_U_buff = (doublecomplex *) tmp_ptr;
	checkGPUErrors(gpuMallocHost ( &tmp_ptr, sizeof(int_t) * (A_host->bufmax[0])));
	A_gpu->scubufs[streamId].lsub_buf =  (int_t *) tmp_ptr;
	checkGPUErrors(gpuMallocHost ( &tmp_ptr, sizeof(int_t) * (A_host->bufmax[2])));
	A_gpu->scubufs[streamId].usub_buf = (int_t *) tmp_ptr;

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  remain_l_max * sizeof(doublecomplex) )) ;
	A_gpu->scubufs[streamId].Remain_L_buff = (doublecomplex *) tmp_ptr;
	gpu_mem_used += remain_l_max * sizeof(doublecomplex);
	checkGPUErrors(gpuMalloc(  &tmp_ptr,  bigu_size * sizeof(doublecomplex) )) ;
	A_gpu->scubufs[streamId].bigU = (doublecomplex *) tmp_ptr;
	gpu_mem_used += bigu_size * sizeof(doublecomplex);
	checkGPUErrors(gpuMalloc(  &tmp_ptr,  mcb * sizeof(Ublock_info_t) )) ;
	A_gpu->scubufs[streamId].Ublock_info = (Ublock_info_t *) tmp_ptr;
	gpu_mem_used += mcb * sizeof(Ublock_info_t);
	checkGPUErrors(gpuMalloc(  &tmp_ptr,  mrb * sizeof(Remain_info_t) )) ;
	A_gpu->scubufs[streamId].Remain_info = (Remain_info_t *) tmp_ptr;
	gpu_mem_used += mrb * sizeof(Remain_info_t);
	checkGPUErrors(gpuMalloc(  &tmp_ptr,  buffer_size * sizeof(doublecomplex))) ;
	A_gpu->scubufs[streamId].bigV = (doublecomplex *) tmp_ptr;
	gpu_mem_used += buffer_size * sizeof(doublecomplex);
	checkGPUErrors(gpuMalloc(  &tmp_ptr,  A_host->bufmax[0]*sizeof(int_t))) ;
	A_gpu->scubufs[streamId].lsub = (int_t *) tmp_ptr;
	gpu_mem_used += A_host->bufmax[0] * sizeof(int_t);
	checkGPUErrors(gpuMalloc(  &tmp_ptr,  A_host->bufmax[2]*sizeof(int_t))) ;
	A_gpu->scubufs[streamId].usub = (int_t *) tmp_ptr;
	gpu_mem_used += A_host->bufmax[2] * sizeof(int_t);
	
    } /* endfor streamID ... allocate paged-locked memory */

    stat->isOffloaded = (int *) SUPERLU_MALLOC (sizeof(int) * nsupers);
    stat->GemmStart  = (gpuEvent_t *) SUPERLU_MALLOC(sizeof(gpuEvent_t) * nsupers);
    stat->GemmEnd  = (gpuEvent_t *) SUPERLU_MALLOC(sizeof(gpuEvent_t) * nsupers);
    stat->ScatterEnd  = (gpuEvent_t *) SUPERLU_MALLOC(sizeof(gpuEvent_t) * nsupers);
    stat->ePCIeH2D = (gpuEvent_t *) SUPERLU_MALLOC(sizeof(gpuEvent_t) * nsupers);
    stat->ePCIeD2H_Start = (gpuEvent_t *) SUPERLU_MALLOC(sizeof(gpuEvent_t) * nsupers);
    stat->ePCIeD2H_End = (gpuEvent_t *) SUPERLU_MALLOC(sizeof(gpuEvent_t) * nsupers);
    
    for (int i = 0; i < nsupers; ++i)
	{
	    stat->isOffloaded[i] = 0;
	    checkGPUErrors(gpuEventCreate(&(stat->GemmStart[i])));
	    checkGPUErrors(gpuEventCreate(&(stat->GemmEnd[i])));
	    checkGPUErrors(gpuEventCreate(&(stat->ScatterEnd[i])));
	    checkGPUErrors(gpuEventCreate(&(stat->ePCIeH2D[i])));
	    checkGPUErrors(gpuEventCreate(&(stat->ePCIeD2H_Start[i])));
	    checkGPUErrors(gpuEventCreate(&(stat->ePCIeD2H_End[i])));
	}

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

	    if (A_host->Lrowind_bc_ptr[i] != NULL && isNodeInMyGrid[i * Pc + mycol] == 1)
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
	    if (A_host->Lrowind_bc_ptr[i] != NULL && isNodeInMyGrid[i * Pc + mycol] == 1)
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
    checkGPUErrors(gpuMalloc(  &tmp_ptr,  cum_num_l_blocks * sizeof(local_l_blk_info_t))) ;
    A_gpu->local_l_blk_infoVec = (local_l_blk_info_t *) tmp_ptr;
    gpu_mem_used += cum_num_l_blocks * sizeof(local_l_blk_info_t);
    checkGPUErrors(gpuMemcpy( (A_gpu->local_l_blk_infoVec), local_l_blk_infoVec, cum_num_l_blocks * sizeof(local_l_blk_info_t), gpuMemcpyHostToDevice)) ;

    checkGPUErrors(gpuMalloc(  &tmp_ptr,  CEILING(nsupers, Pc)*sizeof(int_t))) ;
    A_gpu->local_l_blk_infoPtr = (int_t *) tmp_ptr;
    gpu_mem_used += CEILING(nsupers, Pc) * sizeof(int_t);
    checkGPUErrors(gpuMemcpy( (A_gpu->local_l_blk_infoPtr), local_l_blk_infoPtr, CEILING(nsupers, Pc)*sizeof(int_t), gpuMemcpyHostToDevice)) ;

    /*---- Copy U data structure to GPU ----*/

    local_u_blk_info_t  *local_u_blk_infoVec;
    int_t  * local_u_blk_infoPtr;
    local_u_blk_infoPtr =  (int_t *) malloc( CEILING(nsupers, Pr) * sizeof(int_t ) );

    /* First pass: count total U blocks */
    int_t cum_num_u_blocks = 0;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
	{

	    if (A_host->Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
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
	    if (A_host->Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
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

	checkGPUErrors(gpuMalloc( &tmp_ptr,  cum_num_u_blocks * sizeof(local_u_blk_info_t))) ;
	A_gpu->local_u_blk_infoVec = (local_u_blk_info_t *) tmp_ptr;
	gpu_mem_used += cum_num_u_blocks * sizeof(local_u_blk_info_t);
	checkGPUErrors(gpuMemcpy( (A_gpu->local_u_blk_infoVec), local_u_blk_infoVec, cum_num_u_blocks * sizeof(local_u_blk_info_t), gpuMemcpyHostToDevice)) ;

	checkGPUErrors(gpuMalloc( &tmp_ptr,  CEILING(nsupers, Pr)*sizeof(int_t))) ;
	A_gpu->local_u_blk_infoPtr = (int_t *) tmp_ptr;
	gpu_mem_used += CEILING(nsupers, Pr) * sizeof(int_t);
	checkGPUErrors(gpuMemcpy( (A_gpu->local_u_blk_infoPtr), local_u_blk_infoPtr, CEILING(nsupers, Pr)*sizeof(int_t), gpuMemcpyHostToDevice)) ;

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
		    if (index_host != NULL)
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
	    if (index_host != NULL && isNodeInMyGrid[lb * Pr + myrow] == 1)
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

#if (PRNTlevel>=2)
	printf("(%d) Number of L blocks in GPU %d, U blocks %d\n",
	       grid3d->iam, num_gpu_l_blocks, num_gpu_u_blocks );
	printf("(%d) elimination order of first block in GPU: L block %d, U block %d\n",
	       grid3d->iam, A_gpu->first_l_block_gpu, A_gpu->first_u_block_gpu);
	printf("(%d) Memory of L %.1f GB, memory for U %.1f GB, Total device memory used %.1f GB, Memory allowed %.1f GB \n", grid3d->iam,
	       l_val_len * sizeof(doublecomplex) * 1e-9,
	       u_val_len * sizeof(doublecomplex) * 1e-9,
	       gpu_mem_used * 1e-9, max_gpu_memory * 1e-9);
	fflush(stdout);
#endif

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

		    if (index_host != NULL)
			{
			    int_t nrbl  =   index_host[0]; /* number of L blocks */
			    int_t len   = index_host[1];   /* LDA of the nzval[] */
			    int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
			    
			    memcpy(&indtemp[temp_LrowindPtr[ljb]] , index_host, len1 * sizeof(int_t)) ;
			}
		}
	}

	checkGPUErrors(gpuMalloc( &tmp_ptr,  l_ind_len * sizeof(int_t))) ;
	A_gpu->LrowindVec = (int_t *) tmp_ptr;
	checkGPUErrors(gpuMemcpy( (A_gpu->LrowindVec), indtemp, l_ind_len * sizeof(int_t), gpuMemcpyHostToDevice)) ;

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  l_val_len * sizeof(doublecomplex)));
	A_gpu->LnzvalVec = (doublecomplex *) tmp_ptr;
	checkGPUErrors(gpuMemset( (A_gpu->LnzvalVec), 0, l_val_len * sizeof(doublecomplex)));

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  l_k * sizeof(int_t))) ;
	A_gpu->LrowindPtr = (int_t *) tmp_ptr;
	checkGPUErrors(gpuMemcpy( (A_gpu->LrowindPtr), temp_LrowindPtr, l_k * sizeof(int_t), gpuMemcpyHostToDevice)) ;

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  l_k * sizeof(int_t))) ;
	A_gpu->LnzvalPtr = (int_t *) tmp_ptr;
	checkGPUErrors(gpuMemcpy( (A_gpu->LnzvalPtr), temp_LnzvalPtr, l_k * sizeof(int_t), gpuMemcpyHostToDevice)) ;

	A_gpu->LnzvalPtr_host = temp_LnzvalPtr;

	int_t *indtemp1 = (int_t *) malloc(sizeof(int_t) * u_ind_len);
	for ( int_t lb = 0; lb < u_k; ++lb)
	{
	    int_t *index_host;
	    index_host =  A_host->Ufstnz_br_ptr[lb];

	    if (index_host != NULL && isNodeInMyGrid[lb * Pr + myrow] == 1)
		{
		    int_t len1 = index_host[2];
		    memcpy(&indtemp1[temp_UrowindPtr[lb]] , index_host, sizeof(int_t)*len1);
		}
	}

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  u_ind_len * sizeof(int_t))) ;
	A_gpu->UrowindVec = (int_t *) tmp_ptr;
	checkGPUErrors(gpuMemcpy( (A_gpu->UrowindVec), indtemp1, u_ind_len * sizeof(int_t), gpuMemcpyHostToDevice)) ;

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  u_val_len * sizeof(doublecomplex)));
	A_gpu->UnzvalVec = (doublecomplex *) tmp_ptr;
	checkGPUErrors(gpuMemset( (A_gpu->UnzvalVec), 0, u_val_len * sizeof(doublecomplex)));

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  u_k * sizeof(int_t))) ;
	A_gpu->UrowindPtr = (int_t *) tmp_ptr;
	checkGPUErrors(gpuMemcpy( (A_gpu->UrowindPtr), temp_UrowindPtr, u_k * sizeof(int_t), gpuMemcpyHostToDevice)) ;

	A_gpu->UnzvalPtr_host = temp_UnzvalPtr;

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  u_k * sizeof(int_t))) ;
	A_gpu->UnzvalPtr = (int_t *) tmp_ptr;
	checkGPUErrors(gpuMemcpy( (A_gpu->UnzvalPtr), temp_UnzvalPtr, u_k * sizeof(int_t), gpuMemcpyHostToDevice)) ;

	checkGPUErrors(gpuMalloc(  &tmp_ptr,  (nsupers + 1)*sizeof(int_t))) ;
	A_gpu->xsup = (int_t *) tmp_ptr;
	checkGPUErrors(gpuMemcpy( (A_gpu->xsup), xsup, (nsupers + 1)*sizeof(int_t), gpuMemcpyHostToDevice)) ;

	checkGPUErrors(gpuMalloc( &tmp_ptr,  sizeof(zLUstruct_gpu_t))) ;
	*dA_gpu = (zLUstruct_gpu_t *) tmp_ptr;
	checkGPUErrors(gpuMemcpy( *dA_gpu, A_gpu, sizeof(zLUstruct_gpu_t), gpuMemcpyHostToDevice)) ;

	free (temp_LrowindPtr);
	free (temp_UrowindPtr);
	free (indtemp1);
	free (indtemp);

} /* end zCopyLUToGPU3D */



int zreduceAllAncestors3d_GPU (
	int_t ilvl, int_t* myNodeCount,
    	int_t** treePerm,
	zLUValSubBuf_t*LUvsb,
	zLUstruct_t* LUstruct,
	gridinfo3d_t* grid3d,
	zsluGPU_t *sluGPU,
	d2Hreduce_t* d2Hred,
	factStat_t *factStat,
	HyP_t* HyP, SCT_t* SCT, SuperLUStat_t *stat
    )
{
    // first synchronize all gpu streams
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
        for (int_t streamId = 0; streamId < sluGPU->nGPUStreams; streamId++)
	{
	    double ttx = SuperLU_timer_();
	    gpuStreamSynchronize(sluGPU->funCallStreams[streamId]);
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
		    zsendLUpanelGPU2HOST(k, d2Hred, sluGPU, stat);
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
    for (int streamId = 0; streamId < sluGPU->nGPUStreams; streamId++)
    {
        double ttx = SuperLU_timer_();
        gpuStreamSynchronize(sluGPU->funCallStreams[streamId]);
        SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
        sluGPU->lastOffloadStream[streamId] = -1;
     }
}
