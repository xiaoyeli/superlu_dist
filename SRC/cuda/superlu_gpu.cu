#define GPU_DEBUG

#include "mpi.h"
#include "omp.h"
// #include "sec_structs.h"
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#undef Reduce
#include "cub/cub.cuh"
//#include <thrust/system/cuda/detail/cub/cub.cuh>

#include "lustruct_gpu.h"
// #include "p3dcomm.h"


extern "C" {
	void cblas_daxpy(const int N, const double alpha, const double *X,
	                 const int incX, double *Y, const int incY);
}

/*error reporting functions */
static
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}


// cublasStatus_t checkCublas(cublasStatus_t result)
// {
// #if defined(DEBUG) || defined(_DEBUG)
// 	if (result != CUBLAS_STATUS_SUCCESS)
// 	{
// 		fprintf(stderr, "CUDA Blas Runtime Error: %s\n", cublasGetErrorString(result));
// 		assert(result == CUBLAS_STATUS_SUCCESS);
// 	}
// #endif
// 	return result;
// }


int_t getnCudaStreams()
{
	// Disabling multiple cuda streams 
	#if 1
		return 1;
	#else 
		char *ttemp;
		ttemp = getenv ("N_CUDA_STREAMS");

		if (ttemp)
			return atoi (ttemp);
		else
			return 1;
	#endif 
}


// #define UNIT_STRIDE

__device__ inline
void device_scatter_l (int_t thread_id,
                       int_t nsupc, int_t temp_nbrow,
                       int_t *usub, int_t iukp, int_t klst,
                       double *nzval, int_t ldv,
                       double *tempv, int_t nbrow,
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

#define THREAD_BLOCK_SIZE  512  /* Sherry: was 192 on Titan */

#define MAX_SUPER_SIZE   512    /* Sherry: was 192. Must be <= THREAD_BLOCK_SIZE */


__device__ inline
void device_scatter_l_2D (int thread_id,
                          int nsupc, int temp_nbrow,
                          int_t *usub, int iukp, int_t klst,
                          double *nzval, int ldv,
                          const double *tempv, int nbrow,
                          // int_t *indirect2_thread
                          int *indirect2_thread,
                          int nnz_cols, int ColPerBlock,
                          int *IndirectJ3
                         )
{
	if ( thread_id < temp_nbrow * ColPerBlock )
	{
		int thread_id_x  = thread_id % temp_nbrow;
		int thread_id_y  = thread_id / temp_nbrow;

#define UNROLL_ITER 8

#pragma unroll 4
		for (int col = thread_id_y; col < nnz_cols ; col += ColPerBlock)
		{
			// nzval[ldv * IndirectJ3[col] + indirect2_thread[thread_id_x]]
			// -= tempv[nbrow * col + thread_id_x];
			atomicAdd(&nzval[ldv * IndirectJ3[col] + indirect2_thread[thread_id_x]],
				-tempv[nbrow * col + thread_id_x]);
		}
	}
}

/* Sherry: this routine is not used */
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

__device__ inline
void device_scatter_u_2D (int thread_id,
                          int temp_nbrow,  int nsupc,
                          double * ucol,
                          int_t * usub, int iukp,
                          int_t ilst, int_t klst,
                          int_t * index, int iuip_lib,
                          double * tempv, int nbrow,
                          int *indirect,
                          int nnz_cols, int ColPerBlock,
                          int *IndirectJ1,
                          int *IndirectJ3
                         )
{
	if ( thread_id < temp_nbrow * ColPerBlock )
	{
		/* 1D threads are logically arranged in 2D shape. */
		int thread_id_x  = thread_id % temp_nbrow;
		int thread_id_y  = thread_id / temp_nbrow;

#pragma unroll 4
		for (int col = thread_id_y; col < nnz_cols ; col += ColPerBlock)
		{
			// ucol[IndirectJ1[IndirectJ3[col]] + indirect[thread_id_x]]
			// -= tempv[nbrow * col + thread_id_x];
			atomicAdd(&ucol[IndirectJ1[IndirectJ3[col]] + indirect[thread_id_x]],
				-tempv[nbrow * col + thread_id_x]);
		}
	}
}


__device__ inline
void device_scatter_u (int_t thread_id,
                       int_t temp_nbrow,  int_t nsupc,
                       double * ucol,
                       int_t * usub, int_t iukp,
                       int_t ilst, int_t klst,
                       int_t * index, int_t iuip_lib,
                       double * tempv, int_t nbrow,
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
		if (segsize)            /* Nonzero segment in U(k.j). */
		{


			if (thread_id < temp_nbrow)
			{
#ifndef UNIT_STRIDE
				ucol[indirect[thread_id]] -= tempv[thread_id];
#else
				/*making access unit strided;
				it doesn't work; it for measurements */
				ucol[thread_id] -= tempv[thread_id];
#endif
			}
			tempv += nbrow;
		}
		ucol += ilst ;
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
    LUstruct_gpu * A_gpu)
{

	/* initializing pointers */
	int_t *xsup = A_gpu->xsup;
	int_t *UrowindPtr = A_gpu->UrowindPtr;
	int_t *UrowindVec = A_gpu->UrowindVec;
	int_t *UnzvalPtr = A_gpu->UnzvalPtr;
	double *UnzvalVec = A_gpu->UnzvalVec;
	int_t *LrowindPtr = A_gpu->LrowindPtr;
	int_t *LrowindVec = A_gpu->LrowindVec;
	int_t *LnzvalPtr = A_gpu->LnzvalPtr;
	double *LnzvalVec = A_gpu->LnzvalVec;
	double *bigV = A_gpu->scubufs[streamId].bigV;
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
	__shared__ int indirect_thread[MAX_SUPER_SIZE];  /* row-wise */
	__shared__ int indirect2_thread[MAX_SUPER_SIZE]; /* row-wise */
	__shared__ int IndirectJ1[THREAD_BLOCK_SIZE];    /* column-wise */
	__shared__ int IndirectJ3[THREAD_BLOCK_SIZE];    /* column-wise */

	/* see CUB page https://nvlabs.github.io/cub/. Implement threads collectives */
	typedef cub::BlockScan<int, THREAD_BLOCK_SIZE> BlockScan; /*1D int data type*/
	__shared__ typename BlockScan::TempStorage temp_storage; /*storage temp*/

	int thread_id = threadIdx.x;

	int iukp = Ublock_info[j].iukp;
	int jb = Ublock_info[j].jb;
	int nsupc = SuperSize (jb);
	int ljb = jb / npcol;

	double *tempv1;
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
		__shared__ int ljb_ind;
		/*do a search ljb_ind at local row lib*/
		int blks_per_threads = CEILING(num_u_blocks, THREAD_BLOCK_SIZE);
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
		double *Unzval_lib = &UnzvalVec[UnzvalPtr[lib]];
		double *ucol = &Unzval_lib[ruip_lib];

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

		__syncthreads();

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
			BlockScan(temp_storage).InclusiveSum(IndirectJ1[thread_id], IndirectJ1[thread_id]);

		if (thread_id < THREAD_BLOCK_SIZE)
			IndirectJ1[thread_id] = -IndirectJ1[thread_id] + ilst * thread_id;

		__syncthreads();

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
		double *nzval;
		int_t *index = &LrowindVec[LrowindPtr[ljb]];
		int num_l_blocks = index[0];
		int ldv = index[1];

		int fnz = FstBlockC (ib);
		int lib = ib / nprow;

		__shared__ int lib_ind;
		/*do a search lib_ind for lib*/
		int blks_per_threads = CEILING(num_l_blocks, THREAD_BLOCK_SIZE);
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
			indirect_thread[rel] = thread_id;
		}
		__syncthreads();

		/* can be precalculated */
		if (thread_id < temp_nbrow)
		{
			rel = lsub[lptr + thread_id] - fnz;
			indirect2_thread[thread_id] = indirect_thread[rel];
		}
		if (thread_id < nnz_cols)
			IndirectJ3[thread_id] = (int) A_gpu->scubufs[streamId].usub_IndirectJ3[cum_ncol + thread_id];
		__syncthreads();

		int ColPerBlock = THREAD_BLOCK_SIZE / temp_nbrow;

		nzval = &LnzvalVec[LnzvalPtr[ljb]] + luptrj;
		device_scatter_l_2D(
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

int_t SchurCompUpdate_GPU(
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
    sluGPU_t *sluGPU, gridinfo_t *grid
)
{

	LUstruct_gpu * A_gpu = sluGPU->A_gpu;
	LUstruct_gpu * dA_gpu = sluGPU->dA_gpu;
	int_t nprow = grid->nprow;
	int_t npcol = grid->npcol;

	cudaStream_t FunCallStream = sluGPU->funCallStreams[streamId];
	cublasHandle_t cublas_handle0 = sluGPU->cublasHandles[streamId];
	int_t * lsub = A_gpu->scubufs[streamId].lsub_buf;
	int_t * usub = A_gpu->scubufs[streamId].usub_buf;
	Remain_info_t *Remain_info = A_gpu->scubufs[streamId].Remain_info_host;
	double * Remain_L_buff = A_gpu->scubufs[streamId].Remain_L_buff_host;
	Ublock_info_t *Ublock_info = A_gpu->scubufs[streamId].Ublock_info_host;
	double * bigU = A_gpu->scubufs[streamId].bigU_host;

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

		for (int_t kk = 0; kk < MAX_SUPER_SIZE; ++kk)
		{
			indirectJ1[kk] = 0;
		}

		for (int_t kk = 0; kk < nsupc; ++kk)
		{
			indirectJ1[kk] = ((klst - usub[iukp + kk]) == 0) ? 0 : 1;
		}

		/*prefix sum - indicates # of nonzero segments up to column kk */
		indirectJ2[0] = indirectJ1[0];
		for (int_t kk = 1; kk < MAX_SUPER_SIZE; ++kk)
		{
			indirectJ2[kk] = indirectJ2[kk - 1] + indirectJ1[kk];
		}

		/* total number of nonzero segments in this supernode */
		int nnz_col = indirectJ2[MAX_SUPER_SIZE - 1];

		/* compactation */
		for (int_t kk = 0; kk < MAX_SUPER_SIZE; ++kk)
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

	//printf("SchurCompUpdate_GPU[3]: jj_cpu %d, nub %d\n", jj_cpu, nub); fflush(stdout);

	/*sizeof RemainLbuf = Rnbuf*knsupc */
	double tTmp = SuperLU_timer_();
	cudaEventRecord(A_gpu->ePCIeH2D[k0], FunCallStream);

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].usub_IndirectJ3,
	                          A_gpu->scubufs[streamId].usub_IndirectJ3_host,
	                          ncols * sizeof(int_t), cudaMemcpyHostToDevice,
	                          FunCallStream)) ;

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].Remain_L_buff, Remain_L_buff,
	                          Remain_lbuf_send_size * sizeof(double),
	                          cudaMemcpyHostToDevice, FunCallStream)) ;

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].bigU, bigU,
	                          bigu_send_size * sizeof(double),
	                          cudaMemcpyHostToDevice, FunCallStream) );

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].Remain_info, Remain_info,
	                          RemainBlk * sizeof(Remain_info_t),
	                          cudaMemcpyHostToDevice, FunCallStream) );

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].Ublock_info, Ublock_info,
	                          mcb * sizeof(Ublock_info_t), cudaMemcpyHostToDevice,
	                          FunCallStream) );

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].lsub, lsub,
	                          lsub_len * sizeof(int_t), cudaMemcpyHostToDevice,
	                          FunCallStream) );

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].usub, usub,
	                          usub_len * sizeof(int_t), cudaMemcpyHostToDevice,
	                          FunCallStream) );

	A_gpu->tHost_PCIeH2D += SuperLU_timer_() - tTmp;
	A_gpu->cPCIeH2D += Remain_lbuf_send_size * sizeof(double)
	                   + bigu_send_size * sizeof(double)
	                   + RemainBlk * sizeof(Remain_info_t)
	                   + mcb * sizeof(Ublock_info_t)
	                   + lsub_len * sizeof(int_t)
	                   + usub_len * sizeof(int_t);

	double alpha = 1.0, beta = 0.0;

	int_t ii_st  = 0;
	int_t ii_end = 0;
	int_t maxGemmBlockDim = (int) sqrt(buffer_size);
	// int_t maxGemmBlockDim = 8000;

	/* Organize GEMM by blocks of [ii_st : ii_end, jj_st : jj_end] that
	   fits in the buffer_size  */
	while (ii_end < RemainBlk)
	{
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

		int_t jj_st = jj_cpu;
		int_t jj_end = jj_cpu;

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

			int_t ncols;
			int_t st_col;
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
				if (nrows * ncols > buffer_size)
				{
					printf("!! Matrix size %lld x %lld exceeds buffer_size \n",
					       nrows, ncols, buffer_size);
					fflush(stdout);
				}
				assert(nrows * ncols <= buffer_size);
				cublasSetStream(cublas_handle0, FunCallStream);
				cudaEventRecord(A_gpu->GemmStart[k0], FunCallStream);
				cublasDgemm(cublas_handle0, CUBLAS_OP_N, CUBLAS_OP_N,
				            nrows, ncols, ldu, &alpha,
				            &A_gpu->scubufs[streamId].Remain_L_buff[(knsupc - ldu) * Rnbrow + st_row], Rnbrow,
				            &A_gpu->scubufs[streamId].bigU[st_col * ldu], ldu,
				            &beta, A_gpu->scubufs[streamId].bigV, nrows);

// #define SCATTER_OPT
#ifdef SCATTER_OPT
				cudaStreamSynchronize(FunCallStream);
#warning this function is synchrnous
#endif
				cudaEventRecord(A_gpu->GemmEnd[k0], FunCallStream);

				A_gpu->GemmFLOPCounter += 2.0 * (double) nrows * ncols * ldu ;

				/*
				 * Scattering the output
				 */
				dim3 dimBlock(THREAD_BLOCK_SIZE);   // 1d thread

				dim3 dimGrid(ii_end - ii_st, jj_end - jj_st);

				Scatter_GPU_kernel <<< dimGrid, dimBlock, 0, FunCallStream>>>
				(streamId, ii_st, ii_end,  jj_st, jj_end, klst,
				 0, nrows, ldt, npcol, nprow, dA_gpu);
#ifdef SCATTER_OPT
				cudaStreamSynchronize(FunCallStream);
#warning this function is synchrnous
#endif

				cudaEventRecord(A_gpu->ScatterEnd[k0], FunCallStream);

				A_gpu->ScatterMOPCounter +=  3.0 * (double) nrows * ncols;
			} /* endif ... none of the matrix dimension is zero. */

		} /* end while jj_end < nub */

	} /* end while (ii_end < RemainBlk) */

	return 0;
} /* end SchurCompUpdate_GPU */


void print_occupany()
{
	int blockSize;   // The launch configurator returned block size
	int minGridSize; /* The minimum grid size needed to achieve the
			    best potential occupancy  */

	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
	                                    Scatter_GPU_kernel, 0, 0);
#if (PRNTlevel>=1)
	printf("Occupancy: MinGridSize %d blocksize %d \n", minGridSize, blockSize);
#endif
}

void printDevProp(cudaDeviceProp devProp)
{
	size_t mfree, mtotal;
	cudaMemGetInfo	(&mfree, &mtotal);
	
	printf("pciBusID:                      %d\n",  devProp.pciBusID);
	printf("pciDeviceID:                   %d\n",  devProp.pciDeviceID);
	printf("GPU Name:                      %s\n",  devProp.name);
	printf("Total global memory:           %zu\n",  devProp.totalGlobalMem);
	printf("Total free memory:             %zu\n",  mfree);
	printf("Clock rate:                    %d\n",  devProp.clockRate);

	return;
}


int
get_mpi_process_per_gpu ()
{


	char *ttemp;
	ttemp = getenv ("MPI_PROCESS_PER_GPU");

	if (ttemp)
		return atol (ttemp);
	else
	{
		printf("MPI_PROCESS_PER_GPU is not set; Using default 1 \n");
		return 1;
	}
}

size_t
get_acc_memory ()
{

	size_t mfree, mtotal;
	cudaMemGetInfo	(&mfree, &mtotal);
#if 0
	printf("Total memory %zu & free memory %zu\n", mtotal, mfree);
#endif
	return (size_t) (0.9 * (double) mfree) / get_mpi_process_per_gpu ();


}


int_t free_LUstruct_gpu (LUstruct_gpu * A_gpu)
{
	checkCuda(cudaFree(A_gpu->LrowindVec));
	checkCuda(cudaFree(A_gpu->LrowindPtr));

	checkCuda(cudaFree(A_gpu->LnzvalVec));
	checkCuda(cudaFree(A_gpu->LnzvalPtr));
	free(A_gpu->LnzvalPtr_host);
	/*freeing the pinned memory*/
	int_t streamId = 0;
	checkCuda (cudaFreeHost (A_gpu->scubufs[streamId].Remain_info_host));
	checkCuda (cudaFreeHost (A_gpu->scubufs[streamId].Ublock_info_host));
	checkCuda (cudaFreeHost (A_gpu->scubufs[streamId].Remain_L_buff_host));
	checkCuda (cudaFreeHost (A_gpu->scubufs[streamId].bigU_host));

	checkCuda(cudaFreeHost(A_gpu->acc_L_buff));
	checkCuda(cudaFreeHost(A_gpu->acc_U_buff));
	checkCuda(cudaFreeHost(A_gpu->scubufs[streamId].lsub_buf));
	checkCuda(cudaFreeHost(A_gpu->scubufs[streamId].usub_buf));


	free(A_gpu->isOffloaded);
	free(A_gpu->GemmStart);
	free(A_gpu->GemmEnd);
	free(A_gpu->ScatterEnd);
	free(A_gpu->ePCIeH2D);

	free(A_gpu->ePCIeD2H_Start);
	free(A_gpu->ePCIeD2H_End);

	checkCuda(cudaFree(A_gpu->UrowindVec));
	checkCuda(cudaFree(A_gpu->UrowindPtr));

	
	checkCuda(cudaFree(A_gpu->UnzvalVec));
	checkCuda(cudaFree(A_gpu->UnzvalPtr));

	// checkCuda(cudaFree(A_gpu->grid));



	checkCuda(cudaFree(A_gpu->scubufs[streamId].bigV));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].bigU));

	checkCuda(cudaFree(A_gpu->scubufs[streamId].Remain_L_buff));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].Ublock_info));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].Remain_info));

	// checkCuda(cudaFree(A_gpu->indirect));
	// checkCuda(cudaFree(A_gpu->indirect2));
	checkCuda(cudaFree(A_gpu->xsup));

	checkCuda(cudaFree(A_gpu->scubufs[streamId].lsub));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].usub));


	checkCuda(cudaFree(A_gpu->local_l_blk_infoVec));
	checkCuda(cudaFree(A_gpu->local_l_blk_infoPtr));
	checkCuda(cudaFree(A_gpu->jib_lookupVec));
	checkCuda(cudaFree(A_gpu->jib_lookupPtr));
	checkCuda(cudaFree(A_gpu->local_u_blk_infoVec));
	checkCuda(cudaFree(A_gpu->local_u_blk_infoPtr));
	checkCuda(cudaFree(A_gpu->ijb_lookupVec));
	checkCuda(cudaFree(A_gpu->ijb_lookupPtr));

	return 0;

}



void dPrint_matrix( char *desc, int_t m, int_t n, double * dA, int_t lda )
{
	double *cPtr = (double *) malloc(sizeof(double) * lda * n);
	checkCuda(cudaMemcpy( cPtr, dA,
	                      lda * n * sizeof(double), cudaMemcpyDeviceToHost)) ;

	int_t i, j;
	printf( "\n %s\n", desc );
	for ( i = 0; i < m; i++ )
	{
		for ( j = 0; j < n; j++ ) printf( " %.3e", cPtr[i + j * lda] );
		printf( "\n" );
	}
	free(cPtr);
}

void printGPUStats(LUstruct_gpu * A_gpu, gridinfo_t* grid)
{
	double tGemm = 0;
	double tScatter = 0;
	double tPCIeH2D = 0;
	double tPCIeD2H = 0;

	for (int_t i = 0; i < A_gpu->nsupers; ++i)
	{
		float milliseconds = 0;

		if (A_gpu->isOffloaded[i])
		{
			cudaEventElapsedTime(&milliseconds, A_gpu->ePCIeH2D[i], A_gpu->GemmStart[i]);
			tPCIeH2D += 1e-3 * (double) milliseconds;
			milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, A_gpu->GemmStart[i], A_gpu->GemmEnd[i]);
			tGemm += 1e-3 * (double) milliseconds;
			milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, A_gpu->GemmEnd[i], A_gpu->ScatterEnd[i]);
			tScatter += 1e-3 * (double) milliseconds;
		}

		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, A_gpu->ePCIeD2H_Start[i], A_gpu->ePCIeD2H_End[i]);
		tPCIeD2H += 1e-3 * (double) milliseconds;
	}

	// gridinfo_t* grid = grid3d->grid2d;
	// DistPrint("Wait_LSend            ", SCT->Wait_LSend_tl / CPU_CLOCK_RATE, "Seconds", grid);

	// DistPrint("GPU: flops_Offloaded    ", 1e-9 *A_gpu->GemmFLOPCounter, "GFlop", grid);
	// DistPrint("GPU: MOP_Offloaded     ", 1e-9 * A_gpu->ScatterMOPCounter, "GMemOp", grid);
	DistPrint("GPU: tGemm            ", tGemm, "Seconds", grid);
	DistPrint("GPU: tScatter         ", tScatter, "Seconds", grid);
	// DistPrint("GPU: MOP_Offloaded     ", 1e-9 * A_gpu->ScatterMOPCounter, "GMemOp", grid);


	DistPrint("GPU: flops_Rate        ", 1e-9 * A_gpu->GemmFLOPCounter / tGemm, "GF/sec", grid);
	DistPrint("GPU: MOP_Rate         ", 8e-9 * A_gpu->ScatterMOPCounter / tScatter, "GB/sec", grid);
	DistPrint("GPU: H2D_Rate         ", 1e-9 * A_gpu->cPCIeH2D / tPCIeH2D, "GB/sec", grid);
	DistPrint("GPU: D2H_Rate         ", 1e-9 * A_gpu->cPCIeD2H / tPCIeD2H, "GB/sec", grid);

	// printf("GPU: Flops offloaded %.3e Time spent %lf Flop rate %lf GF/sec \n",
	//        A_gpu->GemmFLOPCounter, tGemm, 1e-9 * A_gpu->GemmFLOPCounter / tGemm  );
	// printf("GPU: Mop offloaded %.3e Time spent %lf Bandwidth %lf GByte/sec \n",
	//        A_gpu->ScatterMOPCounter, tScatter, 8e-9 * A_gpu->ScatterMOPCounter / tScatter  );
	// printf("PCIe Data Transfer H2D:\n\tData Sent %.3e(GB)\n\tTime observed from CPU %lf\n\tActual time spent %lf\n\tBandwidth %lf GByte/sec \n",
	//        1e-9 * A_gpu->cPCIeH2D, A_gpu->tHost_PCIeH2D, tPCIeH2D, 1e-9 * A_gpu->cPCIeH2D / tPCIeH2D  );
	// printf("PCIe Data Transfer D2H:\n\tData Sent %.3e(GB)\n\tTime observed from CPU %lf\n\tActual time spent %lf\n\tBandwidth %lf GByte/sec \n",
	//        1e-9 * A_gpu->cPCIeD2H, A_gpu->tHost_PCIeD2H, tPCIeD2H, 1e-9 * A_gpu->cPCIeD2H / tPCIeD2H  );
	fflush(stdout);

} /* end printGPUStats */


int_t initSluGPU3D_t(
    sluGPU_t *sluGPU,
    LUstruct_t *LUstruct,
    gridinfo3d_t * grid3d,
    int_t* perm_c_supno,
    int_t n,
    int_t buffer_size,    /* read from env variable MAX_BUFFER_SIZE */
    int_t bigu_size,
    int_t ldt             /* NSUP read from sp_ienv(3) */
)
{

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);               // How many GPUs?
	int device_id = int()(grid3d->iam/get_mpi_process_per_gpu ()) % (deviceCount); //YL: allow multiple MPIs per GPU
	
	char *ttemp;
    ttemp = getenv ("SUPERLU_BIND_MPI_GPU");
    if (ttemp) {
	cudaSetDevice(device_id); 
	}

// #ifdef GPU_DEBUG
// 	if(!grid3d->iam)
// 	{
// 		print_occupany();
// 		printf("NUmber of GPUs =%d, setting gpu to %d-gpu\n", deviceCount,device_id);
// 		cudaDeviceProp devProp;
// 		cudaGetDeviceProperties(&devProp, device_id);
// 		printDevProp(devProp);
// 	}
// #endif
	gridinfo_t* grid = &(grid3d->grid2d);
	// checkCudaErrors(cudaDeviceReset ())     ;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	LocalLU_t *Llu = LUstruct->Llu;
	int_t* isNodeInMyGrid = sluGPU->isNodeInMyGrid;

	sluGPU->nCudaStreams = getnCudaStreams();
	if (!grid->iam)
	{
		if(!grid3d->iam)
		printf("initSluGPU3D_t: Using hardware acceleration, with %d cuda streams \n", sluGPU->nCudaStreams);
		fflush(stdout);
		if ( MAX_SUPER_SIZE < ldt )
		{
			ABORT("MAX_SUPER_SIZE smaller than requested NSUP");
		}
	}

	cudaStreamCreate(&(sluGPU->CopyStream));

	for (int_t streamId = 0; streamId < sluGPU->nCudaStreams; streamId++)
	{
		cudaStreamCreate(&(sluGPU->funCallStreams[streamId]));
		cublasCreate(&(sluGPU->cublasHandles[streamId]));
		sluGPU->lastOffloadStream[streamId] = -1;
	}

	sluGPU->A_gpu = (LUstruct_gpu *) malloc (sizeof(LUstruct_gpu));
	sluGPU->A_gpu->perm_c_supno = perm_c_supno;
	CopyLUToGPU3D (
	    isNodeInMyGrid,
	    Llu,             /* referred to as A_host */
	    sluGPU,
	    Glu_persist, n,
	    grid3d,
	    buffer_size,
	    bigu_size,
	    ldt
	);

	return 0;
} /* end initSluGPU3D_t */

int_t initD2Hreduce(
    int_t next_k,
    d2Hreduce_t* d2Hred,
    int_t last_flag,
    HyP_t* HyP,
    sluGPU_t *sluGPU,
    gridinfo_t *grid,
    LUstruct_t *LUstruct
    , SCT_t* SCT
)
{
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	LocalLU_t *Llu = LUstruct->Llu;
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
			int_t streamIdk0Offload =  lastk0 % sluGPU->nCudaStreams;
			if (sluGPU->lastOffloadStream[streamIdk0Offload] == lastk0 && lastk0 != -1)
			{
				// printf("Waiting for Offload =%d to finish StreamId=%d\n", lastk0, streamIdk0Offload);
				double ttx = SuperLU_timer_();
				cudaStreamSynchronize(sluGPU->funCallStreams[streamIdk0Offload]);
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
			int_t streamIdk0Offload =  lastk0 % sluGPU->nCudaStreams;
			if (sluGPU->lastOffloadStream[streamIdk0Offload] == lastk0 && lastk0 != -1)
			{
				// printf("Waiting for Offload =%d to finish StreamId=%d\n", lastk0, streamIdk0Offload);
				double ttx = SuperLU_timer_();
				cudaStreamSynchronize(sluGPU->funCallStreams[streamIdk0Offload]);
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
}

int_t reduceGPUlu(

    int_t last_flag,
    d2Hreduce_t* d2Hred,
    sluGPU_t *sluGPU,
    SCT_t *SCT,
    gridinfo_t *grid,
    LUstruct_t *LUstruct
)
{

	LocalLU_t *Llu = LUstruct->Llu;
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);
	int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
	double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
	int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
	double** Unzval_br_ptr = Llu->Unzval_br_ptr;

	cudaStream_t CopyStream;
	LUstruct_gpu *A_gpu;
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
		cudaStreamSynchronize(CopyStream);
		SCT->PhiWaitTimer_2 += SuperLU_timer_() - ttx;
	}


	double tt_start = SuperLU_timer_();


	if (last_flag)
	{

		if (mkcol == mycol && Lrowind_bc_ptr[kljb] != NULL )
		{

			kindex = Lrowind_bc_ptr[kljb];
			int_t len = kindex[1];

			if (copyL_kljb)
			{

				double *nzval_host;
				nzval_host = Lnzval_bc_ptr[kljb];
				int_t llen = ksup_size * len;

				double alpha = 1;
				superlu_daxpy (llen, alpha, A_gpu->acc_L_buff, 1, nzval_host, 1);
			}

		}
	}
	if (last_flag)
	{
		if (mkrow == myrow && Ufstnz_br_ptr[kijb] != NULL )
		{

			kindex = Ufstnz_br_ptr[kijb];
			int_t len = kindex[1];

			if (copyU_kljb)
			{

				double *nzval_host;
				nzval_host = Unzval_br_ptr[kijb];

				double alpha = 1;
				superlu_daxpy (len, alpha, A_gpu->acc_U_buff, 1, nzval_host, 1);
			}

		}
	}

	double tt_end = SuperLU_timer_();
	SCT->AssemblyTimer += tt_end - tt_start;
	return 0;
}


int_t waitGPUscu(int_t streamId,  sluGPU_t *sluGPU, SCT_t *SCT)
{
	double ttx = SuperLU_timer_();
	cudaStreamSynchronize(sluGPU->funCallStreams[streamId]);
	SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
	return 0;
}

int_t sendLUpanelGPU2HOST(
    int_t k0,
    d2Hreduce_t* d2Hred,
    sluGPU_t *sluGPU
)
{

	int_t kljb = d2Hred->kljb;
	int_t kijb = d2Hred->kijb;
	int_t copyL_kljb = d2Hred->copyL_kljb;
	int_t copyU_kljb = d2Hred->copyU_kljb;
	int_t l_copy_len = d2Hred->l_copy_len;
	int_t u_copy_len = d2Hred->u_copy_len;
	cudaStream_t CopyStream = sluGPU->CopyStream;;
	LUstruct_gpu *A_gpu = sluGPU->A_gpu;
	double tty = SuperLU_timer_();
	cudaEventRecord(A_gpu->ePCIeD2H_Start[k0], CopyStream);
	if (copyL_kljb)
		checkCuda(cudaMemcpyAsync(A_gpu->acc_L_buff, &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[kljb]],
		                          l_copy_len * sizeof(double), cudaMemcpyDeviceToHost, CopyStream ) );

	if (copyU_kljb)
		checkCuda(cudaMemcpyAsync(A_gpu->acc_U_buff, &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[kijb]],
		                          u_copy_len * sizeof(double), cudaMemcpyDeviceToHost, CopyStream ) );
	cudaEventRecord(A_gpu->ePCIeD2H_End[k0], CopyStream);
	A_gpu->tHost_PCIeD2H += SuperLU_timer_() - tty;
	A_gpu->cPCIeD2H += u_copy_len * sizeof(double) + l_copy_len * sizeof(double);

	return 0;
}

/* Copy L and U panel data structures from host to the host part of the
   data structures in A_gpu.
   GPU is not involved in this routine. */
int_t sendSCUdataHost2GPU(
    int_t streamId,
    int_t* lsub,
    int_t* usub,
    double* bigU,
    int_t bigu_send_size,
    int_t Remain_lbuf_send_size,
    sluGPU_t *sluGPU,
    HyP_t* HyP
)
{
	//{printf("....[enter] sendSCUdataHost2GPU, bigu_send_size %d\n", bigu_send_size); fflush(stdout);}

	int_t usub_len = usub[2];
	int_t lsub_len = lsub[1] + BC_HEADER + lsub[0] * LB_DESCRIPTOR;
	//{printf("....[2] in sendSCUdataHost2GPU, lsub_len %d\n", lsub_len); fflush(stdout);}
	LUstruct_gpu *A_gpu = sluGPU->A_gpu;
	memcpy(A_gpu->scubufs[streamId].lsub_buf, lsub, sizeof(int_t)*lsub_len);
	memcpy(A_gpu->scubufs[streamId].usub_buf, usub, sizeof(int_t)*usub_len);
	memcpy(A_gpu->scubufs[streamId].Remain_info_host, HyP->Remain_info,
	       sizeof(Remain_info_t)*HyP->RemainBlk);
	memcpy(A_gpu->scubufs[streamId].Ublock_info_host, HyP->Ublock_info_Phi,
	       sizeof(Ublock_info_t)*HyP->num_u_blks_Phi);
	memcpy(A_gpu->scubufs[streamId].Remain_L_buff_host, HyP->Remain_L_buff,
	       sizeof(double)*Remain_lbuf_send_size);
	memcpy(A_gpu->scubufs[streamId].bigU_host, bigU,
	       sizeof(double)*bigu_send_size);

	return 0;
}


int_t freeSluGPU(sluGPU_t *sluGPU)
{
	free_LUstruct_gpu (sluGPU->A_gpu);
	return 0;
}


void CopyLUToGPU3D (
    int_t* isNodeInMyGrid,
    LocalLU_t *A_host, /* distributed LU structure on host */
    sluGPU_t *sluGPU,
    Glu_persist_t *Glu_persist, int_t n,
    gridinfo3d_t *grid3d,
    int_t buffer_size, /* bigV size on GPU for Schur complement update */
    int_t bigu_size,
    int_t ldt
)
{
	gridinfo_t* grid = &(grid3d->grid2d);
	LUstruct_gpu * A_gpu =  sluGPU->A_gpu;
	LUstruct_gpu **dA_gpu =  &(sluGPU->dA_gpu);


	int_t *xsup ;
	xsup = Glu_persist->xsup;
	int_t iam = grid->iam;
	int_t nsupers = Glu_persist->supno[n - 1] + 1;
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
	for (int_t streamId = 0; streamId < nCudaStreams; streamId++ )
	{
		void *tmp_ptr;
		checkCudaErrors(cudaMallocHost(  &tmp_ptr, (n) * sizeof(int_t) )) ;
		A_gpu->scubufs[streamId].usub_IndirectJ3_host = (int_t*) tmp_ptr;

		checkCudaErrors(cudaMalloc( &tmp_ptr,  ( n) * sizeof(int_t) ));
		A_gpu->scubufs[streamId].usub_IndirectJ3 =  (int_t*) tmp_ptr;
		gpu_mem_used += ( n) * sizeof(int_t);
		checkCudaErrors(cudaMallocHost(  &tmp_ptr, mrb * sizeof(Remain_info_t) )) ;
		A_gpu->scubufs[streamId].Remain_info_host = (Remain_info_t*)tmp_ptr;
		checkCudaErrors(cudaMallocHost(  &tmp_ptr, mcb * sizeof(Ublock_info_t) )) ;
		A_gpu->scubufs[streamId].Ublock_info_host = (Ublock_info_t*)tmp_ptr;
		checkCudaErrors(cudaMallocHost(  &tmp_ptr,  remain_l_max * sizeof(double) )) ;
		A_gpu->scubufs[streamId].Remain_L_buff_host = (double *) tmp_ptr;
		checkCudaErrors(cudaMallocHost(  &tmp_ptr,  bigu_size * sizeof(double) )) ;
		A_gpu->scubufs[streamId].bigU_host = (double *) tmp_ptr;

		checkCudaErrors(cudaMallocHost ( &tmp_ptr, sizeof(double) * (A_host->bufmax[1])));
		A_gpu->acc_L_buff = (double *) tmp_ptr;
		checkCudaErrors(cudaMallocHost ( &tmp_ptr, sizeof(double) * (A_host->bufmax[3])));
		A_gpu->acc_U_buff = (double *) tmp_ptr;
		checkCudaErrors(cudaMallocHost ( &tmp_ptr, sizeof(int_t) * (A_host->bufmax[0])));
		A_gpu->scubufs[streamId].lsub_buf =  (int_t *) tmp_ptr;
		checkCudaErrors(cudaMallocHost ( &tmp_ptr, sizeof(int_t) * (A_host->bufmax[2])));
		A_gpu->scubufs[streamId].usub_buf = (int_t *) tmp_ptr;

		checkCudaErrors(cudaMalloc(  &tmp_ptr,  remain_l_max * sizeof(double) )) ;
		A_gpu->scubufs[streamId].Remain_L_buff = (double *) tmp_ptr;
		gpu_mem_used += remain_l_max * sizeof(double);
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  bigu_size * sizeof(double) )) ;
		A_gpu->scubufs[streamId].bigU = (double *) tmp_ptr;
		gpu_mem_used += bigu_size * sizeof(double);
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  mcb * sizeof(Ublock_info_t) )) ;
		A_gpu->scubufs[streamId].Ublock_info = (Ublock_info_t *) tmp_ptr;
		gpu_mem_used += mcb * sizeof(Ublock_info_t);
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  mrb * sizeof(Remain_info_t) )) ;
		A_gpu->scubufs[streamId].Remain_info = (Remain_info_t *) tmp_ptr;
		gpu_mem_used += mrb * sizeof(Remain_info_t);
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  buffer_size * sizeof(double))) ;
		A_gpu->scubufs[streamId].bigV = (double *) tmp_ptr;
		gpu_mem_used += buffer_size * sizeof(double);
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  A_host->bufmax[0]*sizeof(int_t))) ;
		A_gpu->scubufs[streamId].lsub = (int_t *) tmp_ptr;
		gpu_mem_used += A_host->bufmax[0] * sizeof(int_t);
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  A_host->bufmax[2]*sizeof(int_t))) ;
		A_gpu->scubufs[streamId].usub = (int_t *) tmp_ptr;
		gpu_mem_used += A_host->bufmax[2] * sizeof(int_t);

	} /* endfor streamID ... allocate paged-locked memory */

	A_gpu->isOffloaded = (int_t *) malloc (sizeof(int_t) * nsupers);
	A_gpu->GemmStart  = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
	A_gpu->GemmEnd  = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
	A_gpu->ScatterEnd  = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
	A_gpu->ePCIeH2D = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
	A_gpu->ePCIeD2H_Start = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
	A_gpu->ePCIeD2H_End = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);

	for (int_t i = 0; i < nsupers; ++i)
	{
		A_gpu->isOffloaded[i] = 0;
		checkCudaErrors(cudaEventCreate(&(A_gpu->GemmStart[i])));
		checkCudaErrors(cudaEventCreate(&(A_gpu->GemmEnd[i])));
		checkCudaErrors(cudaEventCreate(&(A_gpu->ScatterEnd[i])));
		checkCudaErrors(cudaEventCreate(&(A_gpu->ePCIeH2D[i])));
		checkCudaErrors(cudaEventCreate(&(A_gpu->ePCIeH2D[i])));
		checkCudaErrors(cudaEventCreate(&(A_gpu->ePCIeD2H_Start[i])));
		checkCudaErrors(cudaEventCreate(&(A_gpu->ePCIeD2H_End[i])));
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
	checkCudaErrors(cudaMalloc(  &tmp_ptr,  cum_num_l_blocks * sizeof(local_l_blk_info_t))) ;
	A_gpu->local_l_blk_infoVec = (local_l_blk_info_t *) tmp_ptr;
	gpu_mem_used += cum_num_l_blocks * sizeof(local_l_blk_info_t);
	checkCudaErrors(cudaMemcpy( (A_gpu->local_l_blk_infoVec), local_l_blk_infoVec, cum_num_l_blocks * sizeof(local_l_blk_info_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  CEILING(nsupers, Pc)*sizeof(int_t))) ;
	A_gpu->local_l_blk_infoPtr = (int_t *) tmp_ptr;
	gpu_mem_used += CEILING(nsupers, Pc) * sizeof(int_t);
	checkCudaErrors(cudaMemcpy( (A_gpu->local_l_blk_infoPtr), local_l_blk_infoPtr, CEILING(nsupers, Pc)*sizeof(int_t), cudaMemcpyHostToDevice)) ;


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

	checkCudaErrors(cudaMalloc( &tmp_ptr,  cum_num_u_blocks * sizeof(local_u_blk_info_t))) ;
	A_gpu->local_u_blk_infoVec = (local_u_blk_info_t *) tmp_ptr;
	gpu_mem_used += cum_num_u_blocks * sizeof(local_u_blk_info_t);
	checkCudaErrors(cudaMemcpy( (A_gpu->local_u_blk_infoVec), local_u_blk_infoVec, cum_num_u_blocks * sizeof(local_u_blk_info_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc( &tmp_ptr,  CEILING(nsupers, Pr)*sizeof(int_t))) ;
	A_gpu->local_u_blk_infoPtr = (int_t *) tmp_ptr;
	gpu_mem_used += CEILING(nsupers, Pr) * sizeof(int_t);
	checkCudaErrors(cudaMemcpy( (A_gpu->local_u_blk_infoPtr), local_u_blk_infoPtr, CEILING(nsupers, Pr)*sizeof(int_t), cudaMemcpyHostToDevice)) ;


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
				mem_l_block = sizeof(double) * Lnzval_size[ljb];
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
				mem_u_block = sizeof(double) * Unzval_size[lib];
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

#if (PRNTlevel>=1)
	printf("(%d) Number of L blocks in GPU %d, U blocks %d\n",
	       grid3d->iam, num_gpu_l_blocks, num_gpu_u_blocks );
	printf("(%d) elimination order of first block in GPU: L block %d, U block %d\n",
	       grid3d->iam, A_gpu->first_l_block_gpu, A_gpu->first_u_block_gpu);
	printf("(%d) Memory of L %.1f GB, memory for U %.1f GB, Total device memory used %.1f GB, Memory allowed %.1f GB \n", grid3d->iam,
	       l_val_len * sizeof(double) * 1e-9,
	       u_val_len * sizeof(double) * 1e-9,
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

	checkCudaErrors(cudaMalloc( &tmp_ptr,  l_ind_len * sizeof(int_t))) ;
	A_gpu->LrowindVec = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->LrowindVec), indtemp, l_ind_len * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  l_val_len * sizeof(double)));
	A_gpu->LnzvalVec = (double *) tmp_ptr;
	checkCudaErrors(cudaMemset( (A_gpu->LnzvalVec), 0, l_val_len * sizeof(double)));

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  l_k * sizeof(int_t))) ;
	A_gpu->LrowindPtr = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->LrowindPtr), temp_LrowindPtr, l_k * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  l_k * sizeof(int_t))) ;
	A_gpu->LnzvalPtr = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->LnzvalPtr), temp_LnzvalPtr, l_k * sizeof(int_t), cudaMemcpyHostToDevice)) ;

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

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_ind_len * sizeof(int_t))) ;
	A_gpu->UrowindVec = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->UrowindVec), indtemp1, u_ind_len * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_val_len * sizeof(double)));
	A_gpu->UnzvalVec = (double *) tmp_ptr;
	checkCudaErrors(cudaMemset( (A_gpu->UnzvalVec), 0, u_val_len * sizeof(double)));

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_k * sizeof(int_t))) ;
	A_gpu->UrowindPtr = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->UrowindPtr), temp_UrowindPtr, u_k * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	A_gpu->UnzvalPtr_host = temp_UnzvalPtr;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_k * sizeof(int_t))) ;
	A_gpu->UnzvalPtr = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->UnzvalPtr), temp_UnzvalPtr, u_k * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  (nsupers + 1)*sizeof(int_t))) ;
	A_gpu->xsup = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->xsup), xsup, (nsupers + 1)*sizeof(int_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc( &tmp_ptr,  sizeof(LUstruct_gpu))) ;
	*dA_gpu = (LUstruct_gpu *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( *dA_gpu, A_gpu, sizeof(LUstruct_gpu), cudaMemcpyHostToDevice)) ;

	free (temp_LrowindPtr);
	free (temp_UrowindPtr);
	free (indtemp1);
	free (indtemp);

} /* end CopyLUToGPU3D */



int_t reduceAllAncestors3d_GPU(int_t ilvl, int_t* myNodeCount,
                               int_t** treePerm,
                               dLUValSubBuf_t*LUvsb,
                               LUstruct_t* LUstruct,
                               gridinfo3d_t* grid3d,
                               sluGPU_t *sluGPU,
                               d2Hreduce_t* d2Hred,
                               factStat_t *factStat,
                               HyP_t* HyP,
                               SCT_t* SCT )
{


// first synchronize all cuda streams
	int_t superlu_acc_offload =   HyP->superlu_acc_offload;




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
			cudaStreamSynchronize(sluGPU->funCallStreams[streamId]);
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

					initD2Hreduce(k, d2Hred, 1,
					              HyP, sluGPU, grid, LUstruct, SCT);
					int_t copyL_kljb = d2Hred->copyL_kljb;
					int_t copyU_kljb = d2Hred->copyU_kljb;

					double tt_start1 = SuperLU_timer_();
					SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
					if (copyL_kljb || copyU_kljb) SCT->PhiMemCpyCounter++;
					sendLUpanelGPU2HOST(k, d2Hred, sluGPU);
					/*
					    Reduce the LU panels from GPU
					*/
					reduceGPUlu(1, d2Hred,
					            sluGPU, SCT, grid, LUstruct);

					gpuLUreduced[k] = 1;
				}
			}
		}
	} /*if (myGrid == sender)*/

	dreduceAllAncestors3d(ilvl, myNodeCount, treePerm,
	                      LUvsb, LUstruct, grid3d, SCT );
	return 0;
}


void syncAllfunCallStreams(sluGPU_t* sluGPU, SCT_t* SCT)
{
	for (int_t streamId = 0; streamId < sluGPU->nCudaStreams; streamId++)
	{
		double ttx = SuperLU_timer_();
		cudaStreamSynchronize(sluGPU->funCallStreams[streamId]);
		SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
		sluGPU->lastOffloadStream[streamId] = -1;
	}
}
