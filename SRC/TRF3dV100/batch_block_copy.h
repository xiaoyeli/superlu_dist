/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/batch_block_copy.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/

#ifndef __BATCH_BLOCK_COPY_H__
#define __BATCH_BLOCK_COPY_H__

#ifdef __cplusplus
extern "C" {
#endif
// Strided interface
int kblasDcopyBlock_batch_strided(
	cudaStream_t stream, int rows, int cols,
	double* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	double* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
);

int kblasScopyBlock_batch_strided(
	cudaStream_t stream, int rows, int cols,
	float* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	float* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
);

// Array of pointers interface
int kblasDcopyBlock_batch(
	cudaStream_t stream, int rows, int cols,
	double** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	double** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
);

int kblasScopyBlock_batch(
	cudaStream_t stream, int rows, int cols,
	float** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	float** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
);

// Variable batch interface 
int kblasScopyBlock_vbatch(
	cudaStream_t stream, int* rows_batch, int* cols_batch, int max_rows, int max_cols,
	float** dest_array, int* ld_dest, float** src_array, int* ld_src, int ops
);

int kblasDcopyBlock_vbatch(
	cudaStream_t stream, int* rows_batch, int* cols_batch, int max_rows, int max_cols,
	double** dest_array, int* ld_dest, double** src_array, int* ld_src, int ops
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Strided interface
inline int kblas_copyBlock_batch(
	cudaStream_t stream, int rows, int cols,
	double* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	double* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return kblasDcopyBlock_batch_strided(
		stream, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src, ops
	);
}

inline int kblas_copyBlock_batch(
	cudaStream_t stream, int rows, int cols,
	float* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	float* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return kblasScopyBlock_batch_strided(
		stream, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src, ops
	);
}

// Array of pointers interface
inline int kblas_copyBlock_batch(
	cudaStream_t stream, int rows, int cols,
	double** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	double** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
)
{
	return kblasDcopyBlock_batch(
		stream, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest,
		src_array, row_offset_src, col_offset_src, ld_src, ops
	);
}

inline int kblas_copyBlock_batch(
	cudaStream_t stream, int rows, int cols,
	float** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	float** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
)
{
	return kblasScopyBlock_batch(
		stream, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest,
		src_array, row_offset_src, col_offset_src, ld_src, ops
	);
}

// Variable batch interface 
inline int kblas_copyBlock_vbatch(
	cudaStream_t stream, int* rows_batch, int* cols_batch, int max_rows, int max_cols,
	float** dest_array, int* ld_dest, float** src_array, int* ld_src, int ops
)
{
	return kblasScopyBlock_vbatch(
		stream, rows_batch, cols_batch, max_rows, max_cols,
		dest_array, ld_dest, src_array, ld_src, ops
	);
}

inline int kblas_copyBlock_vbatch(
	cudaStream_t stream, int* rows_batch, int* cols_batch, int max_rows, int max_cols,
	double** dest_array, int* ld_dest, double** src_array, int* ld_src, int ops
)
{
	return kblasDcopyBlock_vbatch(
		stream, rows_batch, cols_batch, max_rows, max_cols,
		dest_array, ld_dest, src_array, ld_src, ops
	);
}

// Common interface for array of pointers with dummy strides
template<class T>
inline int kblas_copyBlock_batch(
	cudaStream_t stream, int rows, int cols,
	T** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	T** src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return kblas_copyBlock_batch(
		stream, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest,
		src_array, row_offset_src, col_offset_src, ld_src, ops
	);
}
#endif

#endif //__BATCH_BLOCK_COPY_H__
