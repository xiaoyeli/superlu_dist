#ifndef __BATCH_BLOCK_COPY_H__
#define __BATCH_BLOCK_COPY_H__

/* superlu_defs.h pulls in superlu_dist_config.h (which defines HAVE_CUDA/HAVE_HIP)
 * before gpu_api_utils.h -> gpu_wrapper.h, ensuring gpuStream_t etc. are defined. */
#include "superlu_defs.h"

#ifdef __cplusplus
extern "C" {
#endif
int scopyBlock_vbatch(
    gpuStream_t stream, int* m_batch, int* n_batch, int max_m, int max_n,
    float** dest_ptrs, int* dest_ld_batch, float** src_ptrs, int* src_ld_batch,
    int ops
);

int dcopyBlock_vbatch(
    gpuStream_t stream, int* m_batch, int* n_batch, int max_m, int max_n,
    double** dest_ptrs, int* dest_ld_batch, double** src_ptrs, int* src_ld_batch,
    int ops
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
inline int copyBlock_vbatch(
    gpuStream_t stream, int* m_batch, int* n_batch, int max_m, int max_n,
    float** dest_ptrs, int* dest_ld_batch, float** src_ptrs, int* src_ld_batch,
    int ops
)
{
    return scopyBlock_vbatch(
        stream, m_batch, n_batch, max_m, max_n, dest_ptrs, dest_ld_batch,
        src_ptrs, src_ld_batch, ops
    );
}

inline int copyBlock_vbatch(
    gpuStream_t stream, int* m_batch, int* n_batch, int max_m, int max_n,
    double** dest_ptrs, int* dest_ld_batch, double** src_ptrs, int* src_ld_batch,
    int ops
)
{
    return dcopyBlock_vbatch(
        stream, m_batch, n_batch, max_m, max_n, dest_ptrs, dest_ld_batch,
        src_ptrs, src_ld_batch, ops
    );
}
#endif

#endif //__BATCH_BLOCK_COPY_H__
