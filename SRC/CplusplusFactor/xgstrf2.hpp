#pragma once 
#include "superlu_defs.h"
#include "superlu_ddefs.h"
#include "superlu_sdefs.h"
#include "superlu_blas.hpp"
#pragma GCC push_options
#pragma GCC optimize ("O0")

template<typename Ftype>
void xgstrf2(int_t k, Ftype* diagBlk, int_t LDA, Ftype* BlockUfactor, int_t LDU, 
    threshPivValType<Ftype> thresh, int_t* xsup,
    superlu_dist_options_t *options,
    SuperLUStat_t *stat, int *info
 )
{

    int_t jfst = FstBlockC(k);
    // int_t jlst = FstBlockC(k + 1);
    int_t nsupc = SuperSize(k);
    
    Ftype *ublk_ptr = BlockUfactor;
    Ftype *ujrow = BlockUfactor;
    int_t luptr = 0;       /* Point_t to the diagonal entries. */
    int cols_left = nsupc; /* supernode size */

    for (int_t j = 0; j < nsupc; ++j) /* for each column in panel */
    {
        /* Diagonal pivot */
        int_t i = luptr;
        /* Not to replace zero pivot.  */
        if (options->ReplaceTinyPivot == YES)
        {
            // if (fabs(diagBlk[i]) < thresh)
            if (std::sqrt(sqnorm(diagBlk[i])) < thresh)
            { /* Diagonal */

#if (PRNTlevel >= 2)
                printf("(%d) .. col %d, tiny pivot %e  ",
                       iam, jfst + j, diagBlk[i]);
#endif
                /* Keep the new diagonal entry with the same sign. */
                setDiagToThreshold(&diagBlk[i], thresh);
                // if (diagBlk[i] < 0)
                //     // diagBlk[i] = -thresh;
                //     setDiagToThreshold(&diagBlk[i], -thresh);
                // else
                //     // diagBlk[i] = thresh;
                //     setDiagToThreshold(&diagBlk[i], thresh);
#if (PRNTlevel >= 2)
                printf("replaced by %e\n", diagBlk[i]);
#endif
                ++(stat->TinyPivots);
            }
        }

        for (int_t l = 0; l < cols_left; ++l, i += LDA)
        {
            int_t st = j * LDU + j;
            ublk_ptr[st + l * LDU] = diagBlk[i]; /* copy one row of U */
        }
        Ftype zero = zeroT<Ftype>();
        if (ujrow[0] == zero) /* Test for singularity. */
        {
            *info = j + jfst + 1;
            /* Replace the zero pivot to prevent NaN propagation in the scale
               step, rank-1 update, and downstream operations (pdCompute_Diag_Inv
               computes U^{-1} via local_dtrtri, which divides by diagonal
               elements).  The CUDA path (cusolverDnDgetrf with devIpiv=NULL)
               handles zero pivots implicitly without propagating them.
               Note: diagFactorPackDiagBlockGPU passes a local_info so this
               *info assignment does NOT reach pdgssvx3d's global return value. */
            setDiagToThreshold(&diagBlk[luptr], thresh);
            ujrow[0] = diagBlk[luptr]; /* keep ujrow[] in sync with diagBlk */
            ++(stat->TinyPivots);
        }
        /* Scale the j-th column (runs for both normal pivots and replaced zeros). */
        {
            Ftype temp;
            temp = one<Ftype>() / ujrow[0];
            for (int_t i = luptr + 1; i < luptr - j + nsupc; ++i)
                diagBlk[i] *= temp;
            stat->ops[FACT] += nsupc - j - 1;
        }

        /* Rank-1 update of the trailing submatrix. */
        if (--cols_left)
        {
            /*following must be int*/
            int l = nsupc - j - 1;
            int incx = 1;
            int incy = LDU;
            /* Rank-1 update: A -= x * y^T
               For real types (double/float): use an inline loop instead of
               superlu_ger() / BLAS dger_ to avoid calling an OpenBLAS kernel
               compiled for AVX-512 (skylakex) on a CPU that only supports AVX2
               (e.g. AMD EPYC 7763).  When the matrix dimensions exceed the BLAS
               fallback threshold (~n>=16), OpenBLAS dispatches to an AVX-512
               kernel and raises SIGILL on non-AVX-512 CPUs.
               For complex types the binary operator* is not defined on the C
               struct, so we keep the BLAS path (complex sizes are typically
               smaller and less likely to hit the AVX-512 dispatch). */
            if constexpr (std::is_same<Ftype, double>::value ||
                          std::is_same<Ftype, float>::value) {
                Ftype *x = &diagBlk[luptr + 1];      /* column of L (length l)  */
                Ftype *y = &ujrow[LDU];               /* row of U (stride LDU)   */
                Ftype *A = &diagBlk[luptr + LDA + 1]; /* trailing submatrix      */
                for (int jj = 0; jj < cols_left; ++jj) {
                    Ftype yj = y[jj * LDU];
                    for (int ii = 0; ii < l; ++ii)
                        A[ii + jj * LDA] -= x[ii] * yj;
                }
            } else {
                Ftype alpha = -one<Ftype>();
                superlu_ger<Ftype>(l, cols_left, alpha, &diagBlk[luptr + 1], incx,
                             &ujrow[LDU], incy, &diagBlk[luptr + LDA + 1],
                             LDA);
            }
            stat->ops[FACT] += 2 * l * cols_left;
        }

        ujrow = ujrow + LDU + 1; /* move to next row of U */
        luptr += LDA + 1;           /* move to next column */

    } /* for column j ...  first loop */

    // printf("Coming to local dgstrf2\n");
}


#pragma GCC pop_options