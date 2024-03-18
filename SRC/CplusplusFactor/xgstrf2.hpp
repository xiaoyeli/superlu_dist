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
        }
        else /* Scale the j-th column. */
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
            /* Rank-1 update */
            Ftype alpha = -one<Ftype>();
            superlu_ger<Ftype>(l, cols_left, alpha, &diagBlk[luptr + 1], incx,
                         &ujrow[LDU], incy, &diagBlk[luptr + LDA + 1],
                         LDA);
            stat->ops[FACT] += 2 * l * cols_left;
        }

        ujrow = ujrow + LDU + 1; /* move to next row of U */
        luptr += LDA + 1;           /* move to next column */

    } /* for column j ...  first loop */

    // printf("Coming to local dgstrf2\n");
}


#pragma GCC pop_options