/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Chooses machine-dependent parameters for the local environment
 */
/*
 * File name:		sp_ienv.c
 * History:             Modified from lapack routine ILAENV
 */
#include "superlu_ddefs.h"
#include "machines.h"

/*! \brief

</pre>
    Purpose   
    =======   

    sp_ienv_dist() is inquired to choose machine-dependent integer parameters
    for the local environment. See ISPEC for a description of the parameters.   

    This version provides a set of parameters which should give good,   
    but not optimal, performance on many of the currently available   
    computers.  Users are encouraged to set the environment variable to 
    change the tuning parameters for their particular machines.

    Arguments   
    =========   

    ISPEC   (input) int
            Specifies the parameter to be returned as the value of SP_IENV_DIST.   
            = 1: the panel size w; a panel consists of w consecutive
	         columns of matrix A in the process of Gaussian elimination.
		 The best value depends on machine's cache characters.
            = 2: the relaxation parameter relax; if the number of
	         nodes (columns) in a subtree of the elimination tree is less
		 than relax, this subtree is considered as one supernode,
		 regardless of the their row structures.
            = 3: the maximum size for a supernode, which must be greater
                 than or equal to relaxation parameter (see case 2);
	    = 4: the minimum row dimension for 2-D blocking to be used;
	    = 5: the minimum column dimension for 2-D blocking to be used;
	    = 6: the estimated fills factor for the adjacency structures 
	         of L and U, compared with A;
	    = 7: the minimum value of the product M*N*K for a GEMM call
	         worth being offloaded to accelerator (e.g., GPU, Xeon Phi).
            = 8: the maximum buffer size on GPU that can hold the "dC"
	         matrix in the GEMM call for the Schur complement update.
		 If this is too small, the Schur complement update will be
		 done in multiple partitions, may be slower.
	    = 9: number of GPU streams
	    = 10: whether to offload work to GPU or not

   (SP_IENV_DIST) (output) int
            >= 0: the value of the parameter specified by ISPEC   
            < 0:  if SP_IENV_DIST = -k, the k-th argument had an illegal value.
  
    ===================================================================== 
</pre>
*/

#include <stdlib.h>
#include <stdio.h>

int
sp_ienv_dist(int ispec, superlu_dist_options_t *options)
{
    int i;

    char* ttemp;

    switch (ispec) {
	case 2: 
            ttemp = getenv("SUPERLU_RELAX");
            if(ttemp)
            {
                return(atoi(ttemp));
            }else if(getenv("NREL"))
            {
                return(atoi(getenv("NREL")));
            }
            else {
		options->superlu_relax = 1;
		return (options->superlu_relax);
	    }
            
	case 3: 
	    ttemp = getenv("SUPERLU_MAXSUP"); // take min of MAX_SUPER_SIZE in superlu_defs.h
            if(ttemp)
            {
	        int k = SUPERLU_MIN( atoi(ttemp), MAX_SUPER_SIZE );
                return (k);
            }else if(getenv("NSUP"))
            {
                int k = SUPERLU_MIN( atoi(getenv("NSUP")), MAX_SUPER_SIZE );
                return (k);
            }
            else return (options->superlu_maxsup);
	    
        case 6: 
            ttemp = getenv("SUPERLU_FILL");
            if ( ttemp ) return(atoi(ttemp));
	    else {
		ttemp = getenv("FILL");
		if ( ttemp ) return(atoi(ttemp));
		else return (5);
	    }
        case 7:
	    ttemp = getenv ("SUPERLU_N_GEMM"); // minimum flops of GEMM worth doing on GPU
	    if (ttemp)
		return atoi (ttemp); 
	    else if(getenv("N_GEMM"))
		return(atoi(getenv("N_GEMM")));
	    else 
		return (options->superlu_n_gemm);
        case 8:
  	    ttemp = getenv ("SUPERLU_MAX_BUFFER_SIZE");
	    if (ttemp) 
		return atoi (ttemp);
	    else if(getenv("MAX_BUFFER_SIZE")) 
		return(atoi(getenv("MAX_BUFFER_SIZE")));
	    else 
		return (options->superlu_max_buffer_size);
         case 9:
  	    ttemp = getenv ("SUPERLU_NUM_GPU_STREAMS");
	    if (ttemp) 
		return atoi (ttemp);
	    else return (options->superlu_num_gpu_streams);
         case 10:
  	    ttemp = getenv ("SUPERLU_ACC_OFFLOAD");
	    if (ttemp) 
		return atoi (ttemp);
	    else return (options->superlu_acc_offload);
    }

    /* Invalid value for ISPEC */
    i = 1;
    xerr_dist("sp_ienv", &i);
    return -1;

} /* sp_ienv_dist */

