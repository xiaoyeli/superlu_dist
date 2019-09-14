/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Solves a system of distributed linear equations A*X = B with a
 * general N-by-N matrix A using the LU factors computed previously.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 15, 2008
 * September 18, 2018  version 6.0
 * February 8, 2019  version 6.1.1
 * </pre>
 */
#include "cublas_utils.h" 
#include <math.h>
#include <cusparse.h>
#include <cublas_v2.h>
				 
#include "superlu_ddefs.h"
#ifndef CACHELINE
#define CACHELINE 64  /* bytes, Xeon Phi KNL, Cori haswell, Edision */
#endif

#define cublasCheckErrors(fn) \
    do { \
        cublasStatus_t __err = fn; \
        if (__err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "Fatal cublas error: %d (at %s:%d)\n", \
                (int)(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while(0);

	
	
	

/***************************************************************************//**
    Does sum reduction of n-element array x, leaving total in x[0].
    Contents of x are destroyed in the process.
    With k threads, can reduce array up to 2*k in size.
    Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
    Having n as template parameter allows compiler to evaluate some conditions at compile time.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
__device__ void
magma_sum_reduce( int n, int i, double* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce

	
	
	
	
	
	
/******************************************************************************/
static __device__ void
gemv_device_dlsum_fmod(
    int_t m, int_t n, double alpha,
    const double * __restrict__ A, int_t lda,
    const double * __restrict__ x, int_t incx, double beta,
    double       * __restrict__ y, int_t incy)
{
    if (m <= 0 || n <= 0) return;

    int_t num_threads = DIM_X * DIM_Y;
    int_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // threads are all configurated locally
    int_t tx = thread_id % DIM_X;
    int_t ty = thread_id / DIM_X;

    int_t ind = tx;

    __shared__ double sdata[DIM_X * DIM_Y];


    int_t st = 0;

    int_t ed = min(st+m, CEILING(m,DIM_X)*DIM_X);
    
    int_t iters = CEILING(ed-st,DIM_X) ;

	double zero = 0.0;
	
    for (int_t i=0; i < iters; i++)
    {   
        if (ind < m ) A += ind;

        double res = zero;
        
        if (ind < m )
        {
            for (int_t col=ty; col < n; col += DIM_Y)
            {       
                res += A[col*lda] * x[col*incx];
            }
        }

        if (DIM_X >= num_threads) // indicated 1D threads configuration. Shared memory is not needed, reduction is done naturally
        {
            if (ty == 0 && ind < m)
            {
                y[ind*incy] = alpha*res + beta*y[ind*incy];
            }
        }
        else 
        {
            sdata[ty + tx * DIM_Y] = res;

            __syncthreads(); 

            if ( DIM_Y > 16)
            { 
                magma_sum_reduce(DIM_Y, ty, sdata + tx * DIM_Y);
            }
            else
            {
                if (ty == 0 && ind < m)
                {
                    for (int_t i=1; i < DIM_Y; i++)
                    {
                        sdata[tx * DIM_Y] += sdata[i + tx * DIM_Y]; 
                    }
                }
            }

            if (ty == 0 && ind < m)
            {
                y[ind*incy] = alpha*sdata[tx * DIM_Y] + beta*y[ind*incy];
            }

            __syncthreads();
        }

        if ( ind < m) A -= ind;

        ind += DIM_X;
    }
}

	
	
	

/******************************************************************************/
static __device__ 
void gemm_device_dlsum_fmod(
    int_t M, int_t N, int_t K,
	int_t blx, int_t bly,
    const double* __restrict__ A, int_t LDA,
    const double* __restrict__ B, int_t LDB,
	double rC[THR_N][THR_M],
    double alpha, double beta)
{
// #if (__CUDA_ARCH__ >= 200)
    int_t idx = threadIdx.x;  // thread's m dimension
    int_t idy = threadIdx.y;  // thread's n dimension

    int_t idt = DIM_X * idy + idx;    // thread's global number

    int_t idxA = idt % DIM_XA;    // idx within A
    int_t idyA = idt / DIM_XA;    // idy within A

    int_t idxB = idt % DIM_XB;    // idx within B
    int_t idyB = idt / DIM_XB;    // idy within B

    // int_t blx = blockIdx.x;   // block's m dimension
    // int_t bly = blockIdx.y;   // block's n dimension

	__shared__ double sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ double sB[BLK_N][BLK_K+1];      // +1 always required	
	
    // Registers for the innermost loop
    double rA[THR_M];
    double rB[THR_N];

    double ra[BLK_K/DIM_YA+1][BLK_M/DIM_XA];
    double rb[BLK_N/DIM_YB][BLK_K/DIM_XB+1];
	
	const double *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    const double *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    int_t boundA = (LDA*(K-1) + M) - ( blx*BLK_M  + idyA*LDA + idxA ) -1;
    int_t boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int_t m, n, k, kk;
	double zero = 0.0;

    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rC[n][m] = zero;

    #pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YA)
        #pragma unroll
        for (m = 0; m < BLK_M; m += DIM_XA)
            sA[n+idyA][m+idxA] = fetch(A, m, n, boundA);
    
    #pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB)
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XB)
            sB[n+idyB][m+idxB] = fetch(B, m, n, boundB);
    
    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
		offs_dA += BLK_K*LDA;
		boundA  -= BLK_K*LDA;

		offs_dB += BLK_K;
		boundB  -= BLK_K;

        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_M/DIM_XA; m++)
                ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

        #pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XB; m++)
                rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);
        
        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    fma(rA[m], rB[n], rC[n][m]);
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_M/DIM_XA; m++)
                sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];
        
        #pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XB; m++)
                sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];
        
        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                fma(rA[m], rB[n], rC[n][m]);
            }
        }
    }

    // Store C regs->dev
    // if( beta == make_FloatingPoint_t(0.0,0.0) ) {
        // #pragma unroll
        // for (n = 0; n < THR_N; n++) {
            // int_t coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            // #pragma unroll
            // for (m = 0; m < THR_M; m++) {
                // int_t coord_dCm = blx*BLK_M + m*DIM_X + idx;
                // if (coord_dCm < M && coord_dCn < N) {
                    // int_t offsC = coord_dCn*LDC + coord_dCm;

                    // double &regC = rC[n][m];
                    // double &memC = C[offsC];

                    // // memC = mul(alpha, regC);
                // }
            // }
        // }
    // } else {
        // #pragma unroll
        // for (n = 0; n < THR_N; n++) {
            // int_t coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            // #pragma unroll
            // for (m = 0; m < THR_M; m++) {
                // int_t coord_dCm = blx*BLK_M + m*DIM_X + idx;
                // if (coord_dCm < M && coord_dCn < N) {
                    // int_t offsC = coord_dCn*LDC + coord_dCm;

                    // double &regC = rC[n][m];
                    // double &memC = C[offsC];

                    // // memC = add(mul(alpha, regC), mul(beta, memC));
                // }
            // }
        // }
    // }
// #endif /* (__CUDA_ARCH__ >= 200) */
}



/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Perform local block modifications: lsum[i] -= L_i,k * X[k].
 * </pre>
 */
__global__ void dlsum_fmod_inv_cuda
/************************************************************************/
(
 double *lsum,    /* Sum of local modifications.                        */
 double *x,       /* X array (local)                                    */
 double *rtemp,   /* Result of full matrix-vector multiply.             */
 int   nrhs,      /* Number of right-hand sides.                        */
 int   maxsup,      /* Max supernode size.                        */
 int_t   nsupers,      /* Number of total supernodes.                        */
 int_t *fmod,     /* Modification count for L-solve.                    */
 int_t *xsup,
 gridinfo_t *grid,
 LocalLU_t *Llu
)
{
    double alpha = 1.0, beta = 0.0,malpha=-1.0;
    double *lusup, *lusup1;
    double *dest;
	double *Linv;/* Inverse of diagonal block */
	int    iam, iknsupc, myrow, mycol, krow, nbrow, nbrow1, nbrow_ref, nsupr, nsupr1, p, pi, idx_r,m;
	int_t  k,i, l,ii,jj, ik, il, ikcol, irow, j, lb, lk, rel, lib,lready;
	int_t  *lsub, *lsub1, nlb1, lptr1, luptr1,*lloc;
	int_t  luptr_tmp,luptr_tmp1,lptr1_tmp,maxrecvsz, idx_i, idx_v,idx_n,  idx_l, fmod_tmp, lbstart,lbend,nn,Nchunk,nlb_loc,remainder;
	int thread_id1;
	flops_t ops_loc=0.0;
    MPI_Status status;
    int test_flag;
	yes_no_t done;
	BcTree  *LBtree_ptr = Llu->LBtree_ptr;
	RdTree  *LRtree_ptr = Llu->LRtree_ptr;
	int_t* idx_lsum,idx_lsum1;
	const int Nbk=1;
	__shared__ double rtemp_loc[128]; 
	double temp,temp1;
	int_t ldalsum;
	int_t nleaf_send_tmp;
	int_t lptr;      /* Starting position in lsub[*].                      */
	int_t luptr;     /* Starting position in lusup[*].                     */
	int_t iword = sizeof(int_t);
	int_t dword = sizeof (double);
	int_t aln_d,aln_i;
	aln_d = 1;//ceil(CACHELINE/(double)dword);
	aln_i = 1;//ceil(CACHELINE/(double)iword);
	int   knsupc;    /* Size of supernode k.                               */
	int_t nlb;       /* Number of L blocks.                                */
	int_t  *ilsum = Llu->ilsum; /* Starting position of each supernode in lsum.   */
	
	int_t bid;
	int_t tmp;
	int_t tid = threadIdx.x + threadIdx.y * blockDim.x; 
	int_t ready = 0;
	int_t lock = 0;
	const int block_size = blockDim.x*blockDim.y; /* number of threads per warp*/
	double zero = 0.0;


	double rC[THR_N][THR_M];
	
	cudaError_t error;
	
	bid= blockIdx.x;
    int_t idx = threadIdx.x;  // thread's m dimension
    int_t idy = threadIdx.y;  // thread's n dimension
	int_t ni,mi;
	
	// printf("  Entering kernel:   %i %i %i %i %i %i %i %i\n", threadIdx.x, blockIdx.x, grid->npcol, nsupers,myrow,krow,bid,tid);
	
	
	// rtemp_loc = (double*)malloc(maxsup*nrhs*Nbk*sizeof(double));
	
	if(bid>=CEILING(nsupers, grid->npcol)){
	return;
	}else if(!Llu->Lrowind_bc_ptr[bid]){
	return;
	}
	
	

	lk=bid;
	iam = grid->iam;
	mycol = MYCOL( iam, grid );
	myrow = MYROW( iam, grid );
	k = mycol+lk*grid->npcol;
	knsupc = SuperSize( k );
	lsub = Llu->Lrowind_bc_ptr[lk];
	iam = grid->iam;
	krow = PROW( k, grid );	
	lusup = Llu->Lnzval_bc_ptr[lk];
	lloc = Llu->Lindval_loc_bc_ptr[lk];
	nsupr = lsub[1];
	
	if(myrow==krow){
		nlb = lsub[0] - 1;
		idx_n = 1;
		idx_i = nlb+2;
		idx_v = 2*nlb+3;
		luptr_tmp = lloc[idx_v];
		m = nsupr-knsupc;
	}else{
		nlb = lsub[0];
		idx_n = 0;
		idx_i = nlb;
		idx_v = 2*nlb;
		luptr_tmp = lloc[idx_v];
		m = nsupr;
	}	
	
	// printf("  Before kernel:   %i %i %i %i %i %i %i %i\n", threadIdx.x, blockIdx.x, grid->npcol, nsupers,myrow,krow,bid,tid);
	
	if(myrow==krow){   /* diagonal block performs trsm and forward the message*/

		if(tid==0){  /*only the first thread in a warp handles the lock */

		// printf("bk: %5d r: %5d %5d %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i],myrow,krow);
		// for (i=0 ; i<maxsup ; i++){
			// rtemp_loc[i]=0.0;
		// }	
		
			lib = LBi( k, grid ); /* Local block number, row-wise. */
			tmp=fmod[lib*aln_i];
			while(tmp>0){
				tmp=fmod[lib*aln_i];
				__threadfence();
				// clock();
				// printf("loop: %5d\n",tmp);
				// tmp=threadIdx.x + blockIdx.x * blockDim.x;
				// tmp++;
				
				
			}
		}
		__syncthreads();
		
			
			lib = LBi( k, grid ); /* Local block number, row-wise. */
			il = LSUM_BLK( lib );
			ii = X_BLK( lib );
			
			RHS_ITERATE(j)
				for (i = tid; i < knsupc; i+=block_size)
					x[i + ii + j*knsupc] += lsum[i + il + j*knsupc ];
			// __syncthreads();
			
			
			if(Llu->inv == 1){
			
				Linv = Llu->Linv_bc_ptr[lk];
					
				if(nrhs==1){
				
					for (i = tid; i < knsupc; i+=block_size){					
						temp1=zero;
						for (l=0 ; l<knsupc ; l++){
							temp1+=  Linv[l*knsupc+i]*x[ii+l];
						}								
						lsum[il+i]=temp1; //reuse lsum as temporary output as it's no longer accessed
					}
					// __syncthreads();					
						
					for (i = tid; i < knsupc; i+=block_size){
						x[i + ii] = lsum[il+i];
						// printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
						}					
					// __syncthreads();		
						

					
					// RHS_ITERATE(j){
					
					// for (i = tid; i < knsupc; i+=block_size)
						// rtemp_loc[i]=zero;					
					// __syncthreads(); 
					
									
					// gemv_device_dlsum_fmod(
						// knsupc, knsupc, alpha,
						// Linv, knsupc,
						// &x[ii+j*knsupc], 1, beta,
						// rtemp_loc, 1);											
						
					// __syncthreads(); 
					// // printf("tid %5d knsupc %5d block_size %5d\n",tid,knsupc,block_size);
					// for (i = tid; i < knsupc; i+=block_size){
						// x[i + ii + j*knsupc] = rtemp_loc[i];
						// // printf("lk %5d %lf\n",lk,x[i + ii + j*knsupc]);
						// }
					// }	
					// __syncthreads(); 	
					
				}else{
					__syncthreads(); 	
					for (int_t blx = 0; blx*BLK_M < knsupc; blx++){
						for (int_t bly = 0; bly*BLK_N < nrhs; bly++){
							gemm_device_dlsum_fmod(knsupc, nrhs, knsupc, blx, bly, 
							Linv, knsupc, &x[ii], knsupc, rC,
							alpha, beta);
								#pragma unroll
							for (ni = 0; ni < THR_N; ni++) {
								int_t coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
								#pragma unroll
								for (mi = 0; mi < THR_M; mi++) {
									int_t coord_dCm = blx*BLK_M + mi*DIM_X + idx;
									if (coord_dCm < knsupc && coord_dCn < nrhs) {
										double &regC = rC[ni][mi];
										lsum[coord_dCm + il + coord_dCn*knsupc ]=regC;  //reuse lsum as temporary output as it's no longer accessed
									}//if (coord_dCm < knsupc && coord_dCn < nrhs)
								}
							}						
						}
					}
					__syncthreads(); 	

					RHS_ITERATE(j)
					for (i = tid; i < knsupc; i+=block_size)
						x[i + ii + j*knsupc] = lsum[i + il + j*knsupc ];
					__syncthreads(); 		
				}//if(nrhs==1)
			}
			
			
		__syncthreads();	
	}else{   /* off-diagonal block forward the message*/
		/* waiting for the x subvector and forward*/ 
	}
	
	
	if(nlb>0){
	
			lib = LBi( k, grid ); /* Local block number, row-wise. */
			ii = X_BLK( lib );	
			
			if(nrhs==1){
				luptr_tmp1 = lloc[idx_v];
				lb = 0;
				nbrow=0;
				lptr1_tmp = lloc[lb+idx_i];
				lptr= lptr1_tmp+2;
				nbrow1 = lsub[lptr1_tmp+1];
				ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
				rel = xsup[ik]; /* Global row index of block ik. */
				lk = LBi( ik, grid ); /* Local block number, row-wise. */
				iknsupc = SuperSize( ik );
				il = LSUM_BLK( lk );			
				
				for (i = tid; i < m; i+=block_size){
					while(nbrow+lsub[lptr1_tmp+1]<=i){
						lb++;
						nbrow +=lsub[lptr1_tmp+1];
						lptr1_tmp = lloc[lb+idx_i];
						lptr= lptr1_tmp+2;
						ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
						rel = xsup[ik]; /* Global row index of block ik. */
						lk = LBi( ik, grid ); /* Local block number, row-wise. */
						iknsupc = SuperSize( ik );
						il = LSUM_BLK( lk );				
					}
					
					irow = lsub[lptr+i-nbrow] - rel; /* Relative row. */
					RHS_ITERATE(j){
					temp1=zero;
					for (l=0 ; l<knsupc ; l++){
						temp1+= lusup[luptr_tmp1+l*nsupr+i]*x[ii+j*knsupc+l];
					}								
					temp=atomicAdd(&lsum[il+irow + j*iknsupc],-temp1);
					}
					if(i==nbrow+lsub[lptr1_tmp+1]-1){
						fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
						// __threadfence();
					}
				}
				__syncthreads();
		
			}else {				
				for (lb = 0; lb < nlb; lb++){
					luptr_tmp1 = lloc[lb+idx_v];
					
					// nbrow=0;					
					// lptr1_tmp = lloc[lb+idx_i];
					// nbrow += lsub[lptr1_tmp+1];
				

					lib = LBi( k, grid ); /* Local block number, row-wise. */
					ii = X_BLK( lib );

					lptr1_tmp = lloc[lb+idx_i];
					lptr= lptr1_tmp+2;
					nbrow1 = lsub[lptr1_tmp+1];
					ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
					rel = xsup[ik]; /* Global row index of block ik. */

					lk = LBi( ik, grid ); /* Local block number, row-wise. */

					iknsupc = SuperSize( ik );
					il = LSUM_BLK( lk );

						
					// if(nrhs==1){

						// for (i = tid; i < nbrow1; i+=block_size)
							// rtemp_loc[i]=zero;					
						// __syncthreads(); 
						
					
						// gemv_device_dlsum_fmod(
							// nbrow1, knsupc, alpha,
							// &lusup[luptr_tmp1], nsupr,
							// &x[ii], 1, beta,
							// rtemp_loc, 1);	
						
						// __syncthreads(); 
						// for (i = tid; i < nbrow1; i+=block_size){
							// irow = lsub[lptr+i] - rel; /* Relative row. */
							// temp=atomicAdd(&lsum[il+irow],-rtemp_loc[i]);
							// }
					// }else{	
					
						for (int_t blx = 0; blx*BLK_M < nbrow1; blx++){
							for (int_t bly = 0; bly*BLK_N < nrhs; bly++){
								gemm_device_dlsum_fmod(nbrow1, nrhs, knsupc, blx, bly, 
								&lusup[luptr_tmp1], nsupr, &x[ii], knsupc, rC,
								alpha, beta);
									#pragma unroll
								for (ni = 0; ni < THR_N; ni++) {
									int_t coord_dCn = bly*BLK_N + ni*DIM_Y + idy;
									#pragma unroll
									for (mi = 0; mi < THR_M; mi++) {
										int_t coord_dCm = blx*BLK_M + mi*DIM_X + idx;
										if (coord_dCm < nbrow1 && coord_dCn < nrhs) {
											irow = lsub[lptr+coord_dCm] - rel; /* Relative row. */
											double &regC = rC[ni][mi];
											temp=atomicAdd(&lsum[il+irow + coord_dCn*iknsupc],-regC);
										}
									}
								}						
							}
						}
					// }//if(nrhs==1)
					
					if(tid==0)fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
					
					
		
				}

			}//if(nrhs==1)
		
			
			// if(tid==0){
			// for (lb = tid; lb < nlb; lb+=block_size){
					// lptr1_tmp = lloc[lb+idx_i];
					// ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
					// lk = LBi( ik, grid ); /* Local block number, row-wise. */
					// fmod_tmp=atomicSub(&fmod[lk*aln_i],1);
					// // printf("k: %5d r: %5d\n",mycol+bid*grid->npcol,fmod[2*aln_i]);
			// }
			// }
			__syncthreads();
		// } /*if tid<Nchunk*/
	} /* if nlb>0*/		
		
	
} /* dlsum_fmod_inv_cuda */


void dlsum_fmod_inv_cuda_wrap
(
 int_t nblock,    /*kernel launch parameter*/
 int_t nthread_x,     /*kernel launch parameter*/
 int_t nthread_y,     /*kernel launch parameter*/
 double *lsum,    /* Sum of local modifications.                        */
 double *x,       /* X array (local)                                    */
 double *rtemp,   /* Result of full matrix-vector multiply.             */
 int   nrhs,      /* Number of right-hand sides.                        */
 int   maxsup,      /* Max supernode size.                        */
 int_t   nsupers,      /* Number of total supernodes.                        */
 int_t *fmod,     /* Modification count for L-solve.                    */
 int_t *xsup,
 gridinfo_t *grid,
 LocalLU_t *Llu
){

cudaStream_t sid=0;
int gid=0;
int mycol;
int_t lk,k,knsupc;
dim3 dimBlock(nthread_x, nthread_y);
	
	// printf("pinv %d\n",Llu->inv);
	// fflush(stdout);
	dlsum_fmod_inv_cuda<<< nblock, dimBlock >>>(lsum,x,rtemp,nrhs,maxsup,nsupers,fmod,xsup,grid,Llu);
	// dlsum_fmod_inv_cuda<<< 4, 32 >>>(lsum,x,rtemp,nrhs,maxsup,nsupers,fmod,xsup,grid,Llu);
	cudaDeviceSynchronize();
}

