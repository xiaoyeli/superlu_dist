/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file 
 * \brief 4D example with dense blocks
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 * December 6, 2018
 * </pre>
 */

#include <math.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "superlu_zdefs.h"

/// Return the number of seconds from some start
inline double w_time() {
    return SuperLU_timer_();
    // return std::chrono::duration<double>(
    //            std::chrono::system_clock::now().time_since_epoch())
    //     .count();
}

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program PZTEST.
 *
 * This example illustrates how to use PZGSSVX to apply the ILU(0) factorization.
 * 
 * Five basic steps are required:
 *   1. Initialize the MPI environment and the SuperLU process grid
 *   2. Set up the input matrix and the right-hand side
 *   3. Set the options argument
 *   4. Call pzgssvx
 *   5. Release the process grid and terminate the MPI environment
 *
 * With MPICH,  program may be run by typing:
 *    mpiexec -n <np> pztest [-dim='x y z t']
 * </pre>
 */

int zcreate_matrix_qcd(SuperMatrix *A, int nrhs, doublecomplex **rhs,
                   int *ldb, doublecomplex **x, int *ldx,
                   int dim[4], int block_size, gridinfo3d_t *grid)
{
    SuperMatrix GA;              /* global A */
    int_t    *rowind, *colptr;   /* global */
    doublecomplex   *nzval;             /* global */
    doublecomplex   *nzval_loc;         /* local */
    int_t    *colind, *rowptr;   /* local */
    int_t    m, n, nnz;
    int_t    m_loc, fst_row, nnz_loc;
    int_t    m_loc_fst; /* Record m_loc of the first p-1 processors,
               when mod(m, p) is not zero. */ 
    int_t    row, col, i, j, relpos;
    int      iam;
    int_t      *marker;

    iam = grid->iam;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter dcreate_matrix_qcd()");
#endif

    /* Generate the dimensions of the matrix */
    n = m = (int_t)dim[0] * dim[1] * dim[2] * dim[3] * block_size;

    /* Compute the number of neighbors */
    int neighbors = 1; /* local nonzero */
    for (int d = 0; d < 4; ++d) {
        if (dim[d] > 1) neighbors++;
        if (dim[d] > 2) neighbors++;
    }

    /* Create the matrix */
    nnz = m * block_size * neighbors;
    zallocateA_dist(n, nnz, &nzval, &rowind, &colptr);

    if(!iam){
        printf("matrix size %10d nnz %15d\n",n,nnz);
        fflush(stdout);
    }


    colptr[0] = 0;
      for (int lt=0, col=0, j=0, jL=0, jU=0; lt<dim[3]; ++lt) {
        for (int lz=0; lz<dim[2]; ++lz) {
          for (int ly=0; ly<dim[1]; ++ly) {
            for (int lx=0; lx<dim[0]; ++lx) {
              for (int b=0; b<block_size; ++b, ++col) {
                /* Add local nonzeros */
                for (int bj=0; bj<block_size; ++bj) {
                int_t row = (
                  lx +
                  ly*dim[0] +
                  lz*dim[0]*dim[1] +
                  lt*dim[0]*dim[1]*dim[2]
                  )*block_size + bj;
                    rowind[j] = row;
                nzval[j] = (b == bj ? (doublecomplex){1.0,0.0} : (doublecomplex){1e-5,0.0});
                  j++;
                }
                /* Add neighbors */
                for (int d=0; d<4; ++d) {
                  for (int delta=-1; delta<(dim[d] > 2 ? 2 : (dim[d] > 1 ? 0 : -1)); delta+=2) {
                    for (int bj=0; bj<block_size; ++bj) {
                      int u[4] = {0,0,0,0};
                      u[d] = delta;
                    int_t row = (
                      (lx+u[0]+dim[0])%dim[0] +
                      (ly+u[1]+dim[1])%dim[1]*dim[0] +
                      (lz+u[2]+dim[2])%dim[2]*dim[0]*dim[1] +
                      (lt+u[3]+dim[3])%dim[3]*dim[0]*dim[1]*dim[2]) * block_size + bj;
                        rowind[j] = row;
                    nzval[j] = (b == bj ? (doublecomplex){1e-5,0.0} : (doublecomplex){1e-10,0.0});
                      j++;
                    }
                  }
                }
                colptr[col+1] = j;
              }
            }
          }
        }
      }
#if 0
	if ( !iam ) {
        char newfile[4096] = "/global/cfs/cdirs/m2957/liuyangz/my_research/matrix/qcd.bin";
		printf("Generate binary file: %s\n", newfile);
		fflush(stdout);
		zwrite_binary_withname(n, nnz, nzval, rowind, colptr, newfile);				
    }
#endif



    /* Compute the number of rows to be distributed to local process */
    int grid_vol = grid->nprow * grid->npcol * grid->npdep;
    m_loc = m / grid_vol; 
    m_loc_fst = m_loc;
    /* When m / procs is not an integer */
    if ((m_loc * grid_vol) != m) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
      if (iam == (grid_vol - 1)) /* last proc. gets all*/
      m_loc = m - m_loc * (grid_vol - 1);
    }

    /* Create compressed column matrix for GA. */
    zCreate_CompCol_Matrix_dist(&GA, m, n, nnz, nzval, rowind, colptr,
                SLU_NC, SLU_Z, SLU_GE);

    /* Generate the exact solution and compute the right-hand side. */
    doublecomplex  *b_global=NULL, *xtrue_global=NULL;  /* replicated on all processes */
    if (nrhs > 0 && !(b_global = doublecomplexMalloc_dist(m*nrhs)) )
        ABORT("Malloc fails for b[]");
    if (nrhs > 0 && !(xtrue_global = doublecomplexMalloc_dist(n*nrhs)) )
        ABORT("Malloc fails for xtrue[]");

#if 1
    char trans[1];
    *trans='N';
    zGenXtrue_dist(n, nrhs, xtrue_global, n);
    zFillRHS_dist(trans, nrhs, xtrue_global, n, &GA, b_global, m);
#else 
    for (int_t i=0; i<m*nrhs; ++i) xtrue_global[i] = (doublecomplex){1.0, 0.0};
    double val = 1.0 /* block diag */ +
                 1e-5 * (block_size - 1) /* out of block diag */ +
                 neighbors * 1e-5 /* block diag */ +
                 neighbors * (block_size - 1) * 1e-10 /* out of block diag */;
    for (int_t i=0; i<m*nrhs; ++i) b_global[i] = (doublecomplex){val, 0.0};
#endif

    /*************************************************
     * Change GA to a local A with NR_loc format     *
     *************************************************/

    rowptr = (int_t *) intMalloc_dist(m_loc+1);
    marker = (int_t *) intCalloc_dist(n);

    /* Get counts of each row of GA */
    for (i = 0; i < n; ++i)
      for (j = colptr[i]; j < colptr[i+1]; ++j) ++marker[rowind[j]];
    /* Set up row pointers */
    rowptr[0] = 0;
    fst_row = iam * m_loc_fst;
    nnz_loc = 0;
    for (j = 0; j < m_loc; ++j) {
      row = fst_row + j;
      rowptr[j+1] = rowptr[j] + marker[row];
      marker[j] = rowptr[j];
    }
    nnz_loc = rowptr[m_loc];

    nzval_loc = (doublecomplex *) doublecomplexMalloc_dist(nnz_loc);
    colind = (int_t *) intMalloc_dist(nnz_loc);

    /* Transfer the matrix into the compressed row storage */
    for (i = 0; i < n; ++i) {
      for (j = colptr[i]; j < colptr[i+1]; ++j) {
        row = rowind[j];
        if ( (row>=fst_row) && (row<fst_row+m_loc) ) {
          row = row - fst_row;
          relpos = marker[row];
          colind[relpos] = i;
          nzval_loc[relpos] = nzval[j];
          ++marker[row];
        }
      }
    }


#if ( DEBUGlevel>=2 )
    if ( !iam ) zPrint_CompCol_Matrix_dist(&GA);
#endif   

    /* Destroy GA */
    Destroy_CompCol_Matrix_dist(&GA);

    /******************************************************/
    /* Change GA to a local A with NR_loc format */
    /******************************************************/

    /* Set up the local A in NR_loc format */
    zCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
                   nzval_loc, colind, rowptr,
                   SLU_NR_loc, SLU_Z, SLU_GE);
    
    /* Get the local B */
    if (nrhs > 0 && rhs) {
        if ( !((*rhs) = doublecomplexMalloc_dist(m_loc*nrhs)) )
            ABORT("Malloc fails for rhs[]");
        for (j =0; j < nrhs; ++j) {
            for (i = 0; i < m_loc; ++i) {
                row = fst_row + i;
                (*rhs)[j*m_loc+i] = b_global[j*n+row];
            }
        }
        *ldb = m_loc;
    }

    /* Set the true X */    
    if (nrhs > 0 && x) {
        *ldx = m_loc;
        if ( !((*x) = doublecomplexMalloc_dist(*ldx * nrhs)) )
            ABORT("Malloc fails for x_loc[]");

        /* Get the local part of xtrue_global */
        for (j = 0; j < nrhs; ++j) {
          for (i = 0; i < m_loc; ++i)
            (*x)[i + j*(*ldx)] = xtrue_global[i + fst_row + j*n];
        }
    }

    if (nrhs>0) SUPERLU_FREE(b_global);
    if (nrhs>0) SUPERLU_FREE(xtrue_global);
    SUPERLU_FREE(marker);

#if ( DEBUGlevel>=1 )
    printf("sizeof(NRforamt_loc) %lu\n", sizeof(NRformat_loc));
    CHECK_MALLOC(iam, "Exit zcreate_matrix()");
#endif
    return 0;
}


int main(int argc, char *argv[])
{
    SuperLUStat_t stat;
    SuperMatrix A;
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    zSOLVEstruct_t SOLVEstruct;
    gridinfo3d_t grid;
    double   *berr;
    float output;
    doublecomplex   *b, *xtrue;
    int    m, n;
    int      nprow, npcol, npdep;
    int      iam, info, ldb, ldx, nrhs;
    int omp_mpi_level;
    int_t i;
    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    npdep = 1;  /* Default process columns.   */
    nrhs = 2;   /* Number of right-hand side. */
    double t0=0;
    int nsolves=1;
    
    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    //MPI_Init( &argc, &argv );
    MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level); 
        

#if ( VAMPIR>=1 )
    VT_traceoff(); 
#endif

#if ( VTUNE>=1 )
        __itt_pause();
#endif

    int dim[4] = {2, 1, 1, 1}; // xyzt
    int block_size = 1; // nonzero dense block dimension
    int nrep=10; // number of repetitions to get more accurate performance measurements

    // Get options
    for (int i = 1; i < argc; ++i) {
        if (strncmp("-dim=", argv[i], 5) == 0) {
            if (sscanf(argv[i] + 5, "%d %d %d %d", &dim[0], &dim[1], &dim[2], &dim[3]) != 4) {
                ABORT("-dim= should follow 64 numbers, for instance -dim='2 2 2 2'");
            }
            if (dim[0] < 1 || dim[1] < 1 || dim[2] < 1 || dim[3] < 1) {
                ABORT("One of the dimensions is smaller than one");
            }
        } else if (strncmp("-rep=", argv[i], 5) == 0) {
            if (sscanf(argv[i] + 5, "%d", &nrep) != 1) {
                ABORT("-rep= should follow a number, for instance -rep=3");
            }
            if (nrep < 1) {
                ABORT("The rep should be greater than zero");
            }
        } else if (strncmp("-bs=", argv[i], 4) == 0) {
            if (sscanf(argv[i] + 4, "%d", &block_size) != 1) {
                ABORT("-bs= should follow a number, for instance -bs=3");
            }
            if (block_size < 1) {
                ABORT("The rep should be greater than zero");
            }
        } else if (strncmp("-grid=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d %d %d", &nprow, &npcol, &npdep) != 3) {
                ABORT("-grid= should follow 3 numbers, for instance -grid='2 2 2'");
            }
            if (nprow < 1 || npcol < 1 || npdep < 1) {
                ABORT("One of the grid dimensions is smaller than one");
            }
        } else {
            ABORT("Unknown commandline option");
        }
    }

    int np=1, rank=0;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID. 
       ------------------------------------------------------------*/
    superlu_gridinit3d(MPI_COMM_WORLD, nprow, npcol, npdep, &grid);


    /* Set the default input options: */
    superlu_dist_options_t options;
    set_default_options_dist(&options);
    options.Algo3d = YES;
    /* Turn off permutations */
    options.SolveOnly          = YES;
    options.ILU_level          = 0;
    options.IterRefine        = NOREFINE;


#ifdef GPU_ACC
    /* ------------------------------------------------------------
       INITIALIZE GPU ENVIRONMENT
       ------------------------------------------------------------ */
    int superlu_acc_offload = sp_ienv_dist(10, &options); //get_acc_offload();
    if (superlu_acc_offload) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        double t1 = SuperLU_timer_();
        gpuFree(0);
        double t2 = SuperLU_timer_();
        if(!rank)printf("first gpufree time: %7.4f\n",t2-t1);
        gpublasHandle_t hb;
        gpublasCreate(&hb);
        if(!rank)printf("first blas create time: %7.4f\n",SuperLU_timer_()-t2);
        gpublasDestroy(hb);
	}
#endif



    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if ( (iam >= nprow * npcol * npdep) || (iam == -1) ) goto out;
    if ( !iam ) {
        int v_major, v_minor, v_bugfix;
        superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
        printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

        printf("Process grid:\t\t%d X %d X %d\n", (int)grid.nprow, (int)grid.npcol, (int)grid.npdep);
    }

#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
       ------------------------------------------------------------*/
    /* here, LU is ILU(0) */
    // Mimic symbolic factorization: set up Glu_freeable_t {} structure
    zcreate_matrix_qcd(&A, nrhs, &b, &ldb, &xtrue, &ldx, dim, block_size, &grid);

    if ( !(berr = doubleMalloc_dist(nrhs)) )
        ABORT("Malloc fails for berr[].");

    m = A.nrow;
    n = A.ncol;

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM.
       ------------------------------------------------------------*/
     /* Initialize ScalePermstruct and LUstruct. */
    zScalePermstructInit(m, n, &ScalePermstruct);
    zLUstructInit(n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit(&stat);

    // Set up Identity permutation vectors for pzdistribute
    for(i = 0; i < m; i++) ScalePermstruct.perm_r[i] = i;
    for(i = 0; i < n; i++)ScalePermstruct.perm_c[i] = i;



    if (!iam) {
        print_sp_ienv_dist(&options);
        print_options_dist(&options);
        fflush(stdout);
    }


    for (i = 0; i < n; i++) LUstruct.etree[i] = i+1;
     
    /* Allow to set up supernode partition */
    options.UserDefineSupernode = YES;





    /* Set up supernode partition */
    //ilu_level_symbfact(&options, &A, ScalePermstruct.perm_c, LUstruct.etree, LUstruct.Glu_persist, &Glu_freeable);
    
    if (options.UserDefineSupernode == YES) {
        /* User needs to allocate supno[]/xsup, and fill in the supernode partition
	   User does not need to free them; they are free'd by zDestroy_LU() */
	int_t *supno = intMalloc_dist(n);
	int_t *xsup = intMalloc_dist(n+1);
        for (i = 0; i < n; i++) supno[i] = i/block_size;
        for (i = 0; i < n/block_size; i++) xsup[i] = i*block_size;
        xsup[n/block_size] = n;
	
	/* aliases for superlu internal access */
        LUstruct.Glu_persist->supno = supno; 
	LUstruct.Glu_persist->xsup = xsup;
    }

    ///* 'fact' is set to be DOFACT to enable first-time distribution */
    //options.Fact = DOFACT;
    //output = pzdistribute3d(&options, n, &LU, &ScalePermstruct, &Glu_freeable, &LUstruct, &grid);

    //options.Fact = FACTORED;
	
    /* Call the linear equation solver. */
    t0 = w_time();
    pzgssvx3d(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid, &LUstruct, &SOLVEstruct, berr, &stat, &info);
    if (rank == 0) std::cout << "Time for the first call: " << (w_time() - t0) << std::endl;
    
    if ( grid.zscp.Iam == 0 ) { // process layer 0
        PStatPrint(&options, &stat, &grid.grid2d);        /* Print the statistics. */
    }

    // Second call!!
    options.SolveOnly= NO; // YL: options->SolveOnly will set Fact to DOFACT for distribution 
    options.Fact = FACTORED;

    t0 = w_time();
    for (int i = 0; i < nsolves; ++i)
      pzgssvx3d(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
                &LUstruct, &SOLVEstruct, berr, &stat, &info);
    if (rank == 0) std::cout << "Time to apply ILU(0): " << (w_time() - t0)/nsolves << std::endl;

    if ( info ) {  /* Something is wrong */
        if ( iam==0 ) {
            printf("ERROR: INFO = %d returned from pzgssvx()\n", info);
            fflush(stdout);
        }
    } else {
        /* Check the accuracy of the solution. */
        pzinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
                         nrhs, b, ldb, xtrue, ldx, grid.comm);
    }

    if ( grid.zscp.Iam == 0 ) { // process layer 0
        PStatPrint(&options, &stat, &grid.grid2d);        /* Print the statistics. */
    }
    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

    PStatFree(&stat);
    Destroy_CompRowLoc_Matrix_dist(&A);
    zScalePermstructFree(&ScalePermstruct);
    zDestroy_LU(n, &grid.grid2d, &LUstruct);
    zLUstructFree(&LUstruct);
    zSolveFinalize(&options, &SOLVEstruct);
    SUPERLU_FREE(b);
    SUPERLU_FREE(xtrue);
    SUPERLU_FREE(berr);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
    superlu_gridexit3d(&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
    MPI_Finalize();

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit main()");
#endif

}