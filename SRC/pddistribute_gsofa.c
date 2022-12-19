/*! \file
  Copyright (c) 2003, The Regents of the University of California, through
  Lawrence Berkeley National Laboratory (subject to receipt of any required
  approvals from U.S. Dept. of Energy)
  All rights reserved.
  The source code is distributed under BSD license, see the file License.txt
  at the top-level directory.
  */


/*! @file
 * \brief Re-distribute A on the 2D process mesh.
 * <pre>
 * -- Distributed SuperLU routine (version 2.3) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 15, 2008
 * </pre>
 */ 
 
#include "superlu_ddefs.h"
#define supernodal_Uskipp 1
#define reload_dense_col 1
#define update_communication_str 1
#define UpdateToRecv 1
#define communicate_fsend_bsend 1
// #define testing_block 5576
#define enable_partial_lsub_sorting 1
// #define use_original_sendRLogic 1

// #define enableExtraToRecv 1
// #define debug2_2 1
// #define enable_extra_comm 1

int compare_dist (const void * a, const void * b)
{
    if ( *(int*)a <  *(int*)b ) return -1;
    if ( *(int*)a == *(int*)b ) return 0;
    if ( *(int*)a >  *(int*)b ) return 1;
}
 
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Re-distribute A on the 2D process mesh.
 *
 * Arguments
 * =========
 *
 * A      (input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        A may be overwritten by diag(R)*A*diag(C)*Pc^T.
 *        The type of A can be: Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.
 *
 * ScalePermstruct (input) dScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_freeable (input) *Glu_freeable_t
 *        The global structure describing the graph of L and U.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * colptr (output) int*
 *
 * rowind (output) int*
 *
 * a      (output) double*
 *
 * Return value
 * ============
 *   > 0, working storage (in bytes) required to perform redistribution.
 *        (excluding LU factor size)
 * </pre>
 */
    int_t
dReDistribute_A_gsofa(SuperMatrix *A, dScalePermstruct_t *ScalePermstruct,
        Glu_freeable_t *Glu_freeable, int_t *xsup, int_t *supno,
        gridinfo_t *grid, int_t *colptr[], int_t *rowind[],
        double *a[])
{
    NRformat_loc *Astore;
    int_t  *perm_r; /* row permutation vector */
    int_t  *perm_c; /* column permutation vector */
    int_t  i, irow, fst_row, j, jcol, k, gbi, gbj, n, m_loc, jsize,nnz_tot;
    int_t  nnz_loc;    /* number of local nonzeros */
    int_t  SendCnt; /* number of remote nonzeros to be sent */
    int_t  RecvCnt; /* number of remote nonzeros to be sent */
    int_t  *nnzToSend, *nnzToRecv, maxnnzToRecv;
    int_t  *ia, *ja, **ia_send, *index, *itemp = NULL;
    int_t  *ptr_to_send;
    double *aij, **aij_send, *nzval, *dtemp = NULL;
    double *nzval_a;
    double asum,asum_tot;
    int    iam, it, p, procs, iam_g;
    MPI_Request *send_req;
    MPI_Status  status;


    /* ------------------------------------------------------------
       INITIALIZATION.
       ------------------------------------------------------------*/
    iam = grid->iam;
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter dReDistribute_A()");
#endif
    perm_r = ScalePermstruct->perm_r;
    perm_c = ScalePermstruct->perm_c; 
    procs = grid->nprow * grid->npcol;
    Astore = (NRformat_loc *) A->Store;
  

    n = A->ncol;
    m_loc = Astore->m_loc;
    fst_row = Astore->fst_row;
    //   PrintInt10("AStore->rowptr (dist)",  n+1, Astore->rowptr);
    // PrintInt10("AStore->colind (dist)",  Astore->nnz_loc, Astore->colind);
    nnzToRecv = intCalloc_dist(2*procs);
    nnzToSend = nnzToRecv + procs;


                //  AStore = (NCformat *) A.Store;                  
                //     PrintInt10("AStore->rowptr",  n+1, AStore->rowptr);
                //     PrintInt10("AStore->colind",  AStore->nnz, AStore->colind);
    /* ------------------------------------------------------------
       COUNT THE NUMBER OF NONZEROS TO BE SENT TO EACH PROCESS,
       THEN ALLOCATE SPACE.
       THIS ACCOUNTS FOR THE FIRST PASS OF A.
       ------------------------------------------------------------*/
    //    printf("Going through first pass of A\n");
    //    fflush(stdout);
    //    printf("m_loc: %d\n",m_loc);
    //     fflush(stdout);
    for (i = 0; i < m_loc; ++i) {
        // printf("Accessing i = %d\n",i);
        // fflush(stdout);
        for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
        // printf(" j = %d\n", j);
        // fflush(stdout);
        //    printf("Before accessing  perm_c[perm_r[%d]] \n", i+fst_row);
        //     fflush(stdout);
            irow = perm_c[perm_r[i+fst_row]];  /* Row number in Pc*Pr*A */
            //        printf("IAM:%d After accessing  perm_c[perm_r[%d]] \n", iam, i+fst_row);
            // fflush(stdout);
            //    printf(" jcol = %d\n", jcol);
            // fflush(stdout);
            jcol = Astore->colind[j];
            // printf(" jcol = %d\n", jcol);
            // fflush(stdout);
            gbi = BlockNum( irow );
            gbj = BlockNum( jcol );
            p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );
            ++nnzToSend[p];
        }
    }

    //   printf("IAM:%d After first pass of A\n",iam);
    //    fflush(stdout);
    /* All-to-all communication */
    MPI_Alltoall( nnzToSend, 1, mpi_int_t, nnzToRecv, 1, mpi_int_t,
            grid->comm);

    maxnnzToRecv = 0;
    nnz_loc = SendCnt = RecvCnt = 0;

    for (p = 0; p < procs; ++p) {
        if ( p != iam ) {
            SendCnt += nnzToSend[p];
            RecvCnt += nnzToRecv[p];
            maxnnzToRecv = SUPERLU_MAX( nnzToRecv[p], maxnnzToRecv );
        } else {
            nnz_loc += nnzToRecv[p];
            /*assert(nnzToSend[p] == nnzToRecv[p]);*/
        }
    }
    k = nnz_loc + RecvCnt; /* Total nonzeros ended up in my process. */
    //  printf("Sendcount and RecvCount computed\n");
    //    fflush(stdout);
    /* Allocate space for storing the triplets after redistribution. */
    if ( k ) { /* count can be zero. */
        if ( !(ia = intMalloc_dist(2*k)) )
            ABORT("Malloc fails for ia[].");
        if ( !(aij = doubleMalloc_dist(k)) )
            ABORT("Malloc fails for aij[].");
    }
    ja = ia + k;

    /* Allocate temporary storage for sending/receiving the A triplets. */
    if ( procs > 1 ) {
        if ( !(send_req = (MPI_Request *)
                    SUPERLU_MALLOC(2*procs *sizeof(MPI_Request))) )
            ABORT("Malloc fails for send_req[].");
        if ( !(ia_send = (int_t **) SUPERLU_MALLOC(procs*sizeof(int_t*))) )
            ABORT("Malloc fails for ia_send[].");
        if ( !(aij_send = (double **)SUPERLU_MALLOC(procs*sizeof(double*))) )
            ABORT("Malloc fails for aij_send[].");
        if ( SendCnt ) { /* count can be zero */
            if ( !(index = intMalloc_dist(2*SendCnt)) )
                ABORT("Malloc fails for index[].");
            if ( !(nzval = doubleMalloc_dist(SendCnt)) )
                ABORT("Malloc fails for nzval[].");
        }
        if ( !(ptr_to_send = intCalloc_dist(procs)) )
            ABORT("Malloc fails for ptr_to_send[].");
        if ( maxnnzToRecv ) { /* count can be zero */
            if ( !(itemp = intMalloc_dist(2*maxnnzToRecv)) )
                ABORT("Malloc fails for itemp[].");
            if ( !(dtemp = doubleMalloc_dist(maxnnzToRecv)) )
                ABORT("Malloc fails for dtemp[].");
        }

        for (i = 0, j = 0, p = 0; p < procs; ++p) {
            if ( p != iam ) {
                ia_send[p] = &index[i];
                i += 2 * nnzToSend[p]; /* ia/ja indices alternate */
                aij_send[p] = &nzval[j];
                j += nnzToSend[p];
            }
        }
    } /* if procs > 1 */

    if ( !(*colptr = intCalloc_dist(n+1)) )
        ABORT("Malloc fails for *colptr[].");

    /* ------------------------------------------------------------
       LOAD THE ENTRIES OF A INTO THE (IA,JA,AIJ) STRUCTURES TO SEND.
       THIS ACCOUNTS FOR THE SECOND PASS OF A.
       ------------------------------------------------------------*/
    //         printf("Forming the triplets (IA, JA, AIJ)\n");
    //    fflush(stdout);
    nnz_loc = 0; /* Reset the local nonzero count. */
    nzval_a = Astore->nzval;
    for (i = 0; i < m_loc; ++i) {
        for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
            irow = perm_c[perm_r[i+fst_row]];  /* Row number in Pc*Pr*A */
            jcol = Astore->colind[j];
            gbi = BlockNum( irow );
            gbj = BlockNum( jcol );
            p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );

            if ( p != iam ) { /* remote */
                k = ptr_to_send[p];
                ia_send[p][k] = irow;
                ia_send[p][k + nnzToSend[p]] = jcol;
                aij_send[p][k] = nzval_a[j];
                ++ptr_to_send[p];
            } else {          /* local */
                ia[nnz_loc] = irow;
                ja[nnz_loc] = jcol;
                aij[nnz_loc] = nzval_a[j];
                ++nnz_loc;
                ++(*colptr)[jcol]; /* Count nonzeros in each column */
            }
        }
    }

// printf("Performing redistribution!\n");
// fflush(stdout);
    /* ------------------------------------------------------------
       PERFORM REDISTRIBUTION. THIS INVOLVES ALL-TO-ALL COMMUNICATION.
NOTE: Can possibly use MPI_Alltoallv.
------------------------------------------------------------*/
    for (p = 0; p < procs; ++p) {
        if ( p != iam && nnzToSend[p]>0 ) {  // cause two of the tests to hang
            //	if ( p != iam ) {
            it = 2*nnzToSend[p];
            MPI_Isend( ia_send[p], it, mpi_int_t,
                    p, iam, grid->comm, &send_req[p] );
            it = nnzToSend[p];
            MPI_Isend( aij_send[p], it, MPI_DOUBLE,
                    p, iam+procs, grid->comm, &send_req[procs+p] );
        }
        }

        for (p = 0; p < procs; ++p) {
            if ( p != iam && nnzToRecv[p]>0 ) {
                //if ( p != iam ) {
                it = 2*nnzToRecv[p];
                MPI_Recv( itemp, it, mpi_int_t, p, p, grid->comm, &status );
                it = nnzToRecv[p];
                MPI_Recv( dtemp, it, MPI_DOUBLE, p, p+procs,
                        grid->comm, &status );
                for (i = 0; i < nnzToRecv[p]; ++i) {
                    ia[nnz_loc] = itemp[i];
                    jcol = itemp[i + nnzToRecv[p]];
                    /*assert(jcol<n);*/
                    ja[nnz_loc] = jcol;
                    aij[nnz_loc] = dtemp[i];
                    ++nnz_loc;
                    ++(*colptr)[jcol]; /* Count nonzeros in each column */
                }
            }
            }

            for (p = 0; p < procs; ++p) {
                if ( p != iam && nnzToSend[p] > 0 ) {
                    //if ( p != iam ) {
                    MPI_Wait( &send_req[p], &status);
                    MPI_Wait( &send_req[procs+p], &status);
                }
                }

                /* ------------------------------------------------------------
                   DEALLOCATE TEMPORARY STORAGE
                   ------------------------------------------------------------*/

                SUPERLU_FREE(nnzToRecv);

                if ( procs > 1 ) {
                    SUPERLU_FREE(send_req);
                    SUPERLU_FREE(ia_send);
                    SUPERLU_FREE(aij_send);
                    if ( SendCnt ) {
                        SUPERLU_FREE(index);
                        SUPERLU_FREE(nzval);
                    }
                    SUPERLU_FREE(ptr_to_send);
                    if ( maxnnzToRecv ) {
                        SUPERLU_FREE(itemp);
                        SUPERLU_FREE(dtemp);
                    }
                }

                /* ------------------------------------------------------------
                   CONVERT THE TRIPLET FORMAT INTO THE CCS FORMAT.
                   ------------------------------------------------------------*/
                if ( nnz_loc ) { /* nnz_loc can be zero */
                    if ( !(*rowind = intMalloc_dist(nnz_loc)) )
                        ABORT("Malloc fails for *rowind[].");
                    if ( !(*a = doubleMalloc_dist(nnz_loc)) )
                        ABORT("Malloc fails for *a[].");
                }

                /* Initialize the array of column pointers */
                k = 0;
                jsize = (*colptr)[0];
                (*colptr)[0] = 0;
                for (j = 1; j < n; ++j) {
                    k += jsize;
                    jsize = (*colptr)[j];
                    (*colptr)[j] = k;
                }

                /* Copy the triplets into the column oriented storage */
                for (i = 0; i < nnz_loc; ++i) {
                    j = ja[i];
                    k = (*colptr)[j];
                    (*rowind)[k] = ia[i];
                    (*a)[k] = aij[i];
                    ++(*colptr)[j];
                }

                /* Reset the column pointers to the beginning of each column */
                for (j = n; j > 0; --j) (*colptr)[j] = (*colptr)[j-1];
                (*colptr)[0] = 0;

                if ( nnz_loc ) {
                    SUPERLU_FREE(ia);
                    SUPERLU_FREE(aij);
                }

#if ( DEBUGlevel>=1 )
                CHECK_MALLOC(iam, "Exit dReDistribute_A_gsofa()");
#endif

                return 0;
            } /* dReDistribute_A */

            float
                pddistribute_gsofa(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
                        dScalePermstruct_t *ScalePermstruct,
                        Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct,
                        gridinfo_t *grid)
                /*
                 * -- Distributed SuperLU routine (version 2.0) --
                 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
                 * March 15, 2003
                 *
                 *
                 * Purpose
                 * =======
                 *   Distribute the matrix onto the 2D process mesh.
                 *
                 * Arguments
                 * =========
                 *
                 * options (input) superlu_dist_options_t*
                 *        options->Fact Specifies whether or not the L and U structures will be re-used.
                 *        = SamePattern_SameRowPerm: L and U structures are input, and
                 *                                   unchanged on exit.
                 *        = DOFACT or SamePattern: L and U structures are computed and output.
                 *
                 * n      (input) int
                 *        Dimension of the matrix.
                 *
                 * A      (input) SuperMatrix*
                 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
                 *        A may be overwritten by diag(R)*A*diag(C)*Pc^T. The type of A can be:
                 *        Stype = SLU_NR_loc; Dtype = SLU_D; Mtype = SLU_GE.
                 *
                 * ScalePermstruct (input) dScalePermstruct_t*
                 *        The data structure to store the scaling and permutation vectors
                 *        describing the transformations performed to the original matrix A.
                 *
                 * Glu_freeable (input) *Glu_freeable_t
                 *        The global structure describing the graph of L and U.
                 *
                 * LUstruct (input) dLUstruct_t*
                 *        Data structures for L and U factors.
                 *
                 * grid   (input) gridinfo_t*
                 *        The 2D process mesh.
                 *
                 * Return value
                 * ============
                 *   > 0, working storage required (in bytes).
                 *
                 */
                {
                    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
                    dLocalLU_t *Llu = LUstruct->Llu;
                    int_t bnnz, fsupc, fsupc1, i, ii, irow, istart, j, ib, jb, jj, k, k1,
                          len, len1, nsupc;
                    int_t nprocs;/*gSoFa*/
                    int_t lib;  /* local block row number */
                    int_t nlb;  /* local block rows*/
                    int_t ljb;  /* local block column number */
                    int_t nrbl; /* number of L blocks in current block column */
                    int_t nrbu; /* number of U blocks in current block column */
                    int_t gb;   /* global block number; 0 < gb <= nsuper */
                    int_t lb;   /* local block number; 0 < lb <= ceil(NSUPERS/Pr) */
                    int_t ub,gik,iklrow,fnz;
                    int iam, jbrow, kcol, krow, mycol, myrow, pc, pr;
                    int p, jbcol,ljb_i, ljb_j; /*gSoFa*/
                    int_t mybufmax[NBUFFERS];
                    NRformat_loc *Astore;
                    double *a;
                    int_t *asub, *xa;
                    int_t *xa_begin, *xa_end;
                    int_t *xsup = Glu_persist->xsup;    /* supernode and column mapping */
                    int_t *supno = Glu_persist->supno;
                    int_t *lsub, *xlsub, *usub, *usub1, *xusub;
                    int_t nsupers;
                    int_t nsupers_i, nsupers_j, nsupers_ij; /*gSoFa*/
                    int_t next_lind;      /* next available position in index[*] */
                    int_t next_lval;      /* next available position in nzval[*] */
                    int_t *index;         /* indices consist of headers and row subscripts */
                    int_t *index_srt;         /* indices consist of headers and row subscripts */
                    int   *index1;        /* temporary pointer to array of int */
                    double *lusup, *lusup_srt, *uval; /* nonzero values in L and U */
                    double **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc) */
                    int_t  **Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc) */
                    int_t   **Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
                    int_t   *Unnz; /* size ceil(NSUPERS/Pc)                 */
                    double **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr) */
                    int_t  **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr) */

                    C_Tree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */ 
                    C_Tree  *LRtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
                    C_Tree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
                    C_Tree  *URtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
                    int msgsize;

                    int_t  *Urbs,*Urbs1; /* Number of row blocks in each block column of U. */
                    Ucb_indptr_t **Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
                    int_t  **Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */
                    /*-- Counts to be used in factorization. --*/
                    int  *ToRecv, *ToSendD, **ToSendR;
                    int **SendRReq; //gSoFa communication structure

                    /*-- Counts to be used in lower triangular solve. --*/
                    int_t  *fmod;          /* Modification count for L-solve.        */
                    int_t  **fsendx_plist; /* Column process list to send down Xk.   */
                    int_t* Request_fsendx_plist; /* gSoFa for communication to set up fsendx_plist */
                    int_t* Recv_Request_fsendx_plist; /* gSoFa for communication to set up fsendx_plist */
                    int_t* Request_bsendx_plist; /* gSoFa for communication to set up bsendx_plist */
                    int_t* Recv_Request_bsendx_plist; /* gSoFa for communication to set up bsendx_plist */
                    int_t  nfrecvx = 0;    /* Number of Xk I will receive.           */
                    int_t  nfsendx = 0;    /* Number of Xk I will send               */
                    int_t  kseen;

                    /*-- Counts to be used in upper triangular solve. --*/
                    int_t  *bmod;          /* Modification count for U-solve.        */
                    int_t  **bsendx_plist; /* Column process list to send down Xk.   */
                    int_t  nbrecvx = 0;    /* Number of Xk I will receive.           */
                    int_t  nbsendx = 0;    /* Number of Xk I will send               */
                    int_t  *ilsum;         /* starting position of each supernode in
                                              the full array (local)                 */

                    /*-- Auxiliary arrays; freed on return --*/
                    int_t *rb_marker;  /* block hit marker; size ceil(NSUPERS/Pr)           */
                    int_t *Urb_length; /* U block length; size ceil(NSUPERS/Pr)             */
                    int_t *Urb_indptr; /* pointers to U index[]; size ceil(NSUPERS/Pr)      */
                    int_t *Urb_fstnz;  /* # of fstnz in a block row; size ceil(NSUPERS/Pr)  */
                    int_t *Ucbs;       /* number of column blocks in a block row            */
                    int_t *Lrb_length; /* L block length; size ceil(NSUPERS/Pr)             */
                    int_t *Lrb_number; /* global block number; size ceil(NSUPERS/Pr)        */
                    int_t *Lrb_indptr; /* pointers to L index[]; size ceil(NSUPERS/Pr)      */
                    int_t *Lrb_valptr; /* pointers to L nzval[]; size ceil(NSUPERS/Pr)      */
                    int_t *ActiveFlag;
                    int_t *ActiveFlagAll;
                    int_t Iactive;
                    int *ranks;
                    int_t *idxs;
                    int_t **nzrows;
                    double rseed;
                    int rank_cnt,rank_cnt_ref,Root;
                    double *dense, *dense_col; /* SPA */
                    double zero = 0.0;
                    int_t ldaspa;     /* LDA of SPA */
                    int_t iword, dword;
                    float mem_use = 0.0;
                    float memTRS = 0.; /* memory allocated for storing the meta-data for triangular solve (positive number)*/
                    float memStrLU, memA,
                          memDist = 0.; /* memory used for redistributing the data, which does
                                           not include the memory for the numerical values
                                           of L and U (positive number)*/
                    float  memNLU = 0.; /* memory allocated for storing the numerical values of L and U, that will be used in the numeric
                                           factorization (positive number) */

                    int_t *mod_bit;
                    int_t *frecv, *brecv, *lloc;
                    double **Linv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
                    double **Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
                    double *SeedSTD_BC,*SeedSTD_RD;
                    int_t idx_indx,idx_lusup;
                    int_t nbrow;
                    int_t  ik, il, lk, rel, knsupc, idx_r;
                    int_t  lptr1_tmp, idx_i, idx_v,m, uu;
                    int_t nub;
                    int tag;
                    int *ptrToRecv, *nnzToRecv, *ptrToSend, *nnzToSend;//from distributed module for gSoFa 
                    int_t *recvBuf; //from distributed module for gSoFa 
                    int_t* SendBufSendR; //for distributed module for gSoFa 
                    int_t* recvBufSendR; //for distributed module for gSoFa 
                    int_t* RequestToRecv; //for distributed module for gSoFa 
                    int_t* RecvBufRequestToRecv; //for distributed module for gSoFa 

                    int *ptrToRecvReq, *nnzToRecvReq, *ptrToSendReq, *nnzToSendReq;//from distributed module for gSoFa 
                    int_t *recvBufReq; //from distributed module for gSoFa 
                    int_t NLocalBLCol; //For gSofa Communication structures

#if ( PRNTlevel>=1 )
                    int_t nLblocks = 0, nUblocks = 0;
#endif
#if ( PROFlevel>=1 )
                    double t, t_u, t_l;
                    int_t u_blks;
#endif

                    /* Initialization. */
                    iam = grid->iam;
                    myrow = MYROW( iam, grid );
                    mycol = MYCOL( iam, grid );
                    nprocs = grid->npcol * grid->nprow;//~gSoFa
                    for (i = 0; i < NBUFFERS; ++i) mybufmax[i] = 0;
                    nsupers  = supno[n-1] + 1;
                    Astore   = (NRformat_loc *) A->Store;

                    int_t* xusub_begin = Glu_freeable->xusub_begin;
                    int_t* xusub_end = Glu_freeable->xusub_end;
                    int_t* xlsub_begin = Glu_freeable->xlsub_begin;
                    int_t* xlsub_end = Glu_freeable->xlsub_end;
                    //#if ( PRNTlevel>=1 )
                    iword = sizeof(int_t);
                    dword = sizeof(double);
                    //#endif

                    // printf("Inside distribution module\n");
                    // fflush(stdout);

#if ( DEBUGlevel>=1 )
                    CHECK_MALLOC(iam, "Enter pddistribute()");
#endif
#if ( PROFlevel>=1 )
                    t = SuperLU_timer_();
#endif

                    dReDistribute_A_gsofa(A, ScalePermstruct, Glu_freeable, xsup, supno,
                            grid, &xa, &asub, &a);

                    // printf("IAM: %d After calling dReDistribute_A()  module!\n", iam);
                    // fflush(stdout);

#if ( PROFlevel>=1 )
                    t = SuperLU_timer_() - t;
                    if ( !iam ) printf("--------\n"
                            ".. Phase 1 - ReDistribute_A time: %.2f\t\n", t);
#endif

                    if ( options->Fact == SamePattern_SameRowPerm ) {

#if ( PROFlevel>=1 )
                        t_l = t_u = 0; u_blks = 0;
#endif
  
                        // printf("IAM: %d propagate the new values of A into the existing L and U data structures. \n",iam);
                        // fflush(stdout);
                        /* We can propagate the new values of A into the existing
                           L and U data structures.            */
                        ilsum = Llu->ilsum;
                        ldaspa = Llu->ldalsum;
                        if ( !(dense = doubleCalloc_dist(ldaspa * sp_ienv_dist(3,options))) )
                            ABORT("Calloc fails for SPA dense[].");
                        nrbu = CEILING( nsupers, grid->nprow ); /* No. of local block rows */
                        if ( !(Urb_length = intCalloc_dist(nrbu)) )
                            ABORT("Calloc fails for Urb_length[].");
                        if ( !(Urb_indptr = intMalloc_dist(nrbu)) )
                            ABORT("Malloc fails for Urb_indptr[].");
                        Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
                        Lindval_loc_bc_ptr = Llu->Lindval_loc_bc_ptr;
                        Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
                        Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
                        Unzval_br_ptr = Llu->Unzval_br_ptr;
                        Unnz = Llu->Unnz;

                        mem_use += 2.0*nrbu*iword + ldaspa*sp_ienv_dist(3,options)*dword;

#if ( PROFlevel>=1 )
                        t = SuperLU_timer_();
#endif

                        // printf("IAM: %d Before Initializing uval to zero \n",iam, jb);
                        // fflush(stdout);
                        /* Initialize Uval to zero. */
                        for (lb = 0; lb < nrbu; ++lb) {
                            Urb_indptr[lb] = BR_HEADER; /* Skip header in U index[]. */
                            index = Ufstnz_br_ptr[lb];
                            if ( index ) {
                                uval = Unzval_br_ptr[lb];
                                len = index[1];
                                for (i = 0; i < len; ++i) uval[i] = zero;
                            } /* if index != NULL */
                        } /* for lb ... */

                        for (jb = 0; jb < nsupers; ++jb) { /* Loop through each block column */
                        // printf("IAM: %d Working on supernodal column %d\n",iam, jb);
                        // fflush(stdout);
                            pc = PCOL( jb, grid );
                            if ( mycol == pc ) { /* Block column jb in my process column */
                                fsupc = FstBlockC( jb );
                                nsupc = SuperSize( jb );

                                /* Scatter A into SPA (for L), or into U directly. */
                                for (j = fsupc, dense_col = dense; j < FstBlockC(jb+1); ++j) {
                                    for (i = xa[j]; i < xa[j+1]; ++i) {
                                        irow = asub[i];
                                        gb = BlockNum( irow );
                                        if ( myrow == PROW( gb, grid ) ) {
                                            lb = LBi( gb, grid );
                                            if ( gb < jb ) { /* in U */
                                                index = Ufstnz_br_ptr[lb];
                                                uval = Unzval_br_ptr[lb];
                                                while (  (k = index[Urb_indptr[lb]]) < jb ) {
                                                    /* Skip nonzero values in this block */
                                                    Urb_length[lb] += index[Urb_indptr[lb]+1];
                                                    /* Move pointer to the next block */
                                                    Urb_indptr[lb] += UB_DESCRIPTOR
                                                        + SuperSize( k );
                                                }
                                                /*assert(k == jb);*/
                                                /* start fstnz */
                                                istart = Urb_indptr[lb] + UB_DESCRIPTOR;
                                                len = Urb_length[lb];
                                                fsupc1 = FstBlockC( gb+1 );
                                                k = j - fsupc;
                                                /* Sum the lengths of the leading columns */
                                                for (jj = 0; jj < k; ++jj)
                                                    len += fsupc1 - index[istart++];
                                                /*assert(irow>=index[istart]);*/
                                                uval[len + irow - index[istart]] = a[i];
                                            } else { /* in L; put in SPA first */
                                                irow = ilsum[lb] + irow - FstBlockC( gb );
                                                dense_col[irow] = a[i];
                                            }
                                        }
                                    } /* for i ... */
                                    dense_col += ldaspa;
                                } /* for j ... */

#if ( PROFlevel>=1 )
                                t_u += SuperLU_timer_() - t;
                                t = SuperLU_timer_();
#endif

                                /* Gather the values of A from SPA into Lnzval[]. */
                                ljb = LBj( jb, grid ); /* Local block number */

                                index = Lrowind_bc_ptr[ljb];

                                if ( index ) {
                                    nrbl = index[0];   /* Number of row blocks. */
                                    len = index[1];    /* LDA of lusup[]. */
                                    lusup = Lnzval_bc_ptr[ljb];
                                    next_lind = BC_HEADER;
                                    next_lval = 0;
                                    for (jj = 0; jj < nrbl; ++jj) {
                                        gb = index[next_lind++];
                                        len1 = index[next_lind++]; /* Rows in the block. */
                                        lb = LBi( gb, grid );
                                        for (bnnz = 0; bnnz < len1; ++bnnz) {
                                            irow = index[next_lind++]; /* Global index. */
                                            irow = ilsum[lb] + irow - FstBlockC( gb );
                                            k = next_lval++;
                                            for (j = 0, dense_col = dense; j < nsupc; ++j) {
                                                lusup[k] = dense_col[irow];
                                                dense_col[irow] = zero;
                                                k += len;
                                                dense_col += ldaspa;
                                            }
                                        } /* for bnnz ... */
                                    } /* for jj ... */
                                } /* if index ... */
#if ( PROFlevel>=1 )
                                t_l += SuperLU_timer_() - t;
#endif
                            } /* if mycol == pc */
                        } /* for jb ... */

                        SUPERLU_FREE(dense);
                        SUPERLU_FREE(Urb_length);
                        SUPERLU_FREE(Urb_indptr);
#if ( PROFlevel>=1 )
                        if ( !iam ) printf(".. 2nd distribute time: L %.2f\tU %.2f\tu_blks %d\tnrbu %d\n",
                                t_l, t_u, u_blks, nrbu);
#endif

                    } else { /* fact is not SamePattern_SameRowPerm */
                        /* ------------------------------------------------------------
                           FIRST TIME CREATING THE L AND U DATA STRUCTURES.
                           ------------------------------------------------------------*/

#if ( PROFlevel>=1 )
                        t_l = t_u = 0; u_blks = 0;
#endif
                        /* We first need to set up the L and U data structures and then
                         * propagate the values of A into them.
                         */
                        lsub = Glu_freeable->lsub;    /* compressed L subscripts */
                        xlsub = Glu_freeable->xlsub;
                        usub = Glu_freeable->usub;    /* compressed U subscripts */
                        xusub = Glu_freeable->xusub;
        //   printf("Process: %d, at line: %d\n",iam, __LINE__);
        //                 fflush(stdout);
                        if ( !(ToRecv = (int *) SUPERLU_MALLOC(nsupers * sizeof(int))) )
                            ABORT("Malloc fails for ToRecv[].");
                        for (i = 0; i < nsupers; ++i) ToRecv[i] = 0;

                        k = CEILING( nsupers, grid->npcol );/* Number of local column blocks */
                        if ( !(ToSendR = (int **) SUPERLU_MALLOC(k*sizeof(int*))) )
                            ABORT("Malloc fails for ToSendR[].");

                        int max_process = SUPERLU_MAX( grid->nprow, grid->npcol);//2 of 1 x 2 configuration

                        NLocalBLCol = CEILING( nsupers, grid->npcol );/* Number of local column blocks */
                        if ( !(recvBufSendR = (int_t *) SUPERLU_MALLOC(NLocalBLCol*max_process*iword)) ) { //k is corresponding to the number of processes
                            fprintf (stderr, "Malloc fails for recvBuf[].");
                            return (memDist + memNLU + memTRS);
                        }

                        if ( !(SendBufSendR = (int_t *) SUPERLU_MALLOC(NLocalBLCol*max_process*iword)) ) { //k is corresponding to the number of processes
                            fprintf (stderr, "Malloc fails for recvBuf[].");
                            return (memDist + memNLU + memTRS);
                        }

                        // if ( !(RequestToRecv = (int_t *) SUPERLU_MALLOC(nsupers*nprocs*iword)) ) { //k is corresponding to the number of processes
                        // 		fprintf (stderr, "Malloc fails for RequestToRecv[].");
                        // 		return (memDist + memNLU + memTRS);
                        // 	}

                        // 		if ( !(RecvBufRequestToRecv = (int_t *) SUPERLU_MALLOC(nsupers*nprocs*iword)) ) { //k is corresponding to the number of processes
                        // 		fprintf (stderr, "Malloc fails for RecvBufRequestToRecv[].");
                        // 		return (memDist + memNLU + memTRS);
                        // 	}

                        if ( !(RequestToRecv = intCalloc_dist(nsupers*nprocs)) )
                            ABORT("Calloc fails for RequestToRecv[].");

                        if ( !(RecvBufRequestToRecv = intCalloc_dist(nsupers*nprocs)) )
                            ABORT("Calloc fails for RecvBufRequestToRecv[].");

                        if ( !(SendRReq = (int **) SUPERLU_MALLOC(k*sizeof(int*))) ) //gSoFa Structure to request ToSendR
                            ABORT("Malloc fails for SendRReq[].");

                        j = k * grid->npcol;
                        if ( !(index1 = SUPERLU_MALLOC(j * sizeof(int))) )
                            ABORT("Malloc fails for index[].");

                        mem_use += (float) k*sizeof(int_t*) + (j + nsupers)*iword;

                        for (i = 0; i < j; ++i) index1[i] = EMPTY;
                        for (i = 0,j = 0; i < k; ++i, j += grid->npcol) ToSendR[i] = &index1[j];
                        for (i = 0,j = 0; i < k; ++i, j += grid->npcol) SendRReq[i] = &index1[j];//gSoFa initialize SendRReq =-1
                        for (int l=0; l< NLocalBLCol*max_process; l++)

                        {
                            // for (i = 0,j = 0; i < k; ++i, j += grid->npcol) 
                            // {
                            recvBufSendR[l] = -1;//gSoFa initialize SendRReq =-1
                            SendBufSendR[l] = -1;//gSoFa initialize SendRReq =-1
                            // }
                        }
                        k = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
                        // printf("Process: %d, at line: %d\n",iam, __LINE__);
                        // fflush(stdout);
                        /* Pointers to the beginning of each block row of U. */
                        if ( !(Unzval_br_ptr =
                                    (double**)SUPERLU_MALLOC(k * sizeof(double*))) )
                            ABORT("Malloc fails for Unzval_br_ptr[].");
                        if ( !(Ufstnz_br_ptr = (int_t**)SUPERLU_MALLOC(k * sizeof(int_t*))) )
                            ABORT("Malloc fails for Ufstnz_br_ptr[].");

                        if ( !(ToSendD = SUPERLU_MALLOC(k * sizeof(int))) )
                            ABORT("Malloc fails for ToSendD[].");
                        for (i = 0; i < k; ++i) ToSendD[i] = NO;
                        if ( !(ilsum = intMalloc_dist(k+1)) )
                            ABORT("Malloc fails for ilsum[].");

                        /* Auxiliary arrays used to set up U block data structures.
                           They are freed on return. */
                        if ( !(rb_marker = intCalloc_dist(k)) )
                            ABORT("Calloc fails for rb_marker[].");
                        if ( !(Urb_length = intCalloc_dist(k)) )
                            ABORT("Calloc fails for Urb_length[].");
                        if ( !(Urb_indptr = intMalloc_dist(k)) )
                            ABORT("Malloc fails for Urb_indptr[].");
                        if ( !(Urb_fstnz = intCalloc_dist(k)) )
                            ABORT("Calloc fails for Urb_fstnz[].");
                        if ( !(Ucbs = intCalloc_dist(k)) )
                            ABORT("Calloc fails for Ucbs[].");

                        mem_use += 2.0*k*sizeof(int_t*) + (7*k+1)*iword;

                        /* Compute ldaspa and ilsum[]. */
                        ldaspa = 0;
                        ilsum[0] = 0;
                        for (gb = 0; gb < nsupers; ++gb) {
                            if ( myrow == PROW( gb, grid ) ) {
                                i = SuperSize( gb );
                                ldaspa += i;
                                lb = LBi( gb, grid );
                                ilsum[lb + 1] = ilsum[lb] + i;
                            }
                        }

#if ( PROFlevel>=1 )
                        t = SuperLU_timer_();
#endif
                        /* ------------------------------------------------------------
                           COUNT NUMBER OF ROW BLOCKS AND THE LENGTH OF EACH BLOCK IN U.
                           THIS ACCOUNTS FOR ONE-PASS PROCESSING OF G(U).
                           ------------------------------------------------------------*/
        //   printf("Process: %d, at line: %d\n",iam, __LINE__);
        //                 fflush(stdout);
                        /* Loop through each supernode column. */
                        for (jb = 0; jb < nsupers; ++jb) {
                            // if (iam == 0) 
                            // {
                            // 	printf("Process:%d reaches Milestone 0 for jb:%d\n",iam,jb);
                            // 	fflush(stdout);
                            // }
                            pc = PCOL( jb, grid ); //PCOL(bnum,grid) ( (bnum) % grid->npcol ). COLUMN processes that own block jb
                            // 		if (iam == 2) 
                            // {
                            // 	printf("Process:%d reaches Milestone 0.1 for jb:%d\n",iam,jb);
                            // 	fflush(stdout);
                            // }
                            // 	if (pc==1)
                            // {
                            // 	printf("Process:%d pc=1\n",iam);
                            // 	fflush(stdout);
                            // }
                            // 	if (pc==0)
                            // {
                            // 	printf("Process:%d pc=0\n",iam);
                            // 	fflush(stdout);
                            // }
                            fsupc = FstBlockC( jb );
                            nsupc = SuperSize( jb );
        //   printf("Process: %d, at line: %d\n",iam, __LINE__);
        //                 fflush(stdout);
                            /* Loop through each column in the block. */
                            for (j = fsupc; j < fsupc + nsupc; ++j) {
                                // if (iam == 0) 
                                // {
                                // 	printf("Process:%d reaches Milestone 0.2 for jb:%d\n",iam,jb);
                                // 	fflush(stdout);
                                // }

                                // if ((pc==1) && (iam==0))
                                // {
                                // 	printf("Process:%d pc:%d fsupc:%d nsupc:%d xusub_begin[%d]:%d xusub_end[%d]:%d\n",iam, pc, fsupc, nsupc,j,xusub_begin[j],j,xusub_end[j]);
                                // 	fflush(stdout);
                                // }
                                /* usub[*] contains only "first nonzero" in each segment. */
                                int gbi = BlockNum( j); //gSoFa
                                for (i = xusub_begin[j]; i < xusub_end[j]; ++i) { 
                                    // 												if (iam == 2) 
                                    // {
                                    // 	printf("Process:%d reaches Milestone 0.3 for jb:%d\n",iam,jb);
                                    // 	fflush(stdout);
                                    // }
                                    //note: usub[i] only has the non-zeros of the current process for gSoFa. Any other processes that need current block need to request the block separately.
                                    //This process then, set the ToSendR[local_jb][target_pc]
                                    // if (iam ==0 ) 
                                    // {
                                    // 	printf("Process:%d i: %d xusub_begin[%d]:%d xusub_end[%d]:%d\n",iam, i,j,xusub_begin[j],j,xusub_end[j]);
                                    // 	fflush(stdout);
                                    // }
                                    irow = usub[i]; /* First nonzero of the segment. */
                                    // if (iam ==2 ) 
                                    // {
                                    // 	printf("Process:%d irow: %d\n",iam, irow);
                                    // 	fflush(stdout);
                                    // }
                                    gb = BlockNum( irow ); //#define BlockNum(i)     ( supno[i] )
                                    // 	if (iam ==2 ) 
                                    // {
                                    // 	printf("Process:%d gb: %d\n",iam, gb);
                                    // 	fflush(stdout);
                                    // }
                                    // int owner_process_irow = PNUM( PROW(gbi,grid), PCOL(gb,grid), grid ); //gSoFa
                                    // if (iam == 0) 
                                    // {
                                    // 	printf("Process:%d reaches Milestone 0.325 for jb:%d\n",iam,jb);
                                    // 	fflush(stdout);
                                    // }
                                    // int owner_process_irow = PNUM(PCOL(gb,grid),  PROW(gbi,grid),  grid ); //gSoFa
                                    // if (iam == 2) 
                                    // {
                                    // 	printf("Process:%d reaches Milestone 0.35 for jb:%d\n",iam,jb);
                                    // 	fflush(stdout);
                                    // }
                                    // int gbj = BlockNum_gSoFa(irow,Glu_persist->supno);
#ifdef supernodal_Uskipp
                                    int_t i_supernode = Glu_persist->supno[irow];        
                                    // Note i_supernode is equal to jb
                                    if (jb == i_supernode)
                                    {
                                        if (SuperSize( i_supernode ) !=1) //#define SuperSize(bnum) ( xsup[bnum+1]-xsup[bnum] )
                                        {
                                            //Skipping the columns that are already included into the supernodes in the L structure                       

                                            if (gb == jb) 
                                            {
                                                // printf("Process:%d Skipping logic i_supernode:%d jb:%d gb:%d!\n",iam,i_supernode, jb, gb);
                                                // 	fflush(stdout);
                                                // if (LBj( gb, grid ) == 813)
                                                // {
                                                // 	printf("Process:%d Continue for the next row for ljb:%d!\n",iam,ljb);
                                                // 	fflush(stdout);
                                                // }
                                                continue;//Skip the indices which belongs to a non-singular supernode                              
                                            }
                                        }
                                    }
#endif
                                    kcol = PCOL( gb, grid ); // it is assigned as pc in parallel module ( (bnum) % grid->npcol )
                                    //column process that owns the block gb. Send request to kcol proceses.
                                    //kcol needs to send to current process
                                    ljb = LBj( gb, grid ); //( (bnum)/grid->npcol )/* Global to local block columnwise*/
                                    //Note: Here ljb is local block in owner process column

                                    // if (pc==1)
                                    // {
                                    // 	printf("Process:%d pc=1\n",iam);
                                    // 	fflush(stdout);
                                    // }
                                    // 	if (pc==0)
                                    // {
                                    // 	printf("Process:%d pc=0\n",iam);
                                    // 	fflush(stdout);
                                    // }

                                    // printf("At process: %d  mycol:%d kcol:%d pc:%d \n",iam,mycol, kcol, pc);
                                    // fflush(stdout);

                                    // if ((iam==1) && (gb==9))
                                    // {
                                    // 	printf("Process:%d detected gb==9 while processing U part at block column:%d\n",iam,jb);
                                    // 	fflush(stdout);
                                    // }
                                    // if (owner_process_irow != )
                                    // {

                                    // }
                                    // if (iam == 0) 
                                    // {
                                    // 	printf("Process:%d reaches Milestone 0.4 for jb:%d\n",iam,jb);
                                    // 	fflush(stdout);
                                    // }
                                    // if ( mycol == kcol && mycol != pc ) //here PC should be pc (process column) of the target process//pc ==> Column processes that own the block column
#ifndef use_original_sendRLogic
                                    if ( (mycol == kcol) && (mycol != pc) ) 
                                    {
                                        ToSendR[ljb][pc] = YES;
                                        // printf("At process: %d ToSendR[%d][%d] set to yes mycol:%d kcol:%d pc:%d \n",iam,ljb,pc,mycol, kcol, pc);
                                        // fflush(stdout);
                                    }
                                    // else if ((mycol != kcol ) && (kcol != pc))
                                    else if ((mycol != kcol ) && (mycol == pc))
#else
                                        if ((mycol != kcol ) && (kcol != pc))
#endif
                                            // if ((mycol != kcol ) && (kcol != pc))
                                            // if (mycol != kcol ) //here PC should be pc (process column) of the target process//pc ==> Column processes that own the block column
                                        {
                                            //Note: kcol is owner column


                                            //if my column is not the process column that owns the block then send request 
                                            //Note: mycol == kcol means this process owns the block in current block column and is authorized to send the non-zero structure to target
                                            //column pc given that the target pc is not same as current process column (mycol). Otherwise, the process would send non-zero structure to itself
                                            // ToSendR[ljb][pc] = YES;

                                            // int max_process = SUPERLU_MAX( grid->nprow, grid->npcol);
                                            // int offset =  kcol * nsupers;
                                            // recvBuf[ljb +  kcol * nsupers] = YES;

                                            // if (iam == 0) 
                                            // {
                                            // 	printf("Process:%d reaches Milestone 0.5 for jb:%d\n",iam,jb);
                                            // 	fflush(stdout);
                                            // }

                                            // printf("Process: %d Before assinging SendBufSendR[] at ljb:%d\n",iam,ljb);
                                            // fflush(stdout);
                                            // printf("Process:%d c%d\n",iam,mycol);
                                            // fflush(stdout);
                                            // SendBufSendR[ljb +  kcol * NLocalBLCol] = YES;
                                            SendBufSendR[ljb +  kcol * NLocalBLCol] = 1;

                                            // if (ljb==813)
                                            // {
                                            // 	if (kcol==2)
                                            // 	{
                                            // 		printf("Source Detected possible problametic ToSendR[%d][%d] set in target process column:%d source Process:%d \n",ljb,mycol,kcol,iam);
                                            // 		fflush(stdout);
                                            // 	}
                                            // }

                                            // printf("Process: %d After assinging SendBufSendR[] at ljb:%d\n",iam,ljb);
                                            // fflush(stdout);
                                            // SendBufSendR[ljb * max_process +  kcol] = YES;
                                            // SendBufSendR[gb +  kcol * nsupers] = YES;


                                            // SendRReq[kcol][ljb] = YES;//Request ljb block from kcol process that owns it.
                                            // printf("At process: %d ToSendR[%d][%d] set to yes\n",iam,ljb,pc);
                                            // if (iam==0)
                                            // {
                                            // 	if (jb == 4)
                                            // 	{
                                            // 		printf("At process: %d blockColumn:%d irow:%d globalblock:%d ljb:%d OwnerProcessColumn (kcol):%d SendBufSendR[%d] set to yes\n",iam, jb, irow, gb, ljb, kcol, ljb +  kcol * NLocalBLCol);
                                            // 		// printf("At process: %d blockColumn:%d irow:%d globalblock:%d ljb:%d OwnerProcessColumn (kcol):%d SendBufSendR[%d] set to yes\n",iam, jb, irow, gb, ljb, kcol, gb +  kcol * nsupers);
                                            // 		fflush(stdout);
                                            // 	}
                                            // }
                                        }

                                    pr = PROW( gb, grid ); // ( (bnum) % grid->nprow )
                                    lb = LBi( gb, grid );  //( (bnum)/grid->nprow )/* Global to local block rowwise */
                                    // if ((jb == 14) && (iam==2))
                                    // 		{
                                    // 			if (irow==1)
                                    // 			{
                                    // 				printf("(14, 1) detected by P2!\n");
                                    // 				fflush(stdout);
                                    // 			}
                                    // 		}

                                    if ( mycol == pc ) {
                                        //For jb == 11, If all process had all structures. P1 and P3 will pass this logic for irow  =9 at jb =11
                                        // But P1 won't pass this structure. Because it doesnot have (11, 9) CSC. And hence, ToRecv[gb] = ToRecv[9] = remains 0 
                                        if  ( myrow == pr ) {

                                            //All processes that satisfy this condition should follow all the steps as below.


                                            // For Procecss 1 , myrow = 0, for P3 myrow = 1
                                            //For irow = 9, pr =1  ==> , P3 has non-zero at (11, 9) CSC and will only enter here. 
                                            ToSendD[lb] = YES;
                                            /* Count nonzeros in entire block row. */

                                            Urb_length[lb] += FstBlockC( gb+1 ) - irow;
                                            if (rb_marker[lb] <= jb) {/* First see the block */
                                                rb_marker[lb] = jb + 1;
                                                Urb_fstnz[lb] += nsupc;
                                                ++Ucbs[lb]; /* Number of column blocks
                                                               in block row lb. */
#if ( PRNTlevel>=1 )
                                                ++nUblocks;
#endif
                                            }

#ifdef UpdateToRecv
                                            // if ((jb == 14) && (iam==2))
                                            // {
                                            // 	if (irow==1)
                                            // 	{
                                            // 		printf("(14, 1) detected by P2!\n");
                                            // 		printf("owner_process_irow: %d \n",owner_process_irow);
                                            // 		fflush(stdout);
                                            // 	}
                                            // }

                                            // if (owner_process_irow == iam)
                                            // {
                                            //if I  own the block of the irow.
                                            // I've to also tell  other processes in my column  to set their ToRecv[gb] =1
                                            for (int process_id=0; process_id < nprocs; process_id++)
                                            {
                                                if (iam != process_id)
                                                {
                                                    // int process_col = MYCOL( process_id, grid );
                                                    int process_id_myrow = MYROW( process_id, grid );
                                                    int process_id_mycol = MYCOL( process_id, grid );
                                                    if (pc == process_id_mycol)
                                                    {
                                                        if (pr == process_id_myrow)
                                                        {

                                                            RequestToRecv[process_id*nsupers + gb ] = 1;
                                                        }
                                                        else
                                                        {
                                                            RequestToRecv[process_id*nsupers + gb ] = 2;
                                                        }

                                                        // if (gb == testing_block)
                                                        // {
                                                        //     printf("Process: %d sets its RequestToRecv[%d]: %d in original logic Target process:%d!\n",iam,process_id*nsupers + gb,RequestToRecv[process_id*nsupers + gb],process_id);
                                                        //     fflush(stdout);
                                                        // }
                                                    }
                                                }
                                            }
                                            // }
#endif

                                            ToRecv[gb] = 1;

                                            // if (gb == testing_block)
                                            // {
                                            // 	printf("Process: %d sets its ToRecv[%d]: %d in original logic!\n",iam,gb,ToRecv[gb]);
                                            // 	fflush(stdout);
                                            // }
                                            // ToRecv[gb] = 2;
                                            // printf("")
                                        } else 
                                        {
#ifdef UpdateToRecv
                                            // if ((jb == 14) && (iam==2))
                                            // {
                                            // 	if (irow==1)
                                            // 	{
                                            // 		printf("(14, 1) detected by P2!\n");
                                            // 		fflush(stdout);
                                            // 	}
                                            // }
                                            // if (owner_process_irow == iam)
                                            // {
                                            //if I own the block of the irow.
                                            // I've to also tell  other processes in my column  to set their ToRecv[gb] =2 
                                            for (int process_id=0; process_id < nprocs; process_id++)
                                            {
                                                if (iam != process_id)
                                                {
                                                    // int process_col = MYCOL( process_id, grid );
                                                    int process_id_myrow = MYROW( process_id, grid );
                                                    int process_id_mycol = MYCOL( process_id, grid );
                                                    if (pc == process_id_mycol)
                                                    {
                                                        if (pr == process_id_myrow)
                                                        {

                                                            RequestToRecv[process_id*nsupers + gb ] = 1;
                                                        }
                                                        else
                                                        {
                                                            RequestToRecv[process_id*nsupers + gb ] = 2;
                                                        }
                                                    }
                                                    // if (gb == testing_block)
                                                    // {
                                                    //     printf("Process: %d sets its RequestToRecv[%d]: %d in original logic Target process:%d!\n",iam,process_id*nsupers + gb,RequestToRecv[process_id*nsupers + gb],process_id);
                                                    //     fflush(stdout);
                                                    // }
                                                }
                                            }

                                            // }
#endif
                                            ToRecv[gb] = 2; /* Do I need 0, 1, 2 ? */
                                            // if (gb == testing_block)
                                            // {
                                            //     printf("Process: %d sets its ToRecv[%d]: %d in original logic!\n",iam,gb,ToRecv[gb]);
                                            //     fflush(stdout);
                                            // }
                                        }
                                    }
                                } /* for i ... */

                            } /* for j ... */
                            // printf("Process:%d finished processing jb:%d for L block structures!\n",iam, jb);
                            // fflush(stdout);
                        } /* for jb ... */
        //   printf("Process: %d, at line: %d\n",iam, __LINE__);
        //                 fflush(stdout);
                        /* Set up the initial pointers for each block row in U. */
                        nrbu = CEILING( nsupers, grid->nprow );/* Number of local block rows */
                        for (lb = 0; lb < nrbu; ++lb) {
                            len = Urb_length[lb];
                            rb_marker[lb] = 0; /* Reset block marker. */
                            if ( len ) {
                                /* Add room for descriptors */
                                len1 = Urb_fstnz[lb] + BR_HEADER + Ucbs[lb] * UB_DESCRIPTOR;
                                if ( !(index = intMalloc_dist(len1+1)) )
                                    ABORT("Malloc fails for Uindex[].");
                                Ufstnz_br_ptr[lb] = index;
                                if ( !(Unzval_br_ptr[lb] = doubleMalloc_dist(len)) )
                                    ABORT("Malloc fails for Unzval_br_ptr[*][].");
                                mybufmax[2] = SUPERLU_MAX( mybufmax[2], len1 );
                                mybufmax[3] = SUPERLU_MAX( mybufmax[3], len );
                                index[0] = Ucbs[lb]; /* Number of column blocks */
                                index[1] = len;      /* Total length of nzval[] */
                                index[2] = len1;     /* Total length of index[] */
                                index[len1] = -1;    /* End marker */
                            } else {
                                Ufstnz_br_ptr[lb] = NULL;
                                Unzval_br_ptr[lb] = NULL;
                            }
                            Urb_length[lb] = 0; /* Reset block length. */
                            Urb_indptr[lb] = BR_HEADER; /* Skip header in U index[]. */
                            Urb_fstnz[lb] = BR_HEADER;
                        } /* for lb ... */

                        SUPERLU_FREE(Ucbs);
        //   printf("Process: %d, at line: %d\n",iam, __LINE__);
        //                 fflush(stdout);
#if ( PROFlevel>=1 )
                        t = SuperLU_timer_() - t;
                        if ( !iam) printf(".. Phase 2 - setup U strut time: %.2f\t\n", t);
#endif

                        mem_use -= 2.0*k * iword;

                        /* Auxiliary arrays used to set up L block data structures.
                           They are freed on return.
                           k is the number of local row blocks.   */
                        if ( !(Lrb_length = intCalloc_dist(k)) )
                            ABORT("Calloc fails for Lrb_length[].");
                        if ( !(Lrb_number = intMalloc_dist(k)) )
                            ABORT("Malloc fails for Lrb_number[].");
                        if ( !(Lrb_indptr = intMalloc_dist(k)) )
                            ABORT("Malloc fails for Lrb_indptr[].");
                        if ( !(Lrb_valptr = intMalloc_dist(k)) )
                            ABORT("Malloc fails for Lrb_valptr[].");
                        if ( !(dense = doubleCalloc_dist(ldaspa * sp_ienv_dist(3,options))) )
                            ABORT("Calloc fails for SPA dense[].");

                        /* These counts will be used for triangular solves. */
                        if ( !(fmod = intCalloc_dist(k)) )
                            ABORT("Calloc fails for fmod[].");
                        if ( !(bmod = intCalloc_dist(k)) )
                            ABORT("Calloc fails for bmod[].");

                        /* ------------------------------------------------ */
                        mem_use += 6.0*k*iword + ldaspa*sp_ienv_dist(3,options)*dword;

                        k = CEILING( nsupers, grid->npcol );/* Number of local block columns */

                        /* Pointers to the beginning of each block column of L. */
                        if ( !(Lnzval_bc_ptr =
                                    (double**)SUPERLU_MALLOC(k * sizeof(double*))) )
                            ABORT("Malloc fails for Lnzval_bc_ptr[].");
                        if ( !(Lrowind_bc_ptr = (int_t**)SUPERLU_MALLOC(k * sizeof(int_t*))) )
                            ABORT("Malloc fails for Lrowind_bc_ptr[].");
                        Lrowind_bc_ptr[k-1] = NULL;

                        if ( !(Lindval_loc_bc_ptr =
                                    (int_t**)SUPERLU_MALLOC(k * sizeof(int_t*))) )
                            ABORT("Malloc fails for Lindval_loc_bc_ptr[].");
                        Lindval_loc_bc_ptr[k-1] = NULL;

                        if ( !(Linv_bc_ptr =
                                    (double**)SUPERLU_MALLOC(k * sizeof(double*))) ) {
                            fprintf(stderr, "Malloc fails for Linv_bc_ptr[].");
                        }
                        if ( !(Uinv_bc_ptr =
                                    (double**)SUPERLU_MALLOC(k * sizeof(double*))) ) {
                            fprintf(stderr, "Malloc fails for Uinv_bc_ptr[].");
                        }
                        Linv_bc_ptr[k-1] = NULL;
                        Uinv_bc_ptr[k-1] = NULL;

                        if ( !(Unnz =
                                    (int_t*)SUPERLU_MALLOC(k * sizeof(int_t))) )
                            ABORT("Malloc fails for Unnz[].");


                        /* These lists of processes will be used for triangular solves. */
                        if ( !(fsendx_plist = (int_t **) SUPERLU_MALLOC(k*sizeof(int_t*))) )
                            ABORT("Malloc fails for fsendx_plist[].");
                        len = k * grid->nprow;
                        if ( !(index = intMalloc_dist(len)) )
                            ABORT("Malloc fails for fsendx_plist[0]");

                        //gSoFa
                        if ( !(Request_fsendx_plist = intCalloc_dist(len)) )
                            ABORT("Malloc fails for Request_fsendx_plist[]");

                        if ( !(Recv_Request_fsendx_plist = intCalloc_dist(len)) )
                            ABORT("Malloc fails for Recv_Request_fsendx_plist[]");

                        if ( !(Request_bsendx_plist = intCalloc_dist(len)) )
                            ABORT("Malloc fails for Request_bsendx_plist[]");

                        if ( !(Recv_Request_bsendx_plist = intCalloc_dist(len)) )
                            ABORT("Malloc fails for Recv_Request_bsendx_plist[]");


                        for (i = 0; i < len; ++i) 
                        {
                            Request_fsendx_plist[i] = EMPTY;
                            Recv_Request_fsendx_plist[i] = EMPTY;
                            Request_bsendx_plist[i] = EMPTY;
                            Recv_Request_bsendx_plist[i] = EMPTY;
                        }
                        // for (i = 0; i < len; ++i) index[i] = EMPTY;
                        //~gSoFa
                        for (i = 0; i < len; ++i) index[i] = EMPTY;
                        for (i = 0, j = 0; i < k; ++i, j += grid->nprow)
                            fsendx_plist[i] = &index[j];
                        if ( !(bsendx_plist = (int_t **) SUPERLU_MALLOC(k*sizeof(int_t*))) )
                            ABORT("Malloc fails for bsendx_plist[].");
                        if ( !(index = intMalloc_dist(len)) )
                            ABORT("Malloc fails for bsendx_plist[0]");
                        for (i = 0; i < len; ++i) index[i] = EMPTY;
                        for (i = 0, j = 0; i < k; ++i, j += grid->nprow)
                            bsendx_plist[i] = &index[j];
                        /* -------------------------------------------------------------- */
                        mem_use += 4.0*k*sizeof(int_t*) + 2.0*len*iword;
                        memTRS += k*sizeof(int_t*) + 2.0*k*sizeof(double*) + k*iword;  //acount for Lindval_loc_bc_ptr, Unnz, Linv_bc_ptr,Uinv_bc_ptr

                        /*------------------------------------------------------------
                          PROPAGATE ROW SUBSCRIPTS AND VALUES OF A INTO L AND U BLOCKS.
                          THIS ACCOUNTS FOR ONE-PASS PROCESSING OF A, L AND U.
                          ------------------------------------------------------------*/
        //   printf("Process: %d, at line: %d\n",iam, __LINE__);
        //                 fflush(stdout);

                        for (jb = 0; jb < nsupers; ++jb) { /* for each block column ... */
                            pc = PCOL( jb, grid );
                            // if (iam == 0) 
                            // {
                            // 	printf("Process:%d reaches Milestone 1 for jb: %d\n",iam, jb);
                            // 	fflush(stdout);
                            // }
                            if ( mycol == pc ) { /* Block column jb in my process column */
                                fsupc = FstBlockC( jb );
                                nsupc = SuperSize( jb );
                                ljb = LBj( jb, grid ); /* Local block number */  //( (bnum)/grid->npcol )/* Global to local block columnwise*/


                                /* Scatter A into SPA. */
                                for (j = fsupc, dense_col = dense; j < FstBlockC(jb+1); ++j) {
                                    for (i = xa[j]; i < xa[j+1]; ++i) {
                                        irow = asub[i];
                                        gb = BlockNum( irow );
                                        if ( myrow == PROW( gb, grid ) ) {
                                            lb = LBi( gb, grid );
                                            irow = ilsum[lb] + irow - FstBlockC( gb );
                                            dense_col[irow] = a[i];
                                        }
                                    }
                                    dense_col += ldaspa;
                                } /* for j ... */

                                jbrow = PROW( jb, grid );
//  printf("Process: %d, jb:%d at line: %d\n",iam, jb, __LINE__);
//                         fflush(stdout);
                                /*------------------------------------------------
                                 * SET UP U BLOCKS.
                                 *------------------------------------------------*/
                                // if (iam == 0) 
                                // {
                                // printf("Process:%d reaches Milestone 2 for jb: %d\n",iam, jb);
                                // fflush(stdout);
                                // }
#if ( PROFlevel>=1 )
                                t = SuperLU_timer_();
#endif
                                kseen = 0;
                                dense_col = dense;
                                /* Loop through each column in the block column. */
                                for (j = fsupc; j < FstBlockC( jb+1 ); ++j) {
                                    istart = xusub_begin[j];
                                    /* NOTE: Only the first nonzero index of the segment
                                       is stored in usub[]. */
                                    for (i = istart; i < xusub_end[j]; ++i) {
                                        irow = usub[i]; /* First nonzero in the segment. */
#ifdef supernodal_Uskipp
                                        int_t i_supernode = Glu_persist->supno[irow];

                                        gb = BlockNum( irow );

                                        if (jb == i_supernode)
                                        {
                                            if (SuperSize( i_supernode ) !=1)
                                            {
                                                //Skipping the columns that are already included into the supernodes in the L structure
                                                // if (SuperSize( jb ) !=1)  continue;
                                                if (gb == jb) 
                                                {
                                                    continue;
                                                }            

                                            }

                                        }

#endif
                                        gb = BlockNum( irow );
                                        pr = PROW( gb, grid );
                                        //gSoFa
                                        int pc_gSoFa = PCOL( jb, grid );

                                        p = PNUM(pr, pc_gSoFa, grid);
                                        //~gSoFa

                                        if ( pr != jbrow &&
                                                myrow == jbrow &&  /* diag. proc. owning jb */
                                                bsendx_plist[ljb][pr] == EMPTY ) {
                                            bsendx_plist[ljb][pr] = YES;
                                            ++nbsendx;
                                        }
                                        else if ((p == iam) && (myrow != jbrow)) // If I own the block in the L panel, but the diagonal block is owned by another processe row.
                                            // else if ((myrow != jbrow) && (pr != jbrow))
                                        {
                                            // Request_bsendx_plist[ljb + pr * NLocalBLCol] = YES;
                                            // Request_bsendx_plist[ljb * grid->nprow + pr] = YES;
                                            // 				printf("Process:%d reaches Milestone 2.1 for jb: %d\n",iam, jb);
                                            // fflush(stdout);
                                            Request_bsendx_plist[ljb + jbrow * NLocalBLCol] = YES;
                                            // 				printf("Process:%d reaches Milestone 2.2 (after Request_bsendx_plist) for jb: %d\n",iam, jb);
                                            // fflush(stdout);
                                        }
                                        if ( myrow == pr ) {
                                            lb = LBi( gb, grid ); /* Local block number */
                                            index = Ufstnz_br_ptr[lb];
                                            uval = Unzval_br_ptr[lb];
                                            fsupc1 = FstBlockC( gb+1 );
                                            if (rb_marker[lb] <= jb) { /* First time see
                                                                          the block       */
                                                rb_marker[lb] = jb + 1;
                                                Urb_indptr[lb] = Urb_fstnz[lb];;
                                                index[Urb_indptr[lb]] = jb; /* Descriptor */
                                                Urb_indptr[lb] += UB_DESCRIPTOR;
                                                /* Record the first location in index[] of the
                                                   next block */
                                                Urb_fstnz[lb] = Urb_indptr[lb] + nsupc;
                                                len = Urb_indptr[lb];/* Start fstnz in index */
                                                index[len-1] = 0;
                                                for (k = 0; k < nsupc; ++k)
                                                    index[len+k] = fsupc1;
                                                if ( gb != jb )/* Exclude diagonal block. */
                                                    ++bmod[lb];/* Mod. count for back solve */
                                                if ( kseen == 0 && myrow != jbrow ) {
                                                    ++nbrecvx;
                                                    kseen = 1;
                                                }
                                            } else { /* Already saw the block */
                                                len = Urb_indptr[lb];/* Start fstnz in index */
                                            }
                                            jj = j - fsupc;
                                            index[len+jj] = irow;
                                            /* Load the numerical values */
                                            k = fsupc1 - irow; /* No. of nonzeros in segment */
                                            index[len-1] += k; /* Increment block length in
                                                                  Descriptor */
                                            irow = ilsum[lb] + irow - FstBlockC( gb );
                                            for (ii = 0; ii < k; ++ii) {
                                                uval[Urb_length[lb]++] = dense_col[irow + ii];
                                                dense_col[irow + ii] = zero;
                                            }
                                        } /* if myrow == pr ... */
                                    } /* for i ... */
                                    dense_col += ldaspa;
                                } /* for j ... */

#if ( PROFlevel>=1 )
                                t_u += SuperLU_timer_() - t;
                                t = SuperLU_timer_();
#endif
//  printf("Process: %d, jb:%d at line: %d\n",iam, jb, __LINE__);
//                         fflush(stdout);
                                /*------------------------------------------------
                                 * SET UP L BLOCKS.
                                 *------------------------------------------------*/

                                // if (iam == 0) 
                                // {
                                // 	printf("Process:%d reaches Milestone 3 for jb: %d\n",iam, jb);
                                // 	fflush(stdout);
                                // }

                                //gSoFa same as in parallel symbolic factorization
#ifdef enable_partial_lsub_sorting
                                /* sort the indices of the diagonal block at the beginning of xlsub */

                                if (myrow == jbrow) 
                                {
                                    qsort(&lsub[xlsub_begin[fsupc]], xlsub_end[fsupc] - xlsub_begin[fsupc], sizeof(int), compare_dist);
                                }
#endif
                                /* Count number of blocks and length of each block. */
                                nrbl = 0;
                                len = 0; /* Number of row subscripts I own. */
                                kseen = 0;
                                istart = xlsub_begin[fsupc];
                                for (i = istart; i < xlsub_end[fsupc]; ++i) {
                                    irow = lsub[i];
                                    gb = BlockNum( irow ); /* Global block number */
                                    pr = PROW( gb, grid ); /* Process row owning this block */
                                    // if ((irow == 11) && (iam==0))
                                    // {
                                    // 	if (jb==2)
                                    // 	{
                                    // 		printf("Process: %d detected irow: %d at jb=0\n",iam,irow);
                                    // 		fflush(stdout);
                                    // 	}
                                    // }
                                    int pc_gSoFa = PCOL( jb, grid );

                                    p = PNUM(pr, pc_gSoFa, grid);

                                    if ( pr != jbrow &&	 // anil ==> not diagonal non-zero
                                            myrow == jbrow &&  /* diag. proc. owning jb */
                                            fsendx_plist[ljb][pr] == EMPTY /* first time */ ) {
                                        // if ((iam == 0)) 
                                        // {
                                        // 	printf("Process:%d reaches Milestone 4(original condition) for jb: %d\n",iam, jb);
                                        // 	fflush(stdout);
                                        // }

                                        // if  (iam==0)
                                        // {

                                        // 	printf("Local Before: Process: %d irow:%d fsendx_plist[%d][%d]: %d\n",iam,irow, ljb,pr, fsendx_plist[ljb][pr]);
                                        // 	fflush(stdout);

                                        // }

                                        // if (pr != myrow)
                                        // // if (pr != jbrow)
                                        // {


                                        // 	Request_fsendx_plist[ljb + pr * NLocalBLCol] = YES;


                                        // }
                                        // else
                                        // {											
                                        fsendx_plist[ljb][pr] = YES;
                                        ++nfsendx;


                                        // if  (iam==0)
                                        // {

                                        // 	printf("Local After: Process: %d irow:%d fsendx_plist[%d][%d]: %d\n",iam,irow, ljb,pr, fsendx_plist[ljb][pr]);
                                        // 	fflush(stdout);

                                        // }
                                        // }
                                    }
                                    else if ((p == iam) && (myrow != jbrow)) // If I own the block in the L panel, but the diagonal block is owned by another processe row.
                                        // else if ((myrow == jbrow) && (pr != jbrow) && (pr == myrow))
                                        // else if ((myrow != jbrow) && (pr != jbrow))
                                    {
                                        // Request_fsendx_plist[jbrow + pr * NLocalBLCol] = YES;
                                        // Request_fsendx_plist[ljb * grid->nprow + pr] = YES;
                                        // Request_fsendx_plist[ljb * grid->nprow + pr] = YES;

                                        Request_fsendx_plist[ljb + jbrow * NLocalBLCol] = YES;
                                        // Request_fsendx_plist[ljb + pr * NLocalBLCol] = YES;
                                        // if ((iam == 5) && (jb==3))
                                        // {
                                        // 	printf("Process:%d reaches Milestone 4(new condition) for jb: %d\n",iam, jb);
                                        // 	fflush(stdout);
                                        // }
                                        //here ljb should be ljb of target process i.e. of jbrow.
                                        // Request_fsendx_plist[ljb + jbrow * NLocalBLCol] = YES;
                                        //ljb = ( (jb)/grid->npcol )/* Global to local block columnwise*/
                                    }

                                    if ( myrow == pr ) {
                                        lb = LBi( gb, grid );  /* Local block number */
                                        if (rb_marker[lb] <= jb) { /* First see this block */
                                            rb_marker[lb] = jb + 1;
                                            Lrb_length[lb] = 1;
                                            Lrb_number[nrbl++] = gb;
                                            if ( gb != jb ) /* Exclude diagonal block. */
                                                ++fmod[lb]; /* Mod. count for forward solve */
                                            if ( kseen == 0 && myrow != jbrow ) {
                                                ++nfrecvx;
                                                kseen = 1;
                                            }
#if ( PRNTlevel>=1 )
                                            ++nLblocks;
#endif
                                        } else {
                                            ++Lrb_length[lb];
                                        }
                                        ++len;
                                    }
                                } /* for i ... */

                                if ( nrbl ) { /* Do not ensure the blocks are sorted! */
                                    /* Set up the initial pointers for each block in
                                       index[] and nzval[]. */
                                    /* Add room for descriptors */
                                    len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
                                    if ( !(index = intMalloc_dist(len1)) )
                                        ABORT("Malloc fails for index[]");
                                    if (!(lusup = (double*)SUPERLU_MALLOC(len*nsupc * sizeof(double))))
                                        ABORT("Malloc fails for lusup[]");
                                    if ( !(Lindval_loc_bc_ptr[ljb] = intCalloc_dist(nrbl*3)) )
                                        ABORT("Malloc fails for Lindval_loc_bc_ptr[ljb][]");
                                    if (!(Linv_bc_ptr[ljb] = (double*)SUPERLU_MALLOC(nsupc*nsupc * sizeof(double))))
                                        ABORT("Malloc fails for Linv_bc_ptr[ljb][]");
                                    if (!(Uinv_bc_ptr[ljb] = (double*)SUPERLU_MALLOC(nsupc*nsupc * sizeof(double))))
                                        ABORT("Malloc fails for Uinv_bc_ptr[ljb][]");
                                    mybufmax[0] = SUPERLU_MAX( mybufmax[0], len1 );
                                    mybufmax[1] = SUPERLU_MAX( mybufmax[1], len*nsupc );
                                    mybufmax[4] = SUPERLU_MAX( mybufmax[4], len );
                                    memTRS += nrbl*3.0*iword + 2.0*nsupc*nsupc*dword;  //acount for Lindval_loc_bc_ptr[ljb],Linv_bc_ptr[ljb],Uinv_bc_ptr[ljb]
                                    index[0] = nrbl;  /* Number of row blocks */
                                    index[1] = len;   /* LDA of the nzval[] */
                                    next_lind = BC_HEADER;
                                    next_lval = 0;
                                    for (k = 0; k < nrbl; ++k) {
                                        gb = Lrb_number[k];
                                        // printf("Checking local block:%d \n",ljb);
                                        // 	fflush(stdout);
                                        // if (ljb==161)
                                        // {
                                        //     printf("Buggy block in EPB2. Supernode not constructed accurately!\n");
                                        //     fflush(stdout);
                                        // }
                                        lb = LBi( gb, grid );
                                        len = Lrb_length[lb];
                                        Lindval_loc_bc_ptr[ljb][k] = lb;
                                        Lindval_loc_bc_ptr[ljb][k+nrbl] = next_lind;
                                        Lindval_loc_bc_ptr[ljb][k+nrbl*2] = next_lval;
                                        Lrb_length[lb] = 0;  /* Reset vector of block length */
                                        index[next_lind++] = gb; /* Descriptor */
                                        index[next_lind++] = len;
                                        Lrb_indptr[lb] = next_lind;
                                        Lrb_valptr[lb] = next_lval;
                                        // if (ljb==161)
                                        // {
                                        //     printf("Values written into the supernoddal blocks:\n lb:%d \n next_lind:%d next_lval:%d (gb:%d) (len:%d) next_lval:%d \n",lb,next_lind,next_lval,gb,len,next_lval);
                                        //     fflush(stdout);
                                        // }
                                        next_lind += len;
                                        next_lval += len;


                                    } 
                                    /* Propagate the compressed row subscripts to Lindex[],
                                       and the initial values of A from SPA into Lnzval[]. */
                                    len = index[1];  /* LDA of lusup[] */
                                    // if (ljb==161)
                                    // {
                                    //     printf("Propagate the compressed row subscripts to Lindex\n");
                                    //     fflush(stdout);
                                    //     printf("index[0](nrbl):%d\n",index[0]);

                                    //     printf("index[1] (len):%d\n",len);
                                    //     fflush(stdout);
                                    // }
									// if (fsupc==49)
									// {
									// 	printf("for fsupc 49 Working on column:%d\n local column(ljb):%d\n",fsupc, ljb);
									// }
                                    for (i = istart; i < xlsub_end[fsupc]; ++i) {
                                        irow = lsub[i];
                                        gb = BlockNum( irow );
                                        if ( myrow == PROW( gb, grid ) ) {
                                            lb = LBi( gb, grid );
                                            k = Lrb_indptr[lb]++; /* Random access a block */
                                            index[k] = irow;
                                            // if (fsupc==49)
                                            // {
                                            //     printf("for fsupc 49 Adding index[k] (irow):%d\n",irow);
                                            //     fflush(stdout);
                                            // }
                                            k = Lrb_valptr[lb]++;
                                            irow = ilsum[lb] + irow - FstBlockC( gb );
                                            for (j = 0, dense_col = dense; j < nsupc; ++j) {
                                                lusup[k] = dense_col[irow];
												// if (fsupc==49)
												// {
												// 	printf("for fsupc 49 Writing :%lf at irow:%d\n",dense_col[irow],irow);
												// }
                                                // if (ljb==161)
                                                // {
                                                //     printf("lusup[k] (irow):%d\n",irow);
                                                //     fflush(stdout);
                                                // }
                                                dense_col[irow] = 0.0;
                                                k += len;
                                                dense_col += ldaspa;
                                            }
                                        }
                                    } /* for i ... */
                                    // fflush(stdout);
                                    Lrowind_bc_ptr[ljb] = index;
                                    Lnzval_bc_ptr[ljb] = lusup;

                                    /* sort Lindval_loc_bc_ptr[ljb], Lrowind_bc_ptr[ljb]
                                       and Lnzval_bc_ptr[ljb] here.  */
                                    if(nrbl>1){
                                        krow = PROW( jb, grid );
                                        if(myrow==krow){ /* skip the diagonal block */
                                            uu=nrbl-2;
                                            lloc = &Lindval_loc_bc_ptr[ljb][1];
                                        }else{
                                            uu=nrbl-1;
                                            lloc = Lindval_loc_bc_ptr[ljb];
                                        }
                                        quickSortM(lloc,0,uu,nrbl,0,3);
                                    }


                                    if ( !(index_srt = intMalloc_dist(len1)) )
                                        ABORT("Malloc fails for index_srt[]");
                                    if (!(lusup_srt = (double*)SUPERLU_MALLOC(len*nsupc * sizeof(double))))
                                        ABORT("Malloc fails for lusup_srt[]");

                                    idx_indx = BC_HEADER;
                                    idx_lusup = 0;
                                    for (jj=0;jj<BC_HEADER;jj++)
                                        index_srt[jj] = index[jj];

                                    for(i=0;i<nrbl;i++){
                                        nbrow = index[Lindval_loc_bc_ptr[ljb][i+nrbl]+1];
                                        for (jj=0;jj<LB_DESCRIPTOR+nbrow;jj++){
                                            index_srt[idx_indx++] = index[Lindval_loc_bc_ptr[ljb][i+nrbl]+jj];
                                        }

                                        Lindval_loc_bc_ptr[ljb][i+nrbl] = idx_indx - LB_DESCRIPTOR - nbrow;

                                        for (jj=0;jj<nbrow;jj++){
                                            k=idx_lusup;
                                            k1=Lindval_loc_bc_ptr[ljb][i+nrbl*2]+jj;
                                            for (j = 0; j < nsupc; ++j) {
                                                lusup_srt[k] = lusup[k1];
                                                k += len;
                                                k1 += len;
                                            }
                                            idx_lusup++;
                                        }
                                        Lindval_loc_bc_ptr[ljb][i+nrbl*2] = idx_lusup - nbrow;
                                    }

                                    SUPERLU_FREE(lusup);
                                    SUPERLU_FREE(index);

                                    Lrowind_bc_ptr[ljb] = index_srt;
                                    Lnzval_bc_ptr[ljb] = lusup_srt;

                                    // if(ljb==0)
                                    // for (jj=0;jj<nrbl*3;jj++){
                                    // printf("iam %5d Lindval %5d\n",iam, Lindval_loc_bc_ptr[ljb][jj]);
                                    // fflush(stdout);
                                    // }
                                    // for (jj=0;jj<nrbl;jj++){
                                    // printf("iam %5d Lindval %5d\n",iam, index[Lindval_loc_bc_ptr[ljb][jj+nrbl]]);
                                    // fflush(stdout);

                                    // }
                                } else {
                                    Lrowind_bc_ptr[ljb] = NULL;
                                    Lnzval_bc_ptr[ljb] = NULL;
                                    Linv_bc_ptr[ljb] = NULL;
                                    Uinv_bc_ptr[ljb] = NULL;
                                    Lindval_loc_bc_ptr[ljb] = NULL;
                                } /* if nrbl ... */
#if ( PROFlevel>=1 )
                                t_l += SuperLU_timer_() - t;
#endif
                            } /* if mycol == pc */
                            // printf("Process:%d finished SETTING UP L AND U block structures for  jb: %d!\n",iam, jb);
                            // fflush(stdout);
                            // if (iam == 0) 
                            // {
                            // 	printf("Process:%d reaches Milestone 5(original condition) for jb: %d\n",iam, jb);
                            // 	fflush(stdout);
                            // }
                        } /* for jb ... */
        //   printf("Process: %d, at line: %d\n",iam, __LINE__);
        //                 fflush(stdout);
                        //gSoFa update fsendx_plist and bsendx_plist locally from the values received from remote processes.
#ifdef communicate_fsend_bsend
                        // MPI_Allreduce( Request_fsendx_plist, Recv_Request_fsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t, MPI_MAX, grid->rscp.comm);

                        // MPI_Allreduce( Request_fsendx_plist, Recv_Request_fsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t, MPI_MAX, grid->cscp.comm);

                        // MPI_Allreduce( Request_bsendx_plist, Recv_Request_bsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t, MPI_MAX, grid->cscp.comm);


                        MPI_Alltoall(Request_fsendx_plist,NLocalBLCol,mpi_int_t,Recv_Request_fsendx_plist, NLocalBLCol, mpi_int_t,grid->cscp.comm);
                        MPI_Alltoall(Request_bsendx_plist,NLocalBLCol,mpi_int_t,Recv_Request_bsendx_plist, NLocalBLCol, mpi_int_t,grid->cscp.comm);

                        // MPI_Allreduce( Request_fsendx_plist, Recv_Request_fsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t, MPI_MAX, grid->rscp.comm);
                        // MPI_Allreduce( Request_bsendx_plist, Recv_Request_bsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t, MPI_MAX, grid->rscp.comm);

                        for (int p_row=0; p_row < grid->nprow; p_row++)		
                        {
                            // if (p_row != myrow)
                            {
                                for (int local_jb=0; local_jb < NLocalBLCol; local_jb++)
                                {

                                    // if (iam ==4)
                                    // {
                                    // 	printf("Request_fsendx_plist[%d]: %d", p_row * NLocalBLCol + local_jb, Request_fsendx_plist[p_row * NLocalBLCol + local_jb]);
                                    // }
                                    // if (Recv_Request_fsendx_plist[local_jb * grid->nprow + p_row] ==1)
                                    if (Recv_Request_fsendx_plist[local_jb + p_row * NLocalBLCol] == 1)
                                    {
                                        // fsendx_plist[iter][i] = Recv_Request_fsendx_plist[iter * grid->nprow +i];
                                        // Request_fsendx_plist[iter + i * NLocalBLCol] = YES;
                                        if (fsendx_plist[local_jb][p_row] == EMPTY)
                                        {
                                            // fsendx_plist[local_jb][p_row] = Recv_Request_fsendx_plist[local_jb * grid->nprow + p_row];
                                            fsendx_plist[local_jb][p_row] = YES;
                                            ++nfsendx;
                                            // if (iam ==5) && 
                                        }
                                    }
                                    if (Recv_Request_bsendx_plist[local_jb + p_row * NLocalBLCol] == 1)
                                    {
                                        if (bsendx_plist[local_jb][p_row] == EMPTY)
                                        {
                                            // bsendx_plist[local_jb][p_row] = Recv_Request_bsendx_plist[local_jb + p_row * NLocalBLCol];
                                            bsendx_plist[local_jb][p_row] = YES;
                                            ++nbsendx;
                                        }
                                    }

                                }
                            }

                        }
                        //~update fsendx structure

#endif 
                        //~gSoFa update fsendx_plist and bsendx_plist locally from the values received from remote processes.


#ifdef update_communication_str
                        /* ~gSoFa~ update the communication related data structures for numerical factorization given the structures are not global to all the processes ~gSoFa~*/
                        /* exchange information about bsendx_plist in between column of processors */

                        /*First communicate all the requests of processes on the row block from the diagonal process and update ToSendR[]*/
                        //  Update SendRReq using MPIALLToALL into ToSendR
                        //Accumulate SendRReq into ToSendR 

                        NLocalBLCol = CEILING( nsupers, grid->npcol );/* Number of local column blocks */

                        //Todebug
                        // Print SendBufSendR 
                        // MPI_Alltoall(SendBufSendR,NLocalBLCol*iword,mpi_int_t,recvBufSendR, NLocalBLCol*iword, mpi_int_t,grid->comm);
                        // 	if (iam==0)
                        // {
                        // 	printf("Process:%d SendBufSendR[17]: %d\n",iam, SendBufSendR[17]);
                        // 	fflush(stdout);
                        // }

                        //Has issues in the communication.
                        //need to change grid->comm
                        //Note:  Use MPI_Comm_create to create process grid for the communication
                        // MPI_Alltoall(SendBufSendR,NLocalBLCol,mpi_int_t,recvBufSendR, NLocalBLCol, mpi_int_t,grid->comm);
                        MPI_Alltoall(SendBufSendR,NLocalBLCol,mpi_int_t,recvBufSendR, NLocalBLCol, mpi_int_t,grid->rscp.comm);



                        // MPI_Alltoall(SendBufSendR,nsupers,mpi_int_t,recvBufSendR, nsupers, mpi_int_t,grid->comm);

#ifdef UpdateToRecv
                        MPI_Alltoall(RequestToRecv,nsupers,mpi_int_t,RecvBufRequestToRecv, nsupers, mpi_int_t,grid->comm);
                        for (int process_id=0; process_id < nprocs; process_id++)
                        {
                            if (iam != process_id)
                            {
                                for (int global_block =0; global_block < nsupers; global_block++)
                                {
                                    if ((ToRecv[global_block] != 1) && (ToRecv[global_block] != 2))
                                    {
                                        ToRecv[global_block] = RecvBufRequestToRecv[process_id*nsupers+global_block];	

                                        if (ToRecv[global_block]==1)
                                        {
                                            int local_block = LBi( global_block, grid );
                                            ToSendD[local_block] = YES;

                                        }
                                        // if (global_block == testing_block)
                                        // {
                                        // 	printf("Process: %d sets its ToRecv[%d]: %d\n",iam,global_block,ToRecv[global_block]);
                                        // 	fflush(stdout);
                                        // }
                                        // printf("Process: %d set value of ToRecv[%d]: %d\n",iam, global_block,ToRecv[global_block]);
                                        // fflush(stdout);
                                    }
                                }

                            }

                        }
#endif

                        // Print recvBufSendR 
                        //Compare the result

                        //recvBufSendR will be copied to the ToSendR[][] data structure of the process

                        // 	for (int l=0; l< NLocalBLCol; l++)
                        // {

                        // 	//l local block index
                        // 	//i the target process
                        // 	//Invalid write error at below line
                        // 	ToSendR[l][i] = recvBufSendR[i*nsupers+l];

                        // }

                        // if (iam==1)
                        // {
                        // 	printf("Process:%d recvBufSendR[17]: %d\n",iam, recvBufSendR[17]);
                        // 	fflush(stdout);
                        // }

#ifndef debug2_2
                        for (int l=0; l< NLocalBLCol; l++)
                        {
                            for (i = 0; i < grid->npcol; ++i) 
                            {
                                //l local block index
                                //i the target process
                                //Invalid write error at below line
                                // ToSendR[l][i] = recvBufSendR[i*nsupers+l];

                                // printf("New sendTOR\n");
                                // fflush(stdout);
#ifdef use_original_sendRLogic
                                ToSendR[l][i] = recvBufSendR[i*NLocalBLCol+l];
#else
                                if (recvBufSendR[i*NLocalBLCol+l]==1)
                                    // if (recvBufSendR[i*NLocalBLCol+l])
                                {
                                    // printf("Process column: %d received request from process column: %d for its local block column:%d\n",mycol,i,ljb);
                                    // fflush(stdout);
                                    // if (mycol != i)
                                    {
                                        ToSendR[l][i] = YES;
                                    }
                                }
#endif
                                // if (iam==2)
                                // {
                                // if ((i==1) && (l==813))
                                // {
                                // 	printf("Detected possible problametic ToSendR[%d][%d] set in Process:%d \n",l,i,iam);
                                // 	fflush(stdout);
                                // }
                                // }
                            }
                        }
#else 
                        for (i = 0; i < grid->npcol; ++i) 
                        {
                            for (int l=0; l< NLocalBLCol; l++)
                            {

                                //l local block index
                                //i the target process
                                //Invalid write error at below line
                                // ToSendR[l][i] = recvBufSendR[i*nsupers+l];


                                // ToSendR[l][i] = recvBufSendR[l* grid->npcol+i];
                                // printf("New sendTOR\n");
                                // fflush(stdout);
                                ToSendR[l][i] = recvBufSendR[l + i* NLocalBLCol];//Chnaged while debugging for large datasets 1 x 6 configs
                            }
                        }
#endif

                        // for (int l=0; l< NLocalBLCol; l++)
                        // {
                        // 	for (i = 0,j = 0; i < k; ++i, j += grid->npcol) 
                        // 	{
                        // 		//l local block index
                        // 		//i the target process
                        // 		//Invalid write error at below line
                        // 		// ToSendR[l][i] = recvBufSendR[i*nsupers+l];
                        // 		ToSendR[l][i] = recvBufSendR[i*NLocalBLCol+l];
                        // 	}
                        // }

                        //ToSendR is then distributed among other process grids

                        k = SUPERLU_MAX( grid->nprow, grid->npcol);
                        if ( !(recvBuf = (int_t *) SUPERLU_MALLOC(nsupers*k*iword)) ) {
                            fprintf (stderr, "Malloc fails for recvBuf[].");
                            return (memDist + memNLU + memTRS);
                        }
                        if ( !(nnzToRecv = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
                            fprintf (stderr, "Malloc fails for nnzToRecv[].");
                            return (memDist + memNLU + memTRS);
                        }
                        if ( !(ptrToRecv = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
                            fprintf (stderr, "Malloc fails for ptrToRecv[].");
                            return (memDist + memNLU + memTRS);
                        }
                        if ( !(nnzToSend = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
                            fprintf (stderr, "Malloc fails for nnzToRecv[].");
                            return (memDist + memNLU + memTRS);
                        }
                        if ( !(ptrToSend = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
                            fprintf (stderr, "Malloc fails for ptrToRecv[].");
                            return (memDist + memNLU + memTRS);
                        }

                        if (memDist < (nsupers*k*iword +4*nprocs * sizeof(int)))
                            memDist = nsupers*k*iword +4*nprocs * sizeof(int);



                        for (p = 0; p < nprocs; p++)
                            nnzToRecv[p] = 0;

                        for (jb = 0; jb < nsupers; jb++) {
                            jbcol = PCOL( jb, grid );
                            jbrow = PROW( jb, grid );
                            p = PNUM(jbrow, jbcol, grid);
                            nnzToRecv[p] += grid->npcol;
                        }
                        i = 0;
                        for (p = 0; p < nprocs; p++) {
                            ptrToRecv[p] = i;
                            i += nnzToRecv[p];
                            ptrToSend[p] = 0;
                            if (p != iam)
                                nnzToSend[p] = nnzToRecv[iam];
                            else
                                nnzToSend[p] = 0;
                        }
                        nnzToRecv[iam] = 0;
                        i = ptrToRecv[iam];



                        //Note: The nnzToSend, ptrToSend, nnzToRecv and ptrToRecv maynot be correct. for the gSoFa communication of ToSendR[]
                        //   MPI_Alltoallv (&(SendRReq[ptrToRecv[iam]]), nnzToSend, ptrToSend, mpi_int_t,
                        // 		 ToSendR, nnzToRecv, ptrToRecv, mpi_int_t, grid->comm);
                        //Use single dimension array for MPI all to all

                        // for (i = 0,j = 0; i < NLocalBLCol; ++i, j += grid->npcol) ToSendR[i] = &index1[j];
                        // int MPI_Alltoall(const void *sendbuf, int sendcount,
                        //     MPI_Datatype sendtype, void *recvbuf, int recvcount,
                        //     MPI_Datatype recvtype, MPI_Comm comm);

                        //    MPI_Alltoallv (&(recvBuf[ptrToRecv[iam]]), nnzToSend, ptrToSend, mpi_int_t,
                        //  recvBuf, nnzToRecv, ptrToRecv, mpi_int_t, grid->comm);

                        //recvBuf has the SendRReq
                        //Transfer the recvBuf into ToSendR


                        //anil 
                        /*Communicate for schur complement region*/

                        for (jb = 0; jb < nsupers; jb++) {
                            jbcol = PCOL( jb, grid );
                            jbrow = PROW( jb, grid );
                            p = PNUM(jbrow, jbcol, grid);
                            if (p == iam) {//if Iam the owner process of the diagonal block //anil
                                ljb_j = LBj( jb, grid ); /* Local block number column wise */
                                // if(iam == 1) 
                                // {
                                // 	printf("Process: %d is updating its recvBuf!\n",iam);
                                // 	fflush(stdout);
                                // }
                                for (j = 0; j < grid->npcol; j++, i++)
                                {
                                    recvBuf[i] = ToSendR[ljb_j][j];
                                    // printf("Process: %d has recvBuf[%d]: %d!\n",iam,i,recvBuf[i]);
                                    // fflush(stdout);
                                }

                            }
                        }

                        MPI_Alltoallv (&(recvBuf[ptrToRecv[iam]]), nnzToSend, ptrToSend, mpi_int_t,
                                recvBuf, nnzToRecv, ptrToRecv, mpi_int_t, grid->comm);

                        //   printf("Process: %d is at MPIallToall!\n",iam);
                        //   fflush(stdout);


                        for (jb = 0; jb < nsupers; jb++) {
                            jbcol = PCOL( jb, grid );
                            jbrow = PROW( jb, grid );
                            p = PNUM(jbrow, jbcol, grid);
                            ljb_j = LBj( jb, grid ); /* Local block number column wise */
                            ljb_i = LBi( jb, grid ); /* Local block number row wise */
                            /* (myrow == jbrow) {
                               if (ToSendD[ljb_i] == YES)
                               ToRecv[jb] = 1;
                               }
                               else {
                               if (recvBuf[ptrToRecv[p] + mycol] == YES)
                               ToRecv[jb] = 2;
                               } */
#ifdef enableExtraToRecv
                            //    if (iam==1)
                            //    {
                            // 	   PrintInt10("Process 1 recvBuf:", nsupers*k*iword, recvBuf);
                            // 	   fflush(stdout);
                            //    }
                            if (recvBuf[ptrToRecv[p] + mycol] == YES) {
                                if (myrow == jbrow)
                                    ToRecv[jb] = 1;
                                else
                                    ToRecv[jb] = 2;
                            }
#endif

                            if (mycol == jbcol) {
                                // if(iam == 1) 
                                // {
                                // 	printf("Process: %d is updating its ToSendR after MPIallToall!\n",iam);
                                // 	fflush(stdout);
                                // }
                                for (i = 0, j = ptrToRecv[p]; i < grid->npcol; i++, j++)
                                {
                                    ToSendR[ljb_j][i] = recvBuf[j];
                                    // if(iam == 1) 
                                    // {
                                    // 	printf("Process: %d is ToSendR[%d][%d ]: %d!\n",iam, ljb_j,i,ToSendR[ljb_j][i]);
                                    // 	fflush(stdout);
                                    // }
                                }
                                ToSendR[ljb_j][mycol] = EMPTY; //~Anil why??
                            }
                            ptrToRecv[p] += grid->npcol;
                        }
                        //   if (iam==1)
                        //    {
                        // 	   PrintInt10("Process 1 recvBuf:", nsupers*k*iword, recvBuf);
                        // 	   fflush(stdout);
                        //    }

#ifdef enable_extra_comm
                        /* exchange information about bsendx_plist in between column of processors */
                        MPI_Allreduce ((*bsendx_plist), recvBuf, nsupers_j * grid->nprow, mpi_int_t,
                                MPI_MAX, grid->cscp.comm);

                        for (jb = 0; jb < nsupers; jb ++) {
                            jbcol = PCOL( jb, grid);
                            jbrow = PROW( jb, grid);
                            if (mycol == jbcol) {
                                ljb_j = LBj( jb, grid ); /* Local block number column wise */
                                if (myrow == jbrow ) {
                                    for (k = ljb_j * grid->nprow; k < (ljb_j+1) * grid->nprow; k++) {
                                        (*bsendx_plist)[k] = recvBuf[k];
                                        if ((*bsendx_plist)[k] != EMPTY)
                                        {
                                            // nbsendx ++; //Commented for gSoFa
                                        }
                                    }
                                }
                                else {
                                    for (k = ljb_j * grid->nprow; k < (ljb_j+1) * grid->nprow; k++)
                                        (*bsendx_plist)[k] = EMPTY;
                                }
                            }
                        }
                        /* ~gSoFa~ update the communication related data structures for numerical factorization */
#endif
#endif
                        /////////////////////////////////////////////////////////////////

                        /* Set up additional pointers for the index and value arrays of U.
                           nub is the number of local block columns. */
                        nub = CEILING( nsupers, grid->npcol); /* Number of local block columns. */
                        if ( !(Urbs = (int_t *) intCalloc_dist(2*nub)) )
                            ABORT("Malloc fails for Urbs[]"); /* Record number of nonzero
                                                                 blocks in a block column. */
                        Urbs1 = Urbs + nub;
                        if ( !(Ucb_indptr = SUPERLU_MALLOC(nub * sizeof(Ucb_indptr_t *))) )
                            ABORT("Malloc fails for Ucb_indptr[]");
                        if ( !(Ucb_valptr = SUPERLU_MALLOC(nub * sizeof(int_t *))) )
                            ABORT("Malloc fails for Ucb_valptr[]");
                        nlb = CEILING( nsupers, grid->nprow ); /* Number of local block rows. */

                        /* Count number of row blocks in a block column.
                           One pass of the skeleton graph of U. */
                        for (lk = 0; lk < nlb; ++lk) {
                            usub1 = Ufstnz_br_ptr[lk];
                            if ( usub1 ) { /* Not an empty block row. */
                                /* usub1[0] -- number of column blocks in this block row. */
                                i = BR_HEADER; /* Pointer in index array. */
                                for (lb = 0; lb < usub1[0]; ++lb) { /* For all column blocks. */
                                    k = usub1[i];            /* Global block number */
                                    ++Urbs[LBj(k,grid)];
                                    i += UB_DESCRIPTOR + SuperSize( k );
                                }
                            }
                        }

                        /* Set up the vertical linked lists for the row blocks.
                           One pass of the skeleton graph of U. */
                        for (lb = 0; lb < nub; ++lb) {
                            if ( Urbs[lb] ) { /* Not an empty block column. */
                                if ( !(Ucb_indptr[lb]
                                            = SUPERLU_MALLOC(Urbs[lb] * sizeof(Ucb_indptr_t))) )
                                    ABORT("Malloc fails for Ucb_indptr[lb][]");
                                if ( !(Ucb_valptr[lb] = (int_t *) intMalloc_dist(Urbs[lb])) )
                                    ABORT("Malloc fails for Ucb_valptr[lb][]");
                            }
                        }
                        for (lk = 0; lk < nlb; ++lk) { /* For each block row. */
                            usub1 = Ufstnz_br_ptr[lk];
                            if ( usub1 ) { /* Not an empty block row. */
                                i = BR_HEADER; /* Pointer in index array. */
                                j = 0;         /* Pointer in nzval array. */

                                for (lb = 0; lb < usub1[0]; ++lb) { /* For all column blocks. */
                                    k = usub1[i];          /* Global block number, column-wise. */
                                    ljb = LBj( k, grid ); /* Local block number, column-wise. */
                                    Ucb_indptr[ljb][Urbs1[ljb]].lbnum = lk;

                                    Ucb_indptr[ljb][Urbs1[ljb]].indpos = i;
                                    Ucb_valptr[ljb][Urbs1[ljb]] = j;

                                    ++Urbs1[ljb];
                                    j += usub1[i+1];
                                    i += UB_DESCRIPTOR + SuperSize( k );
                                }
                            }
                        }


                        /* Count the nnzs per block column */
                        for (lb = 0; lb < nub; ++lb) {
                            Unnz[lb] = 0;
                            k = lb * grid->npcol + mycol;/* Global block number, column-wise. */
                            knsupc = SuperSize( k );
                            for (ub = 0; ub < Urbs[lb]; ++ub) {
                                ik = Ucb_indptr[lb][ub].lbnum; /* Local block number, row-wise. */
                                i = Ucb_indptr[lb][ub].indpos; /* Start of the block in usub[]. */
                                i += UB_DESCRIPTOR;
                                gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
                                iklrow = FstBlockC( gik+1 );
                                for (jj = 0; jj < knsupc; ++jj) {
                                    fnz = Ufstnz_br_ptr[ik][i + jj];
                                    if ( fnz < iklrow ) {
                                        Unnz[lb] +=iklrow-fnz;
                                    }
                                } /* for jj ... */
                            }
                        }

                        /////////////////////////////////////////////////////////////////

#if ( PROFlevel>=1 )
                        t = SuperLU_timer_();
#endif
                        /* construct the Bcast tree for L ... */

                        k = CEILING( nsupers, grid->npcol );/* Number of local block columns */
                        if ( !(LBtree_ptr = (C_Tree*)SUPERLU_MALLOC(k * sizeof(C_Tree))) )
                            ABORT("Malloc fails for LBtree_ptr[].");
                        if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
                            ABORT("Calloc fails for ActiveFlag[].");
                        if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
                            ABORT("Malloc fails for ranks[].");
                        if ( !(SeedSTD_BC = (double*)SUPERLU_MALLOC(k * sizeof(double))) )
                            ABORT("Malloc fails for SeedSTD_BC[].");


                        for (i=0;i<k;i++){
                            SeedSTD_BC[i]=rand();
                        }

                        MPI_Allreduce(MPI_IN_PLACE,&SeedSTD_BC[0],k,MPI_DOUBLE,MPI_MAX,grid->cscp.comm);

                        for (ljb = 0; ljb <k ; ++ljb) {
                            C_BcTree_Nullify(&LBtree_ptr[ljb]);
                            // LBtree_ptr[ljb]=NULL;
                        }


                        if ( !(ActiveFlagAll = intMalloc_dist(grid->nprow*k)) )
                            ABORT("Calloc fails for ActiveFlag[].");
                        memTRS += k*sizeof(C_Tree) + k*dword + grid->nprow*k*iword;  //acount for LBtree_ptr, SeedSTD_BC, ActiveFlagAll
                        for (j=0;j<grid->nprow*k;++j)ActiveFlagAll[j]=3*nsupers;
                        for (ljb = 0; ljb < k; ++ljb) { /* for each local block column ... */
                            jb = mycol+ljb*grid->npcol;  /* not sure */
                            if(jb<nsupers){
                                pc = PCOL( jb, grid );
                                fsupc = FstBlockC( jb );
                                nsupc = SuperSize( jb );

                                istart = xlsub_begin[fsupc];
                                for (i = istart; i < xlsub_end[fsupc]; ++i) {
                                    irow = lsub[i];
                                    gb = BlockNum( irow );
                                    pr = PROW( gb, grid );
                                    ActiveFlagAll[pr+ljb*grid->nprow]=SUPERLU_MIN(ActiveFlagAll[pr+ljb*grid->nprow],gb);
                                } /* for j ... */
                            }
                        }

                        //gSoFa
                        MPI_Allreduce(MPI_IN_PLACE,ActiveFlagAll,grid->nprow*k,mpi_int_t,MPI_MIN,grid->cscp.comm);
                        //~gSoFa

                        // printf("Process: %d\n",iam);
                        // PrintInt10("ActiveFlagAll for L Broadcast tree", grid->nprow*k, ActiveFlagAll);
                        // fflush(stdout);

                        for (ljb = 0; ljb < k; ++ljb) { /* for each local block column ... */

                            jb = mycol+ljb*grid->npcol;  /* not sure */
                            if(jb<nsupers){
                                pc = PCOL( jb, grid );

                                for (j=0;j<grid->nprow;++j)ActiveFlag[j]=ActiveFlagAll[j+ljb*grid->nprow];
                                for (j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
                                for (j=0;j<grid->nprow;++j)ranks[j]=-1;

                                Root=-1;
                                Iactive = 0;
                                for (j=0;j<grid->nprow;++j){
                                    if(ActiveFlag[j]!=3*nsupers){
                                        gb = ActiveFlag[j];
                                        pr = PROW( gb, grid );
                                        if(gb==jb)Root=pr;
                                        if(myrow==pr)Iactive=1;
                                    }
                                }


                                quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,0,2);

                                if(Iactive==1){
                                    // printf("jb %5d damn\n",jb);
                                    // fflush(stdout);
                                    assert( Root>-1 );
                                    rank_cnt = 1;
                                    ranks[0]=Root;
                                    for (j = 0; j < grid->nprow; ++j){
                                        if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
                                            ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
                                            ++rank_cnt;
                                        }
                                    }
                                    // if (iam==0)
                                    // {
                                    // 	// printf("rank_cnt BEFORE BC_TREE CREATION:%d!\n",rank_cnt);

                                    // 	// PrintInt10("ActiveFlag", grid->nprow * 2, ActiveFlag);    
                                    // 	// PrintInt10("ActiveFlagAll", grid->nprow * k, ActiveFlagAll);    
                                    // 	PrintInt10("Process 0 ranks", rank_cnt, ranks);    
                                    // 	fflush(stdout);
                                    // }

                                    if(rank_cnt>1){

                                        for (ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                                        {
                                            ranks[ii] = PNUM( ranks[ii], pc, grid );
                                            // ranks[ii] = PNUM( ranks[ii], pc, grid );
                                            // if (iam==0)
                                            // {
                                            // 	printf("After update in jb=%d Process:%d ranks[%d]:%d\n",jb, iam, ii,ranks[ii]);
                                            // 	fflush(stdout);
                                            // }
                                        }


                                        // rseed=rand();
                                        // rseed=1.0;
                                        msgsize = SuperSize( jb );
                                        // if (iam==0)
                                        // {
                                        // 	printf("Process:%d SeedSTD_BC[%d]:%lf msgsize:%d rank_cnt:%d \n",iam, ljb,SeedSTD_BC[ljb], msgsize, rank_cnt);
                                        // 	// fflush(stdout);
                                        // 	printf("BCTree_CREATION!\n");
                                        // 	fflush(stdout);
                                        // }
                                        // LBtree_ptr[ljb] = BcTree_Create(grid->comm, ranks, rank_cnt, msgsize,SeedSTD_BC[ljb],'d');
                                        // BcTree_SetTag(LBtree_ptr[ljb],BC_L,'d');
                                        C_BcTree_Create(&LBtree_ptr[ljb], grid->comm, ranks, rank_cnt, msgsize, 'd');
				                        LBtree_ptr[ljb].tag_=BC_L;

                                        // printf("iam %5d btree rank_cnt %5d \n",iam,rank_cnt);
                                        // fflush(stdout);

                                        // if(iam==15 || iam==3){
                                        // printf("iam %5d btree lk %5d tag %5d root %5d\n",iam, ljb,jb,BcTree_IsRoot(LBtree_ptr[ljb],'d'));
                                        // fflush(stdout);
                                        // }

                                        // #if ( PRNTlevel>=1 )
                                        if(Root==myrow){
                                            rank_cnt_ref=1;
                                            for (j = 0; j < grid->nprow; ++j) {
                                                if ( fsendx_plist[ljb][j] != EMPTY ) {
                                                    ++rank_cnt_ref;
                                                }
                                            }
                                            assert(rank_cnt==rank_cnt_ref);

                                            // printf("Partial Bcast Procs: col%7d np%4d\n",jb,rank_cnt);

                                            // // printf("Partial Bcast Procs: %4d %4d: ",iam, rank_cnt);
                                            // // for(j=0;j<rank_cnt;++j)printf("%4d",ranks[j]);
                                            // // printf("\n");
                                        }
                                        // #endif
                                    }
                                }
                            }
                        }


                        SUPERLU_FREE(ActiveFlag);
                        SUPERLU_FREE(ActiveFlagAll);
                        SUPERLU_FREE(ranks);
                        SUPERLU_FREE(SeedSTD_BC);
                        memTRS -= k*dword + grid->nprow*k*iword;  //acount for SeedSTD_BC, ActiveFlagAll

#if ( PROFlevel>=1 )
                        t = SuperLU_timer_() - t;
                        if ( !iam) printf(".. Construct Bcast tree for L: %.2f\t\n", t);
#endif

#if 0
                        //gSoFa update fsendx_plist and bsendx_plist locally from the values received from remote processes.
#ifdef communicate_fsend_bsend
                        // MPI_Allreduce( Request_fsendx_plist, Recv_Request_fsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t, MPI_MAX, grid->rscp.comm);
                        MPI_Allreduce( Request_fsendx_plist, Recv_Request_fsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t, MPI_MAX, grid->cscp.comm);
                        MPI_Allreduce( Request_bsendx_plist, Recv_Request_bsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t, MPI_MAX, grid->cscp.comm);
                        // MPI_Alltoall(Request_fsendx_plist,NLocalBLCol*grid->nprow,mpi_int_t,Recv_Request_fsendx_plist, NLocalBLCol*grid->nprow, mpi_int_t,grid->rscp.comm);
                        //update fsendx structure
                        // if (iam==0)
                        // {
                        // 	  PrintInt10("Recv_Request_fsendx_plist",NLocalBLCol*grid->nprow, Recv_Request_fsendx_plist);    
                        // 	fflush(stdout);
                        // }
                        for (i=0; i < grid->nprow; i++)
                        {
                            if (i != myrow)
                            {
                                for (int iter=0; iter < NLocalBLCol; iter++)
                                {
                                    // if (Recv_Request_fsendx_plist[iter*grind->nprow+i] ==0)
                                    // {
                                    // 	fsendx_plist[iter][i] = -1;
                                    // }
                                    // else
                                    // if (Recv_Request_fsendx_plist[iter * (grid->nprow) + i] ==1)
                                    // if (iam ==0)
                                    // {
                                    // 	printf("Recv_Request_fsendx_plist[%d + %d * %d]:%d\n", iter, i, NLocalBLCol, Recv_Request_fsendx_plist[iter + i * NLocalBLCol]);
                                    // 	printf("Before: For fsendx_plist[%d][%d]: %d\n",iter, i, fsendx_plist[iter][i]);
                                    // 	fflush(stdout);
                                    // }
                                    if (Recv_Request_fsendx_plist[iter + i * NLocalBLCol] ==1)
                                    {
                                        // fsendx_plist[iter][i] = Recv_Request_fsendx_plist[iter * grid->nprow +i];
                                        // Request_fsendx_plist[iter + i * NLocalBLCol] = YES;
                                        if (fsendx_plist[iter][i] == EMPTY)
                                        {
                                            fsendx_plist[iter][i] = Recv_Request_fsendx_plist[iter + i * NLocalBLCol];
                                            ++nfsendx;
                                        }
                                        // if (iam ==0)
                                        // {
                                        // 		printf("After: For fsendx_plist[%d][%d]: %d\n",iter, i, fsendx_plist[iter][i]);
                                        // fflush(stdout);
                                        // }
                                        // if  ((iam==0) && (iter==1))
                                        // 					{

                                        // 							printf("Process: %d irow:%d fsendx_plist[%d][%d]: %d\n",iam,irow, iter,i, fsendx_plist[iter][i]);
                                        // 							fflush(stdout);

                                        // 					}
                                    }
                                    if (Recv_Request_bsendx_plist[iter + i * NLocalBLCol] == 1)
                                    {
                                        if (bsendx_plist[iter][i] == EMPTY)
                                        {
                                            bsendx_plist[iter][i] = Recv_Request_bsendx_plist[iter + i * NLocalBLCol];
                                            ++nbsendx;
                                        }
                                    }

                                }
                            }

                        }
                        //~update fsendx structure

#endif
                        //~gSoFa update fsendx_plist and bsendx_plist locally from the values received from remote processes.
#endif
#if ( PROFlevel>=1 )
                        t = SuperLU_timer_();
#endif
                        /* construct the Reduce tree for L ... */
                        /* the following is used as reference */
                        nlb = CEILING( nsupers, grid->nprow );/* Number of local block rows */
                        if ( !(mod_bit = int32Malloc_dist(nlb)) )
                            ABORT("Malloc fails for mod_bit[].");
                        if ( !(frecv = int32Malloc_dist(nlb)) )
                            ABORT("Malloc fails for frecv[].");

                        for (k = 0; k < nlb; ++k) mod_bit[k] = 0;
                        for (k = 0; k < nsupers; ++k) {
                            pr = PROW( k, grid );
                            if ( myrow == pr ) {
                                lib = LBi( k, grid );    /* local block number */
                                kcol = PCOL( k, grid );
                                if (mycol == kcol || fmod[lib] )
                                    mod_bit[lib] = 1;  /* contribution from off-diagonal and diagonal*/
                            }
                        }
                        /* Every process receives the count, but it is only useful on the
                           diagonal processes.  */
                        // MPI_Allreduce( mod_bit, frecv, nlb, mpi_int_t, MPI_SUM, grid->rscp.comm);
                        MPI_Allreduce( mod_bit, frecv, nlb, MPI_INT, MPI_SUM, grid->rscp.comm);



                        k = CEILING( nsupers, grid->nprow );/* Number of local block rows */
                        if ( !(LRtree_ptr = (C_Tree*)SUPERLU_MALLOC(k * sizeof(C_Tree))) )
                            ABORT("Malloc fails for LRtree_ptr[].");
                        if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
                            ABORT("Calloc fails for ActiveFlag[].");
                        if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
                            ABORT("Malloc fails for ranks[].");

                        // if ( !(idxs = intCalloc_dist(nsupers)) )
                        // ABORT("Calloc fails for idxs[].");

                        // if ( !(nzrows = (int_t**)SUPERLU_MALLOC(nsupers * sizeof(int_t*))) )
                        // ABORT("Malloc fails for nzrows[].");

                        if ( !(SeedSTD_RD = (double*)SUPERLU_MALLOC(k * sizeof(double))) )
                            ABORT("Malloc fails for SeedSTD_RD[].");

                        for (i=0;i<k;i++){
                            SeedSTD_RD[i]=rand();
                        }

                        MPI_Allreduce(MPI_IN_PLACE,&SeedSTD_RD[0],k,MPI_DOUBLE,MPI_MAX,grid->rscp.comm);


                        // for (jb = 0; jb < nsupers; ++jb) { /* for each block column ... */
                        // fsupc = FstBlockC( jb );
                        // len=xlsub[fsupc+1]-xlsub[fsupc];
                        // idxs[jb] = len-1;
                        // if(len>0){
                        // if ( !(nzrows[jb] = intMalloc_dist(len)) )
                        // ABORT("Malloc fails for nzrows[jb]");
                        // for(i=xlsub[fsupc];i<xlsub[fsupc+1];++i){
                        // irow = lsub[i];
                        // nzrows[jb][i-xlsub[fsupc]]=irow;
                        // }
                        // quickSort(nzrows[jb],0,len-1,0);
                        // }
                        // else{
                        // nzrows[jb] = NULL;
                        // }
                        // }


                        for (lib = 0; lib <k ; ++lib) {
                            // LRtree_ptr[lib]=NULL;
                            C_RdTree_Nullify(&LRtree_ptr[lib]);
                        }


                        if ( !(ActiveFlagAll = intMalloc_dist(grid->npcol*k)) )
                            ABORT("Calloc fails for ActiveFlagAll[].");
                        for (j=0;j<grid->npcol*k;++j)ActiveFlagAll[j]=-3*nsupers;
                        memTRS += k*sizeof(C_Tree) + k*dword + grid->npcol*k*iword;  //acount for LRtree_ptr, SeedSTD_RD, ActiveFlagAll
                        for (jb = 0; jb < nsupers; ++jb) { /* for each block column ... */
                            fsupc = FstBlockC( jb );
                            pc = PCOL( jb, grid );
                            for(i=xlsub_begin[fsupc];i<xlsub_end[fsupc];++i){
                                irow = lsub[i];
                                ib = BlockNum( irow );
                                pr = PROW( ib, grid );
                                if ( myrow == pr ) { /* Block row ib in my process row */
                                    lib = LBi( ib, grid ); /* Local block number */
                                    ActiveFlagAll[pc+lib*grid->npcol]=SUPERLU_MAX(ActiveFlagAll[pc+lib*grid->npcol],jb);
                                }
                            }
                        }
                        //gSoFa
                        MPI_Allreduce(MPI_IN_PLACE,ActiveFlagAll,grid->npcol*k,mpi_int_t,MPI_MAX,grid->rscp.comm);
                        //~gSoFa
                        // printf("Process: %d\n",iam);
                        // PrintInt10("ActiveFlagAll for L Reduction tree", grid->npcol*k, ActiveFlagAll);
                        // fflush(stdout);

                        for (lib=0;lib<k;++lib){
                            ib = myrow+lib*grid->nprow;  /* not sure */
                            if(ib<nsupers){
                                pr = PROW( ib, grid );
                                for (j=0;j<grid->npcol;++j)ActiveFlag[j]=ActiveFlagAll[j+lib*grid->npcol];;
                                for (j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
                                for (j=0;j<grid->npcol;++j)ranks[j]=-1;
                                Root=-1;
                                Iactive = 0;

                                for (j=0;j<grid->npcol;++j){
                                    if(ActiveFlag[j]!=-3*nsupers){
                                        jb = ActiveFlag[j];
                                        pc = PCOL( jb, grid );
                                        if(jb==ib)Root=pc;
                                        if(mycol==pc)Iactive=1;
                                    }
                                }


                                quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,1,2);

                                if(Iactive==1){
                                    assert( Root>-1 );
                                    rank_cnt = 1;
                                    ranks[0]=Root;
                                    for (j = 0; j < grid->npcol; ++j){
                                        if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
                                            ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
                                            ++rank_cnt;
                                        }
                                    }

                                    // 						if (iam==0)
                                    // 			{
                                    // 		printf("rank_cnt BEFORE RdTree_Create CREATION:%d!\n",rank_cnt);

                                    // 		        PrintInt10("RdTree_Create ActiveFlag", grid->nprow * 2, ActiveFlag);    
                                    //   PrintInt10("RdTree_Create ActiveFlagAll", grid->nprow * k, ActiveFlagAll);    
                                    //   fflush(stdout);
                                    // 			}
                                    if(rank_cnt>1){

                                        for (ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                                            ranks[ii] = PNUM( pr, ranks[ii], grid );

                                        // rseed=rand();
                                        // rseed=1.0;
                                        msgsize = SuperSize( ib );

                                        // if(ib==0){

                                        // LRtree_ptr[lib] = RdTree_Create(grid->comm, ranks, rank_cnt, msgsize,SeedSTD_RD[lib],'d');
                                        // RdTree_SetTag(LRtree_ptr[lib], RD_L,'d');
                                        C_RdTree_Create(&LRtree_ptr[lib], grid->comm, ranks, rank_cnt, msgsize, 'd');
					                    LRtree_ptr[lib].tag_=RD_L;
                                        // }

                                        // printf("iam %5d rtree rank_cnt %5d \n",iam,rank_cnt);
                                        // fflush(stdout);

                                        // if(ib==15  || ib ==16){

                                        // if(iam==15 || iam==3){
                                        // printf("iam %5d rtree lk %5d tag %5d root %5d\n",iam,lib,ib,RdTree_IsRoot(LRtree_ptr[lib],'d'));
                                        // fflush(stdout);
                                        // }


                                        // #if ( PRNTlevel>=1 )
                                        // if(Root==mycol){
                                        // assert(rank_cnt==frecv[lib]);
                                        // printf("Partial Reduce Procs: row%7d np%4d\n",ib,rank_cnt);
                                        // // printf("Partial Reduce Procs: %4d %4d: ",iam, rank_cnt);
                                        // // // for(j=0;j<rank_cnt;++j)printf("%4d",ranks[j]);
                                        // // printf("\n");
                                        // }
                                        // #endif
                                    }
                                    }
                                }
                            }

                            SUPERLU_FREE(mod_bit);
                            SUPERLU_FREE(frecv);


                            SUPERLU_FREE(ActiveFlag);
                            SUPERLU_FREE(ActiveFlagAll);
                            SUPERLU_FREE(ranks);
                            // SUPERLU_FREE(idxs);
                            SUPERLU_FREE(SeedSTD_RD);
                            // for(i=0;i<nsupers;++i){
                            // if(nzrows[i])SUPERLU_FREE(nzrows[i]);
                            // }
                            // SUPERLU_FREE(nzrows);
                            memTRS -= k*dword + grid->nprow*k*iword;  //acount for SeedSTD_RD, ActiveFlagAll
                            ////////////////////////////////////////////////////////

#if ( PROFlevel>=1 )
                            t = SuperLU_timer_() - t;
                            if ( !iam) printf(".. Construct Reduce tree for L: %.2f\t\n", t);
#endif

#if ( PROFlevel>=1 )
                            t = SuperLU_timer_();
#endif

                            /* construct the Bcast tree for U ... */

                            k = CEILING( nsupers, grid->npcol );/* Number of local block columns */
                            if ( !(UBtree_ptr = (C_Tree*)SUPERLU_MALLOC(k * sizeof(C_Tree))) )
                                ABORT("Malloc fails for UBtree_ptr[].");
                            if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
                                ABORT("Calloc fails for ActiveFlag[].");
                            if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
                                ABORT("Malloc fails for ranks[].");
                            if ( !(SeedSTD_BC = (double*)SUPERLU_MALLOC(k * sizeof(double))) )
                                ABORT("Malloc fails for SeedSTD_BC[].");

                            for (i=0;i<k;i++){
                                SeedSTD_BC[i]=rand();
                            }

                            MPI_Allreduce(MPI_IN_PLACE,&SeedSTD_BC[0],k,MPI_DOUBLE,MPI_MAX,grid->cscp.comm);


                            for (ljb = 0; ljb <k ; ++ljb) {
                                // UBtree_ptr[ljb]=NULL;
                                C_BcTree_Nullify(&UBtree_ptr[ljb]);
                            }

                            if ( !(ActiveFlagAll = intMalloc_dist(grid->nprow*k)) )
                                ABORT("Calloc fails for ActiveFlagAll[].");
                            for (j=0;j<grid->nprow*k;++j)ActiveFlagAll[j]=-3*nsupers;
                            memTRS += k*sizeof(C_Tree) + k*dword + grid->nprow*k*iword;  //acount for UBtree_ptr, SeedSTD_BC, ActiveFlagAll

                            for (ljb = 0; ljb < k; ++ljb) { /* for each local block column ... */
                                jb = mycol+ljb*grid->npcol;  /* not sure */
                                if(jb<nsupers){
                                    pc = PCOL( jb, grid );

                                    fsupc = FstBlockC( jb );
                                    for (j = fsupc; j < FstBlockC( jb+1 ); ++j) {
                                        istart = xusub_begin[j];
                                        /* NOTE: Only the first nonzero index of the segment
                                           is stored in usub[]. */
                                        for (i = istart; i < xusub_end[j]; ++i) {
                                            irow = usub[i]; /* First nonzero in the segment. */
                                            gb = BlockNum( irow );
#ifdef supernodal_Uskipp
                                            int_t i_supernode = Glu_persist->supno[irow];
                                            if (jb == i_supernode)
                                            {
                                                if (SuperSize( i_supernode ) !=1)
                                                {
                                                    //Skipping the columns that are already included into the supernodes in the L structure
                                                    //    if (SuperSize( jb ) !=1)  continue;
                                                    if (gb == jb) continue;
                                                }
                                            }
#endif
                                            pr = PROW( gb, grid );
                                            ActiveFlagAll[pr+ljb*grid->nprow]=SUPERLU_MAX(ActiveFlagAll[pr+ljb*grid->nprow],gb);
                                            // printf("gb:%5d jb: %5d nsupers: %5d\n",gb,jb,nsupers);
                                            // fflush(stdout);
                                            //if(gb==jb)Root=pr;
                                        }


                                    }
                                    pr = PROW( jb, grid ); // take care of diagonal node stored as L
                                    // printf("jb %5d current: %5d",jb,ActiveFlagAll[pr+ljb*grid->nprow]);
                                    // fflush(stdout);
                                    ActiveFlagAll[pr+ljb*grid->nprow]=SUPERLU_MAX(ActiveFlagAll[pr+ljb*grid->nprow],jb);
                                }
                            }


                            //gSoFa
                            MPI_Allreduce(MPI_IN_PLACE,ActiveFlagAll,grid->nprow*k,mpi_int_t,MPI_MAX,grid->cscp.comm);
                            //~gSoFa	

                            // printf("Process: %d\n",iam);
                            // PrintInt10("ActiveFlagAll for U Broadcast tree", grid->nprow*k, ActiveFlagAll);
                            // fflush(stdout);


                            for (ljb = 0; ljb < k; ++ljb) { /* for each block column ... */
                                jb = mycol+ljb*grid->npcol;  /* not sure */
                                if(jb<nsupers){
                                    pc = PCOL( jb, grid );
                                    // if ( mycol == pc ) { /* Block column jb in my process column */

                                    for (j=0;j<grid->nprow;++j)ActiveFlag[j]=ActiveFlagAll[j+ljb*grid->nprow];
                                    for (j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
                                    for (j=0;j<grid->nprow;++j)ranks[j]=-1;

                                    Root=-1;
                                    Iactive = 0;
                                    for (j=0;j<grid->nprow;++j){
                                        if(ActiveFlag[j]!=-3*nsupers){
                                            gb = ActiveFlag[j];
                                            pr = PROW( gb, grid );
                                            if(gb==jb)Root=pr;
                                            if(myrow==pr)Iactive=1;
                                        }
                                    }

                                    quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,1,2);
                                    // printf("jb: %5d Iactive %5d\n",jb,Iactive);
                                    // fflush(stdout);
                                    if(Iactive==1){
                                        // printf("root:%5d jb: %5d\n",Root,jb);
                                        // fflush(stdout);
                                        assert( Root>-1 );
                                        rank_cnt = 1;
                                        ranks[0]=Root;
                                        for (j = 0; j < grid->nprow; ++j){
                                            if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
                                                ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
                                                ++rank_cnt;
                                            }
                                        }
                                        // printf("jb: %5d rank_cnt %5d\n",jb,rank_cnt);
                                        // fflush(stdout);
                                        if(rank_cnt>1){
                                            for (ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                                                ranks[ii] = PNUM( ranks[ii], pc, grid );

                                            // rseed=rand();
                                            // rseed=1.0;
                                            msgsize = SuperSize( jb );
                                            // UBtree_ptr[ljb] = BcTree_Create(grid->comm, ranks, rank_cnt, msgsize,SeedSTD_BC[ljb],'d');
                                            // BcTree_SetTag(UBtree_ptr[ljb],BC_U,'d');
                                            C_BcTree_Create(&UBtree_ptr[ljb], grid->comm, ranks, rank_cnt, msgsize, 'd');
				                            UBtree_ptr[ljb].tag_=BC_U;

                                            // printf("iam %5d btree rank_cnt %5d \n",iam,rank_cnt);
                                            // fflush(stdout);

                                            if(Root==myrow){
                                                rank_cnt_ref=1;
                                                for (j = 0; j < grid->nprow; ++j) {
                                                    // printf("ljb %5d j %5d nprow %5d\n",ljb,j,grid->nprow);
                                                    // fflush(stdout);
                                                    if ( bsendx_plist[ljb][j] != EMPTY ) {
                                                        ++rank_cnt_ref;
                                                    }
                                                }
                                                // printf("ljb %5d rank_cnt %5d rank_cnt_ref %5d\n",ljb,rank_cnt,rank_cnt_ref);
                                                // fflush(stdout);
                                                assert(rank_cnt==rank_cnt_ref);
                                            }
                                        }
                                    }
                                }
                                }
                                SUPERLU_FREE(ActiveFlag);
                                SUPERLU_FREE(ActiveFlagAll);
                                SUPERLU_FREE(ranks);
                                SUPERLU_FREE(SeedSTD_BC);
                                memTRS -= k*dword + grid->nprow*k*iword;  //acount for SeedSTD_BC, ActiveFlagAll

#if ( PROFlevel>=1 )
                                t = SuperLU_timer_() - t;
                                if ( !iam) printf(".. Construct Bcast tree for U: %.2f\t\n", t);
#endif

#if ( PROFlevel>=1 )
                                t = SuperLU_timer_();
#endif
                                /* construct the Reduce tree for U ... */
                                /* the following is used as reference */
                                nlb = CEILING( nsupers, grid->nprow );/* Number of local block rows */
                                if ( !(mod_bit = int32Malloc_dist(nlb)) )
                                    ABORT("Malloc fails for mod_bit[].");
                                if ( !(brecv = int32Malloc_dist(nlb)) )
                                    ABORT("Malloc fails for brecv[].");

                                for (k = 0; k < nlb; ++k) mod_bit[k] = 0;
                                for (k = 0; k < nsupers; ++k) {
                                    pr = PROW( k, grid );
                                    if ( myrow == pr ) {
                                        lib = LBi( k, grid );    /* local block number */
                                        kcol = PCOL( k, grid );
                                        if (mycol == kcol || bmod[lib] )
                                            mod_bit[lib] = 1;  /* contribution from off-diagonal and diagonal*/
                                    }
                                }
                                /* Every process receives the count, but it is only useful on the
                                   diagonal processes.  */
                                //gSoFa
                                MPI_Allreduce( mod_bit, brecv, nlb, mpi_int_t, MPI_SUM, grid->rscp.comm);
                                //~gSoFa


                                k = CEILING( nsupers, grid->nprow );/* Number of local block rows */
                                if ( !(URtree_ptr = (C_Tree*)SUPERLU_MALLOC(k * sizeof(C_Tree))) )
                                    ABORT("Malloc fails for URtree_ptr[].");
                                if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
                                    ABORT("Calloc fails for ActiveFlag[].");
                                if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
                                    ABORT("Malloc fails for ranks[].");

                                // if ( !(idxs = intCalloc_dist(nsupers)) )
                                // ABORT("Calloc fails for idxs[].");

                                // if ( !(nzrows = (int_t**)SUPERLU_MALLOC(nsupers * sizeof(int_t*))) )
                                // ABORT("Malloc fails for nzrows[].");

                                if ( !(SeedSTD_RD = (double*)SUPERLU_MALLOC(k * sizeof(double))) )
                                    ABORT("Malloc fails for SeedSTD_RD[].");

                                for (i=0;i<k;i++){
                                    SeedSTD_RD[i]=rand();
                                }

                                MPI_Allreduce(MPI_IN_PLACE,&SeedSTD_RD[0],k,MPI_DOUBLE,MPI_MAX,grid->rscp.comm);


                                // for (jb = 0; jb < nsupers; ++jb) { /* for each block column ... */
                                // fsupc = FstBlockC( jb );
                                // len=0;
                                // for (j = fsupc; j < FstBlockC( jb+1 ); ++j) {
                                // istart = xusub[j];
                                // /* NOTE: Only the first nonzero index of the segment
                                // is stored in usub[]. */
                                // len +=  xusub[j+1] - xusub[j];
                                // }

                                // idxs[jb] = len-1;

                                // if(len>0){
                                // if ( !(nzrows[jb] = intMalloc_dist(len)) )
                                // ABORT("Malloc fails for nzrows[jb]");

                                // fsupc = FstBlockC( jb );

                                // len=0;

                                // for (j = fsupc; j < FstBlockC( jb+1 ); ++j) {
                                // istart = xusub[j];
                                // /* NOTE: Only the first nonzero index of the segment
                                // is stored in usub[]. */
                                // for (i = istart; i < xusub[j+1]; ++i) {
                                // irow = usub[i]; /* First nonzero in the segment. */
                                // nzrows[jb][len]=irow;
                                // len++;
                                // }
                                // }
                                // quickSort(nzrows[jb],0,len-1,0);
                                // }
                                // else{
                                // nzrows[jb] = NULL;
                                // }
                                // }


                                for (lib = 0; lib <k ; ++lib) {
                                    // URtree_ptr[lib]=NULL;
                                    C_RdTree_Nullify(&URtree_ptr[lib]);
                                }


                                if ( !(ActiveFlagAll = intMalloc_dist(grid->npcol*k)) )
                                    ABORT("Calloc fails for ActiveFlagAll[].");
                                for (j=0;j<grid->npcol*k;++j)ActiveFlagAll[j]=3*nsupers;
                                memTRS += k*sizeof(C_Tree) + k*dword + grid->npcol*k*iword;  //acount for URtree_ptr, SeedSTD_RD, ActiveFlagAll

                                for (jb = 0; jb < nsupers; ++jb) { /* for each block column ... */
                                    fsupc = FstBlockC( jb );
                                    pc = PCOL( jb, grid );

                                    fsupc = FstBlockC( jb );
                                    for (j = fsupc; j < FstBlockC( jb+1 ); ++j) {
                                        istart = xusub_begin[j];
                                        /* NOTE: Only the first nonzero index of the segment
                                           is stored in usub[]. */
                                        for (i = istart; i < xusub_end[j]; ++i) {
                                            irow = usub[i]; /* First nonzero in the segment. */
                                            ib = BlockNum( irow );
#ifdef supernodal_Uskipp

                                            int_t i_supernode = Glu_persist->supno[irow];


                                            if (jb == i_supernode) // Only skip the rows in the diagonal block
                                            {
                                                if (SuperSize( i_supernode ) !=1)
                                                {
                                                    //Skipping the columns that are already included into the supernodes in the L structure
                                                    if (ib == jb) continue;
                                                }
                                            }

#endif
                                            pr = PROW( ib, grid );
                                            if ( myrow == pr ) { /* Block row ib in my process row */
                                                lib = LBi( ib, grid ); /* Local block number */
                                                ActiveFlagAll[pc+lib*grid->npcol]=SUPERLU_MIN(ActiveFlagAll[pc+lib*grid->npcol],jb);
                                            }
                                        }
                                    }

                                    pr = PROW( jb, grid );
                                    if ( myrow == pr ) { /* Block row ib in my process row */
                                        lib = LBi( jb, grid ); /* Local block number */
                                        ActiveFlagAll[pc+lib*grid->npcol]=SUPERLU_MIN(ActiveFlagAll[pc+lib*grid->npcol],jb);
                                    }
                                }

                                //gSoFa
                                MPI_Allreduce(MPI_IN_PLACE,ActiveFlagAll,grid->npcol*k,mpi_int_t,MPI_MIN,grid->rscp.comm);
                                //~gSoFa

                                // printf("Process: %d\n",iam);
                                // PrintInt10("ActiveFlagAll for U Reduction tree", grid->npcol*k, ActiveFlagAll);
                                // fflush(stdout);

                                for (lib=0;lib<k;++lib){
                                    ib = myrow+lib*grid->nprow;  /* not sure */
                                    if(ib<nsupers){
                                        pr = PROW( ib, grid );
                                        for (j=0;j<grid->npcol;++j)ActiveFlag[j]=ActiveFlagAll[j+lib*grid->npcol];;
                                        for (j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
                                        for (j=0;j<grid->npcol;++j)ranks[j]=-1;
                                        Root=-1;
                                        Iactive = 0;

                                        for (j=0;j<grid->npcol;++j){
                                            if(ActiveFlag[j]!=3*nsupers){
                                                jb = ActiveFlag[j];
                                                pc = PCOL( jb, grid );
                                                if(jb==ib)Root=pc;
                                                if(mycol==pc)Iactive=1;
                                            }
                                        }

                                        quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,0,2);

                                        if(Iactive==1){
                                            assert( Root>-1 );
                                            rank_cnt = 1;
                                            ranks[0]=Root;
                                            for (j = 0; j < grid->npcol; ++j){
                                                if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
                                                    ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
                                                    ++rank_cnt;
                                                }
                                            }
                                            if(rank_cnt>1){

                                                for (ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
                                                    ranks[ii] = PNUM( pr, ranks[ii], grid );

                                                // rseed=rand();
                                                // rseed=1.0;
                                                msgsize = SuperSize( ib );

                                                // if(ib==0){

                                                // URtree_ptr[lib] = RdTree_Create(grid->comm, ranks, rank_cnt, msgsize,SeedSTD_RD[lib],'d');
                                                // RdTree_SetTag(URtree_ptr[lib], RD_U,'d');
                                                C_RdTree_Create(&URtree_ptr[lib], grid->comm, ranks, rank_cnt, msgsize, 'd');
					                            URtree_ptr[lib].tag_=RD_U;
                                                // }

                                                // #if ( PRNTlevel>=1 )
                                                if(Root==mycol){
                                                    // printf("Partial Reduce Procs: %4d %4d %5d \n",iam, rank_cnt,brecv[lib]);
                                                    // fflush(stdout);
                                                    assert(rank_cnt==brecv[lib]);
                                                    // printf("Partial Reduce Procs: row%7d np%4d\n",ib,rank_cnt);
                                                    // printf("Partial Reduce Procs: %4d %4d: ",iam, rank_cnt);
                                                    // // for(j=0;j<rank_cnt;++j)printf("%4d",ranks[j]);
                                                    // printf("\n");
                                                }
                                                // #endif
                                            }
                                        }
                                    }
                                }
                                SUPERLU_FREE(mod_bit);
                                SUPERLU_FREE(brecv);


                                SUPERLU_FREE(ActiveFlag);
                                SUPERLU_FREE(ActiveFlagAll);
                                SUPERLU_FREE(ranks);
                                // SUPERLU_FREE(idxs);
                                SUPERLU_FREE(SeedSTD_RD);
                                // for(i=0;i<nsupers;++i){
                                // if(nzrows[i])SUPERLU_FREE(nzrows[i]);
                                // }
                                // SUPERLU_FREE(nzrows);

                                memTRS -= k*dword + grid->nprow*k*iword;  //acount for SeedSTD_RD, ActiveFlagAll

#if ( PROFlevel>=1 )
                                t = SuperLU_timer_() - t;
                                if ( !iam) printf(".. Construct Reduce tree for U: %.2f\t\n", t);
#endif

                                ////////////////////////////////////////////////////////


                                Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
    //                             	Llu->Lrowind_bc_dat = Lrowind_bc_dat;
	// Llu->Lrowind_bc_offset = Lrowind_bc_offset;
	// Llu->Lrowind_bc_cnt = Lrowind_bc_cnt;

                                Llu->Lindval_loc_bc_ptr = Lindval_loc_bc_ptr;
    //                             	Llu->Lindval_loc_bc_dat = Lindval_loc_bc_dat;
	// Llu->Lindval_loc_bc_offset = Lindval_loc_bc_offset;
	// Llu->Lindval_loc_bc_cnt = Lindval_loc_bc_cnt;
                                Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
    //                             	Llu->Lnzval_bc_dat = Lnzval_bc_dat;
	// Llu->Lnzval_bc_offset = Lnzval_bc_offset;
	// Llu->Lnzval_bc_cnt = Lnzval_bc_cnt;

                                Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
    //                                 Llu->Ufstnz_br_dat = Ufstnz_br_dat;  
    // Llu->Ufstnz_br_offset = Ufstnz_br_offset;  
    // Llu->Ufstnz_br_cnt = Ufstnz_br_cnt;  

                                Llu->Unzval_br_ptr = Unzval_br_ptr;
    //                             	Llu->Unzval_br_dat = Unzval_br_dat;
	// Llu->Unzval_br_offset = Unzval_br_offset;
	// Llu->Unzval_br_cnt = Unzval_br_cnt;

                                Llu->Unnz = Unnz;
                                Llu->ToRecv = ToRecv;
                                Llu->ToSendD = ToSendD;
                                Llu->ToSendR = ToSendR;
                                Llu->fmod = fmod;
                                Llu->fsendx_plist = fsendx_plist;
                                Llu->nfrecvx = nfrecvx;
                                Llu->nfsendx = nfsendx;
                                Llu->bmod = bmod;
                                Llu->bsendx_plist = bsendx_plist;
                                Llu->nbrecvx = nbrecvx;
                                Llu->nbsendx = nbsendx;
                                Llu->ilsum = ilsum;
                                Llu->ldalsum = ldaspa;

                                Llu->LRtree_ptr = LRtree_ptr;
                                Llu->LBtree_ptr = LBtree_ptr;
                                Llu->URtree_ptr = URtree_ptr;
                                Llu->UBtree_ptr = UBtree_ptr;
                                Llu->Linv_bc_ptr = Linv_bc_ptr;
    //                             	Llu->Linv_bc_dat = Linv_bc_dat;
	// Llu->Linv_bc_offset = Linv_bc_offset;
	// Llu->Linv_bc_cnt = Linv_bc_cnt;
                                Llu->Uinv_bc_ptr = Uinv_bc_ptr;
    //                             	Llu->Uinv_bc_dat = Uinv_bc_dat;
	// Llu->Uinv_bc_offset = Uinv_bc_offset;
	// Llu->Uinv_bc_cnt = Uinv_bc_cnt;	
                                Llu->Urbs = Urbs;
                                Llu->Ucb_indptr = Ucb_indptr;
    //                             	Llu->Ucb_inddat = Ucb_inddat;
	// Llu->Ucb_indoffset = Ucb_indoffset;
	// Llu->Ucb_indcnt = Ucb_indcnt;
                                Llu->Ucb_valptr = Ucb_valptr;
    //                             	Llu->Ucb_valdat = Ucb_valdat;
	// Llu->Ucb_valoffset = Ucb_valoffset;
	// Llu->Ucb_valcnt = Ucb_valcnt;


#ifdef GPU_ACC

	checkGPU(gpuMalloc( (void**)&Llu->d_xsup, (n+1) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_xsup, xsup, (n+1) * sizeof(int_t), gpuMemcpyHostToDevice));
	checkGPU(gpuMalloc( (void**)&Llu->d_LRtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_LBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_URtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree)));
	checkGPU(gpuMalloc( (void**)&Llu->d_UBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree)));	
	checkGPU(gpuMemcpy(Llu->d_LRtree_ptr, Llu->LRtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree), gpuMemcpyHostToDevice));	
	checkGPU(gpuMemcpy(Llu->d_LBtree_ptr, Llu->LBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree), gpuMemcpyHostToDevice));			
	checkGPU(gpuMemcpy(Llu->d_URtree_ptr, Llu->URtree_ptr, CEILING( nsupers, grid->nprow ) * sizeof(C_Tree), gpuMemcpyHostToDevice));	
	checkGPU(gpuMemcpy(Llu->d_UBtree_ptr, Llu->UBtree_ptr, CEILING( nsupers, grid->npcol ) * sizeof(C_Tree), gpuMemcpyHostToDevice));		
	checkGPU(gpuMalloc( (void**)&Llu->d_Lrowind_bc_dat, (Llu->Lrowind_bc_cnt) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_Lrowind_bc_dat, Llu->Lrowind_bc_dat, (Llu->Lrowind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));	
	checkGPU(gpuMalloc( (void**)&Llu->d_Lindval_loc_bc_dat, (Llu->Lindval_loc_bc_cnt) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_Lindval_loc_bc_dat, Llu->Lindval_loc_bc_dat, (Llu->Lindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));	
	checkGPU(gpuMalloc( (void**)&Llu->d_Lrowind_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lrowind_bc_offset, Llu->Lrowind_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));	
	checkGPU(gpuMalloc( (void**)&Llu->d_Lindval_loc_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lindval_loc_bc_offset, Llu->Lindval_loc_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));	
	checkGPU(gpuMalloc( (void**)&Llu->d_Lnzval_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Lnzval_bc_offset, Llu->Lnzval_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));	
	
	// some dummy allocation to avoid checking whether they are null pointers later
	checkGPU(gpuMalloc( (void**)&Llu->d_Ucolind_bc_dat, sizeof(int_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Ucolind_bc_offset, sizeof(int64_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Unzval_bc_dat, sizeof(double)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Unzval_bc_offset, sizeof(int64_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uindval_loc_bc_dat, sizeof(int_t)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uindval_loc_bc_offset, sizeof(int_t)));


	checkGPU(gpuMalloc( (void**)&Llu->d_Linv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Linv_bc_offset, Llu->Linv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));	
	checkGPU(gpuMalloc( (void**)&Llu->d_Uinv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int)));
	checkGPU(gpuMemcpy(Llu->d_Uinv_bc_offset, Llu->Uinv_bc_offset, CEILING( nsupers, grid->npcol ) * sizeof(long int), gpuMemcpyHostToDevice));		
	checkGPU(gpuMalloc( (void**)&Llu->d_ilsum, (CEILING( nsupers, grid->nprow )+1) * sizeof(int_t)));
	checkGPU(gpuMemcpy(Llu->d_ilsum, Llu->ilsum, (CEILING( nsupers, grid->nprow )+1) * sizeof(int_t), gpuMemcpyHostToDevice));


	/* gpuMemcpy for the following is performed in pxgssvx */
	checkGPU(gpuMalloc( (void**)&Llu->d_Lnzval_bc_dat, (Llu->Lnzval_bc_cnt) * sizeof(double)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Linv_bc_dat, (Llu->Linv_bc_cnt) * sizeof(double)));
	checkGPU(gpuMalloc( (void**)&Llu->d_Uinv_bc_dat, (Llu->Uinv_bc_cnt) * sizeof(double)));
	
#endif


                                int k3 = CEILING( nsupers, grid->nprow );/* Number of local block rows */
                                // for (i=0;i<k3;i++)
                                // {
                                // 	//  TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Llu->LRtree_ptr[i];

                                // 	printf("Reduction tree destination count:%d\n ",RdTree_GetDestCount(&LRtree_ptr[i],'d'));
                                // 	printf("Reduction tree message size:%d\n ",RdTree_GetMsgSize(&LRtree_ptr[lib],'d'));
                                // 	printf("Broadcast tree message size:%d\n ", BcTree_GetMsgSize(&LBtree_ptr[lib],'d'));
                                // 	printf("Reduction tree message size:%d\n ", RdTree_IsRoot(&LRtree_ptr[lib],'d'));


                                // 	//  RdTree_IsRoot(LRtree_ptr[lib],'d');

                                // }


#if ( PRNTlevel>=1 )
                                if ( !iam ) printf(".. # L blocks " IFMT "\t# U blocks " IFMT "\n",
                                        nLblocks, nUblocks);
#endif

                                SUPERLU_FREE(rb_marker);
                                SUPERLU_FREE(Urb_fstnz);
                                SUPERLU_FREE(Urb_length);
                                SUPERLU_FREE(Urb_indptr);
                                SUPERLU_FREE(Lrb_length);
                                SUPERLU_FREE(Lrb_number);
                                SUPERLU_FREE(Lrb_indptr);
                                SUPERLU_FREE(Lrb_valptr);
                                SUPERLU_FREE(dense);

                                /* Find the maximum buffer size. */
                                MPI_Allreduce(mybufmax, Llu->bufmax, NBUFFERS, mpi_int_t,
                                        MPI_MAX, grid->comm);

                                k = CEILING( nsupers, grid->nprow );/* Number of local block rows */
                                if ( !(Llu->mod_bit = int32Malloc_dist(k)) )
                                    ABORT("Malloc fails for mod_bit[].");

#if ( PROFlevel>=1 )
                                if ( !iam ) printf(".. 1st distribute time:\n "
                                        "\tL\t%.2f\n\tU\t%.2f\n"
                                        "\tu_blks %d\tnrbu %d\n--------\n",
                                        t_l, t_u, u_blks, nrbu);
#endif

                            } /* else fact != SamePattern_SameRowPerm */

                            if ( xa[A->ncol] > 0 ) { /* may not have any entries on this process. */
                                SUPERLU_FREE(asub);
                                SUPERLU_FREE(a);
                            }
                            SUPERLU_FREE(xa);

#if ( DEBUGlevel>=1 )
                            /* Memory allocated but not freed:
                               ilsum, fmod, fsendx_plist, bmod, bsendx_plist  */
                            CHECK_MALLOC(iam, "Exit pddistribute()");
#endif

                            return (mem_use+memTRS);

                        } /* PDDISTRIBUTE */
