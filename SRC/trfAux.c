/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Auxiliary routines to support 3D algorithms
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Oak Ridge National Lab
 * May 12, 2021
 * </pre>
 */

#include "superlu_ddefs.h"

#if 0
#include "pdgstrf3d.h"
#include "trfAux.h"
#endif


int_t getslu25D_enabled()
{
    if ( getenv("SLU25D") != NULL)
    {
        return atoi(getenv("SLU25D"));
    }
    else
    {
        return 0;
    }
}

int getNsupers(int n, Glu_persist_t *Glu_persist)
{
    int nsupers = Glu_persist->supno[n - 1] + 1;
    return nsupers;
}

int set_tag_ub()
{
    void *attr_val;
    int flag;
    MPI_Comm_get_attr (MPI_COMM_WORLD, MPI_TAG_UB, &attr_val, &flag);
    if (!flag)
    {
        fprintf (stderr, "Could not get TAG_UB\n");
        exit(-1);
    }
    return ( *(int_t *) attr_val );
}

int getNumThreads(int iam)
{
    int num_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel default(shared)
    {
        #pragma omp master
        {
            num_threads = omp_get_num_threads ();

        }
    }
#endif

#if ( PRNTlevel>=1 )    
    if (!iam) printf(".. Starting with %d openMP threads \n", num_threads );
#endif    
    return num_threads;
}


#if 0  //**** Sherry: following two routines are old, the new ones are in util.c
int_t num_full_cols_U(int_t kk,  int_t **Ufstnz_br_ptr, int_t *xsup,
                      gridinfo_t *grid, int_t *perm_u)
{
    int_t lk = LBi (kk, grid);
    int_t *usub = Ufstnz_br_ptr[lk];

    if (usub == NULL)
    {
        /* code */
        return 0;
    }
    int_t iukp = BR_HEADER;   /* Skip header; Pointer to index[] of U(k,:) */
    int_t rukp = 0;           /* Pointer to nzval[] of U(k,:) */
    int_t nub = usub[0];      /* Number of blocks in the block row U(k,:) */

    int_t klst = FstBlockC (kk + 1);
    int_t iukp0 = iukp;
    int_t rukp0 = rukp;
    int_t jb, ljb;
    int_t nsupc;
    int_t temp_ncols = 0;
    int_t segsize;

    temp_ncols = 0;

    for (int_t j = 0; j < nub; ++j)
    {
        arrive_at_ublock(
            j, &iukp, &rukp, &jb, &ljb, &nsupc,
            iukp0, rukp0, usub, perm_u, xsup, grid
        );

        for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
        {
            segsize = klst - usub[jj];
            if ( segsize ) ++temp_ncols;
        }
    }
    return temp_ncols;
}

// Sherry: this is old; new version is in util.c 
int_t estimate_bigu_size( int_t nsupers, int_t ldt, int_t**Ufstnz_br_ptr,
                          Glu_persist_t *Glu_persist,  gridinfo_t* grid, int_t* perm_u)
{

    int_t iam = grid->iam;

    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);

    int_t* xsup = Glu_persist->xsup;

    int ncols = 0;
    int_t ldu = 0;

    /*initilize perm_u*/
    for (int i = 0; i < nsupers; ++i)
    {
        perm_u[i] = i;
    }

    for (int lk = myrow; lk < nsupers; lk += Pr )
    {
        ncols = SUPERLU_MAX(ncols, num_full_cols_U(lk, Ufstnz_br_ptr,
						   xsup, grid, perm_u, &ldu));
    }

    int_t max_ncols = 0;

    MPI_Allreduce(&ncols, &max_ncols, 1, mpi_int_t, MPI_MAX, grid->cscp.comm);

    printf("max_ncols =%d, bigu_size=%ld\n", (int) max_ncols, (long long) ldt * max_ncols);
    return ldt * max_ncols;
} /* old estimate_bigu_size. New one is in util.c */
#endif /**** end old ones ****/

int_t getBigUSize(superlu_dist_options_t *options,
		  int_t nsupers, gridinfo_t *grid, int_t **Lrowind_bc_ptr)
//LUstruct_t *LUstruct)
{

    int_t Pr = grid->nprow;
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);


    /* Following circuit is for finding maximum block size */
    int local_max_row_size = 0;
    int max_row_size;

    for (int_t i = 0; i < nsupers; ++i)
    {
        int_t tpc = PCOL (i, grid);
        if (mycol == tpc)
        {
            int_t lk = LBj (i, grid);
            //int_t* lsub = LUstruct->Llu->Lrowind_bc_ptr[lk];
            int_t* lsub = Lrowind_bc_ptr[lk];
            if (lsub != NULL)
            {
                local_max_row_size = SUPERLU_MAX (local_max_row_size, lsub[1]);
            }
        }

    }

    /* Max row size is global reduction of within A row */
    MPI_Allreduce (&local_max_row_size, &max_row_size, 1, MPI_INT, MPI_MAX,
                   (grid->rscp.comm));


    // int_t Threads_per_process = get_thread_per_process ();

    /*Buffer size is max of of look ahead window*/


    int_t bigu_size =
        8 * sp_ienv_dist (3, options) * (max_row_size) * SUPERLU_MAX(Pr / Pc, 1);

    return bigu_size;
}

int_t* getFactPerm(int_t nsupers)
{
    int_t* perm = INT_T_ALLOC(nsupers);

    for (int_t i = 0; i < nsupers; ++i)
    {
        /* code */
        perm[i] = i;
    }

    return perm;
}

int_t* getFactIperm(int_t* perm, int_t nsupers)
{
    int_t* iperm = INT_T_ALLOC(nsupers);

    for (int_t i = 0; i < nsupers; ++i)
    {
        /* code */
        iperm[perm[i]] = i;
    }

    return iperm;
}

int_t* getPerm_c_supno(int_t nsupers, superlu_dist_options_t *options,
		       int_t *etree, Glu_persist_t *Glu_persist,
		       int_t** Lrowind_bc_ptr, int_t** Ufstnz_br_ptr,
		       gridinfo_t *grid)

{
    /*I do not understand the following code in detail,
    I have just written a wrapper around it*/

    int_t* perm_c_supno;
    //Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    //LocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);

    int_t *etree_supno_l, *etree_supno, *blocks, *blockr, *Ublock, *Urows, *Lblock, *Lrows,
          *sf_block, *sf_block_l, *nnodes_l, *nnodes_u, *edag_supno_l, *recvbuf,
          **edag_supno;
    int_t  i, ib,  jb,
           lb,
           nlb, il, iu;
    int  ncb, nrb, p, pr, pc, nblocks;
    int_t *index;
    int nnodes, *sendcnts, *sdispls, *recvcnts, *rdispls, *srows, *rrows;
    int_t j, k, krow,  yourcol;
    etree_node *head, *tail, *ptr;
    int *num_child;
    nblocks = 0;
    ncb = nsupers / Pc;
    nrb = nsupers / Pr;
    /* ================================================== *
     * static scheduling of j-th step of LU-factorization *
     * ================================================== */
    if ( options->lookahead_etree == YES && /* use e-tree of symmetrized matrix, and      */
            (options->ParSymbFact == NO ||      /* 1) symmetric fact with serial symbolic, or */
             (options->SymPattern == YES &&      /* 2) symmetric pattern, and                  */
              options->RowPerm == NOROWPERM)) )  /*    no rowperm to destroy the symmetry      */
    {
        /* if symmetric pattern or using e-tree of |A^T|+|A|,
        then we can use a simple tree structure for static schduling */

        if ( options->ParSymbFact == NO )
        {
            /* Use the etree computed from serial symb. fact., and turn it
            into supernodal tree.  */
            //int_t *etree = LUstruct->etree;
#if ( PRNTlevel>=1 )
            if ( grid->iam == 0 ) printf( " === using column e-tree ===\n" );
#endif

            /* look for the first off-diagonal blocks */
            etree_supno = SUPERLU_MALLOC( nsupers * sizeof(int_t) );
            for ( i = 0; i < nsupers; i++ ) etree_supno[i] = nsupers;
            for ( j = 0, lb = 0; lb < nsupers; lb++ )
            {
                for ( k = 0; k < SuperSize(lb); k++ )
                {
                    jb = Glu_persist->supno[etree[j + k]];
                    if ( jb != lb ) etree_supno[lb] = SUPERLU_MIN( etree_supno[lb], jb );
                }
                j += SuperSize(lb);
            }
        }
        else     /* ParSymbFACT==YES and SymPattern==YES  and RowPerm == NOROWPERM */
        {
            /* Compute an "etree" based on struct(L),
            assuming struct(U) = struct(L').   */
#if ( PRNTlevel>=1 )
            if ( grid->iam == 0 ) printf( " === using supernodal e-tree ===\n" );
#endif

            /* find the first block in each supernodal-column of local L-factor */
            etree_supno_l = SUPERLU_MALLOC( nsupers * sizeof(int_t) );
            for ( i = 0; i < nsupers; i++ ) etree_supno_l[i] = nsupers;
            for ( lb = 0; lb < ncb; lb++ )
            {
                jb = lb * grid->npcol + mycol;
                //index = Llu->Lrowind_bc_ptr[lb];
                index = Lrowind_bc_ptr[lb];
                if ( index )   /* Not an empty column */
                {
                    i = index[0];
                    k = BC_HEADER;
                    krow = PROW( jb, grid );
                    if ( krow == myrow )  /* skip the diagonal block */
                    {
                        k += LB_DESCRIPTOR + index[k + 1];
                        i--;
                    }
                    if ( i > 0 )
                    {
                        etree_supno_l[jb] = index[k];
                        k += LB_DESCRIPTOR + index[k + 1];
                        i --;
                    }

                    for ( j = 0; j < i; j++ )
                    {
                        etree_supno_l[jb] = SUPERLU_MIN( etree_supno_l[jb], index[k] );
                        k += LB_DESCRIPTOR + index[k + 1];
                    }
                }
            }
            if ( mycol < nsupers % grid->npcol )
            {
                jb = ncb * grid->npcol + mycol;
                //index = Llu->Lrowind_bc_ptr[ncb];
                index = Lrowind_bc_ptr[ncb];
                if ( index )   /* Not an empty column */
                {
                    i = index[0];
                    k = BC_HEADER;
                    krow = PROW( jb, grid );
                    if ( krow == myrow )  /* skip the diagonal block */
		    {
                        k += LB_DESCRIPTOR + index[k + 1];
                        i--;
                    }
                    if ( i > 0 )
                    {
                        etree_supno_l[jb] = index[k];
                        k += LB_DESCRIPTOR + index[k + 1];
                        i --;
                    }
                    for ( j = 0; j < i; j++ )
                    {
                        etree_supno_l[jb] = SUPERLU_MIN( etree_supno_l[jb], index[k] );
                        k += LB_DESCRIPTOR + index[k + 1];
                    }
                }
            }

            /* form global e-tree */
            etree_supno = SUPERLU_MALLOC( nsupers * sizeof(int_t) );
            MPI_Allreduce( etree_supno_l, etree_supno, nsupers, mpi_int_t, MPI_MIN, grid->comm );
            SUPERLU_FREE(etree_supno_l);
        }

        /* initialize the num of child for each node */
        num_child = SUPERLU_MALLOC( nsupers * sizeof(int_t) );
        for ( i = 0; i < nsupers; i++ ) num_child[i] = 0;
        for ( i = 0; i < nsupers; i++ ) if ( etree_supno[i] != nsupers ) num_child[etree_supno[i]] ++;

        /* push initial leaves to the fifo queue */
        nnodes = 0;
        for ( i = 0; i < nsupers; i++ )
        {
            if ( num_child[i] == 0 )
            {
                ptr = SUPERLU_MALLOC( sizeof(etree_node) );
                ptr->id = i;
                ptr->next = NULL;
                /*printf( " == push leaf %d (%d) ==\n",i,nnodes );*/
                nnodes ++;

                if ( nnodes == 1 )
                {
                    head = ptr;
                    tail = ptr;
                }
                else
                {
                    tail->next = ptr;
                    tail = ptr;
                }
            }
        }

        /* process fifo queue, and compute the ordering */
        i = 0;
        perm_c_supno  = SUPERLU_MALLOC( nsupers * sizeof(int_t) );
        while ( nnodes > 0 )
        {
            ptr = head;  j = ptr->id;
            head = ptr->next;
            perm_c_supno[i] = j;
            SUPERLU_FREE(ptr);
            i++; nnodes --;

            if ( etree_supno[j] != nsupers )
            {
                num_child[etree_supno[j]] --;
                if ( num_child[etree_supno[j]] == 0 )
                {
                    nnodes ++;

                    ptr = SUPERLU_MALLOC( sizeof(etree_node) );
                    ptr->id = etree_supno[j];
                    ptr->next = NULL;

                    /*printf( "=== push %d ===\n",ptr->id );*/
                    if ( nnodes == 1 )
                    {
                        head = ptr;
                        tail = ptr;
                    }
                    else
                    {
                        tail->next = ptr;
                        tail = ptr;
                    }
                }
            }
            /*printf( "\n" );*/
        }
        SUPERLU_FREE(num_child);
        SUPERLU_FREE(etree_supno);

    }
    else     /* Unsymmetric pattern */
    {
        /* Need to process both L- and U-factors, use the symmetrically
        pruned graph of L & U instead of tree (very naive implementation) */
        int nrbp1 = nrb + 1;

        /* allocate some workspace */
        if ( !(sendcnts = SUPERLU_MALLOC( (4 + 2 * nrbp1) * Pr * Pc * sizeof(int))) )
            ABORT("Malloc fails for sendcnts[].");
        sdispls  = &sendcnts[Pr * Pc];
        recvcnts = &sdispls [Pr * Pc];
        rdispls  = &recvcnts[Pr * Pc];
        srows    = &rdispls [Pr * Pc];
        rrows    = &srows   [Pr * Pc * nrbp1];

        myrow = MYROW( iam, grid );
#if ( PRNTlevel>=1 )
        if ( grid->iam == 0 ) printf( " === using DAG ===\n" );
#endif

        /* send supno block of local U-factor to a processor *
        * who owns the corresponding block of L-factor      */

        /* srows   : # of block to send to a processor from each supno row */
        /* sendcnts: total # of blocks to send to a processor              */
        for (p = 0; p < Pr * Pc * nrbp1; p++) srows[p] = 0;
        for (p = 0; p < Pr * Pc; p++ ) sendcnts[p] = 0;

        /* sending blocks of U-factors corresponding to L-factors */
        /* count the number of blocks to send */
        for (lb = 0; lb < nrb; ++lb)
        {
            jb = lb * Pr + myrow;
            pc = jb % Pc;
            //index = Llu->Ufstnz_br_ptr[lb];
            index = Ufstnz_br_ptr[lb];

            if ( index )   /* Not an empty row */
            {
                k = BR_HEADER;
                nblocks += index[0];
                for (j = 0; j < index[0]; ++j)
                {
                    ib = index[k];
                    pr = ib % Pr;
                    p  = pr * Pc + pc;
                    sendcnts[p] ++;
                    srows[p * nrbp1 + lb] ++;

                    k += UB_DESCRIPTOR + SuperSize( index[k] );
                }
            }
        }
        if ( myrow < nsupers % grid->nprow )
        {
            jb = nrb * Pr + myrow;
            pc = jb % Pc;
            //index = Llu->Ufstnz_br_ptr[nrb];
	    index = Ufstnz_br_ptr[nrb];

            if ( index )   /* Not an empty row */
            {
                k = BR_HEADER;
                nblocks += index[0];
                for (j = 0; j < index[0]; ++j)
                {
                    ib = index[k];
                    pr = ib % Pr;
                    p  = pr * Pc + pc;
                    sendcnts[p] ++;
                    srows[p * nrbp1 + nrb] ++;
                    k += UB_DESCRIPTOR + SuperSize( index[k] );
                }
            }
        }

        /* insert blocks to send */
        sdispls[0] = 0;
        for ( p = 1; p < Pr * Pc; p++ ) sdispls[p] = sdispls[p - 1] + sendcnts[p - 1];
        if ( !(blocks = intMalloc_dist( nblocks )) ) ABORT("Malloc fails for blocks[].");
        for (lb = 0; lb < nrb; ++lb)
        {
            jb = lb * Pr + myrow;
            pc = jb % Pc;
            //index = Llu->Ufstnz_br_ptr[lb];
            index = Ufstnz_br_ptr[lb];

            if ( index )   /* Not an empty row */
            {
                k = BR_HEADER;
                for (j = 0; j < index[0]; ++j)
                {
                    ib = index[k];
                    pr = ib % Pr;
                    p  = pr * Pc + pc;
                    blocks[sdispls[p]] = ib;
                    sdispls[p] ++;

                    k += UB_DESCRIPTOR + SuperSize( index[k] );
                }
            }
        }
        if ( myrow < nsupers % grid->nprow )
        {
            jb = nrb * Pr + myrow;
            pc = jb % Pc;
            //index = Llu->Ufstnz_br_ptr[nrb];
            index = Ufstnz_br_ptr[nrb];

            if ( index )   /* Not an empty row */
            {
                k = BR_HEADER;
                for (j = 0; j < index[0]; ++j)
                {
                    ib = index[k];
                    pr = ib % Pr;
                    p  = pr * Pc + pc;
                    blocks[sdispls[p]] = ib;
                    sdispls[p] ++;

                    k += UB_DESCRIPTOR + SuperSize( index[k] );
                }
            }
        }

        /* communication */
        MPI_Alltoall( sendcnts,  1, MPI_INT, recvcnts,   1, MPI_INT, grid->comm );
        MPI_Alltoall( srows, nrbp1, MPI_INT, rrows,  nrbp1, MPI_INT, grid->comm );

        nblocks = recvcnts[0];
        rdispls[0] = sdispls[0] = 0;
        for ( p = 1; p < Pr * Pc; p++ )
        {
            rdispls[p] = rdispls[p - 1] + recvcnts[p - 1];
            sdispls[p] = sdispls[p - 1] + sendcnts[p - 1];
            nblocks += recvcnts[p];
        }

        if ( !(blockr =  intMalloc_dist( nblocks )) )  ABORT("Malloc fails for blockr[].");
        MPI_Alltoallv( blocks, sendcnts, sdispls, mpi_int_t, blockr, recvcnts, rdispls, mpi_int_t, grid->comm );
        SUPERLU_FREE( blocks );

        /* store the received U-blocks by rows */
        nlb = nsupers / Pc;
        if ( !(Ublock = intMalloc_dist( nblocks )) )  ABORT("Malloc fails for Ublock[].");
        if ( !(Urows  = intMalloc_dist( 1 + nlb )) )  ABORT("Malloc fails for Urows[].");
        k = 0;
        for (jb = 0; jb < nlb; jb++ )
        {
            j = jb * Pc + mycol;
            pr = j % Pr;
            lb = j / Pr;
            Urows[jb] = 0;

            for ( pc = 0; pc < Pc; pc++ )
            {
                p = pr * Pc + pc; /* the processor owning this block of U-factor */

                for ( i = rdispls[p]; i < rdispls[p] + rrows[p * nrbp1 + lb]; i++)
                {
                    Ublock[k] = blockr[i];
                    k++; Urows[jb] ++;
                }
                rdispls[p] += rrows[p * nrbp1 + lb];
            }
            /* sort by the column indices to make things easier for later on */

#ifdef ISORT
            isort1( Urows[jb], &(Ublock[k - Urows[jb]]) );
#else
            qsort( &(Ublock[k - Urows[jb]]), (size_t)(Urows[jb]), sizeof(int_t), &superlu_sort_perm );
#endif
        }
        if ( mycol < nsupers % grid->npcol )
        {
            j = nlb * Pc + mycol;
            pr = j % Pr;
            lb = j / Pr;
            Urows[nlb] = 0;

            for ( pc = 0; pc < Pc; pc++ )
            {
                p = pr * Pc + pc;
                for ( i = rdispls[p]; i < rdispls[p] + rrows[p * nrbp1 + lb]; i++)
                {
                    Ublock[k] = blockr[i];
                    k++; Urows[nlb] ++;
                }
                rdispls[p] += rrows[p * nrb + lb];
            }
#ifdef ISORT
            isort1( Urows[nlb], &(Ublock[k - Urows[nlb]]) );
#else
            qsort( &(Ublock[k - Urows[nlb]]), (size_t)(Urows[nlb]), sizeof(int_t), &superlu_sort_perm );
#endif
        }
        SUPERLU_FREE( blockr );

        /* sort the block in L-factor */
        nblocks = 0;
        for ( lb = 0; lb < ncb; lb++ )
        {
            jb = lb * Pc + mycol;
            //index = Llu->Lrowind_bc_ptr[lb];
            index = Lrowind_bc_ptr[lb];
            if ( index )   /* Not an empty column */
            {
                nblocks += index[0];
            }
        }
        if ( mycol < nsupers % grid->npcol )
        {
            jb = ncb * Pc + mycol;
            //index = Llu->Lrowind_bc_ptr[ncb];
            index = Lrowind_bc_ptr[ncb];
            if ( index )   /* Not an empty column */
            {
                nblocks += index[0];
            }
        }

        if ( !(Lblock = intMalloc_dist( nblocks )) ) ABORT("Malloc fails for Lblock[].");
        if ( !(Lrows  = intMalloc_dist( 1 + ncb )) ) ABORT("Malloc fails for Lrows[].");
        for ( lb = 0; lb <= ncb; lb++ ) Lrows[lb] = 0;
        nblocks = 0;
        for ( lb = 0; lb < ncb; lb++ )
        {
            Lrows[lb] = 0;

            jb = lb * Pc + mycol;
            //index = Llu->Lrowind_bc_ptr[lb];
            index = Lrowind_bc_ptr[lb];
            if ( index )   /* Not an empty column */
            {
                i = index[0];
                k = BC_HEADER;
                krow = PROW( jb, grid );
                if ( krow == myrow )  /* skip the diagonal block */
                {
                    k += LB_DESCRIPTOR + index[k + 1];
                    i--;
                }

                for ( j = 0; j < i; j++ )
                {
                    Lblock[nblocks] = index[k];
                    Lrows[lb] ++;
                    nblocks++;

                    k += LB_DESCRIPTOR + index[k + 1];
                }
            }
#ifdef ISORT
            isort1( Lrows[lb], &(Lblock[nblocks - Lrows[lb]]) );
#else
            qsort( &(Lblock[nblocks - Lrows[lb]]), (size_t)(Lrows[lb]), sizeof(int_t), &superlu_sort_perm );
#endif
        }
        if ( mycol < nsupers % grid->npcol )
        {
            Lrows[ncb] = 0;
            jb = ncb * Pc + mycol;
            //index = Llu->Lrowind_bc_ptr[ncb];
            index = Lrowind_bc_ptr[ncb];
            if ( index )   /* Not an empty column */
            {
                i = index[0];
                k = BC_HEADER;
                krow = PROW( jb, grid );
                if ( krow == myrow )  /* skip the diagonal block */
                {
                    k += LB_DESCRIPTOR + index[k + 1];
                    i--;
                }
                for ( j = 0; j < i; j++ )
                {
                    Lblock[nblocks] = index[k];
                    Lrows[ncb] ++;
                    nblocks++;
                    k += LB_DESCRIPTOR + index[k + 1];
                }
#ifdef ISORT
                isort1( Lrows[ncb], &(Lblock[nblocks - Lrows[ncb]]) );
#else
                qsort( &(Lblock[nblocks - Lrows[ncb]]), (size_t)(Lrows[ncb]), sizeof(int_t), &superlu_sort_perm );
#endif
            }
        }

        /* look for the first local symmetric nonzero block match */
        if ( !(sf_block   = intMalloc_dist( nsupers )) )
            ABORT("Malloc fails for sf_block[].");
        if ( !(sf_block_l = intMalloc_dist( nsupers )) )
            ABORT("Malloc fails for sf_block_l[].");
        for ( lb = 0; lb < nsupers; lb++ ) sf_block_l[lb] = nsupers;
        i = 0; j = 0;
        for ( jb = 0; jb < nlb; jb++ )
        {
            if ( Urows[jb] > 0 )
            {
                ib = i + Urows[jb];
                lb = jb * Pc + mycol;
                for ( k = 0; k < Lrows[jb]; k++ )
                {
                    while ( Ublock[i] < Lblock[j] && i + 1 < ib ) i++;

                    if ( Ublock[i] == Lblock[j] )
                    {
                        sf_block_l[lb] = Lblock[j];
                        j += (Lrows[jb] - k);
                        k = Lrows[jb];
                    }
                    else
                    {
                        j++;
                    }
                }
                i = ib;
            }
            else
            {
                j += Lrows[jb];
            }
        }
        if ( mycol < nsupers % grid->npcol )
        {
            if ( Urows[nlb] > 0 )
            {
                ib = i + Urows[nlb];
                lb = nlb * Pc + mycol;
                for ( k = 0; k < Lrows[nlb]; k++ )
                {
                    while ( Ublock[i] < Lblock[j] && i + 1 < ib ) i++;

                    if ( Ublock[i] == Lblock[j] )
                    {
                        sf_block_l[lb] = Lblock[j];
                        j += (Lrows[nlb] - k);
                        k = Lrows[nlb];
                    }
                    else
                    {
                        j++;
                    }
                }
                i = ib;
            }
            else
            {
                j += Lrows[nlb];
            }
        }
        /* compute the first global symmetric matchs */
        MPI_Allreduce( sf_block_l, sf_block, nsupers, mpi_int_t, MPI_MIN, grid->comm );
        SUPERLU_FREE( sf_block_l );

        /* count number of nodes in DAG (i.e., the number of blocks on and above the first match) */
        if ( !(nnodes_l = intMalloc_dist( nsupers )) )
            ABORT("Malloc fails for nnodes_l[].");
        if ( !(nnodes_u = intMalloc_dist( nsupers )) )
            ABORT("Malloc fails for nnodes_u[].");
        for ( lb = 0; lb < nsupers; lb++ ) nnodes_l[lb] = 0;
        for ( lb = 0; lb < nsupers; lb++ ) nnodes_u[lb] = 0;

        nblocks = 0;
        /* from U-factor */
        for (i = 0, jb = 0; jb < nlb; jb++ )
        {
            lb = jb * Pc + mycol;
            ib = i + Urows[jb];
            while ( i < ib )
            {
                if ( Ublock[i] <= sf_block[lb] )
                {
                    nnodes_u[lb] ++;
                    i++; nblocks++;
                }
                else     /* get out*/
                {
                    i = ib;
                }
            }
            i = ib;
        }
        if ( mycol < nsupers % grid->npcol )
        {
            lb = nlb * Pc + mycol;
            ib = i + Urows[nlb];
            while ( i < ib )
            {
                if ( Ublock[i] <= sf_block[lb] )
                {
                    nnodes_u[lb] ++;
                    i++; nblocks++;
                }
                else     /* get out*/
                {
                    i = ib;
                }
            }
            i = ib;
        }

        /* from L-factor */
        for (i = 0, jb = 0; jb < nlb; jb++ )
        {
            lb = jb * Pc + mycol;
            ib = i + Lrows[jb];
            while ( i < ib )
            {
                if ( Lblock[i] < sf_block[lb] )
                {
                    nnodes_l[lb] ++;
                    i++; nblocks++;
                }
                else
                {
                    i = ib;
                }
            }
            i = ib;
        }
        if ( mycol < nsupers % grid->npcol )
        {
            lb = nlb * Pc + mycol;
            ib = i + Lrows[nlb];
            while ( i < ib )
            {
                if ( Lblock[i] < sf_block[lb] )
                {
                    nnodes_l[lb] ++;
                    i++; nblocks++;
                }
                else
                {
                    i = ib;
                }
            }
            i = ib;
        }

#ifdef USE_ALLGATHER
        /* insert local nodes in DAG */
        if ( !(edag_supno_l = intMalloc_dist( nsupers + nblocks )) )
            ABORT("Malloc fails for edag_supno_l[].");
        iu = il = nblocks = 0;
        for ( lb = 0; lb < nsupers; lb++ )
        {
            j  = lb / Pc;
            pc = lb % Pc;

            edag_supno_l[nblocks] = nnodes_l[lb] + nnodes_u[lb]; nblocks ++;
            if ( mycol == pc )
            {
                /* from U-factor */
                ib = iu + Urows[j];
                for ( jb = 0; jb < nnodes_u[lb]; jb++ )
                {
                    edag_supno_l[nblocks] = Ublock[iu];
                    iu++; nblocks++;
                }
                iu = ib;

                /* from L-factor */
                ib = il + Lrows[j];
                for ( jb = 0; jb < nnodes_l[lb]; jb++ )
                {
                    edag_supno_l[nblocks] = Lblock[il];
                    il++; nblocks++;
                }
                il = ib;
            }
        }
        SUPERLU_FREE( nnodes_u );

        /* form global DAG on each processor */
        MPI_Allgather( &nblocks, 1, MPI_INT, recvcnts, 1, MPI_INT, grid->comm );
        nblocks = recvcnts[0];
        rdispls[0] = 0;
        for ( lb = 1; lb < Pc * Pr; lb++ )
        {
            rdispls[lb] = nblocks;
            nblocks += recvcnts[lb];
        }
        if ( !(recvbuf = intMalloc_dist( nblocks )) )
            ABORT("Malloc fails for recvbuf[].");
        MPI_Allgatherv( edag_supno_l, recvcnts[iam], mpi_int_t,
                        recvbuf, recvcnts, rdispls, mpi_int_t, grid->comm );
        SUPERLU_FREE(edag_supno_l);

        if ( !(edag_supno = SUPERLU_MALLOC( nsupers * sizeof(int_t*) )) )
            ABORT("Malloc fails for edag_supno[].");
        k = 0;
        for ( lb = 0; lb < nsupers; lb++ ) nnodes_l[lb] = 0;
        for ( p = 0; p < Pc * Pr; p++ )
        {
            for ( lb = 0; lb < nsupers; lb++ )
            {
                nnodes_l[lb] += recvbuf[k];
                k += (1 + recvbuf[k]);
            }
        }
        for ( lb = 0; lb < nsupers; lb++ )
        {
            if ( nnodes_l[lb] > 0 )
                if ( !(edag_supno[lb] = intMalloc_dist( nnodes_l[lb] )) )
                    ABORT("Malloc fails for edag_supno[lb].");
            nnodes_l[lb] = 0;
        }
        k = 0;
        for ( p = 0; p < Pc * Pr; p++ )
        {
            for ( lb = 0; lb < nsupers; lb++ )
            {
                jb = k + recvbuf[k] + 1;
                k ++;
                for ( ; k < jb; k++ )
                {
                    edag_supno[lb][nnodes_l[lb]] = recvbuf[k];
                    nnodes_l[lb] ++;
                }
            }
        }
        SUPERLU_FREE(recvbuf);
#else
        int nlsupers = nsupers / Pc;
        if ( mycol < nsupers % Pc ) nlsupers ++;

        /* insert local nodes in DAG */
        if ( !(edag_supno_l = intMalloc_dist( nlsupers + nblocks )) )
            ABORT("Malloc fails for edag_supno_l[].");
        iu = il = nblocks = 0;
        for ( lb = 0; lb < nsupers; lb++ )
        {
            j  = lb / Pc;
            pc = lb % Pc;
            if ( mycol == pc )
            {
                edag_supno_l[nblocks] = nnodes_l[lb] + nnodes_u[lb]; nblocks ++;
                /* from U-factor */
                ib = iu + Urows[j];
                for ( jb = 0; jb < nnodes_u[lb]; jb++ )
                {
                    edag_supno_l[nblocks] = Ublock[iu];
                    iu++; nblocks++;
                }
                iu = ib;

                /* from L-factor */
                ib = il + Lrows[j];
                for ( jb = 0; jb < nnodes_l[lb]; jb++ )
                {
                    edag_supno_l[nblocks] = Lblock[il];
                    il++; nblocks++;
                }
                il = ib;
            }
            else if ( nnodes_l[lb] + nnodes_u[lb] != 0 )
		printf( " # %d: nnodes[%d]=%d+%d\n", grid->iam, 
			(int) lb, (int) nnodes_l[lb], (int) nnodes_u[lb] );
        }
        SUPERLU_FREE( nnodes_u );
        /* form global DAG on each processor */
        MPI_Allgather( &nblocks, 1, MPI_INT, recvcnts, 1, MPI_INT, grid->comm );
        nblocks = recvcnts[0];
        rdispls[0] = 0;
        for ( lb = 1; lb < Pc * Pr; lb++ )
        {
            rdispls[lb] = nblocks;
            nblocks += recvcnts[lb];
        }
        if ( !(recvbuf = intMalloc_dist( nblocks )) )
            ABORT("Malloc fails for recvbuf[].");

        MPI_Allgatherv( edag_supno_l, recvcnts[iam], mpi_int_t,
                        recvbuf, recvcnts, rdispls, mpi_int_t, grid->comm );
        SUPERLU_FREE(edag_supno_l);

        if ( !(edag_supno = SUPERLU_MALLOC( nsupers * sizeof(int_t*) )) )
            ABORT("Malloc fails for edag_supno[].");
        k = 0;
        for ( lb = 0; lb < nsupers; lb++ ) nnodes_l[lb] = 0;
        for ( p = 0; p < Pc * Pr; p++ )
        {
            yourcol = MYCOL( p, grid );

            for ( lb = 0; lb < nsupers; lb++ )
            {
                j  = lb / Pc;
                pc = lb % Pc;
                if ( yourcol == pc )
                {
                    nnodes_l[lb] += recvbuf[k];
                    k += (1 + recvbuf[k]);
                }
            }
        }
        for ( lb = 0; lb < nsupers; lb++ )
        {
            if ( nnodes_l[lb] > 0 )
                if ( !(edag_supno[lb] = intMalloc_dist( nnodes_l[lb] )) )
                    ABORT("Malloc fails for edag_supno[lb].");
            nnodes_l[lb] = 0;
        }
        k = 0;
        for ( p = 0; p < Pc * Pr; p++ )
        {
            yourcol = MYCOL( p, grid );

            for ( lb = 0; lb < nsupers; lb++ )
            {
                j  = lb / Pc;
                pc = lb % Pc;
                if ( yourcol == pc )
                {
                    jb = k + recvbuf[k] + 1;
                    k ++;
                    for ( ; k < jb; k++ )
                    {
                        edag_supno[lb][nnodes_l[lb]] = recvbuf[k];
                        nnodes_l[lb] ++;
                    }
                }
            }
        }
        SUPERLU_FREE(recvbuf);
#endif

        /* initialize the num of child for each node */
        num_child = SUPERLU_MALLOC( nsupers * sizeof(int_t) );
        for ( i = 0; i < nsupers; i++ ) num_child[i] = 0;
        for ( i = 0; i < nsupers; i++ )
        {
            for ( jb = 0; jb < nnodes_l[i]; jb++ )
            {
                num_child[edag_supno[i][jb]]++;
            }
        }

        /* push initial leaves to the fifo queue */
        nnodes = 0;
        for ( i = 0; i < nsupers; i++ )
        {
            if ( num_child[i] == 0 )
            {
                ptr = SUPERLU_MALLOC( sizeof(etree_node) );
                ptr->id = i;
                ptr->next = NULL;
                /*printf( " == push leaf %d (%d) ==\n",i,nnodes );*/
                nnodes ++;

                if ( nnodes == 1 )
                {
                    head = ptr;
                    tail = ptr;
                }
                else
                {
                    tail->next = ptr;
                    tail = ptr;
                }
            }
        }

        /* process fifo queue, and compute the ordering */
        i = 0;
        perm_c_supno  = SUPERLU_MALLOC( nsupers * sizeof(int_t) );
        while ( nnodes > 0 )
        {

            /*printf( "=== pop %d (%d) ===\n",head->id,i );*/
            ptr = head;  j = ptr->id;
            head = ptr->next;

            perm_c_supno[i] = j;
            SUPERLU_FREE(ptr);
            i++; nnodes --;

            for ( jb = 0; jb < nnodes_l[j]; jb++ )
            {
                num_child[edag_supno[j][jb]]--;
                if ( num_child[edag_supno[j][jb]] == 0 )
                {
                    nnodes ++;

                    ptr = SUPERLU_MALLOC( sizeof(etree_node) );
                    ptr->id = edag_supno[j][jb];
                    ptr->next = NULL;

                    /*printf( "=== push %d ===\n",ptr->id );*/
                    if ( nnodes == 1 )
                    {
                        head = ptr;
                        tail = ptr;
                    }
                    else
                    {
                        tail->next = ptr;
                        tail = ptr;
                    }
                }
            }
            /*printf( "\n" );*/
        }
        SUPERLU_FREE(num_child);

        for ( lb = 0; lb < nsupers; lb++ ) if ( nnodes_l[lb] > 0 ) SUPERLU_FREE(edag_supno[lb] );
        SUPERLU_FREE(edag_supno);
        SUPERLU_FREE(nnodes_l);
        SUPERLU_FREE(sendcnts);
        SUPERLU_FREE(sf_block);
        SUPERLU_FREE(Ublock);
        SUPERLU_FREE(Urows);
        SUPERLU_FREE(Lblock);
        SUPERLU_FREE(Lrows);
    }
    /* ======================== *
     * end of static scheduling *
     * ======================== */

    return perm_c_supno;
} /* getPerm_c_supno */


int_t Trs2_InitUblock_info(int_t klst, int_t nb,
			    Ublock_info_t *Ublock_info,
			    int_t *usub,
			    Glu_persist_t *Glu_persist, SuperLUStat_t *stat )
{
    int_t *xsup = Glu_persist->xsup;
    int_t iukp, rukp;
    iukp = BR_HEADER;
    rukp = 0;

    for (int_t b = 0; b < nb; ++b)
    {
        int_t gb = usub[iukp];
        int_t nsupc = SuperSize (gb);

        Ublock_info[b].iukp = iukp;
        Ublock_info[b].rukp = rukp;
        // Ublock_info[b].nsupc = nsupc;

        iukp += UB_DESCRIPTOR;
	/* Sherry: can remove this loop for rukp
	   rukp += usub[iukp-1];
	 */
       for (int_t j = 0; j < nsupc; ++j)
        {
            int_t segsize = klst - usub[iukp++];
            rukp += segsize;
            stat->ops[FACT] += segsize * (segsize + 1);
        }
    }
    return 0;
}

void getSCUweight(int_t nsupers, treeList_t* treeList, int_t* xsup,
		  int_t** Lrowind_bc_ptr, int_t** Ufstnz_br_ptr,
		  gridinfo3d_t * grid3d
		  )
{
    gridinfo_t* grid = &(grid3d->grid2d);
    //int_t** Lrowind_bc_ptr = LUstruct->Llu->Lrowind_bc_ptr;
    //int_t** Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    //int_t* xsup = LUstruct->Glu_persist->xsup;

    int_t * perm_u = INT_T_ALLOC(nsupers);
    int_t * mylsize = INT_T_ALLOC(nsupers);
    int_t * myusize = INT_T_ALLOC(nsupers);
    // int_t * maxlsize = INT_T_ALLOC(nsupers);
    // int_t * maxusize = INT_T_ALLOC(nsupers);
    int ldu;

    for (int i = 0; i < nsupers; ++i)
    {
        perm_u[i] = i;
        mylsize[i] = 0;
        myusize[i] = 0;
    }

    for (int_t k = 0; k < nsupers ; ++k)
    {
        treeList[k].scuWeight = 0.0;
        int_t iam = grid->iam;
        int_t myrow = MYROW (iam, grid);
        int_t mycol = MYCOL (iam, grid);
        // int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
	int_t ldu;

        if (myrow == krow)
        {
            /* code */
            myusize[k] = num_full_cols_U(k,  Ufstnz_br_ptr, xsup, grid,
					 perm_u, &ldu);
        }

        if (mycol == kcol)
        {
            /* code */
            int_t lk = LBj( k, grid ); /* Local block number */
            int_t  *lsub;
            // double* lnzval;
            lsub = Lrowind_bc_ptr[lk];
            if (lsub)
            {
                /* code */
                mylsize[k] = lsub[1];
            }
        }
    }

    // int_t maxlsize = 0;
    MPI_Allreduce( MPI_IN_PLACE, mylsize, nsupers, mpi_int_t, MPI_MAX, grid->comm );
    // int_t maxusize = 0;
    MPI_Allreduce(  MPI_IN_PLACE, myusize, nsupers, mpi_int_t, MPI_MAX, grid->comm );

    for (int_t k = 0; k < nsupers ; ++k)
    {

        treeList[k].scuWeight = 0.0;
        int_t ksupc = SuperSize(k);
        treeList[k].scuWeight = 1.0 * ksupc * mylsize[k] * myusize[k];
    }

    SUPERLU_FREE(mylsize);
    SUPERLU_FREE(myusize);
    SUPERLU_FREE(perm_u);

} /* getSCUweight */

