

#include "superlu_ddefs.h"
//#include "pddistribute3d.h"

/**
 * Propagates the new values of A into the existing L and U data structures.
 *
 * @param Llu The local L and U data structures.
 * @param xa The index array of A.
 * @param asub The row subscripts of A.
 * @param a The numerical values of A.
 * @param options The options array.
 * @param grid The process grid.
 * @param mem_use The memory usage.
 *
 * @return void
 */
void dpropagate_A_to_LU3d(
    dLUstruct_t *LUstruct,
    int_t *xa,
    int_t *asub,
    double *a,
    superlu_dist_options_t* options,
    gridinfo3d_t *grid3d,
    int_t nsupers,
    float *mem_use)
{
    /* Initialization. */
    gridinfo_t *grid = &(grid3d->grid2d);
    dLocalLU_t *Llu = LUstruct->Llu;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int_t *xsup = Glu_persist->xsup; /* supernode and column mapping */
    int_t *supno = Glu_persist->supno;
    int iam = grid->iam;
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);

    // Initialize variables for profiling
    #if (PROFlevel >= 1)
    double t_l = 0.0, t_u = 0.0, t = 0.0;
    int_t u_blks = 0;
    #endif

    int_t *ilsum = Llu->ilsum;
    int_t ldaspa = Llu->ldalsum;

    // Allocate space for dense storage
    double *dense = doubleCalloc_dist(ldaspa * sp_ienv_dist(3, options));
    if (dense == NULL) {
        ABORT("Calloc fails for SPA dense[].");
    }

    // Calculate number of local block rows
    int_t nrbu = CEILING(nsupers, grid->nprow);

    // Allocate space for row lengths and pointers
    int_t *Urb_length = intCalloc_dist(nrbu);
    if (Urb_length == NULL) {
        ABORT("Calloc fails for Urb_length[].");
    }

    int_t *Urb_indptr = intMalloc_dist(nrbu);
    if (Urb_indptr == NULL) {
        ABORT("Malloc fails for Urb_indptr[].");
    }

    // Get pointers to the local LU data
    int_t **Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    int_t **Lindval_loc_bc_ptr = Llu->Lindval_loc_bc_ptr;
    double **Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double **Unzval_br_ptr = Llu->Unzval_br_ptr;
    int_t *Unnz = Llu->Unnz;

    // Calculate size of integer and double types
    int_t iword = sizeof(int_t);
    int_t dword = sizeof(double);

    float memTRS = 0.0;

    // Start timer for profiling
    #if (PROFlevel >= 1)
    t = SuperLU_timer_();
    #endif

    // Initialize Uval to zero
    for (int_t lb = 0; lb < nrbu; ++lb) {
        Urb_indptr[lb] = BR_HEADER;
        int_t *index = Ufstnz_br_ptr[lb];
        if (index != NULL) {
            double *uval = Unzval_br_ptr[lb];
            int_t len = index[1];
            for (int_t i = 0; i < len; ++i) {
                uval[i] = 0.0;
            } /* for i ... */
        } /* if index ... */
    } /* for lb ... */

    for (int_t jb = 0; jb < nsupers; ++jb) {
        int_t pc = PCOL(jb, grid);
        if (mycol == pc) {
            int_t fsupc = FstBlockC(jb);
            int_t nsupc = SuperSize(jb);

            for (int_t j = fsupc; j < FstBlockC(jb + 1); ++j) {
                double *dense_col = dense + (j - fsupc) * ldaspa;
                for (int_t i = xa[j]; i < xa[j + 1]; ++i) {
                    int_t irow = asub[i];
                    int_t gb = BlockNum(irow);
                    if (myrow == PROW(gb, grid)) {
                        int_t lb = LBi(gb, grid);
                        if (gb < jb) {
                            int_t *index = Ufstnz_br_ptr[lb];
                            double *uval = Unzval_br_ptr[lb];
                            while ( index[Urb_indptr[lb]] < jb) {
                                Urb_length[lb] += index[Urb_indptr[lb] + 1];
                                Urb_indptr[lb] += UB_DESCRIPTOR + SuperSize(index[Urb_indptr[lb]]);
                            }
                            int_t istart = Urb_indptr[lb] + UB_DESCRIPTOR;
                            int_t len = Urb_length[lb];
                            int_t fsupc1 = FstBlockC(gb + 1);
                            int_t k = j - fsupc;
                            for (int_t jj = 0; jj < k; ++jj) {
                                len += fsupc1 - index[istart++];
                            }
                            uval[len + irow - index[istart]] = a[i];
                        } else {
                            irow = ilsum[lb] + irow - FstBlockC(gb);
                            dense_col[irow] = a[i];
                        }
                    } /* if myrow == PROW(gb) */
                } /* for i ... */
            } /* for j ... */
#if (PROFlevel >= 1)
                t_u += SuperLU_timer_() - t;
                t = SuperLU_timer_();
#endif


            int_t ljb = LBj(jb, grid);
            int_t *index = Lrowind_bc_ptr[ljb];
            if (index != NULL) {
                int_t nrbl = index[0];
                int_t len = index[1];
                double *lusup = Lnzval_bc_ptr[ljb];
                int_t next_lind = BC_HEADER;
                int_t next_lval = 0;
                for (int_t jj = 0; jj < nrbl; ++jj) {
                    int_t gb = index[next_lind++];
                    int_t len1 = index[next_lind++];
                    int_t lb = LBi(gb, grid);
                    for (int_t bnnz = 0; bnnz < len1; ++bnnz) {
                        int_t irow = index[next_lind++];
                        irow = ilsum[lb] + irow - FstBlockC(gb);
                        int_t k = next_lval++;
                        for (int_t j = 0; j < nsupc; ++j) {
                            double *dense_col = dense + j * ldaspa;
                            lusup[k] = dense_col[irow];
                            dense_col[irow] = 0.0;
                            k += len;
                        } /* for j ... */
                    } /* for bnnz ... */
                } /* for jj ... */
            } /* if L is not empty */
#if (PROFlevel >= 1)
            t_l += SuperLU_timer_() - t;
#endif
        } /* if mycol == pc */
    } /* for jb ... */

    SUPERLU_FREE(dense);
    SUPERLU_FREE(Urb_length);
    SUPERLU_FREE(Urb_indptr);

    #if (PROFlevel >= 1)
    if (!iam) {
        printf(".. 2nd distribute time: L %.2f\tU %.2f\tu_blks %d\tnrbu %d\n",
               t_l, t_u, u_blks, nrbu);
    }
    #endif
}



/**
 * Function: computeLDAspa_Ilsum
 *
 * This function computes the Local Distributed Storages (LDS) for the L factor in the LU decomposition.
 * It also updates the input array 'ilsum' to hold the prefix sum of the size of each supernode.
 *  Compute ldaspa and ilsum[]. */
        // ldaspa = 0;
        // ilsum[0] = 0;
        // for (int_t gb = 0; gb < nsupers; ++gb)
        // {
        //     if (myrow == PROW(gb, grid))
        //     {
        //         i = SuperSize(gb);
        //         ldaspa += i;
        //         lb = LBi(gb, grid);
        //         ilsum[lb + 1] = ilsum[lb] + i;
        //     }
        // }
/*
 * Inputs:
 * - nsupers: The total number of supernodes in the LU decomposition.
 * - ilsum: An array of size 'nsupers'. On output, ilsum[i] will hold the sum of the sizes of the first 'i' supernodes.
 * - LUstruct: A pointer to the LU decomposition structure, which holds the information about the LU decomposition.
 * - grid3d: A pointer to the 3D process grid structure, which holds the information about the 3D process grid.
 *
 * Outputs:
 * - The function returns the Local Distributed Storage (LDS) for the L factor.
 * - The array 'ilsum' is updated in-place to hold the prefix sum of the size of each supernode.
 */
int_t computeLDAspa_Ilsum( int_t nsupers, int_t* ilsum,  dLUstruct_t *LUstruct, gridinfo3d_t* grid3d)
{
    // Extract the supernode and column mapping from the LU structure
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int_t *xsup = Glu_persist->xsup;

    // Extract the 2D grid from the 3D grid structure
    gridinfo_t *grid = &(grid3d->grid2d);

    // Determine the row of the current process in the 2D process grid
    int myrow = MYROW( grid->iam, grid );

    // Initialize ldaspa and the first element of ilsum
    int_t ldaspa = 0;
    ilsum[0] = 0;

    // Loop over all supernodes
    for (int_t gb = 0; gb < nsupers; ++gb)
    {
        // If the current process row is the owner of this supernode
        if (myrow == PROW(gb, grid))
        {
            // Get the size of the supernode
            int_t i = SuperSize(gb);

            // Add the size to the total
            ldaspa += i;

            // Find the local block number of the supernode
            int_t lb = LBi(gb, grid);

            // Update the prefix sum array
            ilsum[lb + 1] = ilsum[lb] + i;
        }
    }

    // Return the total size
    return ldaspa;
}


/**
 * The function propagate_blocks performs data propagation in the form of values through blocks
 * of a supernodal sparse matrix. It sends values to particular blocks and marks blocks to receive
 * new values depending on certain conditions. It also counts the number of non-zeros in each block row
 * and keeps track of the number of column blocks in each block row.
 *
 * Input:
 *   nsupers: Total number of supernodes, i.e., the number of block columns in the original matrix.
 *   grid: The process grid. This grid contains the mapping of the matrix blocks to the process grid.
 *   xusub: An array of indices, each pointing to the start of each column in the usub array.
 *   usub: An array containing the row indices of non-zero elements of the supernodal matrix.
 *   ToSendR: A 2D flag array indicating which blocks need to be sent to other processes, organized by column.
 *   ToSendD: A flag array indicating which blocks are to be sent to other processes.
 *   Urb_length: An array containing the number of non-zero elements in each block row of the matrix.
 *   rb_marker: A marker array to track the last block column visited in each block row.
 *   Urb_fstnz: An array containing the number of first non-zero elements in each block row of the matrix.
 *   Ucbs: An array counting the number of column blocks in each block row.
 *   ToRecv: A flag array indicating which blocks are to be received from other processes.
 *
 * Output:
 *   ToSendR, ToSendD, Urb_length, rb_marker, Urb_fstnz, Ucbs, ToRecv are modified in place. Their new values
 *   depend on the conditions checked during the loops.
 *
 * @return void
 */
void propagateDataThroughMatrixBlocks(int_t nsupers, Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct, gridinfo3d_t* grid3d,
int_t *Urb_length, int_t *rb_marker, int_t *Urb_fstnz, int_t *Ucbs,
int **ToSendR,  int *ToSendD,  int *ToRecv)
{
    int_t* usub = Glu_freeable->usub; /* compressed U subscripts */
    int_t* xusub = Glu_freeable->xusub;
        // Extract the supernode and column mapping from the LU structure
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    int_t *xsup = Glu_persist->xsup;
    int_t *supno = Glu_persist->supno;

    // Extract the 2D grid from the 3D grid structure
    gridinfo_t *grid = &(grid3d->grid2d);

    // Iterate through all supernodes
    for (int_t jb = 0; jb < nsupers; ++jb)
    {
        // For each supernode, get its details such as its process column and its size.
        int_t pc = PCOL(jb, grid);
        int_t fsupc = FstBlockC(jb);
        int_t nsupc = SuperSize(jb);

        // Loop through each column in the supernode.
        for (int_t j = fsupc; j < fsupc + nsupc; ++j)
        {
            // Inside each column, loop through the non-zero elements.
            for (int_t i = xusub[j]; i < xusub[j + 1]; ++i)
            {
                int_t irow = usub[i]; // The row index of the current non-zero element
                int_t gb = BlockNum(irow);
                int_t kcol = PCOL(gb, grid);
                int_t ljb = LBj(gb, grid);

                int_t mycol = MYCOL(grid->iam, grid);
                int_t myrow = MYROW(grid->iam, grid);

                // Check the conditions to send or receive data.
                if (mycol == kcol && mycol != pc)
                    ToSendR[ljb][pc] = YES;

                int_t pr = PROW(gb, grid);
                int_t lb = LBi(gb, grid);

                // Check more conditions to send or receive data.
                if (mycol == pc)
                {
                    if (myrow == pr)
                    {
                        ToSendD[lb] = YES;
                        Urb_length[lb] += FstBlockC(gb + 1) - irow;

                        if (rb_marker[lb] <= jb)
                        {
                            rb_marker[lb] = jb + 1;
                            Urb_fstnz[lb] += nsupc;
                            ++Ucbs[lb];
                        }
                        ToRecv[gb] = 1;
                    }
                    else
                    {
                        ToRecv[gb] = 2;
                    }
                }
            }
        }
    }
}


int_t checkDist3DLUStruct(  dLUstruct_t* LUstruct, gridinfo3d_t* grid3d)
{
    dtrf3Dpartition_t*  trf3Dpartition = LUstruct->trf3Dpart;
    gridinfo_t *grid = &(grid3d->grid2d);
    int iam = grid->iam;
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int myGrid = grid3d->zscp.Iam;

    sForest_t** sForests = trf3Dpartition->sForests;
    int_t*  gNodeCount = getNodeCountsFr(maxLvl, sForests);
    int_t** gNodeLists = getNodeListFr(maxLvl, sForests);


    for (int grid_id =1 ; grid_id < grid3d->zscp.Np; ++grid_id)
    {
        int first_tree = grid3d->zscp.Np - 1 + grid_id;
        while( first_tree >0 )
        {
            int_t* tree = gNodeLists[first_tree];
            int_t tree_size = gNodeCount[first_tree];
            int grid_zero = 0;
            int result = dist_checkArrayEq(tree, tree_size, mpi_int_t, grid_zero, grid_id, grid3d->zscp.comm, compareInt_t);
            if (myGrid == grid_id && result)
            {
                printf("Check tree list failed: tree_id=%d, grid_id =%d, Iam=(%d, %d) \n",
                    first_tree, grid_id, grid3d->zscp.Iam, grid3d->zscp.Iam);
                exit(1);
            }

            for(int_t node=0; node<tree_size; ++node)
            {
                int_t node_id = tree[node];
                if( PROW(node_id, grid) ==myrow)
                {
                    // check U index list
                    int_t lk = LBi(node_id, grid);
                    int_t* usub = LUstruct->Llu->Ufstnz_br_ptr[lk];
                    int_t usub_size = 0, uval_size = 0;
                    if (usub != NULL)
                    {
                        uval_size = usub[1];
                        usub_size = usub[2];
                    }

                    // check U index
                    int result = dist_checkArrayEq(usub, usub_size, mpi_int_t,
                        grid_zero, grid_id, grid3d->zscp.comm, compareInt_t);
                    if (myGrid == grid_id && result)
                    {
                        printf("Check U index failed: node_id=%d, grid_id =%d, Iam=(%d, %d) \n",
			       (int)node_id, grid_id, grid3d->zscp.Iam, grid3d->zscp.Iam);
                        exit(1);
                    }

                    if(usub==NULL) continue;

                    // check U value
                    double* uval = LUstruct->Llu->Unzval_br_ptr[lk];
                    result = dist_checkArrayEq(uval, uval_size, MPI_DOUBLE,
                        grid_zero, grid_id, grid3d->zscp.comm, compareDouble);
                    if (myGrid == grid_id && result)
                    {
                        printf("Check U value failed: node_id=%d, grid_id =%d, Iam=(%d, %d) \n",
			       (int)node_id, grid_id, grid3d->zscp.Iam, grid3d->zscp.Iam);
                        exit(1);
                    }

                } /* Checking U panel */

                if( PCOL(node_id, grid) ==mycol)
                {
                    int_t lk = LBj(node_id, grid);
                    int_t* lsub = LUstruct->Llu->Lrowind_bc_ptr[lk];
                    double* lusup = LUstruct->Llu->Lnzval_bc_ptr[lk];
                    int_t lsub_size = 0, lval_size = 0;
                    if (lsub != NULL)
                    {
                        int_t nrbl, len;
			            nrbl  =   lsub[0]; /*number of L blocks */
                        len   = lsub[1];       /* LDA of the nzval[] */
                        lsub_size  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
                        int_t* xsup = LUstruct->Glu_persist->xsup; // for SuperSize()
                        lval_size  = SuperSize(node_id) * len;
                    }

                    // check L index
                    int result = dist_checkArrayEq(lsub, lsub_size, mpi_int_t,
                        grid_zero, grid_id, grid3d->zscp.comm, compareInt_t);
                    if (myGrid == grid_id && result)
                    {
                        printf("Check L index failed: node_id=%d, grid_id =%d, Iam=(%d, %d) \n",
			       (int)node_id, grid_id, grid3d->zscp.Iam, grid3d->zscp.Iam);
                        exit(1);
                    }

                    if(lsub==NULL) continue;

                    // check L value
                    double* lval = LUstruct->Llu->Lnzval_bc_ptr[lk];
                    result = dist_checkArrayEq(lval, lval_size, MPI_DOUBLE,
                        grid_zero, grid_id, grid3d->zscp.comm, compareDouble);
                    if (myGrid == grid_id && result)
                    {
                        printf("Check L value failed: node_id=%d, grid_id =%d, Iam=(%d, %d) \n",
			       (int)node_id, grid_id, grid3d->zscp.Iam, grid3d->zscp.Iam);
                        exit(1);
                    }
                }/* Check L panel*/


            }
            first_tree = (first_tree-1)/2;
        }
    }

    // Now check if I am only allocating the memory for the blocks I own
    if ( myGrid )
    // if(0)
    {
        SupernodeToGridMap_t *superGridMap = trf3Dpartition->superGridMap;
        // int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
        int_t nsupers = trf3Dpartition->nsupers;
        for(int k =0; k < nsupers; k++)
        {
            if(superGridMap[k] == NOT_IN_GRID)
            {
                // all pointer should be NULL
                int krow = PROW(k, grid);
                int kcol = PCOL(k, grid);
                if(myrow == krow)
                {
                    int lk = LBi(k, grid);
                    int_t* usub = LUstruct->Llu->Ufstnz_br_ptr[lk];
                    double* uval = LUstruct->Llu->Unzval_br_ptr[lk];
                    if(usub != NULL || uval != NULL)
                    {
                        printf("Check 3D LU structure failed: node_id=%d, grid_id =%d, Iam=(%d, %d) \n",
			       k, myGrid, grid3d->zscp.Iam, grid3d->zscp.Iam);
                        exit(1);
                    }
                }

                if (mycol == kcol)
                {
                    int lk = LBj(k, grid);
                    int_t* lsub = LUstruct->Llu->Lrowind_bc_ptr[lk];
                    double* lusup = LUstruct->Llu->Lnzval_bc_ptr[lk];
                    if(lsub != NULL || lusup != NULL)
                    {
                        printf("Check 3D LU structure failed: node_id=%d, grid_id =%d, Iam=(%d, %d) \n",
                            k, myGrid, grid3d->zscp.Iam, grid3d->zscp.Iam);
                        exit(1);
                    }
                }
            }
        } /* end for k ... */
    }
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Exit checkDist3DLUStruct()");
#endif
    return 0;
}

