#include "superlu_ddefs.h"

sForest_t **compute_sForests(int_t nsupers, superlu_dist_options_t *options, dLUstruct_t *LUstruct, gridinfo3d_t *grid3d)
{
    // Calculation of supernodal etree
    int_t *setree = supernodal_etree(nsupers, LUstruct->etree, LUstruct->Glu_persist->supno, LUstruct->Glu_persist->xsup);

    // Conversion of supernodal etree to list
    treeList_t *treeList = setree2list(nsupers, setree);

    // Calculation of tree weight
    calcTreeWeight(nsupers, setree, treeList, LUstruct->Glu_persist->xsup);

    // Calculation of maximum level
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    // Generation of forests
    sForest_t **sForests = getForests(maxLvl, nsupers, setree, treeList);

    // Allocate trf3d data structure
    // LUstruct->trf3Dpart = (dtrf3Dpartition_t *)SUPERLU_MALLOC(sizeof(dtrf3Dpartition_t));
    // dtrf3Dpartition_t *trf3Dpart = LUstruct->trf3Dpart;

    return sForests;
}

/**
 * @file allocBcastArray.c
 * @brief Function to allocate and broadcast an array in a MPI environment.
 */

#include <mpi.h>
#include <stdlib.h>

/**
 * @brief Allocates and broadcasts an array in a MPI environment.
 *
 * This function sends the size from the root process to all other processes in
 * the communicator. If the process is not the root, it receives the size from
 * the root and allocates the array. Then, the function broadcasts the array
 * from the root process to all other processes in the communicator.
 *
 * @param array Pointer to the array to be allocated and broadcasted.
 * @param size The size of the array.
 * @param comm The MPI communicator.
 * @param root The root process.
 */
void allocBcastArray(void **array, int size, int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank); // Get the rank of the current process

    // Check if the size is valid
    if (rank == root)
    {
        if (size <= 0)
        {
            fprintf(stderr, "Error: Size should be a positive integer.\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
        
    }

    // Send the size from root to all other processes in the communicator
    MPI_Bcast(&size, 1, MPI_INT, root, comm);

    // If I am not the root, receive the size from the root and allocate the array
    if (rank != root)
    {
        *array = SUPERLU_MALLOC(size);
        if (*array == NULL)
        {
            fprintf(stderr, "Error: Failed to allocate memory.\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Then broadcast the array from the root to all other processes
    MPI_Bcast(*array, size, MPI_BYTE, root, comm);
}

// function to broad permuted sparse matrix and symbolic factorization data from
// 2d to 3d grid

void bcastPermutedSparseA(SuperMatrix *A, 
                          dScalePermstruct_t *ScalePermstruct,
                          Glu_freeable_t *Glu_freeable, 
                          dLUstruct_t *LUstruct, gridinfo3d_t *grid3d)
{
    int_t m = A->nrow;
	int_t n = A->ncol;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    NRformat_loc *Astore   = (NRformat_loc *) A->Store;
    // check if the varaibles are not NULL
    if (A == NULL || ScalePermstruct == NULL ||
        Glu_freeable == NULL || LUstruct == NULL  || grid3d == NULL ||  
        Glu_persist == NULL || Llu == NULL || Astore == NULL)
    {
        fprintf(stderr, "Error: Invalid arguments to bcastPermutedSparseA().\n");
        return;
    }

    // list of all the arrays to be broadcasted 
    // A, ScalePermstruct, Glu_freeable, LUstruct
    int_t nsupers;
    
    if (!grid3d->zscp.Iam)
        nsupers = Glu_persist->supno[n-1] + 1;
    // broadcast the number of supernodes
    MPI_Bcast(&nsupers, 1, mpi_int_t, 0, grid3d->zscp.comm);
    
    /* ==== Broadcasting GLU_persist   ======= */
    // what is the size of xsup and supno?
    allocBcastArray( &(Glu_persist->xsup), (nsupers+1)*sizeof(int_t), 
        0, grid3d->zscp.comm);
    allocBcastArray( &(Glu_persist->supno), (n)*sizeof(int_t), 
        0, grid3d->zscp.comm);
    int_t *xsup = Glu_persist->xsup;    /* supernode and column mapping */
    int_t *supno = Glu_persist->supno;
    

    /* ==== Broadcasting ScalePermstruct ======= */
//     typedef struct {
//     DiagScale_t DiagScale; // enum 1
//     double *R;  (double*) dimension (A->nrow)
//     double *C;   (double*) dimension (A->ncol)
//     int_t  *perm_r; (int_t*) dimension (A->nrow)
//     int_t  *perm_c; (int_t*) dimension (A->ncol)
// } dScalePermstruct_t;

    MPI_Bcast(&(ScalePermstruct->DiagScale), sizeof(DiagScale_t), MPI_BYTE, 0, grid3d->zscp.comm);
    allocBcastArray ( &(ScalePermstruct->perm_r), m*sizeof(int_t), 
        0, grid3d->zscp.comm);
    allocBcastArray ( &(ScalePermstruct->perm_c), n*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray ( &(ScalePermstruct->R), m*sizeof(double),
        0, grid3d->zscp.comm);
    allocBcastArray ( &(ScalePermstruct->C), n*sizeof(double),
        0, grid3d->zscp.comm);
    

    /* ==== Broadcasting Glu_freeable ======= */
//     typedef struct {
//     int_t     *lsub;     /* compressed L subscripts */
//     int_t     *xlsub;        // i think its size is n+1
//     int_t     *usub;     /* compressed U subscripts i think nzumax*/
//     int_t     *xusub;
//     int_t     nzlmax;    /* current max size of lsub */
//     int_t     nzumax;    /*    "    "    "      usub */
//     LU_space_t MemModel; /* 0 - system malloc'd; 1 - user provided */
//     //int_t     *llvl;     /* keep track of level in L for level-based ILU */
//     //int_t     *ulvl;     /* keep track of level in U for level-based ILU */
//     int64_t nnzLU;   /* number of nonzeros in L+U*/
// } Glu_freeable_t;

    allocBcastArray( &(Glu_freeable->lsub), Glu_freeable->nzlmax*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray( &(Glu_freeable->xlsub), (n+1)*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray( &(Glu_freeable->usub), Glu_freeable->nzumax*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray( &(Glu_freeable->xusub), (n+1)*sizeof(int_t),
        0, grid3d->zscp.comm);
    MPI_Bcast(&(Glu_freeable->nzlmax), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(&(Glu_freeable->nzumax), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(&(Glu_freeable->nnzLU), sizeof(int64_t), MPI_BYTE, 0, grid3d->zscp.comm);

    if(grid3d->zscp.Iam)
    {
        Glu_freeable->MemModel = SYSTEM;
    }
    
    
    /* ==== Broadcasting permuted sparse matrix ======= */
    // Astore = (NRformat_loc *) A->Store;
//     /typedef struct {
//     int_t nnz_loc;   /* number of nonzeros in the local submatrix */
//     int_t m_loc;     /* number of rows local to this processor */
//     int_t fst_row;   /* global index of the first row */
//     void  *nzval;    /* pointer to array of nonzero values, packed by row */
//     int_t *rowptr;   /* pointer to array of beginning of rows in nzval[] 
// 			and colind[]  */
//     int_t *colind;   /* pointer to array of column indices of the nonzeros */
//                      /* Note:
// 			Zero-based indexing is used;
// 			rowptr[] has n_loc + 1 entries, the last one pointing
// 			beyond the last row, so that rowptr[n_loc] = nnz_loc.*/
// } NRformat_loc;


    // NRformat_loc *Astore = (NRformat_loc *) A->Store;
    MPI_Bcast(&(Astore->nnz_loc), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(&(Astore->m_loc), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(&(Astore->fst_row), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    allocBcastArray( &(Astore->nzval), Astore->nnz_loc*sizeof(double),
        0, grid3d->zscp.comm);
    allocBcastArray( &(Astore->rowptr), (Astore->m_loc+1)*sizeof(int_t), 
        0, grid3d->zscp.comm);
    allocBcastArray( &(Astore->colind), Astore->nnz_loc*sizeof(int_t), 
        0, grid3d->zscp.comm);


}