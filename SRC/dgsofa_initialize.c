/*! \file
 * Copyright (c) 2003, The Regents of the University of California, through
 * Lawrence Berkeley National Laboratory (subject to receipt of any required 
 * approvals from U.S. Dept. of Energy) 
 *
 * All rights reserved. 
 * 
 * The source code is distributed under BSD license, see the file License.txt
 * at the top-level directory.
 * */
#include <stdlib.h>
// #include "cholnzcnt.h"
#include "superlu_ddefs.h"
#include <cublasXt.h>

// #define DEBUGlevel 0
#define asynch_initialization 1

// #define allocated_frontier_size 306749905 //For N_source: 1024 pre2 dataset allocation (50% worse space complexity possible)
#define allocation_threshold_factor 0.4
// #define allocation_threshold_factor 1
#define async_graphload 1
#define Enable_supernodal_relaxation 1
#define initializeCSR 1
// #define enable_supernodal_graph 1

// #define Debug_enabled 1
// #define enablegSoFaMPI 1
// #define merge_csr 1


#define printProcess 50000
#define printNBlockColumns 11948
// #define all_process_gSoFa 1 //All processes selected for gSoFa
// #define autodetect_gSoFa_parameters 1


int compare (const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b  );//in ascending order
}
int_t detect_device(int_t iam_gSoFa, int_t iam, int_t nprs, int_t* init_cuda, int_t is_gSoFa)
{
    int ngprs = 0;//Number of gSoFa Process per resource
    cudaGetDeviceCount(&ngprs);
    if (nprs < ngprs)
    {
        ngprs = nprs;
    }   


    if ((iam >= ngprs) && (iam < ngprs + nprs))
    {
        init_cuda = 1;
        return -1;
    }

    return (iam_gSoFa%ngprs);
    // int localgpu = mygSoFaOffset;
    //  H_ERR(cudaSetDevice(localgpu));
    //  int_t device;
    //  cudaGetDevice(&device);
    //  return device;
}

void Initialize_Cuda()
{
#if 0
    double start_cublas_init = SuperLU_timer_();
    //Optimizing the time for Cuda stream creation
    //  printf("Making sure the cublas library is fully loaded and initialized!\n");
    cudaFree(0);
    cublasHandle_t hb;
    cublasCreate(&hb);
    cublasDestroy(hb);
    //~Optimizing the time for Cuda stream creation
    double cublas_init_time = SuperLU_timer_() - start_cublas_init;
    printf("Cublas initialization time: %lf s\n",cublas_init_time);
    // cudaStream_t* stream = (cudaStream_t*) malloc ((N_GPU_gSoFa_process)*sizeof(cudaStream_t));
#endif
    cudaDeviceReset();
}

int_t BlockNum_gSoFa(int_t i, int_t*supno) 
{
    return supno[i];
} 

#if 0
struct gSoFa_pair
{
    int ind;
    // char hostname[MPI_MAX_PROCESSOR_NAME+1];
    char* hostname;
};
int compare_hosts (const void *a, const void *b)
{
    return (((struct gSoFa_pair *) a)->hostname - ((struct gSoFa_pair *) b)->hostname);
}
#endif

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * Form the structure of A'+A. A is an n-by-n matrix in column oriented
 * format represented by (colptr, rowind). The output A'+A is in column
 * oriented format (symmetrically, also row oriented), represented by
 * (b_colptr, b_rowind).
 * </pre>
 */
void
a1t_plus_a1_dist(
        const int_t n,    /* number of columns in matrix A. */
        const int_t nz,   /* number of nonzeros in matrix A */
        int_t *colptr_beg,    /* column pointer begin of size n for matrix A. */
        int_t *colptr_end,    /* column pointer end of size n for matrix A. */
        int_t *rowind,    /* row indices of size nz for matrix A. */
        int_t *bnz,       /* out - on exit, returns the actual number of
                             nonzeros in matrix A'+A. */
        int_t **b_colptr, /* out - size n+1 */
        int_t **b_rowind  /* out - size *bnz */
        )
{

    register int_t i, j, k, col, num_nz;
    int_t *t_colptr, *t_rowind; /* a column oriented form of T = A' */
    int_t *marker;

    if ( !(marker = (int_t*) SUPERLU_MALLOC( n * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for marker[]");
    if ( !(t_colptr = (int_t*) SUPERLU_MALLOC( (n+1) * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for t_colptr[]");
    if ( !(t_rowind = (int_t*) SUPERLU_MALLOC( nz * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails t_rowind[]");

    /* Get counts of each column of T, and set up column pointers */
    for (i = 0; i < n; ++i) marker[i] = 0;
    for (j = 0; j < n; ++j) {
        // for (i = colptr[j]; i < colptr[j+1]; ++i)
        //     ++marker[rowind[i]];
        for (i = colptr_beg[j]; i < colptr_end[j]; ++i)
            ++marker[rowind[i]];
    }

    t_colptr[0] = 0;
    for (i = 0; i < n; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];
        marker[i] = t_colptr[i];
    }

    /* Transpose the matrix from A to T */
    for (j = 0; j < n; ++j) {
        for (i = colptr_beg[j]; i < colptr_end[j]; ++i) {
            col = rowind[i];
            t_rowind[marker[col]] = j;
            ++marker[col];
        }
    }


    /* ----------------------------------------------------------------
       compute B = A + T, where column j of B is:
       Struct (B_*j) = Struct (A_*k) UNION Struct (T_*k)
       do not include the diagonal entry
       ---------------------------------------------------------------- */

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;

    /* First pass determines number of nonzeros in B */
    num_nz = 0;
    for (j = 0; j < n; ++j) {
        /* Flag the diagonal so it's not included in the B matrix */
        marker[j] = j;

        /* Add pattern of column A_*k to B_*j */
        for (i = colptr_beg[j]; i < colptr_end[j]; ++i) {
            k = rowind[i];
            if ( marker[k] != j ) {
                marker[k] = j;
                ++num_nz;
            }
        }

        /* Add pattern of column T_*k to B_*j */
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            k = t_rowind[i];
            if ( marker[k] != j ) {
                marker[k] = j;
                ++num_nz;
            }
        }
    }
    *bnz = num_nz;


    /* Allocate storage for A+A' */
    if ( !(*b_colptr = (int_t*) SUPERLU_MALLOC( (n+1) * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for b_colptr[]");
    if ( *bnz ) {
        if ( !(*b_rowind = (int_t*) SUPERLU_MALLOC( *bnz * sizeof(int_t)) ) )
            ABORT("SUPERLU_MALLOC fails for b_rowind[]");
    }

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;

    /* Compute each column of B, one at a time */
    num_nz = 0;
    for (j = 0; j < n; ++j) {
        (*b_colptr)[j] = num_nz;

        /* Flag the diagonal so it's not included in the B matrix */
        marker[j] = j;

        /* Add pattern of column A_*k to B_*j */
        for (i = colptr_beg[j]; i < colptr_end[j]; ++i) {
            k = rowind[i];
            if ( marker[k] != j ) {
                marker[k] = j;
                (*b_rowind)[num_nz++] = k;
            }
        }

        /* Add pattern of column T_*k to B_*j */
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            k = t_rowind[i];
            if ( marker[k] != j ) {
                marker[k] = j;
                (*b_rowind)[num_nz++] = k;
            }
        }
    }
    (*b_colptr)[n] = num_nz;

    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_colptr);
    SUPERLU_FREE(t_rowind);
}


static void relax_supnode
/************************************************************************/
(
 const int_t n, /* number of columns in the matrix (input) */
 int_t       *et,   /* column elimination tree (input) */
 const int_t relax, /* max no of columns allowed in a relaxed snode (input) */
 int_t       *desc, /* number of descendants of each etree node. */
 int_t       *relax_end, /* last column in a supernode (output) */
 int_t	   *relaxed_col /* first column in a supernode (output) */
 //   int_t      *included_non_singleton /* flag if the column is already included in non-singleton supernode */
 )

{
    // int_t* relax_end = *relax_end_ref;
    // int_t* relaxed_col = *relaxed_col_ref;
    // printf("Beginnig of relax module!\n");
    // fflush(stdout);
    register int_t j, parent, nsuper;
    register int_t fsupc; /* beginning of a snode */

    ifill_dist(relax_end, n, EMPTY);
    ifill_dist(relaxed_col, n, EMPTY);//for gSoFa
    ifill_dist(desc, n+1, 0);
    // printf("Done initializing relax_end, relaxed_col and des\n");
    // fflush(stdout);
    nsuper = 0;

    /* Compute the number of descendants of each node in the etree. */
    for (j = 0; j < n; j++) {
        parent = et[j];
        if ( parent != n )  /* not the dummy root */
        {
            desc[parent] += desc[j] + 1;         
        }
    }

    /* Identify the relaxed supernodes by postorder traversal of the etree. */
    for (j = 0; j < n; ) { 
        parent = et[j];
        fsupc = j;
        while ( parent != n && desc[parent] < relax ) {
            j = parent;
            parent = et[j];
            // representative_col[parent] = fsupc;//for gSoFa
        }    
        relax_end[fsupc] = j; /* Last column is recorded. */		
        ifill_dist(&relaxed_col[fsupc], j-fsupc+1, fsupc);//for gSoFa


        ++nsuper;
        ++j;
        /* Search for a new leaf. */
        while ( desc[j] != 0 && j < n ) ++j;
    }

#if ( DEBUGlevel>=1 )
    printf(".. No of relaxed snodes: " IFMT "\trelax: " IFMT "\n", nsuper, relax);
#endif
} /* relax_supnode */



void  sp_colorder_Final(superlu_dist_options_t *options,  SuperMatrix *A, int_t *perm_c, 
        int_t *etree, SuperMatrix *AC, /*AC is the input graph to gSoFa*/
        int_t* invp, 
        int_t* col_cnt,int_t* row_cnt, int_t* nlnz, int_t* part_super_h,struct gSoFa_para_t* gSoFa_para)
{

    NCformat  *Astore;
    NCPformat *ACstore;

    NCPformat *ACstore_temp;
    SuperMatrix *AC_temp = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );

    int_t       *iwork, *post;
    register  int_t n, i;
#if ( DEBUGlevel>=1 )
    int iam;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    CHECK_MALLOC(iam, "Enter sp_colorder()");
#endif

    //  printf("Inside  sp_colorder_Final()!\n");
    //     fflush(stdout);
    //        printf("Inside  sp_colorder_Final() At Line: %d!\n",__LINE__);
    //     fflush(stdout);
    n = A->ncol;

    /* Apply column permutation perm_c to A's column pointers so to
       obtain NCP format in AC = A*Pc.  */
    AC->Stype       = SLU_NCP;
    AC->Dtype       = A->Dtype;
    AC->Mtype       = A->Mtype;
    AC->nrow        = A->nrow;
    AC->ncol        = A->ncol;
    Astore          = A->Store;
    ACstore = AC->Store = (void *) SUPERLU_MALLOC( sizeof(NCPformat) );
    if ( !ACstore ) ABORT("SUPERLU_MALLOC fails for ACstore");
    //    printf("Inside  sp_colorder_Final() At Line: %d!\n",__LINE__);
    //         fflush(stdout);
    AC_temp->Stype       = SLU_NCP;
    AC_temp->Dtype       = A->Dtype;
    AC_temp->Mtype       = A->Mtype;
    AC_temp->nrow        = A->nrow;
    AC_temp->ncol        = A->ncol;
    // Astore          = A->Store;


    ACstore_temp = AC_temp->Store = (void *) SUPERLU_MALLOC( sizeof(NCPformat) );
    if ( !ACstore_temp ) ABORT("SUPERLU_MALLOC fails for ACstore");
    #if (DEBUGlevel == 0)
    printf("Inside  sp_colorder_Final() At Line: %d!\n",__LINE__);
    // fflush(stdout);
    #endif
    ACstore_temp->nnz    = Astore->nnz;
    ACstore_temp->nzval  = Astore->nzval;
    ACstore_temp->rowind = Astore->rowind;
    ACstore_temp->colbeg = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
    if ( !(ACstore_temp->colbeg) ) ABORT("SUPERLU_MALLOC fails for ACstore->colbeg");
    ACstore_temp->colend = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
    if ( !(ACstore_temp->colend) ) ABORT("SUPERLU_MALLOC fails for ACstore->colend");
    
    #if (DEBUGlevel == 0)
    printf("Inside  sp_colorder_Final() At Line: %d!\n",__LINE__);
    // fflush(stdout);
    #endif

#if 1
    ACstore->nnz    = Astore->nnz;
    #if (DEBUGlevel == 0)
    printf("Original nnz before relaxation original edge_count:%d\n",ACstore->nnz);
    fflush(stdout);
    #endif
    ACstore->nzval  = Astore->nzval;
    ACstore->rowind =(int_t*) SUPERLU_MALLOC(ACstore->nnz*sizeof(int_t));
    if ( !(ACstore->rowind) ) ABORT("SUPERLU_MALLOC fails for ACstore->rowind");
    ACstore->colbeg = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
    if ( !(ACstore->colbeg) ) ABORT("SUPERLU_MALLOC fails for ACstore->colbeg");
    ACstore->colend = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
    if ( !(ACstore->colend) ) ABORT("SUPERLU_MALLOC fails for ACstore->colend");
#else

    ACstore->nnz    = ACstore_temp->nnz ;
    ACstore->nzval  =  ACstore_temp->nzval;
    ACstore->rowind =ACstore_temp->rowind;  
    ACstore->colbeg = ACstore_temp->colbeg;   
    ACstore->colend = ACstore_temp->colend;

#endif

    // printf("Inside  sp_colorder_Final() At Line: %d!\n",__LINE__);
    // fflush(stdout);
#if ( DEBUGlevel>=3 )
    if ( !iam ) {
        PrintInt10("pre_order:", n, perm_c);
        check_perm_dist("Initial perm_c", n, perm_c);
    }
#endif      

    for (i = 0; i < n; i++) {
        ACstore_temp->colbeg[perm_c[i]] = Astore->colptr[i]; /*input graph to gSoFa*/
        ACstore_temp->colend[perm_c[i]] = Astore->colptr[i+1]; /*input graph to gSoFa*/

        // ACstore->colbeg[i] = Astore->colptr[i]; 
        // ACstore->colend[i] = Astore->colptr[i+1];
    }
    // printf("Inside  sp_colorder_Final() At Line: %d!\n",__LINE__);
    // fflush(stdout);

    if ( options->Fact == DOFACT 
            || options->Fact == SamePattern ) {
        /* In this case, perm_r[] may be changed, etree(Pr*A + (Pr*A)')
           may be changed, so need to recompute etree.   */
        /* Factor A "from scratch" -- we also compute the etree, and
         * make perm_c consistent with the postorder of the etree.
         */

        iwork = (int_t*) SUPERLU_MALLOC((n+1)*sizeof(int_t)); 
        int_t* gSoFa_rev_perm = (int_t*) SUPERLU_MALLOC((n)*sizeof(int_t)); 
        if ( !iwork ) ABORT("SUPERLU_MALLOC fails for iwork[]");

        /* Compute the etree of Pc*(A'+A)*Pc'. */
        int_t *b_colptr, *b_rowind, bnz;
        int_t j;

        int_t *c_colbeg, *c_colend;
        // int_t* c_colptr;//gSoFa
        // printf("Matrix from the original A input to at_plus_a_dist() !\n");

        // for (i = 0; i < n; i++) {
        //     printf("\nNeighbor of %d: ",i);
        //     for (j = Astore->colptr[i]; j < Astore->colptr[i+1]; j++) {
        //         printf("%d ",Astore->rowind[j]);
        //     }          
        // }
        // printf("\n");
        /* Form B = A + A'. */
        // printf("PERFORMING A+A' INSIDE SP_COLORDER.C!!!\n");
        double start_A_P_AT = SuperLU_timer_();
        at_plus_a_dist(n, Astore->nnz, Astore->colptr, Astore->rowind,
                &bnz, &b_colptr, &b_rowind);
        // printf("Matrix from the original A+A'!\n");

        // for (i = 0; i < n; i++) {
        //     printf("\nNeighbor of %d: ",i);
        //     for (j = b_colptr[i]; j < b_colptr[i+1]; j++) {
        //         printf("%d ",b_rowind[j]);
        //     }          
        // }
        // printf("\n");
        double end_A_P_AT = SuperLU_timer_();
        #if (DEBUGlevel == 0)
        printf("TIME TAKEN FOR A+A' = %f ms\n", (end_A_P_AT - start_A_P_AT)*1000);
        printf("A+A' number of non-zeros: %d\n",bnz);
        #endif

        /* Form C = Pc*B*Pc'. */
        c_colbeg = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
        c_colend = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));

        // int_t* b_colbeg = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
        // int_t* b_colend = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
        if (!(c_colbeg) || !(c_colend) )
            ABORT("SUPERLU_MALLOC fails for c_colbeg/c_colend");
        for (i = 0; i < n; i++) {
            c_colbeg[perm_c[i]] = b_colptr[i]; 
            c_colend[perm_c[i]] = b_colptr[i+1];
        }
        for (j = 0; j < n; ++j) {
            for (i = c_colbeg[j]; i < c_colend[j]; ++i) {
                b_rowind[i] = perm_c[b_rowind[i]];
            }
            iwork[perm_c[j]] = j; /* inverse perm_c */
            gSoFa_rev_perm[perm_c[j]] = j; /* inverse perm_c */
        }

        double start_symmetree = SuperLU_timer_();
        /* Compute etree of C. */
        sp_symetree_dist(c_colbeg, c_colend, b_rowind, n, etree);
        double end_symmetree = SuperLU_timer_();
        #if (DEBUGlevel == 0)
        printf("TIME TAKEN FOR SYMMETREE = %f ms\n", (end_symmetree - start_symmetree)*1000);
        #endif
        /* Restore B to be A+A', without column permutation */
        for (i = 0; i < bnz; ++i)
            b_rowind[i] = iwork[b_rowind[i]];
        //////////////////////////////////////////////////////// Maintainning the permutation while predicting nz count ////////

        /* Post order etree */
        double start_postorder = SuperLU_timer_();
        post = (int_t *) TreePostorder_dist(n, etree);
        double end_postorder = SuperLU_timer_();
        #if (DEBUGlevel == 0)
        printf("TIME TAKEN FOR POSTORDER = %f ms\n", (end_postorder - start_postorder)*1000);
        #endif
        invp  = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));

        for (i = 0; i < n; i++) invp[post[i]] = i;

        /* Renumber etree in postorder */
        for (i = 0; i < n; ++i) iwork[post[i]] = post[etree[i]];
        for (i = 0; i < n; ++i) etree[i] = iwork[i];        


        for (i = 0; i < n; ++i) iwork[post[i]] = ACstore_temp->colbeg[i];
        for (i = 0; i < n; ++i) ACstore_temp->colbeg[i] = iwork[i];  /*input graph to gSoFa*/
        for (i = 0; i < n; ++i) iwork[post[i]] = ACstore_temp->colend[i];
        for (i = 0; i < n; ++i) ACstore_temp->colend[i] = iwork[i];  /*input graph to gSoFa*/

        for (i = 0; i < n; ++i)
            iwork[i] = post[perm_c[i]];  /* product of perm_c and post */
        // #ifdef enable_perm_change
        for (i = 0; i < n; ++i) perm_c[i] = iwork[i];		
        for (i = 0; i < n; ++i) invp[perm_c[i]] = i; /* inverse of perm_c */
        // #endif

        for (j = 0; j < n; ++j) 
        {
            for (i = ACstore_temp->colbeg[j]; i < ACstore_temp->colend[j]; ++i)   /*input graph to gSoFa*/
            {
                int_t irow = ACstore_temp->rowind[i];
                ACstore_temp->rowind[i] = perm_c[irow];  /*input graph to gSoFa*/
            }
        }

        // SUPERLU_FREE(c_colbeg);
        // SUPERLU_FREE(c_colend);

        //Note Neither cholnzcnt() and relax_supnode()  modify etree[]. So we can use the same etree for supernode relaxation.
        //Supernode relaxation is done to accomodate relaxes 0.0 addition before predicting the memory for gSoFa
        //ACstore (CSC) for correct run of gSOFa in relaxed supernodes and b_colptr (A+AT) for correct size prediction in cholnzcnt() need to be expanded
#ifdef Enable_supernodal_relaxation          
        // double time_relax_supnode_start = SuperLU_timer_();
        int relax = sp_ienv_dist(2,options);
        // printf("Relaxation parameter: %d\n",relax);
        // fflush(stdout);
        int* desc_temp;
        int_t* relax_end = intMalloc_dist(n);
        int_t* relaxed_col = intMalloc_dist(n);
        if ( !(desc_temp = intMalloc_dist(n+1)) )
            ABORT("Malloc fails for desc_temp[]");
        gSoFa_para->etree =  etree;
        //  relax_supnode(vert_count, gSoFa_para->etree, relax, desc_temp, gSoFa_para->relax_end,gSoFa_para->relaxed_col);
        relax_supnode(n, gSoFa_para->etree, relax, desc_temp, relax_end, relaxed_col);
        gSoFa_para->relaxed_col = relaxed_col;
        gSoFa_para->relax_end = relax_end;
        SUPERLU_FREE(desc_temp);
        //~Computing relaxed supernodes
        // double time_relax_supnode = SuperLU_timer_() - time_relax_supnode_start;
        // fflush(stdout);


        /* Compute the Union of the relaxed supernode columns*/

        //While computing union, the input to the cholnzcnt() as well as gSoFa() needs to be expanded like in symbfact_SubXpand()

        double t1_relax_supernode = SuperLU_timer_();
        //Use relax_end to relax the supernodes from the symbolic factorization
        int_t* marker = (int_t *) intMalloc_dist(n);
        ifill_dist(marker, n, EMPTY);//Initialize marker to EMPTY (-1)
        // int_t* usub1 = (int_t *) intMalloc_dist(NNZ_U);
        // H_ERR(cudaMallocManaged((void**) &usub,sizeof(int_t)*NNZ_U)); //The prefix sum is inclusive
        int_t original_non_zero = ACstore_temp->nnz;
        int csr_counter = 0;
        for (int jcol=0; jcol<n;)
        {
            //if (( gSoFa_para->relax_end[jcol] != EMPTY ) && (gSoFa_para->relax_end[jcol] > jcol)) // For single process version
            if (gSoFa_para->relax_end[jcol] > jcol)
            {  /* beginning of a relaxed snode */ //Only if non-singleton relaxed supernode                       
                int kcol = gSoFa_para->relax_end[jcol];          /* end of the relaxed snode */
                /* Determine union of the row structure of supernode (j:k). */
                // printf("Computing union of relaxed supernode (%d:%d)\n",jcol,kcol);
                // fflush(stdout);
                // int k_counter = ACstore_temp->colbeg[jcol];
                int_t nonzero_cnt =0;
                ACstore->colbeg[jcol] = csr_counter;
                for (i=jcol; i<= kcol; i++)
                {
                    ACstore->nnz -=  ACstore_temp->colend[i] - ACstore_temp->colbeg[i];
                    // printf("\n ACstore_temp->colbeg[%d]:%d \t ACstore_temp->colend[%d]:%d\n",i,ACstore_temp->colbeg[i],i,ACstore_temp->colend[i]); 
                    // fflush(stdout);
                    for (int k = ACstore_temp->colbeg[i]; k < ACstore_temp->colend[i]; k++)
                    {                                
                        int_t irow = ACstore_temp->rowind[k];
                        int kmark = marker[irow];
                        if ( kmark != kcol ) 
                        {
                            marker[irow] = kcol;
                            ACstore->rowind[csr_counter] = irow;
                            nonzero_cnt++;
                            // usub1[k_counter] = irow;
                            csr_counter++;
                            #if (DEBUGlevel>=1)
                            if (csr_counter > original_non_zero) 
                            {
                                printf("Error! I.CSR counter exceeded original non-zero count.\n");
                                fflush(stdout);
                            }
                            #endif
                        }
                    }							
                }                 
                ACstore->colend[jcol] = csr_counter;
                ACstore->nnz += nonzero_cnt;
                for (i=jcol+1; i<= kcol; i++)
                {
                    etree[i-1] = i;
                    ACstore->colbeg[i] = ACstore->colbeg[jcol];
                    ACstore->colend[i] = ACstore->colend[jcol];
                    ACstore->nnz += nonzero_cnt;
                    // ACstore->nnz update

                }
                jcol = kcol+1;
            }
            else
            {
                //Non-leaf column. Move to the next column for the processing.
                ACstore->colbeg[jcol] = csr_counter;
                for (int k = ACstore_temp->colbeg[jcol]; k < ACstore_temp->colend[jcol]; k++)
                {
                    ACstore->rowind[csr_counter] = ACstore_temp->rowind[k];
                    csr_counter++;
                    #if (DEBUGlevel == 0)
                    if (csr_counter > original_non_zero) 
                    {
                        printf("Error! II.CSR counter exceeded original non-zero count.\n");
                        fflush(stdout);
                    }
                    #endif
                }
                ACstore->colend[jcol] = csr_counter;
                jcol++;
            }
        }
        // ACstore->nnz =  ACstore_temp->nnz;
        // Glu_freeable->usub = usub1;
        //~Implement supernodal relaxation for the column of the leaves in the postordered etree
        double t2_relax_supernode = SuperLU_timer_()-t1_relax_supernode;
        #if (DEBUGlevel == 0)
        printf("Time for relax supernode:%f\n",t2_relax_supernode);
        #endif
#endif
#if (DEBUGlevel == 0)
        printf("Inside  sp_colorder_Final() At Line: %d!\n",__LINE__);
        fflush(stdout);
#endif
#ifdef Enable_supernodal_relaxation
        //recalculate A1+A1_T and update the etree and make it post-ordered
        //A1 is ACstore->colbeg[], ACstore->colend[], ACstore->rowind[]

        /* Compute the etree of Pc*(A'+A)*Pc'. */
        // int_t *b_colptr_1, *b_rowind_1, bnz_1;
        // int_t j_1;
        /*Either un-permute ACstore or un permute A+A'*/
#if (DEBUGlevel == 0)
        printf("Before calling A1+A1'\n");
        #endif
        int_t* gSoFa_begPos  = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
        int_t* gSoFa_csr  = (int_t*) SUPERLU_MALLOC(ACstore->nnz*sizeof(int_t));
        int_t  count=0;
        gSoFa_begPos[0]=0;
        for (i=0; i<n; i++)
        {            
            for (j = ACstore->colbeg[i]; j < ACstore->colend[i]; ++j)
            {
                gSoFa_csr[count] = ACstore->rowind[j];
                count++;
            }
            gSoFa_begPos[i+1]=count;
        }
        a1t_plus_a1_dist(n, ACstore->nnz, ACstore->colbeg, ACstore->colend, ACstore->rowind,
                &bnz, &b_colptr, &b_rowind);

        SUPERLU_FREE(c_colbeg);
        SUPERLU_FREE(c_colend);
#endif
#if (DEBUGlevel == 0)
        printf("Before calling cholnzcnt()'\n");
        fflush(stdout);
        printf("Following is the input matrix to cholnzcont(): \n");
        fflush(stdout);
#endif
        int_t* temp_invp = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
        int_t* temp_perm_c = (int_t*) SUPERLU_MALLOC(n*sizeof(int_t));
        for (i = 0; i < n; ++i)
        {
            // printf("\nNeighbor of vertex: %d:\t",i);
            // fflush(stdout);
            temp_invp[i]= i;
            temp_perm_c[i]= i;

            // for (j = b_colptr[i]; j < b_colptr[i+1]; ++j)            
            // {
            //     printf("%d  ",b_rowind[j]);
            // }
        }
        fflush(stdout);
        double t1 = SuperLU_timer_();
        /*Note (b_colptr and b_rowind are just A + AT. Not column permuted) */
        cholnzcnt(n, b_colptr, b_rowind, temp_invp, temp_perm_c,
                etree, col_cnt, row_cnt, nlnz,
                part_super_h);
        double t2 = SuperLU_timer_() - t1;
#if (DEBUGlevel == 0)
        printf("cholnzcnt time: %f ms\n", t2*1000);
        fflush(stdout);
#endif

        //////////////////////////////////////////////////////// Maintainning the permutation while predicting nz count ////////

        //   #endif
#if ( DEBUGlevel>=3 )
        if ( !iam ) {
            PrintInt10("Pc*post:", n, perm_c);
            check_perm_dist("final perm_c", n, perm_c);
        }
#endif

        SUPERLU_FREE (post);
        SUPERLU_FREE (iwork);
        SUPERLU_FREE(b_colptr);
        if ( bnz ) SUPERLU_FREE(b_rowind);

    } /* end if options->Fact == DOFACT ... */


#if ( DEBUGlevel>=1 )
    /* Memory allocated but not freed:
       ACstore, ACstore->colbeg, ACstore->colend  */
    CHECK_MALLOC(iam, "Exit sp_colorder()");
#endif

}


void Allocate_Initialize_gSoFa_para(struct gSoFa_para_t *gSoFa_para, int_t BLKS_NUM, int_t blockSize, int_t next_front, int_t real_allocation, int_t max_supernode_size, int_t N_GPU_Node,int_t vert_count, int_t N_src_group/*, gridinfo_t grid*/)
{
    gSoFa_para->BLKS_NUM = BLKS_NUM;
    gSoFa_para->blockSize = blockSize;
    gSoFa_para->next_front = next_front;
    gSoFa_para->real_allocation = real_allocation;
    gSoFa_para->N_src_group = N_src_group;
    gSoFa_para->chunk_size = max_supernode_size;
    if (gSoFa_para->N_src_group < gSoFa_para->chunk_size)
    {
        gSoFa_para->chunk_size = gSoFa_para->N_src_group;
    }    
    gSoFa_para->total_num_chunks = ceil(vert_count/(float)gSoFa_para->chunk_size);
    gSoFa_para->N_chunks = (int) ceil (vert_count/(float)gSoFa_para->chunk_size);
    gSoFa_para->num_curr_chunks_per_node = (N_src_group * N_GPU_Node)/gSoFa_para->chunk_size;     
    gSoFa_para->total_num_chunks_per_node = ceil(gSoFa_para->total_num_chunks/(float)gSoFa_para->num_nodes);
    if (gSoFa_para->num_curr_chunks_per_node > gSoFa_para->total_num_chunks_per_node) gSoFa_para->num_curr_chunks_per_node = gSoFa_para->total_num_chunks_per_node;
    gSoFa_para->max_supernode_size = max_supernode_size;
    // gSoFa_para->mysrc = (int_t*)malloc(sizeof(int_t)*N_src_group*N_GPU_Node);
    if ( !(gSoFa_para->mysrc = (int_t*) SUPERLU_MALLOC( N_src_group*N_GPU_Node * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for gSoFa_para->mysrc[]");
    for (int i=0;i<N_GPU_Node*N_src_group;i++)
    {
        gSoFa_para->mysrc[i] = INT_MAX;
    }
    // gSoFa_para->counter = (int_t*)malloc(sizeof(int_t)*N_GPU_Node);
    if ( !(gSoFa_para->counter = (int_t*) SUPERLU_MALLOC( N_GPU_Node * sizeof(int_t)) ) )
        ABORT("SUPERLU_MALLOC fails for gSoFa_para->counter[]");

    H_ERR(cudaMallocManaged((void**)&gSoFa_para->my_chunks,gSoFa_para->num_curr_chunks_per_node*sizeof(int_t),cudaMemAttachGlobal));    
    gSoFa_para->sup_per_gpu = (int_t) ceil (vert_count/(float)N_GPU_Node*gSoFa_para->num_nodes);//Assuming singleton supernodes
    // gSoFa_para->relax_end = intMalloc_dist(vert_count);// for relaxed supernodes in gSofa
    //  gSoFa_para->relaxed_col = intMalloc_dist(vert_count);// for relaxed supernodes in gSofa



} 

void Allocate_Initialize (struct gSoFa_para_t *gSoFa_para, struct aux_device* device_obj,int_t vert_count,int_t edge_count,
        int_t BLKS_NUM,int_t blockSize,int index, int_t next_front,int_t N_src_group,int_t real_allocation,
        int_t N_chunks, int total_num_chunks_per_node, int max_supernode_size, cudaStream_t stream, int_t sup_per_gpu, 
        int_t iam, int_t gpu_id)

{
    device_obj->gpu_id=gpu_id; //Can be moved
    double start_time_unified_allocations = SuperLU_timer_();
    H_ERR(cudaMallocManaged( (void**)&device_obj->fill_in_d, sizeof(int_t)*N_src_group*vert_count,cudaMemAttachGlobal)); //Can be moved. Compute N_src_group in pdgssvx
    H_ERR(cudaMemAdvise(device_obj->fill_in_d, sizeof(int_t)*N_src_group*vert_count, cudaMemAdviseSetPreferredLocation, gpu_id));// Advising for cudaMemAdviseSetAccessedBy
    H_ERR(cudaMallocManaged((void**) &device_obj->nz_row_U_d,sizeof(int_t)*N_src_group,cudaMemAttachGlobal)); //Can be moved. Compute N_src_group in pdgssvx  
    H_ERR(cudaMemAdvise(device_obj->nz_row_U_d, sizeof(int_t)*N_src_group, cudaMemAdviseSetPreferredLocation, gpu_id));// Advising for cudaMemAdviseSetAccessedBy  
    H_ERR(cudaMallocManaged((void**) &device_obj->pass_through_d,sizeof(int_t)*N_src_group,cudaMemAttachGlobal)); //Can be moved. Compute N_src_group in pdgssvx
    H_ERR(cudaMemAdvise(device_obj->pass_through_d, sizeof(int_t)*N_src_group, cudaMemAdviseSetPreferredLocation, gpu_id));// Advising for cudaMemAdviseSetAccessedBy   
    double end_time_unified_allocations = SuperLU_timer_();
    #if (DEBUGlevel == 0)
    printf("GPU:%d Unified allocations time: %f ms\n",gpu_id, (end_time_unified_allocations - start_time_unified_allocations)*1000);
    fflush(stdout);
    #endif
    device_obj->max_id_offset = MAX_VAL-vert_count;//vert_count*group; //Can be moved
    device_obj->group_MAX_VAL = device_obj->max_id_offset + vert_count; //Can be moved
    device_obj->count_group_loop=0;//Can be moved   
    device_obj->last_row_U_count = 0; //Can be moved  
    device_obj->fill_count=0;	//Can be moved
    device_obj->group80_count=0;//Can be moved
    // device_obj->my_supernode = (int*) malloc (vert_count*sizeof(int_t)); //Can be moved
    double start_time_initialize_my_supernode = SuperLU_timer_();
    // H_ERR(cudaMallocHost((void**)&device_obj->my_supernode_d,vert_count*sizeof(int_t))); //Can be moved
    double time_initialize_mallochost =SuperLU_timer_() - start_time_initialize_my_supernode;
    #if (DEBUGlevel == 0)
    printf("time_initialize_mallochost: %f ms\n",time_initialize_mallochost*1000);
    #endif
    double start_time_cuda_malloc= SuperLU_timer_();
    H_ERR(cudaMalloc((void**) &device_obj->next_source_d,sizeof(int_t)));  //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->frontier_size_d,sizeof(int_t)));  //Can be moved  
     H_ERR(cudaMallocManaged((void**) &device_obj->cost_array_d,sizeof(uint_t)*N_src_group*vert_count,cudaMemAttachGlobal)); //Can be moved
    H_ERR(cudaMemAdvise(device_obj->cost_array_d, sizeof(uint_t)*N_src_group*vert_count, cudaMemAdviseSetPreferredLocation, index));// Advising for cudaMemAdviseSetAccessedBy
  H_ERR(cudaMalloc((void**) &device_obj->relaxed_col_d,sizeof(int_t)*vert_count)); //Can be moved
    //  H_ERR(cudaMallocManaged((void**) &device_obj->relaxed_col_d,sizeof(int_t)*vert_count,cudaMemAttachGlobal)); //Can be moved
   H_ERR(cudaMalloc((void**) &device_obj->relax_end_d,sizeof(int_t)*vert_count)); //Can be moved
    // H_ERR(cudaMallocManaged((void**) &device_obj->relax_end_d,sizeof(int_t)*vert_count,cudaMemAttachGlobal)); //Can be moved

#if 0
    H_ERR(cudaMemAdvise((void**) &device_obj->relaxed_col_d, sizeof(int_t)*vert_count, cudaMemAdviseSetPreferredLocation, index));// Advising for cudaMemAdviseSetAccessedBy
    H_ERR(cudaMemAdvise((void**) &device_obj->relax_end_d, sizeof(int_t)*vert_count, cudaMemAdviseSetPreferredLocation, index));// Advising for cudaMemAdviseSetAccessedBy
#endif

    //  H_ERR(cudaMallocHost((void**)&device_obj->cost_array_d,sizeof(uint_t)*N_src_group*vert_count)); //Can be moved
    double end_allocate_cost_array = SuperLU_timer_();
    // printf("GPU:%d Allocate cost array time: %f ms\n",index, (end_allocate_cost_array - start_allocate_cost_array)*1000);
    // fflush(stdout);      
    // H_ERR(cudaMalloc((void**) &device_obj->frontier_d,sizeof(int_t)*real_allocation)); //Can be moved
    // H_ERR(cudaMalloc((void**) &device_obj->next_frontier_d,sizeof(int_t)*real_allocation)); //Can be moved
    // H_ERR(cudaMalloc((void**) &device_obj->src_frontier_d,sizeof(int_t)*real_allocation));  //stores the code (mapping) of source not the source itself //Can be moved
    // H_ERR(cudaMalloc((void**) &device_obj->next_src_frontier_d,sizeof(int_t)*real_allocation)); //Can be moved

    H_ERR(cudaMallocManaged((void**) &device_obj->frontier_d,sizeof(int_t)*real_allocation,cudaMemAttachGlobal)); //Can be moved
    H_ERR(cudaMallocManaged((void**) &device_obj->next_frontier_d,sizeof(int_t)*real_allocation,cudaMemAttachGlobal)); //Can be moved
    H_ERR(cudaMallocManaged((void**) &device_obj->src_frontier_d,sizeof(int_t)*real_allocation,cudaMemAttachGlobal));  //stores the code (mapping) of source not the source itself //Can be moved
    H_ERR(cudaMallocManaged((void**) &device_obj->next_src_frontier_d,sizeof(int_t)*real_allocation,cudaMemAttachGlobal)); //Can be moved

    //Note Prefetch transfers are expensive. Hence, we should use the following code.
    H_ERR(cudaMemAdvise(device_obj->frontier_d, sizeof(int_t)*real_allocation, cudaMemAdviseSetPreferredLocation, gpu_id));// Advising for cudaMemAdviseSetAccessedBy 
    //  H_ERR(cudaMemPrefetchAsync ( device_obj->frontier_d, sizeof(int_t)*real_allocation, index, stream ));
    H_ERR(cudaMemAdvise(device_obj->next_frontier_d, sizeof(int_t)*real_allocation, cudaMemAdviseSetPreferredLocation, gpu_id));// Advising for cudaMemAdviseSetAccessedBy
    //   H_ERR(cudaMemPrefetchAsync ( device_obj->next_frontier_d, sizeof(int_t)*real_allocation, index, stream ));
    H_ERR(cudaMemAdvise(device_obj->src_frontier_d, sizeof(int_t)*real_allocation, cudaMemAdviseSetPreferredLocation, gpu_id));// Advising for cudaMemAdviseSetAccessedBy
    //   H_ERR(cudaMemPrefetchAsync ( device_obj->src_frontier_d, sizeof(int_t)*real_allocation, index, stream ));
    H_ERR(cudaMemAdvise(device_obj->next_src_frontier_d, sizeof(int_t)*real_allocation, cudaMemAdviseSetPreferredLocation, gpu_id));// Advising for cudaMemAdviseSetAccessedBy
    //   H_ERR(cudaMemPrefetchAsync ( device_obj->next_src_frontier_d, sizeof(int_t)*real_allocation, index, stream ));
    /*Note consider using 
      "cudaMemPrefetchAsync ( const void* devPtr, size_t count, int  dstDevice, cudaStream_t stream = 0 )" 
      if MemAdvise doesn't work.*/
      #if (DEBUGlevel == 0)
    printf("GPU:%d Tracker and frontier allocation time: %f ms\n",gpu_id, (SuperLU_timer_() - end_allocate_cost_array)*1000);
    fflush(stdout); 
    #endif
    H_ERR(cudaMalloc((void**) &device_obj->fill_count_d,sizeof(ull_t))); //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->source_d,sizeof(int_t)*N_src_group)); //Can be moved 
    H_ERR(cudaMalloc((void**) &device_obj->lock_d,sizeof(int)*BLKS_NUM)); //size of lock_d is num of blocks //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->next_front_d,sizeof( int_t))); //size of lock_d is num of blocks //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->next_frontier_size_d,sizeof(int_t)*N_src_group)); //Can be moved
    H_ERR(cudaMalloc((void**)&device_obj->Nsup_d,sizeof(int_t))); //Can be moved 
    double start_superObj_alloc = SuperLU_timer_();
    if ( !(device_obj->superObj = (struct Super*)SUPERLU_MALLOC( sup_per_gpu * sizeof(struct Super)) ) )
        ABORT("SUPERLU_MALLOC fails for device_obj->superObj[]");
    // device_obj->superObj = (struct Super*) malloc (sup_per_gpu*sizeof(struct Super)); //Can be moved //Super needs to be defined earlier
    H_ERR(cudaMalloc((void**) &device_obj->superObj_d,sizeof(struct Super)*(sup_per_gpu)));  //Can be moved //Super needs to be defined earlier 
    double end_superObj_alloc = SuperLU_timer_();
    #if (DEBUGlevel == 0)
    printf("GPU:%d SuperObj allocation time: %f ms\n",gpu_id, (end_superObj_alloc - start_superObj_alloc)*1000);
    #endif
    double time_cuda_malloc = SuperLU_timer_();
    #if (DEBUGlevel == 0)
    printf("GPU:%d Real allocation time: %f ms\n",gpu_id, (time_cuda_malloc - start_time_cuda_malloc)*1000);
    fflush(stdout);
    #endif

#ifndef asynch_initialization
#if 0
    H_ERR(cudaMalloc((void**) &device_obj->next_source_d,sizeof(int_t)));  //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->frontier_size_d,sizeof(int_t)));  //Can be moved  
    H_ERR(cudaMalloc((void**) &device_obj->cost_array_d,sizeof(uint_t)*N_src_group*vert_count)); //Can be moved

    double start_time_real_allocation = SuperLU_timer_();
    H_ERR(cudaMalloc((void**) &device_obj->frontier_d,sizeof(int_t)*real_allocation)); //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->next_frontier_d,sizeof(int_t)*real_allocation)); //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->src_frontier_d,sizeof(int_t)*real_allocation));  //stores the code (mapping) of source not the source itself //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->next_src_frontier_d,sizeof(int_t)*real_allocation)); //Can be moved
    double end_time_real_allocation = SuperLU_timer_();
    printf("GPU:%d Real allocation time: %f ms\n",index, (end_time_real_allocation - start_time_real_allocation)*1000);
    H_ERR(cudaMalloc((void**) &device_obj->fill_count_d,sizeof(ull_t))); //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->source_d,sizeof(int_t)*N_src_group)); //Can be moved 
    H_ERR(cudaMalloc((void**) &device_obj->lock_d,sizeof(int)*BLKS_NUM)); //size of lock_d is num of blocks //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->next_front_d,sizeof( int_t))); //size of lock_d is num of blocks //Can be moved
    H_ERR(cudaMalloc((void**) &device_obj->next_frontier_size_d,sizeof(int_t)*N_src_group)); //Can be moved
    H_ERR(cudaMalloc((void**)&device_obj->Nsup_d,sizeof(int_t))); //Can be moved 
    device_obj->superObj = (struct Super*) malloc (vert_count*sizeof(struct Super)); //Can be moved //Super needs to be defined earlier
    H_ERR(cudaMalloc((void**) &device_obj->superObj_d,sizeof(struct Super)*vert_count));  //Can be moved //Super needs to be defined earlier 
#endif

    H_ERR(cudaMemset(device_obj->frontier_size_d, 0, sizeof(int_t))); //Can be moved
    H_ERR(cudaMemset(device_obj->lock_d, 0, sizeof(int)*BLKS_NUM)); //Can be moved
    H_ERR(cudaMemcpy(device_obj->next_front_d,&next_front,sizeof(int_t),cudaMemcpyHostToDevice)); //Can be moved    
    // H_ERR(cudaMemcpy(device_obj->my_supernode_d,device_obj->my_supernode,sizeof(int_t)* vert_count,cudaMemcpyHostToDevice)); //Can be moved
    H_ERR(cudaMemset(device_obj->next_frontier_size_d, 0, sizeof(int_t)*N_src_group));//Can be moved
    H_ERR(cudaMemset(device_obj->fill_in_d, 0, sizeof(int_t)*N_src_group*vert_count));//Can be moved. Compute N_src_group in pdgssvx
    H_ERR(cudaMemset(device_obj->pass_through_d, 0, sizeof(int_t)*N_src_group));//Can be moved. Compute N_src_group in pdgssvx
    H_ERR(cudaMemset(device_obj->nz_row_U_d, 0, sizeof(int_t)*N_src_group));//Can be moved. Compute N_src_group in pdgssvx
    H_ERR(cudaMemset(device_obj->Nsup_d, 0, sizeof(int_t))); //Can be moved	
    H_ERR(cudaMemset(device_obj->fill_count_d, 0, sizeof(ull_t)));//Can be moved
#else
#if 0
    H_ERR(cudaMallocAsync((void**) &device_obj->next_source_d,sizeof(int_t),stream));  //Can be moved
    H_ERR(cudaMallocAsync((void**) &device_obj->frontier_size_d,sizeof(int_t),stream));  //Can be moved   
    H_ERR(cudaMallocAsync((void**) &device_obj->cost_array_d,sizeof(uint_t)*N_src_group*vert_count,stream)); //Can be moved
    double start_time_real_allocation = SuperLU_timer_();
    H_ERR(cudaMallocAsync((void**) &device_obj->frontier_d,sizeof(int_t)*real_allocation,stream)); //Can be moved
    H_ERR(cudaMallocAsync((void**) &device_obj->next_frontier_d,sizeof(int_t)*real_allocation,stream)); //Can be moved
    H_ERR(cudaMallocAsync((void**) &device_obj->src_frontier_d,sizeof(int_t)*real_allocation,stream));  //stores the code (mapping) of source not the source itself //Can be moved
    H_ERR(cudaMallocAsync((void**) &device_obj->next_src_frontier_d,sizeof(int_t)*real_allocation,stream)); //Can be moved
    double end_time_real_allocation = SuperLU_timer_();
    printf("GPU:%d Real allocation time: %f ms\n",index, (end_time_real_allocation - start_time_real_allocation)*1000);
    H_ERR(cudaMallocAsync((void**) &device_obj->fill_count_d,sizeof(ull_t),stream)); //Can be moved
    H_ERR(cudaMallocAsync((void**) &device_obj->source_d,sizeof(int_t)*N_src_group,stream)); //Can be moved 
    H_ERR(cudaMallocAsync((void**) &device_obj->lock_d,sizeof(int)*BLKS_NUM,stream)); //size of lock_d is num of blocks //Can be moved
    H_ERR(cudaMallocAsync((void**) &device_obj->next_front_d,sizeof( int_t),stream)); //size of lock_d is num of blocks //Can be moved
    H_ERR(cudaMallocAsync((void**) &device_obj->next_frontier_size_d,sizeof(int_t)*N_src_group,stream)); //Can be moved
    H_ERR(cudaMallocAsync((void**)&device_obj->Nsup_d,sizeof(int_t),stream)); //Can be moved 
    device_obj->superObj = (struct Super*) malloc (vert_count*sizeof(struct Super)); //Can be moved //Super needs to be defined earlier
    H_ERR(cudaMallocAsync((void**) &device_obj->superObj_d,sizeof(struct Super)*vert_count,stream));  //Can be moved //Super needs to be defined earlier 
#endif

    H_ERR(cudaMemsetAsync(device_obj->frontier_size_d, 0, sizeof(int_t),stream)); //Can be moved
    H_ERR(cudaMemsetAsync(device_obj->lock_d, 0, sizeof(int)*BLKS_NUM,stream)); //Can be moved
    H_ERR(cudaMemcpyAsync(device_obj->next_front_d,&next_front,sizeof(int_t),cudaMemcpyHostToDevice,stream)); //Can be moved
    // H_ERR(cudaMemcpyAsync(device_obj->my_supernode_d,device_obj->my_supernode,sizeof(int_t)* vert_count,cudaMemcpyHostToDevice,stream)); //Can be moved
    H_ERR(cudaMemsetAsync(device_obj->next_frontier_size_d, 0, sizeof(int_t)*N_src_group,stream));//Can be moved         
    H_ERR(cudaMemsetAsync(device_obj->fill_in_d, 0, sizeof(int_t)*N_src_group*vert_count,stream));//Can be moved. Compute N_src_group in pdgssvx
    H_ERR(cudaMemsetAsync(device_obj->pass_through_d, 0, sizeof(int_t)*N_src_group,stream));//Can be moved. Compute N_src_group in pdgssvx
    H_ERR(cudaMemsetAsync(device_obj->nz_row_U_d, 0, sizeof(int_t)*N_src_group,stream));//Can be moved. Compute N_src_group in pdgssvx
    H_ERR(cudaMemsetAsync(device_obj->Nsup_d, 0, sizeof(int_t),stream)); //Can be moved	
    H_ERR(cudaMemsetAsync(device_obj->fill_count_d, 0, sizeof(ull_t),stream));//Can be moved
    H_ERR(cudaMemcpyAsync(device_obj->relaxed_col_d, gSoFa_para->relaxed_col, sizeof(int_t)*vert_count, cudaMemcpyHostToDevice, stream));//For relaxed supernode in gSoFa
    H_ERR(cudaMemcpyAsync(device_obj->relax_end_d, gSoFa_para->relax_end, sizeof(int_t)*vert_count, cudaMemcpyHostToDevice, stream));//For relaxed supernode in gSoFa

#endif
}



int Compute_Src_group(int vert_count) //gSoFa added function
{
    // cout<<"Start finding N_src_per_group"<<endl;
    double temp = 2147483648/(double)(6*vert_count);
    if (temp > vert_count)
    {
        temp=(int)log_2(vert_count/(double)2);
        temp=pow(2,temp);

    }
    else
    {
        temp=(int)log_2(temp);
        temp=pow(2,temp);
    }
    // cout<<"Finished finding N_src_per_group"<<endl;
    return (int)temp;
}


void load_graph_gpu (struct aux_device* device_obj,int_t vert_count,int_t edge_count,int_t* csr,int_t* col_st,int_t* col_ed,
        int_t BLKS_NUM,int_t blockSize,int index, int_t next_front,int_t N_src_group,int_t real_allocation,int_t N_chunks,
        int total_num_chunks_per_node,int max_supernode_size,cudaStream_t stream, int iam)
{

    device_obj->gpu_id=index; //Can be moved
    H_ERR(cudaSetDevice(index));
    // PrintInt10("col_ed:",10,&col_ed[vert_count-10]);
    H_ERR(cudaMalloc((void**) &device_obj->csr_d,sizeof(int_t)*edge_count)); 	//Can be moved
#ifndef async_graphload
    H_ERR(cudaMemcpy(device_obj->csr_d, csr,sizeof(int_t)*edge_count,cudaMemcpyHostToDevice));//To be done after sp_colorder    
#else
    H_ERR(cudaMemcpyAsync(device_obj->csr_d, csr,sizeof(int_t)*edge_count,cudaMemcpyHostToDevice,stream));//To be done after sp_colorder    
#endif
    H_ERR(cudaMalloc((void**) &device_obj->col_st_d,sizeof(int_t)*vert_count)); //Can be moved	    
#ifndef async_graphload
    H_ERR(cudaMemcpy(device_obj->col_st_d, col_st,sizeof(int_t)*vert_count,cudaMemcpyHostToDevice));//To be done after sp_colorder    
#else
    H_ERR(cudaMemcpyAsync(device_obj->col_st_d, col_st,sizeof(int_t)*vert_count,cudaMemcpyHostToDevice,stream));//To be done after sp_colorder    
#endif
    H_ERR(cudaMalloc((void**) &device_obj->col_ed_d,sizeof(int_t)*vert_count)); //Can be moved    
#ifndef async_graphload
    H_ERR(cudaMemcpy(device_obj->col_ed_d, col_ed,sizeof(int_t)*vert_count,cudaMemcpyHostToDevice));//To be done after sp_colorder  
    //        printf("Process:%d Loaded col_ed on GPU %d\n",iam,index);
    // fflush(stdout);
#else   
    H_ERR(cudaMemcpyAsync(device_obj->col_ed_d, col_ed,sizeof(int_t)*vert_count,cudaMemcpyHostToDevice,stream));//To be done after sp_colorder  
    //      printf("Process:%d Asynchronous Loaded col_ed_d on GPU %d\n",iam,index);
    // fflush(stdout);
#endif    
} 

void findBlockDim(int* BLKS_NUM1, int* blockSize1)
{
    int BLKS_NUM,blockSize;
    int cfg_blockSize=128; 

    compute_blocksizes(&BLKS_NUM,&blockSize);        
    BLKS_NUM = (blockSize * BLKS_NUM)/cfg_blockSize;
    int exp=log2((float)BLKS_NUM);
    BLKS_NUM=pow(2,exp);
    blockSize = cfg_blockSize;  

    //  BLKS_NUM=512;
    // blockSize = 128;  

    *BLKS_NUM1=BLKS_NUM;
    *blockSize1=blockSize;
}

void compute_csr_offsets(int_t* NNZ_L, int_t* NNZ_U, Glu_freeable_t*  Glu_freeable, int_t* col_cnt_chol, int_t* row_cnt_chol, int_t n)
{
    //Works on CPU. Gives exclusive prefix sum. Initialization of the csr offsets
    //Non-gSoFa processes computes csr_offsets in CPU while gSoFa processes compute csr_offsets in GPU along with symbolic factorization
    //Compute NNZ_L, NNZ_U,Glu_freeable->xlsub_begin & Glu_freeable->xusub_begin
    int_t sumL = 0;
    int_t sumU = 0;
    for (int i=0; i< n; i++)
    {
        Glu_freeable->xlsub_begin[i] = sumL;
        Glu_freeable->xlsub_end[i] = sumL;
        #ifdef initializeCSR
        Glu_freeable->xlsub_original_nz_offset[i] = sumL;
        #endif

        Glu_freeable->xusub_begin[i] = sumU;
        Glu_freeable->xusub_end[i] = sumU;  
        #ifdef initializeCSR      
        Glu_freeable->xusub_original_nz_offset[i] = sumU;
        #endif

        sumL += row_cnt_chol[i];//Row count used for L
        sumU += col_cnt_chol[i];//Col count used for U
    }
    *NNZ_L = sumL;
    *NNZ_U = sumU;
}


void FreeMemory (struct gSoFa_para_t *gSoFa_para, struct aux_device* device_obj,int_t vert_count,int_t edge_count,
        int_t BLKS_NUM,int_t blockSize,int index, int_t next_front,int_t N_src_group,int_t real_allocation,int_t N_chunks, int total_num_chunks_per_node,int max_supernode_size,cudaStream_t stream, int_t sup_per_gpu, int_t iam, int_t gpu_id)

{ 
    //SuperLU_malloc and SuperLU_free are defined in util.c
    //setting debug level 1 for memory profiling
    H_ERR(cudaFree(device_obj->fill_in_d));
    H_ERR(cudaFree(device_obj->nz_row_U_d));
    H_ERR(cudaFree(device_obj->pass_through_d));   
    H_ERR(cudaFree(device_obj->next_source_d));
    H_ERR(cudaFree(device_obj->frontier_size_d));   
    H_ERR(cudaFree(device_obj->csr_d));
    H_ERR(cudaFree(device_obj->col_st_d));
    H_ERR(cudaFree(device_obj->col_ed_d));
    H_ERR(cudaFree(device_obj->cost_array_d));
    H_ERR(cudaFree(device_obj->relaxed_col_d));
    H_ERR(cudaFree(device_obj->relax_end_d));   
    H_ERR(cudaFree(device_obj->frontier_d));
    H_ERR(cudaFree(device_obj->next_frontier_d));     
    H_ERR(cudaFree(device_obj->src_frontier_d));     
    H_ERR(cudaFree(device_obj->next_src_frontier_d));     
    H_ERR(cudaFree(device_obj->fill_count_d)); 
    H_ERR(cudaFree(device_obj->source_d)); 
    H_ERR(cudaFree(device_obj->lock_d)); 
    H_ERR(cudaFree(device_obj->next_front_d)); 
    H_ERR(cudaFree(device_obj->next_frontier_size_d)); 
    H_ERR(cudaFree(device_obj->Nsup_d)); 
    SUPERLU_FREE(device_obj->superObj);
    H_ERR(cudaFree(device_obj->superObj_d));    

    /*
       Deallocating gSoFa parameters
       */
    SUPERLU_FREE(gSoFa_para->mysrc);
    SUPERLU_FREE(gSoFa_para->counter);
    SUPERLU_FREE(gSoFa_para->relaxed_col);
    H_ERR(cudaFree(gSoFa_para->my_chunks));

}

void dgSoFaInit( struct gSoFa_para_t** gSoFa_para1, int_t nprs)
{
    // int i;
    struct gSoFa_para_t* gSoFa_para;                        
    if ( !(gSoFa_para = (struct gSoFa_para_t*)SUPERLU_MALLOC( sizeof(struct gSoFa_para_t)) ) )
        ABORT("SUPERLU_MALLOC fails for gSoFa_para");
    double block_dim_start = SuperLU_timer_();
    findBlockDim(&gSoFa_para->BLKS_NUM, &gSoFa_para->blockSize);
    // double block_dim_time = SuperLU_timer_()-block_dim_start;
    gSoFa_para->is_gsofa=0;
    gSoFa_para->nprs=nprs;
    int N_GPU_resource = 0;
    H_ERR(cudaGetDeviceCount(&N_GPU_resource));
    gSoFa_para->N_gsofa_process_node = N_GPU_resource;
    if (nprs < N_GPU_resource)
    {
        gSoFa_para->N_gsofa_process_node = nprs;
    }   
    //Create streams for gSoFa         
    gSoFa_para->N_GPU_gSoFa_process = 1;// Number of GPU per gSoFa process in a compute node       
    if ( !(gSoFa_para->gSoFa_stream = (cudaStream_t*)SUPERLU_MALLOC( gSoFa_para->N_GPU_gSoFa_process * sizeof(cudaStream_t)) ) )
        ABORT("SUPERLU_MALLOC fails for gSoFa_para->gSoFa_stream[]");

    // printf("Time for detecting block dimension: %lf ms\n",block_dim_time*1000);
    // fflush(stdout);

    // int_t* test;
    // H_ERR(cudaMalloc((void**) &test,sizeof(int_t)*100));
    // H_ERR(cudaMemsetAsync(test, 0, sizeof(int_t)*100, gSoFa_para->gSoFa_stream[0])); //Can be moved
    *gSoFa_para1 = gSoFa_para;
}
// void dgSoFaInit(struct gSoFa_para_t** gSoFa_para1, gridinfo_t* grid, int_t nprs,Glu_persist_t* Glu_persist, dLUstruct_t* LUstruct, Glu_freeable_t** Glu_freeable1)
// //  dgSoFaInit(&gSoFa_para, grid, nprs, Glu_persist, LUstruct,  &Glu_freeable,&gSoFa_stream,n, iam, is_gsofa,global_collected, A, GA, options, perm_c, GAC, GACstore)
// {
//     struct gSoFa_para_t* gSoFa_para;
//     if ( !(gSoFa_para = (struct gSoFa_para_t*)SUPERLU_MALLOC( sizeof(struct gSoFa_para_t)) ) )
//         ABORT("SUPERLU_MALLOC fails for gSoFa_para");
//     findBlockDim(&gSoFa_para->BLKS_NUM, &gSoFa_para->blockSize);

//     int num_process_gSoFa =0;
//     int num_process; 
//     int_t N_src_group;
//     int_t max_supernode_size;
//     Initialize_gSoFa( grid,  &num_process,  nprs,Glu_persist, LUstruct, &Glu_freeable,
//             gSoFa_para, &gSoFa_stream, n, iam,is_gsofa,&num_process_gSoFa,
//             &N_src_group,global_collected, A, GA, options, perm_c, GAC, GACstore,&max_supernode_size);
//     gSoFa_para->num_process_gSoFa = num_process_gSoFa;
//     gSoFa_para->num_process = num_process;  
//     gSoFa_para->max_supernode_size = max_supernode_size;  
//     gSoFa_para->N_src_group = N_src_group;  
//     *gSoFa_para1 = gSoFa_para; 
//     // gSoFa_para = (struct gSoFa_para_t*)malloc(sizeof(struct gSoFa_para_t));
// }


void init_bcast(int_t* NNZ_L, int_t* NNZ_U, Glu_freeable_t*  Glu_freeable, int_t n, dScalePermstruct_t *ScalePermstruct, gridinfo_t *grid)
{
    /*gSoFa process broadcast the lsub/usub xlsub/xusub, perm_c to non-gSoFa processes*/
    MPI_Bcast(NNZ_L, 1, MPI_INT, 0,  grid->comm);
    MPI_Bcast(NNZ_U, 1, MPI_INT, 0,  grid->comm);
    MPI_Bcast(Glu_freeable->xlsub_begin, n, MPI_INT, 0, grid->comm);
    MPI_Bcast(Glu_freeable->xusub_begin, n, MPI_INT, 0,  grid->comm);
    // if (grid->iam ==0) 
    // {
    //       printf("IAM 0  NNZ_L:%d NNZ_U:%d\n",*NNZ_L,*NNZ_U);
    //     fflush(stdout);
    //     PrintInt10("IAM 0 xlsub_begin", 10, Glu_freeable->xlsub_begin);
    //     fflush(stdout);
    //     PrintInt10("IAM 0 xusub_begin", 10, Glu_freeable->xusub_begin);
    //     fflush(stdout);
    // }
    //             if (grid->iam ==7) 
    // {
    //     printf("IAM 7  NNZ_L:%d NNZ_U:%d\n",NNZ_L,NNZ_U);
    //     fflush(stdout);
    //     PrintInt10("IAM 7 xlsub_begin",10, Glu_freeable->xlsub_begin);
    //     fflush(stdout);
    //     PrintInt10("IAM 7 xusub_begin",10, Glu_freeable->xusub_begin);
    //     fflush(stdout);
    // }
    // MPI_Bcast(ScalePermstruct->perm_c, n, MPI_INT, 0,  grid->comm);
}

void supernodes_bcast(int_t* TotalNSupernode, Glu_persist_t* Glu_persist, int_t n, gridinfo_t *grid, int_t is_gSoFa, int_t ngprs, int_t nprs)
{
    #if 0
    MPI_Bcast(TotalNSupernode, 1, MPI_INT, 0, grid->comm);//Assuming Process 0 is always a gSoFa Process
    printf("Broadcasted TotalNSupernode: %d\n", *TotalNSupernode);
    fflush(stdout);
    MPI_Bcast(Glu_persist->supno, n, MPI_INT, 0, grid->comm);//Assuming Process 0 is always a gSoFa Process
    printf("Broadcasted supno:\n");
    fflush(stdout);
    MPI_Bcast(Glu_persist->xsup, *TotalNSupernode+1, MPI_INT, 0, grid->comm);//Assuming Process 0 is always a gSoFa Process     
    printf("Broadcasted xsup:\n");
    fflush(stdout); 
    #else 
    int num_process = grid->nprow*grid->npcol;
    int_t iam = grid->iam;
    // int_t ngp = grid->gSoFa.Np;
    int first_process_resource = (iam/nprs)*nprs;
    // int_t ngprs; //Number of gSoFa process per node
    if (is_gSoFa)
    {
            
        int last_process_resource = first_process_resource + nprs;
        // int last_process_resource = num_process
            MPI_Request *send_request;  				
    if ( !(send_request = (MPI_Request*) SUPERLU_MALLOC(num_process*sizeof(MPI_Request))))
        ABORT("Malloc fails for send_request[].");							  
    
        // int_t iam_gSoFa = grid->gSoFa.Iam;
        // if (iam_gSoFa%ngprs ==0)
        // {
        //     first_process_resource = 
        // }
    for (int dest_process= ngprs + iam; dest_process < last_process_resource; dest_process+=ngprs)
    {
        //iam sends supernode to dest_process
        #if (DEBUGlevel == 0)
        printf("Process:%d sending supernode information to process:%d\n", iam, dest_process);
        fflush(stdout);
        #endif
         MPI_Isend( TotalNSupernode, 1, MPI_INT,
                    dest_process, iam/*tag*/, grid->comm,&send_request[dest_process]);
         MPI_Isend( Glu_persist->supno, n, MPI_INT,
                    dest_process, iam/*tag*/, grid->comm,&send_request[dest_process]);
         MPI_Isend( Glu_persist->xsup, *TotalNSupernode+1, MPI_INT,
                    dest_process, iam/*tag*/, grid->comm,&send_request[dest_process]);
    }
    }
    else
    {
        //iam receives supernode from gSoFa process
          MPI_Request* recv_request;
          MPI_Status  status; 
          int_t source_process = iam%ngprs + first_process_resource;
           if ( !(recv_request = (MPI_Request*) SUPERLU_MALLOC(num_process*sizeof(MPI_Request))))
        ABORT("Malloc fails for recv_request[].");	
#if (DEBUGlevel == 0)
         printf("Process:%d receiving supernode information from process:%d\n", iam, source_process);
        fflush(stdout);
        #endif
         MPI_Recv(TotalNSupernode, 1, MPI_INT, source_process, source_process, grid->comm, &status );
          MPI_Recv( Glu_persist->supno, n, MPI_INT, source_process, source_process, grid->comm, &status );
           MPI_Recv( Glu_persist->xsup, *TotalNSupernode+1, MPI_INT, source_process, source_process, grid->comm, &status );
    }
    #endif   
}

void Finalize_supernodes(int_t is_gsofa, Glu_freeable_t* Glu_freeable, 
int_t NNZ_L, int_t NNZ_U, struct gSoFa_para_t* gSoFa_para, 
int_t* TotalNSupernode1, gridinfo_t* grid, Glu_persist_t* Glu_persist, int_t n)
{
    int_t TotalNSupernode=0;
    if (!is_gsofa)
    {
        //non-gSoFa process
        // if ( !(Glu_freeable->lsub = intMalloc_dist(NNZ_L)) )
        //     ABORT("Malloc fails for lsub[].");
        // if ( !(Glu_freeable->usub = intMalloc_dist(NNZ_U)) )
        //     ABORT("Malloc fails for usub[].");
        // memcpy(Glu_freeable->xlsub_end, Glu_freeable->xlsub_begin, n*sizeof(int_t));
        // memcpy(Glu_freeable->xusub_end, Glu_freeable->xusub_begin, n*sizeof(int_t));
    }
    else
    {
        //gSoFa process
        //Finalization of xsub and nsup across all gSoFa processes   

        for (int i=0;i<gSoFa_para->N_GPU_gSoFa_process;i++)
        {
            int localgpu = i + gSoFa_para->mygSoFaOffset;
            FreeMemory (gSoFa_para, &(gSoFa_para->dev_mem[i]), gSoFa_para->vert_count, 
                    gSoFa_para->edge_count, gSoFa_para->BLKS_NUM, gSoFa_para->blockSize,
                    i + gSoFa_para->mygSoFaOffset,gSoFa_para->next_front,gSoFa_para->N_src_group,
                    gSoFa_para->real_allocation, gSoFa_para->N_chunks,
                    gSoFa_para->total_num_chunks_per_node,gSoFa_para->max_supernode_size,gSoFa_para->gSoFa_stream[i],
                    gSoFa_para->sup_per_gpu, gSoFa_para->iam_gSoFa,localgpu);

        }
        MPI_Allreduce(&(gSoFa_para->Nsupernode_process), &TotalNSupernode, 1,
                  MPI_INT, MPI_SUM, grid->gSoFa.comm);
        // MPI_Reduce(&(gSoFa_para->Nsupernode_process), &TotalNSupernode, 1, MPI_INT, MPI_SUM, 0, grid->gSoFa.comm);
#if ( DEBUGlevel>=1 )
        printf("Broadcast of TotalNSupernode\n");
        fflush(stdout);
#endif
        // MPI_Bcast(&TotalNSupernode, 1, MPI_INT, 0, grid->gSoFa.comm);//Bcast within gSoFa Processes
        // //~Communicate among the processes gSoFa
#if ( DEBUGlevel>=1 )
        printf("After Broadcast of TotalNSupernode\n");
        fflush(stdout);
#endif
        // double supernode_arrangement_and_communication_start = SuperLU_timer_();

        /***Steps that does which does the supernode arrangement and communication
          xsup[s] is the leading column of the s-th supernode.
          supno[i] is the supernode no to which column i belongs;
          e.g.   supno 0 1 2 2 3 3 3 4 4 4 4 4   (n=12) 
          xsup 0 1 2 4 7 12

          Step1 Make myrepresentative(of supernode) buffer from every working process
          Step2 Stream myrepresentative(of supernode) buffer to Process 0
          Step3 Process 0 sorts the myrepresentative arrays (increasing order)
Note: Size of collected represenattive is the number of supernodes.
Step4: Create  Glu_persist->supno[i] //O(V). Iterate through sorted representative array rep[i] to < rep[i+1] xsup[count] = rep[i]
Step5: Create Glu_persist->xsup[s] //Update xsup as supno is being filled
Step6: Duplicate Glu_persist->supno[] and Glu_persist->xsup[] among all the processes
*/
        //Step1 Make myrepresentative(of supernode) buffer from every working process           
        int* sendRepresentativeBuffer = intMalloc_dist(gSoFa_para->Nsupernode_process);
        for (register int i=0;i<gSoFa_para->Nsupernode_process;i++)
        {
            sendRepresentativeBuffer[i]=gSoFa_para->Supernode_per_process[i].start;
        }
        //Step2 Stream myrepresentative(of supernode) buffer to Process 0
        int_t* displs = intMalloc_dist(gSoFa_para->num_process_gSoFa);
        int_t* rcount = intMalloc_dist(gSoFa_para->num_process_gSoFa);
        for (register int i=0;i<gSoFa_para->num_process_gSoFa;i++)
        {
            displs[i] = i;
            rcount[i] = 1;          
        }
        int* RecvRepresentativeBuffer    = intMalloc_dist(TotalNSupernode);
        //Find respective position in the array for every process receive the representative, Receive individual counts and perfom prefix sum
        int* IndivProcessSupCount = intMalloc_dist(gSoFa_para->num_process_gSoFa);        
        MPI_Allgatherv(&(gSoFa_para->Nsupernode_process),1,MPI_INT,IndivProcessSupCount, rcount,displs,MPI_INT,grid->gSoFa.comm);        
        //Perform prefix sum to receive the representatives of supernode from every process
        displs[0] = 0;
        for (register int i=1;i<gSoFa_para->num_process_gSoFa;i++)
        {
            displs[i] = displs[i-1]+IndivProcessSupCount[i-1];
            //Note: rcount = IndivProcessSupCount //Significant only at root. So no need to duplicate among other processes
        }        
        MPI_Allgatherv(sendRepresentativeBuffer,gSoFa_para->Nsupernode_process,MPI_INT,RecvRepresentativeBuffer, IndivProcessSupCount,displs,MPI_INT,grid->gSoFa.comm);
        //Step3 Process 0 sorts the myrepresentative arrays (increasing order)
        qsort(RecvRepresentativeBuffer, TotalNSupernode, sizeof(int), compare);
        //Note: Size of collected represenattive is the number of supernodes.
        //Step4 and Step 5: Create  Glu_persist->supno[i] //O(V). Iterate through sorted representative array rep[i] to < rep[i+1] xsup[count] = rep[i]        
        int vertIdx=0;
        // int supernodeCnt=0;

        for (int i=0;i<TotalNSupernode;i++)
        {
            int begin = RecvRepresentativeBuffer[i];
            int end = ((i +1) < TotalNSupernode) ? RecvRepresentativeBuffer[i+1]:n;
            Glu_persist->xsup[i]=vertIdx;
            while (begin<end)
            {
                Glu_persist->supno[vertIdx]=i;//O(V)
                begin++;
                vertIdx++;
            }
        } 
        Glu_persist->xsup[TotalNSupernode]=n;//O(Nsup +1)
        // double supernode_arrangement_and_communication = SuperLU_timer_() - supernode_arrangement_and_communication_start;         
    } //Communicator logic ends here

    //After this all idle processes as well as the gSoFa will participate in lsub and usub communication based on supernode information
    //Step6: Duplicate Glu_persist->supno[] and Glu_persist->xsup[] among all the processes (Idle as well as the gSoFa process)
    // printf("Global Process:%d starting the MPI_Bcast \n",iam);
    // fflush(stdout);
#if ( DEBUGlevel>=1 )
    printf("Broadcast of Supernodes\n");
    fflush(stdout);
#endif
    double supernode_Bcast_start_time = SuperLU_timer_();
    supernodes_bcast(&TotalNSupernode, Glu_persist, n, grid,is_gsofa,gSoFa_para->N_gsofa_process_node,gSoFa_para->nprs);//to all the processes from process 0
    double supernode_Bcast_time = SuperLU_timer_() - supernode_Bcast_start_time;
    *TotalNSupernode1 = TotalNSupernode;
#if ( DEBUGlevel>=1 )
    printf("After Broadcast of Supernodes\n");
    fflush(stdout);
#endif
#if (DEBUGlevel == 0)
       if (grid->iam==0) printf("IAM:%d Supernode Bcast time:%lf ms", grid->iam, supernode_Bcast_time*1000);
#endif
}

void merge_csr (int_t RecvCntL, int_t RecvCntU,  Glu_freeable_t* Glu_freeable)
{
    
    
#if ( DEBUGlevel>=1 )
// double merge_csr_start = SuperLU_timer_();
    printf("Process:%d merging lsub with RecvCntL:%d\n",iam,RecvCntL);
    fflush(stdout);
#endif
    for (int i=0;i<RecvCntL;i++)
    {
        int row =  Glu_freeable->itemp_L_rowIdx[i];		
        int jcolIdx = Glu_freeable->xlsub_end[row];		
        Glu_freeable->lsub[jcolIdx] =  Glu_freeable->itemp_L[i];
        Glu_freeable->xlsub_end[row] = ++jcolIdx ; 
    }
#if ( DEBUGlevel>=1 )
    printf("Process:%d Finished merging lsub with RecvCntL:%d\n",iam,RecvCntL);
    fflush(stdout);
    printf("Process:%d merging usub with RecvCntL:%d\n",iam,RecvCntU);
    fflush(stdout);
#endif
    for (int i=0;i<RecvCntU;i++)
    {
        int row =  Glu_freeable->itemp_U_rowIdx[i];	
        int jcolIdx = Glu_freeable->xusub_end[row];		
        Glu_freeable->usub[jcolIdx] =  Glu_freeable->itemp_U[i];
        // printf("Adding row index: %d at column: %d during the lsub communication!\n", itemp_U[i], row);
        // fflush(stdout);
        Glu_freeable->xusub_end[row] = ++jcolIdx ; 
    }
#if ( DEBUGlevel>=1 )
    printf("Process:%d Finished merging usub with RecvCntU:%d\n",iam,RecvCntU);
    fflush(stdout);
    double merge_csr_time = SuperLU_timer_() - merge_csr_start;
    printf("Process:%d Merge CSR Time:%f ms\n",iam,merge_csr_time*1000);
#endif

}

void Communicate_lsub_usub(int_t num_process,Glu_persist_t* Glu_persist, struct gSoFa_para_t* gSoFa_para,Glu_freeable_t* Glu_freeable, gridinfo_t* grid,int_t n )
{
    int_t k;
    int_t iam = grid->iam;

    // Step 1: Prepare sending buffer per process
    //First count the number of nonz-zeros to be sent, then allocate and send to the target processes.
#if (DEBUGlevel >=1)
    printf("Process:%d Allocating space  %d for ToSendL/ToSendU\n",iam,num_process);
    fflush(stdout);
#endif

    // double lsubUsubSendCount_start = SuperLU_timer_();
    int* ToSendL = intCalloc_dist(num_process);
    int* ToSendU = intCalloc_dist(num_process);
    int* ToRecvU = intCalloc_dist(num_process);
    int* ToRecvL = intCalloc_dist(num_process);
    int_t TotalSendCntL=0;
    int_t TotalSendCntU=0; 
    // #ifdef initializeCSR
    // PrintInt10("xlsub_begin",10, Glu_freeable->xlsub_begin);
    // PrintInt10("xlsub_original_nz_offset",10, Glu_freeable->xlsub_original_nz_offset);    
    // PrintInt10("xlsub_end", 10, Glu_freeable->xlsub_end);

    // PrintInt10("xusub_begin",10, Glu_freeable->xusub_begin);
    // PrintInt10("xusub_original_nz_offset",10, Glu_freeable->xusub_original_nz_offset);    
    // PrintInt10("xusub_end", 10, Glu_freeable->xusub_end);
    // fflush(stdout);
    // #endif
    // if (gSoFa_communicator  != MPI_COMM_NULL)    
    {
        ////only gSoFa processes will send the lsub and usub structures
        for (int irow=0;irow<n;irow++)
        {						
            int gbj = BlockNum_gSoFa(irow,Glu_persist->supno);		
            #ifdef initializeCSR            
            for (int jcolIdx= Glu_freeable->xlsub_original_nz_offset[irow]; jcolIdx < Glu_freeable->xlsub_end[irow]; jcolIdx++)
            #else										
            for (int jcolIdx= Glu_freeable->xlsub_begin[irow]; jcolIdx < Glu_freeable->xlsub_end[irow]; jcolIdx++)
            #endif
            {
                int jcol = Glu_freeable->lsub[jcolIdx];//Confirm lsub is coming from gSoFa module				
                if (jcol < n)
                {
#ifdef skipOriginalNonzeros
                    if (gSoFa_para->is_OriginalNZ_L[jcol] == 1) continue; //only communicate the fill-ins								
#endif
                    int gbi = BlockNum_gSoFa( jcol,Glu_persist->supno );		
                    // #ifdef segment_logic
                    // seg_array[gbi] = minimum(jcol,seg_array[gbi]);
                    // #endif						
                    int p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );
                    ++ToSendL[p];
                    TotalSendCntL++;
                }							
            }
            gbj = BlockNum_gSoFa(irow,Glu_persist->supno);
            #ifdef initializeCSR
             for (int jcolIdx= Glu_freeable->xusub_original_nz_offset[irow]; jcolIdx < Glu_freeable->xusub_end[irow]; jcolIdx++)
            #else
            for (int jcolIdx= Glu_freeable->xusub_begin[irow]; jcolIdx < Glu_freeable->xusub_end[irow]; jcolIdx++)
            #endif
            {			
                // printf("Accessing jcolIdx:%d\n",jcolIdx);
                // fflush(stdout);				
                int jcol = Glu_freeable->usub[jcolIdx]; 
                if (jcol < n)
                {						
#ifdef skipOriginalNonzeros
                    if (gSoFa_para->is_OriginalNZ_U[jcol] == 1) continue; //only communicate the fill-ins								
#endif							
                    int gbi = BlockNum_gSoFa( jcol,Glu_persist->supno );
                    int p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );									
                    ++ToSendU[p];
                    TotalSendCntU++;
                }							
            } 
        }
    }//end of if (gSoFa_communicator  != MPI_COMM_NULL)
    
#if (DEBUGlevel >=1)
    double lsubUsubSendCount = SuperLU_timer_() - lsubUsubSendCount_start;
    printf("Process:%d Stage 1 SendCountL and U time: %lf ms \n",iam,lsubUsubSendCount*1000);
#endif
    int_t* sending_BufferL; int_t* sending_BufferU;
    int_t* sending_BufferL_rowIdx; int_t* sending_BufferU_rowIdx;
    if (TotalSendCntL)//Can be zero
    {					
        if ( !(sending_BufferL = intMalloc_dist(TotalSendCntL)) )
            ABORT("Malloc fails for index[].");
        if ( !(sending_BufferL_rowIdx = intMalloc_dist(TotalSendCntL)) )
            ABORT("Malloc fails for index[].");
    }
    if (TotalSendCntU)//Can be zero
    {
        if ( !(sending_BufferU = intMalloc_dist(TotalSendCntU)) )
            ABORT("Malloc fails for index[].");					
        if ( !(sending_BufferU_rowIdx = intMalloc_dist(TotalSendCntU)) )
            ABORT("Malloc fails for index[].");
    }

    int_t** ia_sendL; 
    int_t** ia_sendU;
    int_t** ia_sendL_rowIdx;
    int_t** ia_sendU_rowIdx;
    //Allocation
    if ( !(ia_sendL = (int_t **) SUPERLU_MALLOC(num_process*sizeof(int_t*))) )
        ABORT("Malloc fails for ia_send[].");
    if ( !(ia_sendU = (int_t **) SUPERLU_MALLOC(num_process*sizeof(int_t*))) )
        ABORT("Malloc fails for ia_send[].");
    if ( !(ia_sendL_rowIdx = (int_t **) SUPERLU_MALLOC(num_process*sizeof(int_t*))) )
        ABORT("Malloc fails for ia_send[].");
    if ( !(ia_sendU_rowIdx = (int_t **) SUPERLU_MALLOC(num_process*sizeof(int_t*))) )
        ABORT("Malloc fails for ia_send[].");
    //~allocation

    int j=0; int i=0; 
    for (int p=0;p<num_process;p++)
    {
        // if (p!=myrank)
        {
            ia_sendU[p]= &sending_BufferU[i];
            ia_sendU_rowIdx[p] = &sending_BufferU_rowIdx[i];
            i += ToSendU[p]; 

            ia_sendL[p]= &sending_BufferL[j];
            ia_sendL_rowIdx[p] = &sending_BufferL_rowIdx[j];
            j += ToSendL[p]; 
        }
    }
#if (DEBUGlevel >=1)
    printf("Process:%d Communicate LSub USub Stage 2!!!\n",iam);
    fflush(stdout);					
#endif
    int_t* ptr_to_sendL;
    if ( !(ptr_to_sendL = intCalloc_dist(num_process)) )
        ABORT("Malloc fails for ptr_to_sendL[].");					
    int_t* ptr_to_sendU;
    if ( !(ptr_to_sendU = intCalloc_dist(num_process)) )
        ABORT("Malloc fails for ptr_to_sendU[].");

    //Fill up the sending buffer
    double filling_time_start = SuperLU_timer_();
    // if (gSoFa_communicator  != MPI_COMM_NULL)    
    {
        ////only gSoFa processes will send the lsub and usub structures
        for (int irow=0;irow<n;irow++)
        {					
            int gbj = BlockNum_gSoFa(irow,Glu_persist->supno);
            #ifdef initializeCSR
             for (int jcolIdx= Glu_freeable->xlsub_original_nz_offset[irow]; jcolIdx < Glu_freeable->xlsub_end[irow]; jcolIdx++)           
            #else
            for (int jcolIdx= Glu_freeable->xlsub_begin[irow]; jcolIdx < Glu_freeable->xlsub_end[irow]; jcolIdx++)           
            #endif
            {
                //       printf("Accessing jcolIdx:%d\n",jcolIdx);
                // fflush(stdout);		
                int jcol = Glu_freeable->lsub[jcolIdx];		
#ifdef skipOriginalNonzeros
                if (gSoFa_para->is_OriginalNZ_L[jcol] == 1) continue; //only communicate the fill-ins								
#endif			
                int gbi = BlockNum_gSoFa( jcol,Glu_persist->supno );							
                int p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );
                if ( p != iam ) 
                { /* remote */
                    k = ptr_to_sendL[p];						
                    ia_sendL[p][k] = jcol;
                    ia_sendL_rowIdx[p][k] = irow;//Later need to combine these two arrays
                    ++ptr_to_sendL[p];
                } 
            }						
            gbj = BlockNum_gSoFa(irow,Glu_persist->supno);	
            #ifdef initializeCSR
             for (int jcolIdx= Glu_freeable->xusub_original_nz_offset[irow]; jcolIdx < Glu_freeable->xusub_end[irow]; jcolIdx++)
            #else					
            for (int jcolIdx= Glu_freeable->xusub_begin[irow]; jcolIdx < Glu_freeable->xusub_end[irow]; jcolIdx++)
            #endif
            {
                int jcol = Glu_freeable->usub[jcolIdx];
#ifdef skipOriginalNonzeros
                if (gSoFa_para->is_OriginalNZ_U[jcol] == 1) continue; //only communicate the fill-ins								
#endif
                int gbi = BlockNum_gSoFa( jcol, Glu_persist->supno );
                int p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );
                if ( p != iam ) 
                { /* remote */
                    k = ptr_to_sendU[p];								
                    ia_sendU[p][k] = jcol;
                    ia_sendU_rowIdx[p][k] = irow;//Later need to combine these two arrays
                    ++ptr_to_sendU[p];
                }
            }
        }
#if (DEBUGlevel >=1)
        printf("Process:%d filled U structure to send to remote processes.\n",iam);
        fflush(stdout);
#endif
    } //end of if (gSoFa_communicator  != MPI_COMM_NULL)
    
#if (DEBUGlevel >=1)
    double filling_time = SuperLU_timer_() -filling_time_start;
    printf("Process:%d  LSub USub Stage 2 Filling time: %lf ms!!!\n",iam,filling_time*1000);
    printf("Process:%d Communicate LSub USub Stage 3!!!\n",iam);
    fflush(stdout);
#endif
    // int_t** sending_Buffer;
    MPI_Request *send_request;
    MPI_Request* recv_request;					
    if ( !(send_request = (MPI_Request*) SUPERLU_MALLOC(4*num_process*sizeof(MPI_Request))))
        ABORT("Malloc fails for send_request[].");							
    if ( !(recv_request = (MPI_Request*) SUPERLU_MALLOC(4*num_process*sizeof(MPI_Request))))
        ABORT("Malloc fails for recv_request[].");
    MPI_Status  status; 
#if (DEBUGlevel >=1)
    printf("Process:%d started communication of U structures!\n",iam);
    fflush(stdout);
#endif
    double communication_start = SuperLU_timer_();
    for (int process=0; process < num_process; process++)
    {
        if (process!=iam)
        {
            MPI_Isend( &ToSendU[process], 1, MPI_INT,
                    process, iam/*tag*/, grid->comm,&send_request[process]);							
            MPI_Irecv(&ToRecvU[process], 1, MPI_INT, process, process,  grid->comm, &recv_request[process]);
        }
        else
        {							
            ToRecvU[iam] = ToSendU[iam];
        }
    }
    for(int process=0;process<num_process;process++)
    {
        if (process !=iam)
        {							
            MPI_Wait(&send_request[process], &status);
            MPI_Wait(&recv_request[process], &status);						
        }
    }
#if (DEBUGlevel >=1)
    printf("Process:%d finished communication of U structures!\n",iam);
    fflush(stdout);
#endif
    //Normal send receive strctures
#if (DEBUGlevel >=1)
    printf("Process:%d started communication of L structures!\n",iam);
    fflush(stdout);
#endif
    // int process = iam+1;
    for (int process=0; process < num_process; process++)
    {
        if (process!=iam)
        {
            MPI_Isend( &ToSendL[process], 1, MPI_INT,
                    process, iam/*tag*/, grid->comm,&send_request[process]);
            MPI_Irecv(&ToRecvL[process], 1, MPI_INT, process, process,  grid->comm, &recv_request[process]);
        }
        else
        {
            ToRecvL[iam] = ToSendL[iam];

        }
    }			

    for(int process=0;process<num_process;process++)
    {
        if (process !=iam)
        {
            MPI_Wait(&send_request[process], &status);
            MPI_Wait(&recv_request[process], &status);							
        }
    }
#if (DEBUGlevel >=1)
    printf("Process:%d finished communication of L structures!\n",iam);
    fflush(stdout);
#endif


    //~Normal send receive strctures	
#if (DEBUGlevel >=1)			
    printf("Process:%d Finished communicating the #L and #U to be communcated among the processes!\n",iam);
    fflush(stdout);
#endif
    //     rank    send buf                        recv buf
    // ----    --------                        --------
    //  0      a,b,c          MPI_Alltoall     a,A,#
    //  1      A,B,C        ---------------->  b,B,@
    //  2      #,@,%                           c,C,%

    //Send to the processes
    int_t SendCntU=0; int_t SendCntL=0; int_t RecvCntU=0;int_t RecvCntL=0;
    for (int_t p = 0; p < num_process; ++p) 
    {

        if (p != iam ) 
        {
            SendCntU += ToSendU[p];
            SendCntL += ToSendL[p];
            RecvCntU += ToRecvU[p];//These are count maintained for storing locally
            RecvCntL += ToRecvL[p];//These are count maintained for storing locally							
            // maxnnzToRecvU = SUPERLU_MAX( ToRecvU[p], maxnnzToRecvU );
            // maxnnzToRecvL = SUPERLU_MAX( ToRecvL[p], maxnnzToRecvL );
        } 
    }
#if (DEBUGlevel >=1)
    printf("Process:%d Communicate LSub USub Stage 4!!!\n",iam);
    fflush(stdout);
#endif
    //Step 2: Communicate among all the processes
    //Send the buffered arrays
    for (int_t p = 0; p < num_process; ++p) 
    {
        if ( p != iam ) 
        {
            int itL = ToSendL[p];

            if (itL!=0)
            {
                #if (DEBUGlevel >= 1)
                printf("Process: %d sending itL:%d  values (i, j) to process: %d\n",iam,itL,p);
                fflush(stdout);
                #endif
                MPI_Isend( ia_sendL[p], itL, MPI_INT,
                        p, iam/*tag*/, grid->comm, &send_request[p] );

                MPI_Isend( ia_sendL_rowIdx[p], itL, MPI_INT,
                        p, iam+num_process/*tag*/, grid->comm, &send_request[p+num_process] );
            }
            int itU = ToSendU[p];

            if (itU!=0)
            {
                #if (DEBUGlevel >= 1)
                printf("Process: %d sending itU:%d  values (i, j) to process: %d\n",iam,itU,p);
                fflush(stdout);
                #endif

                MPI_Isend( ia_sendU[p], itU, MPI_INT,
                        p, iam+2*num_process/*tag*/, grid->comm, &send_request[p+2*num_process] );
                MPI_Isend( ia_sendU_rowIdx[p], itU, MPI_INT,
                        p, iam+3*num_process/*tag*/, grid->comm, &send_request[p+3*num_process] );
            }             							
        }
    }
    //Receive the buffered array					
    int_t* ToRecvOffsetL;
    int_t* ToRecvOffsetU;
    if ( !(ToRecvOffsetL = intCalloc_dist(num_process)) )
        ABORT("Malloc fails for ToRecvOffsetL[].");
    if ( !(ToRecvOffsetU = intCalloc_dist(num_process)) )
        ABORT("Malloc fails for ToRecvOffsetU[].");
    int_t* itemp_L; int_t* itemp_U;int_t* itemp_L_rowIdx; int_t* itemp_U_rowIdx;
    if (RecvCntL)//Can be zero
    {
        itemp_L = intMalloc_dist(RecvCntL);					
        itemp_L_rowIdx = intMalloc_dist(RecvCntL);
    } 
    if (RecvCntU)//Can be zero
    {
        itemp_U = intMalloc_dist(RecvCntU);					
        itemp_U_rowIdx = intMalloc_dist(RecvCntU);
    }
    // Glu_freeable->RecvCntU = &RecvCntU; 
    // Glu_freeable->RecvCntL = &RecvCntL; 
    Glu_freeable->itemp_U = itemp_U;
    Glu_freeable->itemp_L = itemp_L;
    Glu_freeable->itemp_U_rowIdx = itemp_U_rowIdx;
    Glu_freeable->itemp_L_rowIdx = itemp_L_rowIdx;
    int_t sum_offsetL=0;
    int_t sum_offsetU=0;
#if (DEBUGlevel >=1)
    printf("Process:%d Communicate LSub USub Stage 5!!!\n",iam);
    fflush(stdout);
#endif
    for (int_t p = 0; p < num_process; ++p) 
    {
        ToRecvOffsetL[p]=sum_offsetL;
        ToRecvOffsetU[p]=sum_offsetU;
        if ( p != iam )
        {
            int itL = ToRecvL[p];
            if (itL!=0)
            {
                #if (DEBUGlevel >=1)
                printf("Process: %d Waiting for itL:%d  values (i, j) from process: %d\n",iam,itL,p);
                fflush(stdout);            
                #endif
                // MPI_Recv( &itemp_L[sum_offsetL], itL, MPI_INT, p, p, grid->comm, &status );//index : Duplicate with the row number. Put the row to the respective lsub and usub
                // MPI_Recv( &itemp_L_rowIdx[sum_offsetL], itL, MPI_INT, p, p+num_process, grid->comm, &status );//index : Duplicate with the row number. Put the row to the respective lsub and usub

                MPI_Irecv( &itemp_L[sum_offsetL], itL, MPI_INT, p, p, grid->comm, &recv_request[p] );//index : Duplicate with the row number. Put the row to the respective lsub and usub
                MPI_Irecv( &itemp_L_rowIdx[sum_offsetL], itL, MPI_INT, p, p+num_process, grid->comm,&recv_request[p + num_process]);
            }
            int itU = ToRecvU[p];                            
            if (itU!=0)
            {
                #if (DEBUGlevel >=1)
                printf("Process: %d Waiting for itU:%d  values (i, j) from process: %d\n",iam,itU,p);
                fflush(stdout);
                #endif
                // MPI_Irecv(&ToRecvU[process], 1, MPI_INT, process, process,  grid->comm, &recv_request[process]);
                 MPI_Irecv( &itemp_U[sum_offsetU], itU, MPI_INT, p, p+2*num_process, grid->comm,&recv_request[p+2*num_process] );
                // MPI_Recv( &itemp_U[sum_offsetU], itU, MPI_INT, p, p+2*num_process, grid->comm, &status );
                MPI_Irecv( &itemp_U_rowIdx[sum_offsetU], itU, MPI_INT, p, p+3*num_process, grid->comm, &recv_request[p+3*num_process] );
                // MPI_Recv( &itemp_U_rowIdx[sum_offsetU], itU, MPI_INT, p, p+3*num_process, grid->comm, &status );
            }
            sum_offsetL += itL;
            sum_offsetU += itU;
        }
    }
    //wait until process receives the required structure from remote process
    for (int_t p = 0; p < num_process; ++p) 
    {
        if ( p != iam )
        {
             if ( ToRecvL[p]!=0)
            {
                MPI_Wait(&recv_request[p], &status);	
                MPI_Wait(&recv_request[p+num_process], &status);	
            }
            if ( ToRecvU[p]!=0)
            {
                MPI_Wait(&recv_request[p+2*num_process], &status);	
                MPI_Wait(&recv_request[p+3*num_process], &status);	
                // MPI_Wait();
            }
        }
    }
    
#if (DEBUGlevel >=1)
    double communication_time = SuperLU_timer_() - communication_start;
    printf("Process: %d Actual communication time: %f ms\n",iam,communication_time*1000);
    fflush(stdout);
#endif
    //Step 3: update to respective supernodal lsub and usub
    // #endif
#if (DEBUGlevel >=1)
    printf("Process:%d LSub USub Communicate Time:%f ms\n",iam,lsubUsubCommunicate*1000);
    printf("Process:%d TotalNSupernode:%d!\n",iam,TotalNSupernode);
    printf("Process:%d finished symbolic factorization!\n",iam);
    fflush(stdout);
#endif
    double receive_csr_merge_start = SuperLU_timer_();
    merge_csr(RecvCntL, RecvCntU, Glu_freeable);
    
#if (DEBUGlevel >=1)
    double receive_csr_merge_time = SuperLU_timer_() - receive_csr_merge_start;
    printf("Process:%d receive_csr_merge_time:%f ms\n",iam,receive_csr_merge_time*1000);
    fflush(stdout);
#endif
}

void U_Segmentation(int_t TotalNSupernode, int_t n, Glu_freeable_t* Glu_freeable, Glu_persist_t* Glu_persist, int_t NNZ_U)
{

#if (DEBUGlevel >=1)		
    printf("Process:%d performing segmentation logic\n",iam);		
    fflush(stdout);
#endif
    int_t* lsub1;    
    if ( !(lsub1 = intMalloc_dist(NNZ_U)) ) //Lsub will later be swapped with usub 
        ABORT("Malloc fails for lsub1[].");
    int_t* segmentFlagArray = intMalloc_dist(TotalNSupernode);
    int_t* flagArray = intMalloc_dist(TotalNSupernode);
    memset(flagArray, 0, TotalNSupernode*sizeof(int_t));              
    for (int iter=0;iter<n;iter++)
    {  
        for (int begin = Glu_freeable->xlsub_begin[iter]; begin < Glu_freeable->xlsub_end[iter];begin++)
        {
            int row = Glu_freeable->lsub[begin];              
            int my_supernode = Glu_persist->supno[row];
            if (flagArray[my_supernode] < (iter+1))
            {
                //the segment my_supernode encountered for the first time in the column iter
                flagArray[my_supernode] = iter+1;
                segmentFlagArray[my_supernode] = row;
            }
            else 
            {
                if ( segmentFlagArray[my_supernode] > row) segmentFlagArray[my_supernode] =row;
            }
        }	
        //Creating the supernodal structure of lsub
        int newEnd = Glu_freeable->xlsub_begin[iter];				
        for (int begin = Glu_freeable->xlsub_begin[iter]; begin < Glu_freeable->xlsub_end[iter];begin++)
        {			
            int row = Glu_freeable->lsub[begin];
            int my_supernode = Glu_persist->supno[row];
            if (row == segmentFlagArray[my_supernode])
            {
                lsub1[newEnd] = row;
                newEnd++;
            }
        }
        Glu_freeable->xlsub_end1[iter] = newEnd;						
    }
    Glu_freeable->xlsub_end = Glu_freeable->xlsub_end1;
    Glu_freeable->lsub = lsub1;
#if (DEBUGlevel >=1)		
    printf("Process:%d finished performing segmentation logic\n",iam);
    // fflush(stdout);
#endif
}

void LSupernodal_graph(Glu_persist_t* Glu_persist, int_t n, Glu_freeable_t* Glu_freeable)
{
    //Loop through each vertex and update its xusub_begin and xusub_end if it falls in a supernode with representative row different than itself
    for (int i=0;i<n;i++)
    {
        int i_supernode = Glu_persist->supno[i];						
        int_t i_representative =  Glu_persist->xsup[i_supernode];						
        if (i_representative != i)
        {
            //If the representative column/row(gSoFa) is not the current column
            Glu_freeable->xusub_begin[i] = Glu_freeable->xusub_begin[i_representative];  
            Glu_freeable->xusub_end[i] = Glu_freeable->xusub_end[i_representative];  					
        }
    }
}

void SwapLU(Glu_freeable_t* Glu_freeable)
{

    int_t* temp= Glu_freeable->xusub_begin;
    Glu_freeable->xusub_begin = Glu_freeable->xlsub_begin;
    Glu_freeable->xlsub_begin = temp;

    temp= Glu_freeable->xusub_end;
    Glu_freeable->xusub_end = Glu_freeable->xlsub_end;
    Glu_freeable->xlsub_end = temp;

    temp= Glu_freeable->usub;
    Glu_freeable->usub = Glu_freeable->lsub;
    Glu_freeable->lsub = temp;

    // temp=     Glu_freeable->RecvCntU;
    // Glu_freeable->RecvCntU =   Glu_freeable->RecvCntL;
    // Glu_freeable->RecvCntL = temp;

    // temp = Glu_freeable->itemp_U;
    // Glu_freeable->itemp_U = Glu_freeable->itemp_L;
    // Glu_freeable->itemp_L = temp;

    temp = Glu_freeable->xlsub;
    Glu_freeable->xlsub = Glu_freeable->xusub;
    Glu_freeable->xusub = temp;

    int_t temp_val = Glu_freeable->nzlmax;
    Glu_freeable->nzlmax = Glu_freeable->nzumax;
    Glu_freeable->nzumax = temp_val;
}


void pre_distribute(int_t NNZ_L, int_t NNZ_U, Glu_freeable_t**  Glu_freeable1, dScalePermstruct_t **ScalePermstruct1, 
        int_t is_gsofa, struct gSoFa_para_t** gSoFa_para1, gridinfo_t *grid, Glu_persist_t** Glu_persist1, int_t iam,
        int_t n)
{
    // double start_predistribution_time = SuperLU_timer_();
    int_t k;
    int_t TotalNSupernode =0;
    Glu_freeable_t*  Glu_freeable = *Glu_freeable1;
    dScalePermstruct_t *ScalePermstruct = *ScalePermstruct1;
    //Broadcast the NNZ_L and NNZ_U to all the processes
#if ( DEBUGlevel>=1 )
    printf("Initial broadcast of NNL and NNU\n");
    fflush(stdout);
#endif

    double t_broadcast_start_time = SuperLU_timer_();  
    // init_bcast(&NNZ_L, &NNZ_U, Glu_freeable, n, ScalePermstruct, grid); 
    double t_broadcast_time = SuperLU_timer_()-t_broadcast_start_time;

#if ( DEBUGlevel>=1 )
    printf("After broadcast of NNL and NNU\n");
    fflush(stdout);
#endif
#if (DEBUGlevel == 0)
    if (iam==0) printf("IAM:%d Time for broadcast: %lf ms\n",iam, t_broadcast_time*1000); 
#endif
    Glu_persist_t* Glu_persist = *Glu_persist1;
    struct gSoFa_para_t* gSoFa_para = *gSoFa_para1;
    int_t num_process = gSoFa_para->num_process;
   
    double t_xsub_nsup_start_time = SuperLU_timer_();
    Finalize_supernodes(is_gsofa, Glu_freeable, NNZ_L, NNZ_U, gSoFa_para,&TotalNSupernode,grid,Glu_persist, n);
    #if (DEBUGlevel == 0)
    printf("IAM:%d Supernode arrangement and communication time + Free gSoFa Memory%f ms\n",iam, (SuperLU_timer_()-t_xsub_nsup_start_time)*1000);
    // fflush(stdout);
    #endif
    double lsubUsubCommunicate_start = SuperLU_timer_();
    Communicate_lsub_usub(num_process, Glu_persist, gSoFa_para, Glu_freeable, grid, n);
    double lsubUsubCommunicate = SuperLU_timer_() - lsubUsubCommunicate_start;
    #if (DEBUGlevel == 0)
    printf("IAM:%d Time for lsub and usub communication: %lf ms\n",iam, lsubUsubCommunicate*1000);
    // fflush(stdout);
    #endif
#ifdef enablegSoFaMPI
    double symbolicFactorization_infoComm_start = SuperLU_timer_();
    MPI_Reduce(&(gSoFa_para->fill_count), &(gSoFa_para->global_fill_in), 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, grid->comm);
    MPI_Bcast(&(gSoFa_para->global_fill_in), 1, MPI_UNSIGNED_LONG_LONG, 0, grid->comm);
    // MPI_Barrier(grid->comm); 
    double symbolicFactorization_infoComm = SuperLU_timer_() -symbolicFactorization_infoComm_start;
    #if (DEBUGlevel == 0)
    printf("Process:%d Symbolic Factorization Info Comm Time:%f ms\n",iam,symbolicFactorization_infoComm*1000);
    // fflush(stdout);
    #endif
#endif
    // printf("Process: %d Final number of processed vertex: %d vertex count:%d\n",iam,*global_processed_vertex, vert_count);

    double segmentation_start = SuperLU_timer_();
    U_Segmentation(TotalNSupernode, n, Glu_freeable, Glu_persist, NNZ_U);
    double segmentation = SuperLU_timer_() - segmentation_start;
    #if (DEBUGlevel == 0)
    printf("Process:%d Segmentation Time:%f ms\n",iam,segmentation*1000);
    // fflush(stdout);
    #endif
    
    #ifdef enable_supernodal_graph
    double enable_supernodal_graph_start = SuperLU_timer_();
    LSupernodal_graph(Glu_persist, n, Glu_freeable);
    double enable_supernodal_graph_time = SuperLU_timer_() - enable_supernodal_graph_start;
    printf("Process:%d Enable Supernodal Graph Time:%f ms\n",iam,enable_supernodal_graph_time*1000);
    #endif
    // fflush(stdout);
    
    double swapLAndUStructures_gSoFa_start = SuperLU_timer_();
    SwapLU(Glu_freeable);//CSR to CSC
    double swapLAndUStructures_gSoFa_time = SuperLU_timer_() - swapLAndUStructures_gSoFa_start;
    #if (DEBUGlevel == 0)
    printf("Process:%d Swap L and U Structures Time:%f ms\n",iam,swapLAndUStructures_gSoFa_time*1000);
    #endif

    Glu_freeable->nnzLU = gSoFa_para->global_fill_in+gSoFa_para->edge_count;
    Glu_freeable->MemModel=SYSTEM;
    // double predistribution_time = SuperLU_timer_()-start_predistribution_time;
    // printf("IAM:%d Pre-distribution time:%lf ms \n",predistribution_time*1000);
    }

void  Initialize_filled_graph(Glu_freeable_t* Glu_freeable, int_t* GACcolbeg, int_t* GACcolend, int_t* GACrowind, int_t n)
{
    int_t offset;
    for (int vtx =0; vtx<n; vtx++)
    {
        offset = Glu_freeable->xusub_end[vtx];
        Glu_freeable->usub[offset] = vtx;
         Glu_freeable->xusub_end[vtx]= offset +1;
         Glu_freeable->xusub_original_nz_offset[vtx] = offset +1;
        for (int nbr_idx =GACcolbeg[vtx]; nbr_idx<GACcolend[vtx]; nbr_idx++)
        {
            int nbr = GACrowind[nbr_idx];
            if (nbr < vtx)
            {
                offset = Glu_freeable->xlsub_end[vtx];
                Glu_freeable->lsub[offset] = nbr;
                Glu_freeable->xlsub_end[vtx] = offset +1;   
                 Glu_freeable->xlsub_original_nz_offset[vtx] = offset +1;   
            }
            else if (nbr > vtx)
            {
                offset = Glu_freeable->xusub_end[vtx];
                Glu_freeable->usub[offset] = nbr;
                Glu_freeable->xusub_end[vtx] = offset +1;  
                Glu_freeable->xusub_original_nz_offset[vtx] = offset +1;                
            }            
        }      

    }
}
void Initialize_gSoFa(gridinfo_t* grid, int_t nprs,Glu_persist_t* Glu_persist, dLUstruct_t* LUstruct, Glu_freeable_t** Glu_freeable1, 
        struct gSoFa_para_t* gSoFa_para, int_t n, int_t iam, int_t is_gsofa, 
        int_t global_collected, SuperMatrix *A,SuperMatrix GA,
        superlu_dist_options_t * options, int_t *perm_c, SuperMatrix GAC,
          NCPformat *GACstore, int_t* NNZ_L, int_t*NNZ_U)
{
    gSoFa_para->vert_count = n;    
    gSoFa_para->num_process =  grid->nprow * grid->npcol;
    gSoFa_para->max_supernode_size = sp_ienv_dist(3,options); 
    int_t num_process = gSoFa_para->num_process;
    int_t* etree = LUstruct->etree;
    gSoFa_para->N_src_group = Compute_Src_group(n);
    #if (DEBUGlevel == 0)
    printf("Computed N_src_group: %d\n",gSoFa_para->N_src_group);
    #endif
    // *num_process = grid->nprow * grid->npcol;
    int num_nodes=0;
    // int N_gsofa_process_node = 0;

    // struct Super* Supernode_per_process;

    if ( !(gSoFa_para->Supernode_per_process = (struct Super *)
                SUPERLU_MALLOC(sizeof(struct Super)*n)) )
        ABORT("Malloc fails for Supernode_per_process.");

    num_nodes = num_process/nprs;
    #if (DEBUGlevel == 0)
    printf("Number of nodes (resources): %d\n",num_nodes);
    #endif
    // fflush(stdout);
    // if (nprs < N_GPU_resource)
    // {
    //     N_gsofa_process_node = nprs;
    // }
    // int_t N_GPU_Node = N_gsofa_process_node;
    // printf("Number of GPU detected: %d\n",N_GPU_resource);
    // printf("Number of gSoFa process: %d\n",N_gsofa_process_node);
    // int total_gSoFa_Processes = N_gsofa_process_node * num_nodes;
    int_t mygSoFaOffset =0;
    int_t mygSoFaRank =-1;

    int_t N_GPU_gSoFa_process = gSoFa_para->N_GPU_gSoFa_process;
    // double detect_gsofa_parameters_time = SuperLU_timer_() - detect_gsofa_parameters;
    // printf("IAM:%d Detect gSoFa parameters time: %f ms\n",iam, detect_gsofa_parameters_time*1000);
    double lsub_usub_allocation_start = SuperLU_timer_();
    Glu_freeable_t* Glu_freeable;
    if ( !(Glu_freeable = (Glu_freeable_t *)
                SUPERLU_MALLOC(sizeof(Glu_freeable_t))) )
        ABORT("Malloc fails for Glu_freeable.");
    double t_csr_allocation_start_time = SuperLU_timer_();
    LUstruct->Glu_persist = Glu_persist;
    Glu_freeable->xlsub_begin   = intMalloc_dist(n);
    Glu_freeable->xlsub_end  = intMalloc_dist(n);
    Glu_freeable->xusub_begin   = intMalloc_dist(n);
    Glu_freeable->xusub_end  = intMalloc_dist(n);
    Glu_persist->xsup = intMalloc_dist(n+1);
    Glu_persist->supno = intMalloc_dist(n+1);
    Glu_freeable->xlsub_end1  = intMalloc_dist(n);
    #ifdef initializeCSR
    Glu_freeable->xlsub_original_nz_offset  = intMalloc_dist(n);
    Glu_freeable->xusub_original_nz_offset  = intMalloc_dist(n);
    #endif
    double t_00 = SuperLU_timer_();
    int iam_gSoFa =-1;
    struct aux_device *dev_mem;
    int_t vert_count = n;
    int BLKS_NUM,blockSize;
    int_t next_front;
    // printf("N_GPU_Node: %d\n",N_GPU_Node);//NGPU per resource used for gSoFa
    // printf("N_gsofa_process_node: %d\n",N_gsofa_process_node);

    int_t real_allocation;
    int_t N_chunks;
    int_t total_num_chunks_per_node;

    int_t myrank=-1;
    int Nsupernode_process = 0;    
    ull_t fill_count=0;
    // ull_t global_fill_in=0;
    // int_t N_processed_source=0;
    // int_t global_N_processed_source=0;
    // double indiv_time=0;
    // double max_time=0;
    // double min_time=0;

    int_t edge_count;
    // double start_predistribution_time;
    double lsub_usub_allocation_time = SuperLU_timer_() - lsub_usub_allocation_start;
    #if (DEBUGlevel == 0)
    printf("IAM:%d lsub_usub_allocation_time: %f ms\n",iam, lsub_usub_allocation_time*1000);
    #endif
    double t_01 = SuperLU_timer_();
    double t_011 =0;
    double t_012 =0;
    int_t num_process_gSoFa  = grid->gSoFa.Np;
    if (is_gsofa)
    {
        t_011 = SuperLU_timer_();
        iam_gSoFa = grid->gSoFa.Iam;
        // *num_process_gSoFa = grid->gSoFa.Np;
        mygSoFaOffset = iam_gSoFa%gSoFa_para->N_gsofa_process_node;
        int localgpu = mygSoFaOffset;
        H_ERR(cudaSetDevice(localgpu));
        // printf("IAM_gSOFA:%d num_process_gSoFa:%d iam:%d localgpu:%d\n",iam_gSoFa,num_process_gSoFa,iam,localgpu);
        // fflush(stdout);
        // printf("Outside Process :%d is gSoFa process. Its local gSoFa rank: %d \n",iam,iam_gSoFa);     
        gSoFa_para->num_nodes=num_process_gSoFa;//num_process;

        double t_blockSize_start = SuperLU_timer_();
        // findBlockDim(&BLKS_NUM,&blockSize);  
        BLKS_NUM= gSoFa_para->BLKS_NUM;    
        blockSize = gSoFa_para->blockSize;           
        // t_011 = SuperLU_timer_();
        // printf("IAM:%d Initial BLKS_NUM:%d blockSize:%d Time:%lf ms\n",iam,BLKS_NUM,blockSize,(SuperLU_timer_()-t_blockSize_start)*1000);


        next_front= (BLKS_NUM*blockSize) >> 5;
        int chunk_size = gSoFa_para->max_supernode_size;
        myrank = iam_gSoFa;
        real_allocation=vert_count*gSoFa_para->N_src_group*allocation_threshold_factor;
        t_012 = SuperLU_timer_();
        double t_ParaStart=SuperLU_timer_();
        // double w_parastart = SuperLU_timer_();
        Allocate_Initialize_gSoFa_para(gSoFa_para, BLKS_NUM,  blockSize,  next_front,  real_allocation, gSoFa_para->max_supernode_size,N_GPU_gSoFa_process,vert_count,gSoFa_para->N_src_group);
        H_ERR(cudaDeviceSynchronize());
        double t_ParaEnd=SuperLU_timer_()-t_ParaStart;
        // double w_paraend = SuperLU_timer_()-w_parastart;
        // printf(": %lf ms\n",t_ParaEnd*1000);
        // printf("Time for allocation and initialization of gSoFa parameters (wtime): %lf ms\n",w_paraend*1000);
        // fflush(stdout);
        N_chunks= gSoFa_para->N_chunks;
        total_num_chunks_per_node=gSoFa_para->total_num_chunks_per_node;
        double device_mem_alloc_start = SuperLU_timer_();
        if ( !(dev_mem = (struct aux_device *)SUPERLU_MALLOC( N_GPU_gSoFa_process * sizeof(struct aux_device)) ) )
            ABORT("SUPERLU_MALLOC fails for device_obj->superObj[]");
        // dev_mem = (struct aux_device *)malloc(N_GPU_gSoFa_process*sizeof(struct aux_device));
        double device_mem_alloc_end = SuperLU_timer_()-device_mem_alloc_start;
        #if (DEBUGlevel == 0)
        printf("Process:%d Time for allocation of device memory: %lf ms\n",iam_gSoFa,device_mem_alloc_end*1000);
        #endif
        // fflush(stdout);
    } //Communicator logic ends here
    //~begin of size prediction
    double t_02 = SuperLU_timer_();
    #if (DEBUGlevel == 0)
    printf("IAM: %d Time First section: %f ms\n",iam, (t_02-t_01)*1000);
    #endif
    double t_gather_start=SuperLU_timer_();
    SuperMatrix GA_gsofa;   
    if (global_collected==0) //Processes will work on collecting the global A
    {
        pdCompRow_loc_to_CompCol_global(0, A, grid, &GA_gsofa);
#ifdef _print_gSoFa_
        printf("Collecting the global matrix!\n");
#endif
    }
    else
    {
        GA_gsofa = GA;                     
    }
    double t_gather_end=SuperLU_timer_()-t_gather_start;
    #if (DEBUGlevel == 0)
    printf("IAM: %d Time for gathering the matrix: %lf ms\n",iam, t_gather_end*1000);
    #endif
    // if(gSoFa_communicator != MPI_COMM_NULL) 
    double t_03 = SuperLU_timer_();
    double t_spcol_start=SuperLU_timer_();

        //gSoFa processes will work on the prediction
#ifdef _print_gSoFa_
        printf("INSIDE MILESTONE 3!\n");
#endif
        double t_size_prediction_start=SuperLU_timer_();
        int_t* rowcnt = intMalloc_dist(n);
        int_t* colcnt = intMalloc_dist(n);
        int_t* nlnz = intMalloc_dist(1);
        int_t* part_super_l = intMalloc_dist(n);
        //OUTPUT of prediction module
        int_t* invp = intMalloc_dist(n); /* inverse of perm_c */
        #if (DEBUGlevel == 0)
        printf("Calling the tailored sp_color!\n");
        #endif
        sp_colorder_Final(options, &GA_gsofa, perm_c, etree, &GAC, invp, colcnt, rowcnt, nlnz, part_super_l,gSoFa_para);//GAC is output, perm_c and etree are updated
        gSoFa_para->etree = etree;
        double t_size_prediction_end=SuperLU_timer_()-t_size_prediction_start;                       
        // printf("After sp_colorder_Final()  distributed matrix A:\n");
                int_t *GACcolbeg, *GACcolend, *GACrowind;
        GACstore = (NCPformat *) GAC.Store;
        GACcolbeg = GACstore->colbeg;
        GACcolend = GACstore->colend;
        GACrowind = GACstore->rowind; 
        edge_count= GACstore->nnz;
    if (is_gsofa)
    {    
#if (DEBUGlevel == 0)
        printf("edge_count Original number of edges (nnz): %d\n",edge_count);
        // fflush(stdout);
        printf("Process: %d Time for sp_col: %lf ms\n",iam_gSoFa, t_size_prediction_end*1000);
        printf("Process: %d Time for size prediction + related allocations: %lf ms\n",iam_gSoFa, t_size_prediction_end*1000);
        #endif
        ull_t Process_TEPS =0;
        // int_t** lsub_CPU;
        // int_t** usub_CPU;
        // int_t** xlsub_begin;
        // int_t** xlsub_end;
        // int_t** xusub_begin;
        // int_t** xusub_end;
        // int_t** xusub_end1;

        // int_t* is_OriginalNZ_L;
        // int_t* is_OriginalNZ_U;
        double allocation_start_time = SuperLU_timer_();
        // cudaStream_t* gSoFa_stream;
        // if ( !(gSoFa_stream = (cudaStream_t*)SUPERLU_MALLOC( N_GPU_gSoFa_process * sizeof(cudaStream_t)) ) )
        //     ABORT("SUPERLU_MALLOC fails for device_obj->superObj[]");
        // gSoFa_stream = (cudaStream_t*) malloc ((N_GPU_gSoFa_process)*sizeof(cudaStream_t));
        for (int i=0;i<N_GPU_gSoFa_process;i++)
        {
            // printf("selecting GPU: %d\n",i);//<<i<<endl;
            int localgpu=0;
            int gpuOffset=0; 
#ifdef  lambda                  
            gpuOffset = myrank*N_GPU_gSoFa_process;
            localgpu = gpuOffset +i;
            int offset = GPUOffset; 
            H_ERR(cudaSetDevice(localgpu + offset));  
#else
            // H_ERR(cudaSetDevice(i)); 
            #if (DEBUGlevel == 0)
            printf("Local process: %d mygSoFaOffset:%d selected GPU: %d\n",iam_gSoFa,mygSoFaOffset,i + mygSoFaOffset);
            // fflush(stdout);
            #endif
            localgpu = i + mygSoFaOffset;
            // H_ERR(cudaSetDevice(localgpu));
#endif
            double t_allocation_start = SuperLU_timer_();
            // H_ERR( cudaStreamCreate(&gSoFa_stream[i]));
            #if (DEBUGlevel == 0)
            printf("gSoFa_para->max_supernode_size : %d",gSoFa_para->max_supernode_size);
            // fflush(stdout);
            #endif
            Allocate_Initialize (gSoFa_para, &dev_mem[i], vert_count, edge_count, BLKS_NUM, blockSize,i + mygSoFaOffset,next_front,gSoFa_para->N_src_group,real_allocation, N_chunks,
                    total_num_chunks_per_node,gSoFa_para->max_supernode_size,gSoFa_para->gSoFa_stream[i],gSoFa_para->sup_per_gpu, iam_gSoFa,localgpu);
            // H_ERR(cudaDeviceSynchronize()); //NREL Debugging
            double t_allocation_end = SuperLU_timer_()-t_allocation_start;
            #if (DEBUGlevel == 0)
            printf("Process:%d Time for allocation of device memory: %lf ms\n",iam_gSoFa,t_allocation_end*1000);
            #endif

        }
        // }
        double allocation_time = SuperLU_timer_()-allocation_start_time; 
        #if (DEBUGlevel == 0)
        printf("Process: %d gSoFa data structures allocations, initializations and cudasetdevice in  pdgssvx.c: %f ms\n",iam_gSoFa, allocation_time*1000);
        #endif
        // printf("Stream create time only in pdgssvx.c: %f ms\n",streamcreate_time*1000);
        // fflush(stdout);
        // #ifdef enable_gsofa
        // /#ifdef enable_gather_original_A
        // printf("IAM:%d N_GPU_gSoFa_process: %d\n",iam,N_GPU_gSoFa_process);
        // if (iam==0)
        // {
        // printf("Process(gSoFa):%d started loading the graph into GPU(s) memory edge_count:%d vert_count:%d\n",iam_gSoFa, edge_count, vert_count);
        // fflush(stdout);


        double t_load_graph_start_time = SuperLU_timer_();
        NCformat* Astore_temp  = GA_gsofa.Store;
        int_t original_edge_count = Astore_temp->nnz;
        // printf("Process (gSoFa):%d original_edge_count:%d new_edge_count:%d\n",iam_gSoFa, original_edge_count, edge_count);
        // fflush(stdout);

        for (int i = 0; i< N_GPU_gSoFa_process; i++)
        {

            int localgpu= i + mygSoFaOffset;
            // printf("IAM:%d selecting GPU: %d N_GPU_gSoFa_process:%d \n",iam, localgpu,N_GPU_gSoFa_process);//<<i<<endl;
            int gpuOffset=0;
#ifdef  lambda                 
            gpuOffset = myrank*N_GPU_gSoFa_process;
            localgpu = gpuOffset +i;
            int offset = GPUOffset; 
            H_ERR(cudaSetDevice(localgpu + offset));  
#else

            // H_ERR(cudaSetDevice(localgpu));
#endif
            load_graph_gpu (&dev_mem[i], vert_count, original_edge_count, GACrowind, GACcolbeg, GACcolend, BLKS_NUM, blockSize,localgpu,next_front,
                    gSoFa_para->N_src_group,real_allocation, N_chunks,total_num_chunks_per_node,gSoFa_para->max_supernode_size,gSoFa_para->gSoFa_stream[i],iam_gSoFa);
            // H_ERR(cudaDeviceSynchronize());//Debugging multiGPU NREL
        }
        gSoFa_para->edge_count = edge_count;
        gSoFa_para->iam_gSoFa = iam_gSoFa;
        gSoFa_para->Nsupernode_process = Nsupernode_process;
        gSoFa_para->colcnt = colcnt;
        gSoFa_para->rowcnt = rowcnt;
        // gSoFa_para->is_OriginalNZ_L = is_OriginalNZ_L;
        // gSoFa_para->is_OriginalNZ_U = is_OriginalNZ_U;
        gSoFa_para->dev_mem = dev_mem;
        gSoFa_para->mygSoFaOffset = mygSoFaOffset;       
        // *gSoFa_stream1 = gSoFa_para->gSoFa_stream;
        double t_load_graph_time = SuperLU_timer_()- t_load_graph_start_time;
        // printf("Process:%d Finished loading the graph into GPU(s) memory\n",iam_gSoFa);
        // fflush(stdout);
        #if (DEBUGlevel == 0)
        printf("Process:%d Time to load graph: %lf ms\n",iam_gSoFa, t_load_graph_time*1000);
        #endif

        // return;
}
else
{
    //Non-gSoFa processes compute the xlsub_begin and xusub_begin
    // int_t NNZ_L = 0;
    // int_t NNZ_U = 0;
        compute_csr_offsets(NNZ_L, NNZ_U, Glu_freeable, colcnt, rowcnt, n);
       
    //         if (grid->iam ==7) 
    // {
    //     printf("IAM 7  NNZ_L:%d NNZ_U:%d\n",*NNZ_L,*NNZ_U);
    //     fflush(stdout);
    //     PrintInt10("IAM 7 xlsub_begin",10, Glu_freeable->xlsub_begin);
    //     fflush(stdout);
    //     PrintInt10("IAM 7 xusub_begin",10, Glu_freeable->xusub_begin);
    //     fflush(stdout);
    // }
//Allocate memory for lsub and usub in the non-gSoFa processes
        if ( !(Glu_freeable->lsub = intMalloc_dist(*NNZ_L)) )
            ABORT("Malloc fails for lsub[].");
        if ( !(Glu_freeable->usub = intMalloc_dist(*NNZ_U)) )
            ABORT("Malloc fails for usub[].");
            #ifdef initializeCSR
    Initialize_filled_graph(Glu_freeable, GACcolbeg, GACcolend, GACrowind, n);
    #endif
// lsub = intMalloc_dist(NNZ_L);
// usub = intMalloc_dist(NNZ_U);
}
gSoFa_para->num_process_gSoFa = num_process_gSoFa;       
*Glu_freeable1 = Glu_freeable; //Required by all the processes
double t_csr_allocation_time = SuperLU_timer_()- t_csr_allocation_start_time;
#if (DEBUGlevel == 0)
printf("IAM:%d Time for csr allocation: %lf ms\n",iam, t_csr_allocation_time*1000);
// fflush(stdout);

double t_04 = SuperLU_timer_();
printf("IAM:%d t_csr:%lf ms t_00:%lf ms t_01:%lf ms t_012:%lf ms t_02:%lf ms t_03:%lf ms t_04:%lf ms \n",iam,t_csr_allocation_time*1000, (t_00 -t_csr_allocation_start_time)*1000, (t_01-t_00)*1000, (t_012-t_011)*1000,(t_02-t_01)*1000,(t_03-t_02)*1000,(t_04-t_03)*1000);
// fflush(stdout);
#endif
}

