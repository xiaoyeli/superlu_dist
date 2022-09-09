/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Communication routines for the 3D algorithm.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * May 12, 2021
 */
#include "superlu_ddefs.h"
//#include "cblas.h"
#if 0
#include "p3dcomm.h"
#include "sec_structs.h"
//#include "load-balance/supernodal_etree.h"
//#include "load-balance/supernodalForest.h"
#include "supernodal_etree.h"
#include "supernodalForest.h"
#include "trfAux.h"
#include "treeFactorization.h"
#include "xtrf3Dpartition.h"
#endif

// #define MPI_MALLOC
#define MPI_INT_ALLOC(a, b) (MPI_Alloc_mem( (b)*sizeof(int_t), MPI_INFO_NULL, &(a) ))
#define MPI_DATATYPE_ALLOC(a, b) (MPI_Alloc_mem((b)*sizeof(double), MPI_INFO_NULL, &(a)))

int_t dAllocLlu(int_t nsupers, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d)
{
    int i;
    int_t Pc = grid3d->npcol;
    int_t Pr = grid3d->nprow;
    
    int_t nbc = CEILING(nsupers, Pc);
    int_t nbr = CEILING(nsupers, Pr);
    
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t   **Lrowind_bc_ptr =
	(int_t**) SUPERLU_MALLOC(sizeof(int_t*)*nbc); 	/* size ceil(NSUPERS/Pc) */
    double  **Lnzval_bc_ptr =
	(double **) SUPERLU_MALLOC(sizeof(double*)*nbc);  /* size ceil(NSUPERS/Pc) */

    for (i = 0; i < nbc ; ++i)
	{
	    /* code */
	    Lrowind_bc_ptr[i] = NULL;
	    Lnzval_bc_ptr[i] = NULL;
	}
    
    int_t   **Ufstnz_br_ptr =
	(int_t**) SUPERLU_MALLOC(sizeof(int_t*)*nbr); /* size ceil(NSUPERS/Pr) */
    double  **Unzval_br_ptr =
	(double **) SUPERLU_MALLOC(sizeof(double*)*nbr); /* size ceil(NSUPERS/Pr) */
    
    for (i = 0; i < nbr ; ++i)
	{
	    /* code */
	    Ufstnz_br_ptr[i] = NULL;
	    Unzval_br_ptr[i] = NULL;
	}

   // Sherry: use int type
                  /* Recv from no one (0), left (1), and up (2).*/
    int *ToRecv = SUPERLU_MALLOC(nsupers * sizeof(int));
    for (i = 0; i < nsupers; ++i) ToRecv[i] = 0;
                  /* Whether need to send down block row. */
    int *ToSendD = SUPERLU_MALLOC(nbr * sizeof(int));
    for (i = 0; i < nbr; ++i) ToSendD[i] = 0;
                  /* List of processes to send right block col. */
    int **ToSendR = (int **) SUPERLU_MALLOC(nbc * sizeof(int*));

    for (int_t i = 0; i < nbc; ++i)
	{
	    /* code */
	    //ToSendR[i] = INT_T_ALLOC(Pc);
	    ToSendR[i] = SUPERLU_MALLOC(Pc * sizeof(int));
	}
    
    /*now setup the pointers*/
    Llu->Lrowind_bc_ptr = Lrowind_bc_ptr ;
    Llu->Lnzval_bc_ptr = Lnzval_bc_ptr ;
    Llu->Ufstnz_br_ptr = Ufstnz_br_ptr ;
    Llu->Unzval_br_ptr = Unzval_br_ptr ;
    Llu->ToRecv = ToRecv ;
    Llu->ToSendD = ToSendD ;
    Llu->ToSendR = ToSendR ;
    
    return 0;
} /* dAllocLlu */

int_t dmpiMallocLUStruct(int_t nsupers, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);
    
    int_t k = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
    for ( int_t lb = 0; lb < k; ++lb)
	{
	    int_t *usub, *usub_new;
	    usub =  Ufstnz_br_ptr[lb];
	    
	    double * uval = Unzval_br_ptr[lb];
	    double * uval_new;
	    
	    /*if non empty set the flag*/
	    if (usub != NULL)
		{
		    int_t lenv, lens;
		    lenv = usub[1];
		    lens = usub[2];
		    
		    MPI_INT_ALLOC(usub_new, lens);
		    memcpy( usub_new, usub, lens * sizeof(int_t));
		    MPI_DATATYPE_ALLOC(uval_new, lenv);
		    memcpy( uval_new, uval, lenv * sizeof(double));
		    Ufstnz_br_ptr[lb] = usub_new;
		    Unzval_br_ptr[lb] = uval_new;
		    SUPERLU_FREE(usub);
		    SUPERLU_FREE(uval);
		}
	} /*for ( int_t lb = 0; lb < k; ++lb)*/
    
    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);
    
    /*start broadcasting blocks*/
    for (int_t jb = 0; jb < nsupers; ++jb)   /* for each block column ... */
	{
	    int_t pc = PCOL( jb, grid );
	    if (mycol == pc)
		{
		    int_t ljb = LBj( jb, grid ); /* Local block number */
		    int_t  *lsub , *lsub_new;
		    double *lnzval, *lnzval_new;
		    lsub = Lrowind_bc_ptr[ljb];
		    lnzval = Lnzval_bc_ptr[ljb];
		    
		    if (lsub)
			{
			    int_t nrbl, len, len1, len2;
			    
			    nrbl  =   lsub[0]; /*number of L blocks */
			    len   = lsub[1];       /* LDA of the nzval[] */
			    len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
			    len2  = SuperSize(jb) * len;
			    
			    MPI_INT_ALLOC(lsub_new, len1);
			    memcpy( lsub_new, lsub, len1 * sizeof(int_t));
			    MPI_DATATYPE_ALLOC(lnzval_new, len2);
			    memcpy( lnzval_new, lnzval, len2 * sizeof(double));
			    Lrowind_bc_ptr[ljb] = lsub_new;
			    SUPERLU_FREE(lsub );
			    Lnzval_bc_ptr[ljb] = lnzval_new;
			    SUPERLU_FREE(lnzval );
			}
		} /* if mycol == pc ... */
	} /* for jb ... */
    
    return 0;
}


int_t dzSendLPanel(int_t k, int_t receiver,
                   dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);
    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);

    int_t pc = PCOL( k, grid );
    if (mycol == pc)
	{
	    int_t lk = LBj( k, grid ); /* Local block number */
	    int_t  *lsub;
	    double* lnzval;
	    lsub = Lrowind_bc_ptr[lk];
	    lnzval = Lnzval_bc_ptr[lk];
	    
	    if (lsub != NULL)
		{
		    int_t len   = lsub[1];       /* LDA of the nzval[] */
		    int_t len2  = SuperSize(k) * len; /* size of nzval of L panel */
		    
		    MPI_Send(lnzval, len2, MPI_DOUBLE, receiver, k, grid3d->zscp.comm);
		    SCT->commVolRed += len2 * sizeof(double);
		}
	}
    return 0;
}


int_t dzRecvLPanel(int_t k, int_t sender, double alpha, double beta,
                    double* Lval_buf,
                    dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT)
{
    
    // A(k) = alpha*A(k) + beta* A^{sender}(k)
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);
    int inc = 1;    
    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);
    
    int_t pc = PCOL( k, grid );
    if (mycol == pc)
	{
	    int_t lk = LBj( k, grid ); /* Local block number */
	    int_t  *lsub;
	    double* lnzval;
	    lsub = Lrowind_bc_ptr[lk];
	    lnzval = Lnzval_bc_ptr[lk];
	    
	    if (lsub != NULL)
		{
		    int len   = lsub[1];       /* LDA of the nzval[] */
		    int len2  = SuperSize(k) * len; /* size of nzval of L panels */
		    
		    MPI_Status status;
		    MPI_Recv(Lval_buf , len2, MPI_DOUBLE, sender, k,
			     grid3d->zscp.comm, &status);
		    
		    /*reduce the updates*/
		    superlu_dscal(len2, alpha, lnzval, 1);
		    superlu_daxpy(len2, beta, Lval_buf, 1, lnzval, 1);
		}
	}

    return 0;
}

int_t dzSendUPanel(int_t k, int_t receiver,
                    dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);
    int_t iam = grid->iam;

    int_t myrow = MYROW (iam, grid);
    int_t pr = PROW( k, grid );
    if (myrow == pr)
	{
	    int_t lk = LBi( k, grid ); /* Local block number */
	    int_t  *usub;
	    double* unzval;
	    usub = Ufstnz_br_ptr[lk];
	    unzval = Unzval_br_ptr[lk];
	    
	    if (usub != NULL)
		{
		    int lenv = usub[1];
		    
		    /* code */
		    MPI_Send(unzval, lenv, MPI_DOUBLE, receiver, k, grid3d->zscp.comm);
		    SCT->commVolRed += lenv * sizeof(double);
		}
	}
	
    return 0;
}


int_t dzRecvUPanel(int_t k, int_t sender, double alpha, double beta,
                    double* Uval_buf, dLUstruct_t* LUstruct,
                    gridinfo3d_t* grid3d, SCT_t* SCT)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);
    int inc = 1;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t pr = PROW( k, grid );

    if (myrow == pr)
	{
	    int_t lk = LBi( k, grid ); /* Local block number */
	    int_t  *usub;
	    double* unzval;
	    usub = Ufstnz_br_ptr[lk];
	    unzval = Unzval_br_ptr[lk];
	    
	    if (usub != NULL)
		{
		    int lenv = usub[1];
		    MPI_Status status;
		    MPI_Recv(Uval_buf , lenv, MPI_DOUBLE, sender, k,
			     grid3d->zscp.comm, &status);
		    
		    /*reduce the updates*/
		    superlu_dscal(lenv, alpha, unzval, 1);
		    superlu_daxpy(lenv, beta, Uval_buf, 1, unzval, 1);
		}
	}
    return 0;
}


int_t dp3dScatter(int_t n, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d)
/* Copies LU structure from layer 0 to all the layers */
{
    gridinfo_t* grid = &(grid3d->grid2d);
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    
    /* broadcast etree */
    int_t *etree = LUstruct->etree;
    MPI_Bcast( etree, n, mpi_int_t, 0,  grid3d->zscp.comm);
    
    int_t nsupers;
    
    if (!grid3d->zscp.Iam)
	nsupers = getNsupers(n, LUstruct->Glu_persist);
    
    /* broadcast nsupers */
    MPI_Bcast( &nsupers, 1, mpi_int_t, 0,  grid3d->zscp.comm);
    
    /* Scatter and alloc Glu_persist */
    if ( grid3d->zscp.Iam ) // all other process layers not equal 0
	dAllocGlu_3d(n, nsupers, LUstruct);
    
    /* broadcast Glu_persist */
    int_t *xsup = LUstruct->Glu_persist->xsup;
    MPI_Bcast( xsup, nsupers + 1, mpi_int_t, 0,  grid3d->zscp.comm);
    
    int_t *supno = LUstruct->Glu_persist->supno;
    MPI_Bcast( supno, n, mpi_int_t, 0,  grid3d->zscp.comm);
    
    /* now broadcast local LU structure */
    /* first allocating space for it */
    if ( grid3d->zscp.Iam ) // all other process layers not equal 0
	dAllocLlu(nsupers, LUstruct, grid3d);
    
    dLocalLU_t *Llu = LUstruct->Llu;
    
    /*scatter all the L blocks and indexes*/
    dscatter3dLPanels( nsupers, LUstruct, grid3d);

    /*scatter all the U blocks and indexes*/
    dscatter3dUPanels( nsupers, LUstruct, grid3d);
    
    int_t* bufmax = Llu->bufmax;
    MPI_Bcast( bufmax, NBUFFERS, mpi_int_t, 0,  grid3d->zscp.comm);
    
    /* now sending tosendR etc */
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int* ToSendD = Llu->ToSendD;
    
    int_t nbr = CEILING(nsupers, Pr);
    int_t nbc = CEILING(nsupers, Pc);
    //    MPI_Bcast( ToRecv, nsupers, mpi_int_t, 0,  grid3d->zscp.comm);
    MPI_Bcast( ToRecv, nsupers, MPI_INT, 0,  grid3d->zscp.comm);
    
    MPI_Bcast( ToSendD, nbr, MPI_INT, 0,  grid3d->zscp.comm);
    for (int_t i = 0; i < nbc; ++i)
	{
	    /* code */
	    MPI_Bcast( ToSendR[i], Pc, MPI_INT, 0,  grid3d->zscp.comm);
	}
    
    //
#ifdef MPI_MALLOC
    // change MY LU struct into MPI malloc based
    if (!grid3d->zscp.Iam)
	mpiMallocLUStruct(nsupers, LUstruct, grid3d);
#endif
    return 0;
} /* dp3dScatter */


int_t dscatter3dUPanels(int_t nsupers,
		       dLUstruct_t * LUstruct, gridinfo3d_t* grid3d)
{

    dLocalLU_t *Llu = LUstruct->Llu;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);
    
    int_t k = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
    for ( int_t lb = 0; lb < k; ++lb) {
	int_t *usub;
	usub =  Ufstnz_br_ptr[lb];
	
	double * uval = Unzval_br_ptr[lb];
	
	int_t flag = 0;
	/*if non empty set the flag*/
	if (!grid3d->zscp.Iam && usub != NULL)
	    flag = 1;
	/*bcast the flag*/
	MPI_Bcast( &flag, 1, mpi_int_t, 0,  grid3d->zscp.comm);
	
	if (flag) {
	    int_t lenv, lens;
	    lenv = 0;
	    lens = 0;
	    
	    if (!grid3d->zscp.Iam)
		{
		    lenv = usub[1];
		    lens = usub[2];
		}
	    
	    /*broadcast the size of sub array*/
	    MPI_Bcast( &lens, 1, mpi_int_t, 0,  grid3d->zscp.comm);
	    MPI_Bcast( &lenv, 1, mpi_int_t, 0,  grid3d->zscp.comm);
	    
	    /*allocate lsub*/
	    if (grid3d->zscp.Iam)
#ifdef MPI_MALLOC
		MPI_INT_ALLOC(usub, lens);
#else
 	        usub = INT_T_ALLOC(lens);
#endif

	    /*bcast usub*/
	    MPI_Bcast( usub, lens, mpi_int_t, 0,  grid3d->zscp.comm);

	    /*allocate uval*/
	    if (grid3d->zscp.Iam)
#ifdef MPI_MALLOC
		MPI_DATATYPE_ALLOC(uval, lenv);
#else
	        uval = doubleMalloc_dist(lenv); //DOUBLE_ALLOC(lenv);
#endif
	    /*broadcast uval*/
	    MPI_Bcast( uval, lenv, MPI_DOUBLE, 0,  grid3d->zscp.comm);
	    
	    /*setup the pointer*/
	    Unzval_br_ptr[lb] = uval;
	    Ufstnz_br_ptr[lb] = usub;
	} /* end if flag */

    } /* end for lb ... */
    return 0;
} /* end dScatter3dUPanels */


int_t dscatter3dLPanels(int_t nsupers,
                       dLUstruct_t * LUstruct, gridinfo3d_t* grid3d)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = LUstruct->Glu_persist->xsup;
    gridinfo_t* grid = &(grid3d->grid2d);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t iam = grid->iam;
    
    int_t mycol = MYCOL (iam, grid);
    
    /*start broadcasting blocks*/
    for (int_t jb = 0; jb < nsupers; ++jb)   /* for each block column ... */
    {
	int_t pc = PCOL( jb, grid );
	if (mycol == pc)
        {
	    int_t ljb = LBj( jb, grid ); /* Local block number */
	    int_t  *lsub;
	    double* lnzval;
	    lsub = Lrowind_bc_ptr[ljb];
	    lnzval = Lnzval_bc_ptr[ljb];
		
	    int_t flag = 0;
	    /*if non empty set the flag*/
	    if (!grid3d->zscp.Iam && lsub != NULL)
		    flag = 1;
            /*bcast the flag*/
	    MPI_Bcast( &flag, 1, mpi_int_t, 0,  grid3d->zscp.comm);
		
            if (flag) {
		int_t nrbl, len, len1, len2;
		if (!grid3d->zscp.Iam)
		    {
			nrbl  =   lsub[0]; /*number of L blocks */
			len   = lsub[1];   /* LDA of the nzval[] */
			len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
			len2  = SuperSize(jb) * len;
		    }

		/*bcast lsub len*/
		MPI_Bcast( &len1, 1, mpi_int_t, 0,  grid3d->zscp.comm);
		    
   	        /*allocate lsub*/
		if (grid3d->zscp.Iam)
#ifdef MPI_MALLOC
		    MPI_INT_ALLOC(lsub, len1);
#else
		    
		    lsub = INT_T_ALLOC(len1);
#endif
		    /*now broadcast lsub*/
		    MPI_Bcast( lsub, len1, mpi_int_t, 0,  grid3d->zscp.comm);

		    /*set up pointer*/
		    Lrowind_bc_ptr[ljb] = lsub;
		    
		    /*bcast lnzval len*/
		    MPI_Bcast( &len2, 1, mpi_int_t, 0,  grid3d->zscp.comm);
		    
		    /*allocate space for nzval*/
		    if (grid3d->zscp.Iam)
#ifdef MPI_MALLOC
			MPI_DATATYPE_ALLOC(lnzval, len2);
#else
		        lnzval = doubleCalloc_dist(len2);
#endif
		    
		    /*bcast nonzero values*/
		    MPI_Bcast( lnzval, len2, MPI_DOUBLE, 0,  grid3d->zscp.comm);
		    
		    /*setup the pointers*/
		    Lnzval_bc_ptr[ljb] = lnzval;

		} /* end if flag */

	} /* end if mycol == pc */
    } /* end for jb ... */

    return 0;
} /* dscatter3dLPanels */

int_t dcollect3dLpanels(int_t layer, int_t nsupers, dLUstruct_t * LUstruct,
		       gridinfo3d_t* grid3d)
{

    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);

    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);

    /*start broadcasting blocks*/
    for (int_t jb = 0; jb < nsupers; ++jb)   /* for each block column ... */
    {
	int_t pc = PCOL( jb, grid );
	if (mycol == pc)
	{
	    int_t ljb = LBj( jb, grid ); /* Local block number */
	    int_t  *lsub;
	    double* lnzval;
	    lsub = Lrowind_bc_ptr[ljb];
	    lnzval = Lnzval_bc_ptr[ljb];
		    
	    if (lsub != NULL)
	    {
	        int_t len   = lsub[1];       /* LDA of the nzval[] */
		int_t len2  = SuperSize(jb) * len; /*size of nzval of L panel */
			    
	        if (grid3d->zscp.Iam == layer)
		{
		    MPI_Send(lnzval, len2, MPI_DOUBLE, 0, jb, grid3d->zscp.comm);
		}
		if (!grid3d->zscp.Iam)
		{
		    MPI_Status status;
		    MPI_Recv(lnzval, len2, MPI_DOUBLE, layer, jb, grid3d->zscp.comm, &status);
		}
	     }
	}
    } /* for jb ... */
    return 0;
}

int_t dcollect3dUpanels(int_t layer, int_t nsupers, dLUstruct_t * LUstruct,
      			 gridinfo3d_t* grid3d)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);
    
    int_t k = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
    for ( int_t lb = 0; lb < k; ++lb)
    {
	int_t *usub;
	usub =  Ufstnz_br_ptr[lb];
	double * uval = Unzval_br_ptr[lb];
	    
	if (usub)
	{
	    /* code */
	    int lenv = usub[1];
	    if (grid3d->zscp.Iam == layer)
		{
		    MPI_Send(uval, lenv, MPI_DOUBLE, 0, lb, grid3d->zscp.comm);
		}
		    
	    if (!grid3d->zscp.Iam)
		{
		    MPI_Status status;
		    MPI_Recv(uval, lenv, MPI_DOUBLE, layer, lb, grid3d->zscp.comm, &status);
		}
	}
    } /* for lb ... */
    return 0;
}

/* Gather the LU factors on layer-0 */
int_t dp3dCollect(int_t layer, int_t n, dLUstruct_t * LUstruct, gridinfo3d_t* grid3d)
{
    int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
    dcollect3dLpanels(layer, nsupers,  LUstruct, grid3d);
    dcollect3dUpanels(layer,  nsupers, LUstruct, grid3d);
    return 0;
}


/* Zero out LU non zero entries */
int_t dzeroSetLU(int_t nnodes, int_t* nodeList, dLUstruct_t *LUstruct,
      		 gridinfo3d_t* grid3d)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    
    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    gridinfo_t* grid = &(grid3d->grid2d);
    
    int_t iam = grid->iam;
    
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    
    /*first setting the L blocks to zero*/
    for (int_t node = 0; node < nnodes; ++node)   /* for each block column ... */
	{
	    
	    int_t jb = nodeList[node];
	    int_t pc = PCOL( jb, grid );
	    if (mycol == pc)
		{
		    int_t ljb = LBj( jb, grid ); /* Local block number */
		    int_t  *lsub;
		    double* lnzval;
		    lsub = Lrowind_bc_ptr[ljb];
		    lnzval = Lnzval_bc_ptr[ljb];
		    
		    if (lsub != NULL)
			{
			    int_t len   = lsub[1];       /* LDA of the nzval[] */
			    int_t len2  = SuperSize(jb) * len;	/*size of nzval of L panel */
			    memset( lnzval, 0, len2 * sizeof(double) );
			}
		}
	}

    for (int_t node = 0; node < nnodes; ++node)   /* for each block column ... */
	{
	    
	    int_t ib = nodeList[node];
	    int_t pr = PROW( ib, grid );
	    if (myrow == pr)
		{
		    int_t lib = LBi( ib, grid ); /* Local block number */
		    int_t  *usub;
		    double* unzval;
		    usub = Ufstnz_br_ptr[lib];
		    unzval = Unzval_br_ptr[lib];
		    
		    if (usub != NULL)
			{
			    int lenv = usub[1];
			    memset( unzval, 0, lenv * sizeof(double) );
			}
		}
	}
    
    return 0;
}


int_t dreduceAncestors3d(int_t sender, int_t receiver,
                        int_t nnodes, int_t* nodeList,
                        double* Lval_buf, double* Uval_buf,
                        dLUstruct_t* LUstruct,  gridinfo3d_t* grid3d, SCT_t* SCT)
{
    double alpha = 1.0, beta = 1.0;	
    int_t myGrid = grid3d->zscp.Iam;
    
    /*first setting the L blocks to zero*/
    for (int_t node = 0; node < nnodes; ++node)   /* for each block column ... */
	{
	    int_t jb = nodeList[node];
	    
	    if (myGrid == sender)
		{
		    dzSendLPanel(jb, receiver, LUstruct,  grid3d, SCT);
		    dzSendUPanel(jb, receiver, LUstruct,  grid3d, SCT);
		}
	    else {
	        dzRecvLPanel(jb, sender, alpha, beta, Lval_buf,
                                LUstruct, grid3d, SCT);
		dzRecvUPanel(jb, sender, alpha, beta, Uval_buf,
                                LUstruct,  grid3d, SCT);
	    }
	    
	}
    return 0;
    
}


int_t dgatherFactoredLU(int_t sender, int_t receiver,
                        int_t nnodes, int_t *nodeList,
                        dLUValSubBuf_t* LUvsb,
                        dLUstruct_t* LUstruct, gridinfo3d_t* grid3d, SCT_t* SCT)
{
    double alpha = 0.0, beta = 1.0;	
    double * Lval_buf  = LUvsb->Lval_buf;
    double * Uval_buf  = LUvsb->Uval_buf;
    int_t myGrid = grid3d->zscp.Iam;
    for (int_t node = 0; node < nnodes; ++node)   /* for each block column ... */
	{
	    int_t jb = nodeList[node];
	    if (myGrid == sender)
		{
		    dzSendLPanel(jb, receiver, LUstruct,  grid3d, SCT);
		    dzSendUPanel(jb, receiver, LUstruct,  grid3d, SCT);
		    
		}
	    else
		{
		    dzRecvLPanel(jb, sender, alpha, beta, Lval_buf,
                                     LUstruct, grid3d, SCT);
		    dzRecvUPanel(jb, sender, alpha, beta, Uval_buf,
                                     LUstruct, grid3d, SCT);
		}
	}
    return 0;
    
}


int_t dinit3DLUstruct( int_t* myTreeIdxs, int_t* myZeroTrIdxs,
                      int_t* nodeCount, int_t** nodeList, dLUstruct_t* LUstruct,
		      gridinfo3d_t* grid3d)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    
    for (int_t lvl = 0; lvl < maxLvl; lvl++)
	{
	    if (myZeroTrIdxs[lvl])
		{
		    /* code */
		    int_t treeId = myTreeIdxs[lvl];
		    dzeroSetLU(nodeCount[treeId], nodeList[treeId], LUstruct, grid3d);
		}
	}
    
    return 0;
}


int dreduceAllAncestors3d(int_t ilvl, int_t* myNodeCount, int_t** treePerm,
                             dLUValSubBuf_t* LUvsb, dLUstruct_t* LUstruct,
                             gridinfo3d_t* grid3d, SCT_t* SCT )
{
    double * Lval_buf  = LUvsb->Lval_buf;
    double * Uval_buf  = LUvsb->Uval_buf;
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;
    
    int_t sender, receiver;
    if ((myGrid % (1 << (ilvl + 1))) == 0)
	{
	    sender = myGrid + (1 << ilvl);
	    receiver = myGrid;
	}
    else
	{
	    sender = myGrid;
	    receiver = myGrid - (1 << ilvl);
	}
    
    /*Reduce all the ancestors*/
    for (int_t alvl = ilvl + 1; alvl < maxLvl; ++alvl)
	{
	    /* code */
	    // int_t atree = myTreeIdxs[alvl];
	    int_t nsAncestor = myNodeCount[alvl];
	    int_t* cAncestorList = treePerm[alvl];
	    double treduce = SuperLU_timer_();
	    dreduceAncestors3d(sender, receiver, nsAncestor, cAncestorList,
			        Lval_buf, Uval_buf, LUstruct, grid3d, SCT);
	    SCT->ancsReduce += SuperLU_timer_() - treduce;
	    
	}
    return 0;
}

int_t dgatherAllFactoredLU( dtrf3Dpartition_t*  trf3Dpartition,
			   dLUstruct_t* LUstruct, gridinfo3d_t* grid3d, SCT_t* SCT )
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;
    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    dLUValSubBuf_t*  LUvsb =  trf3Dpartition->LUvsb;
    int_t*  gNodeCount = getNodeCountsFr(maxLvl, sForests);
    int_t** gNodeLists = getNodeListFr(maxLvl, sForests);
    
    for (int_t ilvl = 0; ilvl < maxLvl - 1; ++ilvl)
	{
	    /* code */
	    int_t sender, receiver;
	    if (!myZeroTrIdxs[ilvl])
		{
		    if ((myGrid % (1 << (ilvl + 1))) == 0)
			{
			    sender = myGrid + (1 << ilvl);
			    receiver = myGrid;
			}
		    else
			{
			    sender = myGrid;
			    receiver = myGrid - (1 << ilvl);
			}
		    
		    for (int_t alvl = 0; alvl <= ilvl; alvl++)
			{
			    int_t diffLvl  = ilvl - alvl;
			    int_t numTrees = 1 << diffLvl;
			    int_t blvl = maxLvl - alvl - 1;
			    int_t st = (1 << blvl) - 1 + (sender >> alvl);
			    
			    for (int_t tr = st; tr < st + numTrees; ++tr)
				{
				    /* code */
				    dgatherFactoredLU(sender, receiver,
						     gNodeCount[tr], gNodeLists[tr],
						     LUvsb,
						     LUstruct, grid3d, SCT );
				}
			}
		    
		}
	} /* for ilvl ... */
    	
    SUPERLU_FREE(gNodeCount); // sherry added
    SUPERLU_FREE(gNodeLists);

    return 0;
} /* dgatherAllFactoredLU */

