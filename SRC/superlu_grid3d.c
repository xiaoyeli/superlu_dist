/*! @file
 * \brief SuperLU grid utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.1.0) --
 * Lawrence Berkeley National Lab, Oak Ridge National Lab
 * May 12, 2021
 * October 5, 2021
 * </pre>
 */

#include "superlu_ddefs.h"

void superlu_gridmap3d(
    MPI_Comm Bcomm, /* The base communicator upon which
                    the new grid is formed. */
    int nprow,
    int npcol,
    int npdep,
    gridinfo3d_t *grid);


/*! \brief All processes in the MPI communicator must call this routine.
 */
void superlu_gridinit3d(MPI_Comm Bcomm, /* The base communicator upon which
					   the new grid is formed. */
                        int nprow, int npcol, int npdep, gridinfo3d_t *grid)
{
    int Np = nprow * npcol * npdep;
    int i, j, info;

    /* Make a list of the processes in the new communicator. */
    //    usermap = (int_t *) SUPERLU_MALLOC(Np*sizeof(int_t));
    //    for (j = 0; j < npcol; ++j)
    //        for (i = 0; i < nprow; ++i) usermap[j*nprow+i] = i*npcol+j;

    /* Check MPI environment initialization. */
    MPI_Initialized( &info );
    if ( !info )
        ABORT("C main program must explicitly call MPI_Init()");

    MPI_Comm_size( Bcomm, &info );
    if ( info < Np )
        ABORT("Number of processes is smaller than NPROW * NPCOL * NPDEP");

    superlu_gridmap3d(Bcomm, nprow, npcol, npdep, grid);

    // SUPERLU_FREE(usermap);
    
#ifdef GPU_ACC
    /* Binding each MPI to a GPU device */
    char *ttemp;
    ttemp = getenv ("SUPERLU_BIND_MPI_GPU");

    if (ttemp) {
	int devs, rank;
	MPI_Comm_rank(Bcomm, &rank); // MPI_COMM_WORLD??
	gpuGetDeviceCount(&devs);  // Returns the number of compute-capable devices
	gpuSetDevice(rank % devs); // Set device to be used for GPU executions
    }
#endif
}


/*! \brief All processes in the MPI communicator must call this routine.
 *  On output, if a process is not in the SuperLU group, the following
 *  values are assigned to it:
 *      grid->comm = MPI_COMM_NULL
 *      grid->iam = -1
 */
void superlu_gridmap3d(
    MPI_Comm Bcomm, /* The base communicator upon which
		       the new grid is formed. */
    int nprow,
    int npcol,
    int npdep,
    gridinfo3d_t *grid)
{
    MPI_Group mpi_base_group, superlu_grp;
    int Np = nprow * npcol * npdep, mycol, myrow;
    int *pranks;
    int i, j, info;

#if 0 // older MPI doesn't support complex in C    
    /* Create datatype in C for MPI complex. */
    if ( SuperLU_MPI_DOUBLE_COMPLEX == MPI_DATATYPE_NULL ) {
        MPI_Type_contiguous( 2, MPI_DOUBLE, &SuperLU_MPI_DOUBLE_COMPLEX );
        MPI_Type_commit( &SuperLU_MPI_DOUBLE_COMPLEX );
    }
#endif
    
    /* Check MPI environment initialization. */
    MPI_Initialized( &info );
    if ( !info )
        ABORT("C main program must explicitly call MPI_Init()");

    /* Make a list of the processes in the new communicator. */
    pranks = (int *) SUPERLU_MALLOC(Np * sizeof(int));
    for (j = 0; j < Np; ++j)
        pranks[j] = j;

    /*
     * Form MPI communicator for all.
     */
    /* Get the group underlying Bcomm. */
    MPI_Comm_group( Bcomm, &mpi_base_group );
    /* Create the new group. */
    MPI_Group_incl( mpi_base_group, Np, pranks, &superlu_grp );
    /* Create the new communicator. */
    /* NOTE: The call is to be executed by all processes in Bcomm,
       even if they do not belong in the new group -- superlu_grp.
       The function returns MPI_COMM_NULL to processes that are not in superlu_grp. */
    MPI_Comm_create( Bcomm, superlu_grp, &grid->comm );

    /* Bail out if I am not in the group, superlu_group. */
    if ( grid->comm == MPI_COMM_NULL ) {
        //grid->comm = Bcomm; do not need to reassign to a valid communicator
        grid->iam = -1;
        //SUPERLU_FREE(pranks);
        //return;
	goto gridmap_out;
    }

    grid->nprow = nprow;
    grid->npcol = npcol;
    grid->npdep = npdep;

    /* Create 3D grid */
    int ndim = 3;
    int dims[3];
    int reorder = 1;
    int periodic[] = {0, 0, 0};
    int coords3d[3];
    int iam;
    MPI_Comm superlu3d_comm;

    if ( getenv("SUPERLU_RANKORDER") && strcmp(getenv("SUPERLU_RANKORDER"), "XY" ) )
    {
	grid->rankorder = 1;  // XY-major 

        dims[0] = nprow;
        dims[1] = npcol;
        dims[2] = npdep;
	
        // create the new communicator
        int error = MPI_Cart_create(grid->comm, ndim, dims, periodic, reorder, &superlu3d_comm);

        // get the coordinate of the processor

        MPI_Comm_rank (superlu3d_comm, &iam);
        grid->iam = iam;
        MPI_Cart_coords(superlu3d_comm, iam, ndim, coords3d);

        int rowc[3] = {1, 0, 0};
        int colc[3] = {0, 1, 0};
        int depc[3] = {0, 0, 1};

	// Partition a communicator into subgroups which form
	// lower-dimensional cartesian subgrids
        MPI_Cart_sub(superlu3d_comm, colc, &(grid->rscp.comm)); /* XZ grids */
        MPI_Cart_sub(superlu3d_comm, rowc, &(grid->cscp.comm)); /* YZ grids */
        MPI_Cart_sub(superlu3d_comm, depc, &(grid->zscp.comm)); /* XY grids */

        grid->cscp.Np = nprow;
        grid->cscp.Iam = coords3d[0];
        grid->rscp.Np = npcol;
        grid->rscp.Iam = coords3d[1];
        grid->zscp.Np = npdep;
        grid->zscp.Iam = coords3d[2];

        //
        grid->nprow = nprow;
        grid->npcol = npcol;
        grid->npdep = npdep;

        // 2D communicator
        int xyc[3] = {1, 1, 0};
        MPI_Cart_sub(superlu3d_comm, xyc, &(grid->grid2d.comm));

    } else { /* default */
        grid->rankorder = 0; // Z-major

        dims[1] = nprow;
        dims[2] = npcol;
        dims[0] = npdep;

        // get the communicator
        int error = MPI_Cart_create(grid->comm, ndim, dims, periodic, reorder, &superlu3d_comm);

        //get the coordinate of the processor

        MPI_Comm_rank (superlu3d_comm, &iam);
        grid->iam = iam;
        MPI_Cart_coords(superlu3d_comm, iam, ndim, coords3d);

	/* printf("(%d) My coordinats are (%d %d %d)\n",
	   iam, coords3d[0], coords3d[1], coords3d[2] );
	fflush(stdout);  */

        // create row communicator

        int rowc[3] = {0, 1, 0};
        int colc[3] = {0, 0, 1};
        int depc[3] = {1, 0, 0};

        MPI_Cart_sub(superlu3d_comm, colc, &(grid->rscp.comm));
        MPI_Cart_sub(superlu3d_comm, rowc, &(grid->cscp.comm));
        MPI_Cart_sub(superlu3d_comm, depc, &(grid->zscp.comm));

	//  2x3: 0,2,4 / 1,3,5  column-major
        grid->cscp.Np = nprow;
        grid->cscp.Iam = coords3d[1];
        grid->rscp.Np = npcol;
        grid->rscp.Iam = coords3d[2];

        grid->zscp.Np = npdep;
        grid->zscp.Iam = coords3d[0];

        grid->nprow = nprow;
        grid->npcol = npcol;
        grid->npdep = npdep;

        // 2D communicator
        int xyc[3] = {0, 1, 1};
        MPI_Cart_sub(superlu3d_comm, xyc, &(grid->grid2d.comm));

    } /* end if SUPERLU_RANKORDER */


    // Initialize grid2d;

    grid->grid2d.rscp = grid->rscp;
    grid->grid2d.cscp = grid->cscp;
    grid->grid2d.nprow = nprow;
    grid->grid2d.npcol = npcol;
    MPI_Comm_rank( grid->grid2d.comm, &(grid->grid2d.iam));

    // grid->grid2d.cscp = grid->cscp;

#if ( PRNTlevel>=1 )
    if ( (grid->zscp).Iam == 0) {
      printf("(3d grid: layer 0) iam %d, grid->grid2d.iam %d\n",
	     grid->iam, (grid->grid2d).iam);
    } 
    fflush(stdout);
#endif

    MPI_Comm_free( &superlu3d_comm );  // Sherry added
    
 gridmap_out:    
    SUPERLU_FREE(pranks);
    MPI_Group_free( &superlu_grp );
    MPI_Group_free( &mpi_base_group );
}

void superlu_gridexit3d(gridinfo3d_t *grid)
{
    if ( grid->comm != MPI_COMM_NULL && grid->comm != MPI_COMM_WORLD ) {
        /* Marks the communicator objects for deallocation. */
        MPI_Comm_free( &grid->rscp.comm );
        MPI_Comm_free( &grid->cscp.comm );
        MPI_Comm_free( &grid->zscp.comm );
        MPI_Comm_free( &grid->grid2d.comm );
        MPI_Comm_free( &grid->comm );
    }
#if 0    
    if ( SuperLU_MPI_DOUBLE_COMPLEX != MPI_DATATYPE_NULL ) {
        MPI_Type_free( &SuperLU_MPI_DOUBLE_COMPLEX );
	SuperLU_MPI_DOUBLE_COMPLEX = MPI_DATATYPE_NULL; /* some MPI system does not set this
							   to be NULL after Type_free */
    }
#endif    
}
