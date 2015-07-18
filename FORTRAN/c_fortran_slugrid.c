#include "superlu_ddefs.h"

#define HANDLE_SIZE 8

void
c_fortran_slugrid_(int *iopt, MPI_Comm *slu_comm, int *nprow, int *npcol,
		   int grid_handle[HANDLE_SIZE])
/*
 * This routine provides a fortran call for initializing and 
 * freeing the SuperLU_DIST processor grid.  The pointer for the grid
 * structure is returned in grid_handle.
 *
 * The input option, iopt, controls the functionality:
 *   iopt=1:  allocate and define a new process grid
 *   iopt=2:  free an existing process grid
 *
 * slu_comm is the base communication handle
 * nprow is the number of processors per process grid row
 * npcol is the number of processors per process grid column
 */

{
    gridinfo_t *grid;

    if ( *iopt == 1 ) {
      /* Allocate the grid structure. */
      grid = (gridinfo_t *) SUPERLU_MALLOC(sizeof(gridinfo_t));

      /* Initialize the process grid. */
      superlu_gridinit(*slu_comm, *nprow, *npcol, grid);

      /* Set the handle passed from fortran, so that the
       * process grid can be reused. */
      grid_handle[0] = (int) grid;

    } else if ( *iopt == 2 ) {
      /* Locate and free the process grid. */
      grid = (gridinfo_t *) grid_handle[0];
      superlu_gridexit(grid);
      SUPERLU_FREE(grid);

    } else {
      fprintf(stderr, "Invalid iopt=%d passed to c_fortran_slugrid()\n", *iopt);
      exit(-1);
    }
}
