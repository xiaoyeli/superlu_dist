#include "mpi.h"
extern MPI_Win bc_winl;
extern MPI_Win rd_winl;
extern MPI_Comm row_comm;
extern MPI_Comm col_comm;
extern int *BufSize;
extern int *BufSize_rd;
extern double *onesidecomm_bc;
extern double *onesidecomm_rd;
extern double *onesidedgemm;
