//
// Created by NanDing on 7/28/22.
//
#ifdef USE_FOMPI
#include "fompi.h"
#else
#include "mpi.h"
#endif

#ifdef USE_FOMPI
extern foMPI_Win bc_winl;
extern foMPI_Win rd_winl;
#else
extern MPI_Win bc_winl;
extern MPI_Win rd_winl;
#endif

extern MPI_Comm row_comm;
extern MPI_Comm col_comm;
extern int *BufSize;
extern int *BufSize_rd;
extern int *keep_validBCQindex;
extern int *keep_validRDQindex;
extern int *BufSize_u;
extern int *BufSize_urd;
extern int *keep_validBCQindex_u;
extern int *keep_validRDQindex_u;
extern int *recv_size_all;
extern int *recv_size_all_u;
extern double* BC_taskq;
extern double* RD_taskq;
