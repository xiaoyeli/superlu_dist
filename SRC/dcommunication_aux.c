/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Communication routines.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology.
 * May 10, 2019
 */
#include "superlu_ddefs.h"
#if 0
#include "sec_structs.h"
#include "communication_aux.h"
#include "compiler.h"
#endif

int_t dIBcast_LPanel
/*broadcasts index array lsub and non-zero value
 array lusup of a newly factored L column to my process row*/
(int_t k, int_t k0, int_t* lsub, double* lusup, gridinfo_t *grid,
 int* msgcnt, MPI_Request *send_req, int **ToSendR, int_t *xsup,
 int tag_ub)
{
    int_t Pc = grid->npcol;
    int_t lk = LBj (k, grid);
    superlu_scope_t *scp = &grid->rscp;  /* The scope of process row. */
    if (lsub)
    {
        msgcnt[0] = lsub[1] + BC_HEADER + lsub[0] * LB_DESCRIPTOR;
        msgcnt[1] = lsub[1] * SuperSize (k);
    }
    else
    {
        msgcnt[0] = msgcnt[1] = 0;
    }

    for (int_t pj = 0; pj < Pc; ++pj)
    {
        if (ToSendR[lk][pj] != EMPTY)
        {


            MPI_Isend (lsub, msgcnt[0], mpi_int_t, pj,
                       SLU_MPI_TAG (0, k0) /* 0 */ ,
                       scp->comm, &send_req[pj]);
            MPI_Isend (lusup, msgcnt[1], MPI_DOUBLE, pj,
                       SLU_MPI_TAG (1, k0) /* 1 */ ,
                       scp->comm, &send_req[pj + Pc]);

        }
    }

    return 0;
}


int_t dBcast_LPanel
/*broadcasts index array lsub and non-zero value
 array lusup of a newly factored L column to my process row*/
(int_t k, int_t k0, int_t* lsub, double* lusup, gridinfo_t *grid,
 int* msgcnt,  int **ToSendR, int_t *xsup , SCT_t* SCT,
 int tag_ub)
{
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    int_t Pc = grid->npcol;
    int_t lk = LBj (k, grid);
    superlu_scope_t *scp = &grid->rscp;  /* The scope of process row. */
    if (lsub)
    {
        msgcnt[0] = lsub[1] + BC_HEADER + lsub[0] * LB_DESCRIPTOR;
        msgcnt[1] = lsub[1] * SuperSize (k);
    }
    else
    {
        msgcnt[0] = msgcnt[1] = 0;
    }

    for (int_t pj = 0; pj < Pc; ++pj)
    {
        if (ToSendR[lk][pj] != EMPTY)
        {


            MPI_Send (lsub, msgcnt[0], mpi_int_t, pj,
                       SLU_MPI_TAG (0, k0) /* 0 */ ,
                       scp->comm);
            MPI_Send (lusup, msgcnt[1], MPI_DOUBLE, pj,
                       SLU_MPI_TAG (1, k0) /* 1 */ ,
                       scp->comm);

        }
    }
    //SCT->Bcast_UPanel_tl += (double) ( _rdtsc() - t1);
    SCT->Bcast_UPanel_tl +=  SuperLU_timer_() - t1;
    return 0;
}



int_t dIBcast_UPanel
/*asynchronously braodcasts U panel to my process row */
(int_t k, int_t k0, int_t* usub, double* uval, gridinfo_t *grid,
 int* msgcnt, MPI_Request *send_req_u, int *ToSendD, int tag_ub )
{

    int_t iam = grid->iam;
    int_t lk = LBi (k, grid);
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    superlu_scope_t *scp = &grid->cscp; /* The scope of process col. */
    if (usub)
    {
        msgcnt[2] = usub[2];
        msgcnt[3] = usub[1];
    }
    else
    {
        msgcnt[2] = msgcnt[3] = 0;
    }

    if (ToSendD[lk] == YES)
    {
        for (int_t pi = 0; pi < Pr; ++pi)
        {
            if (pi != myrow)
            {

                MPI_Isend (usub, msgcnt[2], mpi_int_t, pi,
                           SLU_MPI_TAG (2, k0) /* (4*k0+2)%tag_ub */ ,
                           scp->comm,
                           &send_req_u[pi]);
                MPI_Isend (uval, msgcnt[3], MPI_DOUBLE,
                           pi, SLU_MPI_TAG (3, k0) /* (4*kk0+3)%tag_ub */ ,
                           scp->comm,
                           &send_req_u[pi + Pr]);

            }   /* if pi ... */
        }   /* for pi ... */
    }       /* if ToSendD ... */
    return 0;
}

/*Synchronously braodcasts U panel to my process row */
int_t dBcast_UPanel(int_t k, int_t k0, int_t* usub,
                     double* uval, gridinfo_t *grid,
		    int* msgcnt, int *ToSendD, SCT_t* SCT, int tag_ub)

{
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    int_t iam = grid->iam;
    int_t lk = LBi (k, grid);
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    superlu_scope_t *scp = &grid->cscp; /* The scope of process col. */
    if (usub)
    {
        msgcnt[2] = usub[2];
        msgcnt[3] = usub[1];
    }
    else
    {
        msgcnt[2] = msgcnt[3] = 0;
    }

    if (ToSendD[lk] == YES)
    {
        for (int_t pi = 0; pi < Pr; ++pi)
        {
            if (pi != myrow)
            {
                MPI_Send (usub, msgcnt[2], mpi_int_t, pi,
                          SLU_MPI_TAG (2, k0) /* (4*k0+2)%tag_ub */ ,
                          scp->comm);
                MPI_Send (uval, msgcnt[3], MPI_DOUBLE, pi,
                          SLU_MPI_TAG (3, k0) /* (4*k0+3)%tag_ub */ ,
                          scp->comm);

            }       /* if pi ... */
        }           /* for pi ... */
    }
    //SCT->Bcast_UPanel_tl += (double) ( _rdtsc() - t1);
    SCT->Bcast_UPanel_tl += SuperLU_timer_() - t1;
    return 0;
}

int_t dIrecv_LPanel
/*it places Irecv call for L panel*/
(int_t k, int_t k0,  int_t* Lsub_buf, double* Lval_buf,
 gridinfo_t *grid, MPI_Request *recv_req, dLocalLU_t *Llu, int tag_ub )
{
    int_t kcol = PCOL (k, grid);

    superlu_scope_t *scp = &grid->rscp;  /* The scope of process row. */
    MPI_Irecv (Lsub_buf, Llu->bufmax[0], mpi_int_t, kcol,
               SLU_MPI_TAG (0, k0) /* 0 */ ,
               scp->comm, &recv_req[0]);
    MPI_Irecv (Lval_buf, Llu->bufmax[1], MPI_DOUBLE, kcol,
               SLU_MPI_TAG (1, k0) /* 1 */ ,
               scp->comm, &recv_req[1]);
    return 0;
}


int_t dIrecv_UPanel
/*it places Irecv calls to receive U panels*/
(int_t k, int_t k0, int_t* Usub_buf, double* Uval_buf, dLocalLU_t *Llu,
 gridinfo_t* grid, MPI_Request *recv_req_u, int tag_ub )
{
    int_t krow = PROW (k, grid);
    superlu_scope_t *scp = &grid->cscp;  /* The scope of process column. */
    MPI_Irecv (Usub_buf, Llu->bufmax[2], mpi_int_t, krow,
               SLU_MPI_TAG (2, k0) /* (4*kk0+2)%tag_ub */ ,
               scp->comm, &recv_req_u[0]);
    MPI_Irecv (Uval_buf, Llu->bufmax[3], MPI_DOUBLE, krow,
               SLU_MPI_TAG (3, k0) /* (4*kk0+3)%tag_ub */ ,
               scp->comm, &recv_req_u[1]);

    return 0;
}

int_t dWait_URecv
( MPI_Request *recv_req, int* msgcnt, SCT_t* SCT)
{
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    MPI_Status status;
    MPI_Wait (&recv_req[0], &status);
    MPI_Get_count (&status, mpi_int_t, &msgcnt[2]);
    MPI_Wait (&recv_req[1], &status);
    MPI_Get_count (&status, MPI_DOUBLE, &msgcnt[3]);
    //SCT->Wait_URecv_tl += (double) ( _rdtsc() - t1);
    SCT->Wait_URecv_tl +=  SuperLU_timer_() - t1;
    return 0;
}

int_t dWait_LRecv
/*waits till L blocks have been received*/
(  MPI_Request* recv_req, int* msgcnt, int* msgcntsU, gridinfo_t * grid, SCT_t* SCT)
{
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    MPI_Status status;
    
    if (recv_req[0] != MPI_REQUEST_NULL)
    {
        MPI_Wait (&recv_req[0], &status);
        MPI_Get_count (&status, mpi_int_t, &msgcnt[0]);
        recv_req[0] = MPI_REQUEST_NULL;
    }
    else
    {
        msgcnt[0] = msgcntsU[0];
    }

    if (recv_req[1] != MPI_REQUEST_NULL)
    {
        MPI_Wait (&recv_req[1], &status);
        MPI_Get_count (&status, MPI_DOUBLE, &msgcnt[1]);
        recv_req[1] = MPI_REQUEST_NULL;
    }
    else
    {
        msgcnt[1] = msgcntsU[1];
    }
    //SCT->Wait_LRecv_tl += (double) ( _rdtsc() - t1);
    SCT->Wait_LRecv_tl +=  SuperLU_timer_() - t1;
    return 0;
}


int_t dISend_UDiagBlock(int_t k0, double *ublk_ptr, /*pointer for the diagonal block*/
                       int_t size, /*number of elements to be broadcasted*/
                       MPI_Request *U_diag_blk_send_req,
                       gridinfo_t * grid, int tag_ub)
{
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    MPI_Comm comm = (grid->cscp).comm;
    /** ALWAYS SEND TO ALL OTHERS - TO FIX **/
    for (int_t pr = 0; pr < Pr; ++pr)
    {
        if (pr != myrow)
        {
            /* tag = ((k0<<2)+2) % tag_ub;        */
            /* tag = (4*(nsupers+k0)+2) % tag_ub; */
            MPI_Isend (ublk_ptr, size, MPI_DOUBLE, pr,
                       SLU_MPI_TAG (4, k0) /* tag */ ,
                       comm, U_diag_blk_send_req + pr);
        }
    }

    return 0;
}


int_t dRecv_UDiagBlock(int_t k0, double *ublk_ptr, /*pointer for the diagonal block*/
                      int_t size, /*number of elements to be broadcasted*/
                      int_t src,
                      gridinfo_t * grid, SCT_t* SCT, int tag_ub)
{
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    MPI_Status status;
    MPI_Comm comm = (grid->cscp).comm;
    /* tag = ((k0<<2)+2) % tag_ub;        */
    /* tag = (4*(nsupers+k0)+2) % tag_ub; */

    MPI_Recv (ublk_ptr, size, MPI_DOUBLE, src,
              SLU_MPI_TAG (4, k0), comm, &status);
    //SCT->Recv_UDiagBlock_tl += (double) ( _rdtsc() - t1);
    SCT->Recv_UDiagBlock_tl +=  SuperLU_timer_() - t1;
    return 0;
}


int_t dPackLBlock(int_t k, double* Dest, Glu_persist_t *Glu_persist,
                  gridinfo_t *grid, dLocalLU_t *Llu)
/*Copies src matrix into dest matrix*/
{
    /* Initialization. */
    int_t *xsup = Glu_persist->xsup;
    int_t lk = LBj (k, grid);          /* Local block number */
    double *lusup = Llu->Lnzval_bc_ptr[lk];
    int_t nsupc = SuperSize (k);
    int_t nsupr;
    if (Llu->Lrowind_bc_ptr[lk])
        nsupr = Llu->Lrowind_bc_ptr[lk][1];
    else
        nsupr = 0;
#if 0
    LAPACKE_dlacpy (LAPACK_COL_MAJOR, 'A', nsupc, nsupc, lusup, nsupr, Dest, nsupc);
#else /* Sherry */
    for (int j = 0; j < nsupc; ++j) {
	memcpy( &Dest[j * nsupc], &lusup[j * nsupr], nsupc * sizeof(double) );
    }
#endif
    
    return 0;
}

int_t dISend_LDiagBlock(int_t k0, double *lblk_ptr, /*pointer for the diagonal block*/
                       int_t size,                                        /*number of elements to be broadcasted*/
                       MPI_Request *L_diag_blk_send_req,
                       gridinfo_t * grid, int tag_ub)
{
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    MPI_Comm comm = (grid->rscp).comm; /*Row communicator*/
    /** ALWAYS SEND TO ALL OTHERS - TO FIX **/
    for (int_t pc = 0; pc < Pc; ++pc)
    {
        if (pc != mycol)
        {
            /* tag = ((k0<<2)+2) % tag_ub;        */
            /* tag = (4*(nsupers+k0)+2) % tag_ub; */
            MPI_Isend (lblk_ptr, size, MPI_DOUBLE, pc,
                       SLU_MPI_TAG (5, k0) /* tag */ ,
                       comm, L_diag_blk_send_req + pc);

        }
    }

    return 0;
}


int_t dIRecv_UDiagBlock(int_t k0, double *ublk_ptr, /*pointer for the diagonal block*/
                       int_t size,                                        /*number of elements to be broadcasted*/
                       int_t src,
                       MPI_Request *U_diag_blk_recv_req,
                       gridinfo_t * grid, SCT_t* SCT, int tag_ub)
{
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    MPI_Comm comm = (grid->cscp).comm;
    /* tag = ((k0<<2)+2) % tag_ub;        */
    /* tag = (4*(nsupers+k0)+2) % tag_ub; */

    int_t err = MPI_Irecv (ublk_ptr, size, MPI_DOUBLE, src,
               		   SLU_MPI_TAG (4, k0), comm, U_diag_blk_recv_req);
    if (err==MPI_ERR_COUNT)
    {
        printf("Error in IRecv_UDiagBlock count\n");
    }
    //SCT->Recv_UDiagBlock_tl += (double) ( _rdtsc() - t1);
    SCT->Recv_UDiagBlock_tl += SuperLU_timer_() - t1;
    return 0;
}

int_t dIRecv_LDiagBlock(int_t k0, double *L_blk_ptr, /*pointer for the diagonal block*/
                       int_t size,  /*number of elements to be broadcasted*/
                       int_t src,
                       MPI_Request *L_diag_blk_recv_req,
                       gridinfo_t * grid, SCT_t* SCT, int tag_ub)
{
    //unsigned long long t1 = _rdtsc();
    double t1 = SuperLU_timer_();
    MPI_Comm comm = (grid->rscp).comm;
    /* tag = ((k0<<2)+2) % tag_ub;        */
    /* tag = (4*(nsupers+k0)+2) % tag_ub; */

    int_t err = MPI_Irecv (L_blk_ptr, size, MPI_DOUBLE, src,
                   SLU_MPI_TAG (5, k0),
                   comm, L_diag_blk_recv_req);
    if (err==MPI_ERR_COUNT)
    {
        printf("Error in IRecv_lDiagBlock count\n");
    }
    //SCT->Recv_UDiagBlock_tl += (double) ( _rdtsc() - t1);
    SCT->Recv_UDiagBlock_tl +=  SuperLU_timer_() - t1;
    return 0;
}

#if (MPI_VERSION>2)

/****Ibcast based on mpi ibcast****/
int_t dIBcast_UDiagBlock(int_t k, double *ublk_ptr, /*pointer for the diagonal block*/
                        int_t size,  /*number of elements to be broadcasted*/
                        MPI_Request *L_diag_blk_ibcast_req,
                        gridinfo_t * grid)
{
    int_t  krow = PROW (k, grid);
    MPI_Comm comm = (grid->cscp).comm;

    MPI_Ibcast(ublk_ptr, size, MPI_DOUBLE, krow,comm, L_diag_blk_ibcast_req);
    
    // MPI_Status status;
    // MPI_Wait(L_diag_blk_ibcast_req, &status);
    return 0;
}

int_t dIBcast_LDiagBlock(int_t k, double *lblk_ptr, /*pointer for the diagonal block*/
                        int_t size,  /*number of elements to be broadcasted*/
                        MPI_Request *U_diag_blk_ibcast_req,
                        gridinfo_t * grid)
{
    int_t  kcol = PCOL (k, grid);
    MPI_Comm comm = (grid->rscp).comm;

    MPI_Ibcast(lblk_ptr, size, MPI_DOUBLE, kcol,comm, U_diag_blk_ibcast_req);
    // MPI_Status status;
    // MPI_Wait(U_diag_blk_ibcast_req, &status);
    return 0;
}

#endif 

int_t dUDiagBlockRecvWait( int_t k,  int_t* IrecvPlcd_D, int_t* factored_L,
                           MPI_Request * U_diag_blk_recv_req,
                           gridinfo_t *grid,
                           dLUstruct_t *LUstruct, SCT_t *SCT)
{
    dLocalLU_t *Llu = LUstruct->Llu;

    int_t iam = grid->iam;

    int_t mycol = MYCOL (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);

    int_t kcol = PCOL (k, grid);

    if (IrecvPlcd_D[k] == 1)
    {
        /* code */
        /*factor the L panel*/
        if (mycol == kcol  && factored_L[k] == 0 && iam != pkk)
        {
            factored_L[k] = 1;
            int_t lk = LBj (k, grid);

            int_t nsupr;
            if (Llu->Lrowind_bc_ptr[lk])
                nsupr = Llu->Lrowind_bc_ptr[lk][1];
            else
                nsupr = 0;
            /*wait for communication to finish*/

            // Wait_UDiagBlock_Recv( U_diag_blk_recv_req, SCT);
            int_t flag = 0;
            while (flag == 0)
            {
                flag = Test_UDiagBlock_Recv( U_diag_blk_recv_req, SCT);
            }
        }
    }
    return 0;
}

