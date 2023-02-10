/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Auxiliary routines to support communication in 3D algorithms
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Oak Ridge National Lab
 * May 12, 2021
 * </pre>
 */

#include "superlu_defs.h"
#if 0
#include "sec_structs.h"
#include "communication_aux.h"
#include "compiler.h"
#endif


int_t Wait_LSend
/*wait till broadcast of L finished*/
(int_t k, gridinfo_t *grid, int **ToSendR, MPI_Request *send_req, SCT_t* SCT)
{
    double t1 = SuperLU_timer_();
    int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t lk = LBj (k, grid);
    int_t mycol = MYCOL (iam, grid);
    MPI_Status status;
    for (int_t pj = 0; pj < Pc; ++pj)
    {
        /* Wait for Isend to complete before using lsub/lusup. */
        if (ToSendR[lk][pj] != EMPTY && pj != mycol)
        {
            MPI_Wait (&send_req[pj], &status);
            MPI_Wait (&send_req[pj + Pc], &status);
        }
    }
    SCT->Wait_LSend_tl += ( SuperLU_timer_() - t1);
    return 0;
}


int_t Wait_USend
/*wait till broadcast of U panels finished*/
( MPI_Request *send_req, gridinfo_t *grid, SCT_t* SCT)
{
    double t1 = SuperLU_timer_();
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    MPI_Status status;
    for (int_t pi = 0; pi < Pr; ++pi)
    {
        if (pi != myrow)
        {
            MPI_Wait (&send_req[pi], &status);
            MPI_Wait (&send_req[pi + Pr], &status);
        }
    }
    SCT->Wait_USend_tl += (double) (SuperLU_timer_() - t1);
    return 0;
}


int_t Check_LRecv
/*checks if diagnoal blocks have been received*/

(  MPI_Request* recv_req, int* msgcnt )
{
    int flag0, flag1;
    MPI_Status status;

    flag0 = flag1 = 0;
    if (recv_req[0] != MPI_REQUEST_NULL)
    {
        MPI_Test (&recv_req[0], &flag0, &status);
        if (flag0)
        {
            MPI_Get_count (&status, mpi_int_t, &msgcnt[0]);
            recv_req[0] = MPI_REQUEST_NULL;
        }
    }
    else
        flag0 = 1;
    if (recv_req[1] != MPI_REQUEST_NULL)
    {
        MPI_Test (&recv_req[1], &flag1, &status);
        if (flag1)
        {
            MPI_Get_count (&status, mpi_int_t, &msgcnt[1]);
            recv_req[1] = MPI_REQUEST_NULL;
        }
    }
    else
        flag1 = 1;

    return flag1 && flag0;
}


int_t Wait_UDiagBlockSend(MPI_Request *U_diag_blk_send_req,
                          gridinfo_t * grid, SCT_t* SCT)
{

    double t1 = SuperLU_timer_();
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    MPI_Status status;
    for (int_t pr = 0; pr < Pr; ++pr)
    {
        if (pr != myrow)
        {
            MPI_Wait (U_diag_blk_send_req + pr, &status);
        }
    }
    SCT->Wait_UDiagBlockSend_tl += (double) ( SuperLU_timer_() - t1);
    return 0;
}

int_t Wait_LDiagBlockSend(MPI_Request *L_diag_blk_send_req,
                          gridinfo_t * grid, SCT_t* SCT)
{
    double t1 = SuperLU_timer_();
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    MPI_Status status;
    for (int_t pc = 0; pc < Pc; ++pc)
    {
        if (pc != mycol)
        {
            MPI_Wait (L_diag_blk_send_req + pc, &status);
        }
    }
    SCT->Wait_UDiagBlockSend_tl += (double) ( SuperLU_timer_() - t1);
    return 0;
}


int_t Wait_UDiagBlock_Recv( MPI_Request *request, SCT_t* SCT)
{
    double t1 = SuperLU_timer_();
    MPI_Status status;
    MPI_Wait(request, &status);
    SCT->Wait_UDiagBlock_Recv_tl += (double) ( SuperLU_timer_() - t1);
    return 0;
}

int_t Test_UDiagBlock_Recv( MPI_Request *request, SCT_t* SCT)
{
    double t1 = SuperLU_timer_();
    MPI_Status status;
    int flag;
    MPI_Test(request,&flag, &status);
    SCT->Wait_UDiagBlock_Recv_tl += (double) ( SuperLU_timer_() - t1);
    return flag;

}

int_t Wait_LDiagBlock_Recv( MPI_Request *request, SCT_t* SCT)
{
    double t1 = SuperLU_timer_();
    MPI_Status status;
    MPI_Wait(request, &status);
    SCT->Wait_LDiagBlock_Recv_tl += (double) ( SuperLU_timer_() - t1);
    return 0;

}

int_t Test_LDiagBlock_Recv( MPI_Request *request, SCT_t* SCT)
{
    double t1 = SuperLU_timer_();
    MPI_Status status;
    int flag;
    MPI_Test(request, &flag, &status);
    SCT->Wait_LDiagBlock_Recv_tl += (double) ( SuperLU_timer_() - t1);
    return flag;
}

/*
 * The following are from trfCommWrapper.c.
 */
int Wait_LUDiagSend(int_t k, MPI_Request *U_diag_blk_send_req,
                      MPI_Request *L_diag_blk_send_req,
                      gridinfo_t *grid, SCT_t *SCT)
{
    // Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    // LocalLU_t *Llu = LUstruct->Llu;
    // int_t* xsup = Glu_persist->xsup;

    int iam = grid->iam;

    int pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);

    if (iam == pkk)
    {
        Wait_UDiagBlockSend(U_diag_blk_send_req, grid, SCT);
        Wait_LDiagBlockSend(L_diag_blk_send_req, grid, SCT);
    }

    return 0;
}



int_t LDiagBlockRecvWait( int_t k,   int_t* factored_U,
                          MPI_Request * L_diag_blk_recv_req,
                          gridinfo_t *grid)
{
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t krow = PROW (k, grid);
    
      /*factor the U panel*/
    if (myrow == krow && factored_U[k] == 0 && iam != pkk)
    {
        factored_U[k] = 1;
        MPI_Status status;
        MPI_Wait(L_diag_blk_recv_req, &status);
    }
    return 0;
}

