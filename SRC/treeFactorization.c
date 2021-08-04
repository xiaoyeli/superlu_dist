/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief factorization routines in 3D algorithms
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Oak Ridge National Lab
 * May 12, 2021
 * </pre>
 */

 #include "superlu_ddefs.h"
 #if 0
 #include "treeFactorization.h"
 #include "trfCommWrapper.h"
 #endif

 #if 0 /******** Sherry: Remove extra layer of function calls.  *******/
 int_t sDiagFactIBCast(int_t k,  diagFactBufs_t *dFBuf,
		       factStat_t *factStat,
		       commRequests_t *comReqs,
		       gridinfo_t *grid,
		       superlu_dist_options_t *options,
		       double thresh,
		       LUstruct_t *LUstruct,
		       SuperLUStat_t *stat, int *info,
		       SCT_t *SCT,
		       int tag_ub
		      )
 {
     MPI_Request * U_diag_blk_recv_req  = comReqs->U_diag_blk_recv_req;
     MPI_Request * L_diag_blk_recv_req  = comReqs->L_diag_blk_recv_req;
     MPI_Request * U_diag_blk_send_req  = comReqs->U_diag_blk_send_req;
     MPI_Request * L_diag_blk_send_req  = comReqs->L_diag_blk_send_req;
     int_t * IrecvPlcd_D = factStat->IrecvPlcd_D;

     double * BlockUFactor =  dFBuf->BlockUFactor;
     double * BlockLFactor =  dFBuf->BlockLFactor;
     dDiagFactIBCast(k,  k,   BlockUFactor, BlockLFactor,
		    IrecvPlcd_D,
		    U_diag_blk_recv_req, L_diag_blk_recv_req,
		    U_diag_blk_send_req, L_diag_blk_send_req,
		    grid, options, thresh, LUstruct, stat, info, SCT, tag_ub);
     return 0;
 }
 int_t sLPanelUpdate( int_t k,  diagFactBufs_t *dFBuf,
		      factStat_t *factStat,
		      commRequests_t *comReqs,
		      gridinfo_t *grid,
		      LUstruct_t *LUstruct, SCT_t *SCT)
 {
     MPI_Request * U_diag_blk_recv_req  = comReqs->U_diag_blk_recv_req;
     int_t * IrecvPlcd_D = factStat->IrecvPlcd_D;
     int_t * factored_L = factStat->factored_L;
     double * BlockUFactor =  dFBuf->BlockUFactor;

     dLPanelUpdate( k,  IrecvPlcd_D, factored_L,
		    U_diag_blk_recv_req, BlockUFactor, grid, LUstruct, SCT);
     return 0;
 }

 int_t sUPanelUpdate( int_t k,
		      int_t ldt,
		      diagFactBufs_t *dFBuf,
		      factStat_t *factStat,
		      commRequests_t *comReqs,
		      scuBufs_t* scuBufs,
		      packLUInfo_t* packLUInfo,
		      gridinfo_t *grid,
		      LUstruct_t *LUstruct,
		      SuperLUStat_t *stat, SCT_t *SCT)
 {
     double* bigV = scuBufs->bigV;
     Ublock_info_t* Ublock_info = packLUInfo->Ublock_info;

     MPI_Request * L_diag_blk_recv_req  = comReqs->L_diag_blk_recv_req;

     int_t * factored_U = factStat->factored_U;

     double * BlockLFactor =  dFBuf->BlockLFactor;
     dUPanelUpdate(k, factored_U, L_diag_blk_recv_req, BlockLFactor, bigV, ldt,
		   Ublock_info, grid, LUstruct, stat, SCT);
     return 0;
 }
 int_t sIBcastRecvLPanel(
     int_t k,
     commRequests_t *comReqs,
     LUValSubBuf_t* LUvsb,
     msgs_t* msgs,
     factStat_t *factStat,
     gridinfo_t *grid,
     LUstruct_t *LUstruct, SCT_t *SCT, int tag_ub)

 {
     int* msgcnt  = msgs->msgcnt;
    MPI_Request *send_req  = comReqs->send_req;
    MPI_Request *recv_req  = comReqs->recv_req;
    int_t * Lsub_buf  = LUvsb->Lsub_buf;
    double * Lval_buf  = LUvsb->Lval_buf;
    int_t* factored = factStat->factored;
    dIBcastRecvLPanel(k, k,
		      msgcnt, send_req, recv_req,
		      Lsub_buf, Lval_buf, factored, grid, LUstruct, SCT, tag_ub);
    return 0;
}

int_t sIBcastRecvUPanel(
    int_t k,
    commRequests_t *comReqs,
    LUValSubBuf_t* LUvsb,
    msgs_t* msgs,
    factStat_t *factStat,
    gridinfo_t *grid,
    LUstruct_t *LUstruct, SCT_t *SCT, int tag_ub)
{
    int* msgcnt  = msgs->msgcnt;
    MPI_Request *send_requ  = comReqs->send_requ;
    MPI_Request *recv_requ  = comReqs->recv_requ;
    int_t * Usub_buf  = LUvsb->Usub_buf;
    double * Uval_buf  = LUvsb->Uval_buf;
    dIBcastRecvUPanel(k, k, msgcnt, send_requ, recv_requ, Usub_buf,
		      Uval_buf, grid, LUstruct, SCT, tag_ub);
    return 0;
}
int_t sWaitL(int_t k,
             commRequests_t *comReqs,
             msgs_t* msgs,
             gridinfo_t *grid,
             LUstruct_t *LUstruct, SCT_t *SCT)
{
    int* msgcnt  = msgs->msgcnt;
    int* msgcntU  = msgs->msgcntU;
    MPI_Request *send_req  = comReqs->send_req;
    MPI_Request *recv_req  = comReqs->recv_req;
    dWaitL(k, msgcnt, msgcntU,  send_req, recv_req, grid, LUstruct, SCT);
    return 0;
}
int_t sWaitU(int_t k,
             commRequests_t *comReqs,
             msgs_t* msgs,
             gridinfo_t *grid,
             LUstruct_t *LUstruct, SCT_t *SCT)
{
    int* msgcnt  = msgs->msgcnt;
    MPI_Request *send_requ  = comReqs->send_requ;
    MPI_Request *recv_requ  = comReqs->recv_requ;
    dWaitU(k, msgcnt, send_requ, recv_requ, grid, LUstruct, SCT);
    return 0;
}
int_t sWait_LUDiagSend(int_t k,  commRequests_t *comReqs,
                       gridinfo_t *grid, SCT_t *SCT)
{
    MPI_Request *U_diag_blk_send_req  = comReqs->U_diag_blk_send_req;
    MPI_Request *L_diag_blk_send_req  = comReqs->L_diag_blk_send_req;
    Wait_LUDiagSend(k, U_diag_blk_send_req, L_diag_blk_send_req,  grid, SCT);
    return 0;
}
int_t sSchurComplementSetup(int_t k, msgs_t* msgs,
                            packLUInfo_t* packLUInfo,
                            int_t* gIperm_c_supno, int_t*perm_c_supno,
                            factNodelists_t* fNlists,
                            scuBufs_t* scuBufs, LUValSubBuf_t* LUvsb,
                            gridinfo_t *grid, LUstruct_t *LUstruct)
{
    int_t * Lsub_buf  = LUvsb->Lsub_buf;
    double * Lval_buf  = LUvsb->Lval_buf;
    int_t * Usub_buf  = LUvsb->Usub_buf;
    double * Uval_buf  = LUvsb->Uval_buf;
    Ublock_info_t* Ublock_info = packLUInfo->Ublock_info;
    Remain_info_t*  Remain_info = packLUInfo->Remain_info;
    uPanelInfo_t* uPanelInfo = packLUInfo->uPanelInfo;
    lPanelInfo_t* lPanelInfo = packLUInfo->lPanelInfo;
    int* msgcnt  = msgs->msgcnt;
    int_t* iperm_u  = fNlists->iperm_u;
    int_t* perm_u  = fNlists->perm_u;

    double* bigU = scuBufs->bigU;
    return dSchurComplementSetup(k, msgcnt,
				 Ublock_info, Remain_info, uPanelInfo, lPanelInfo,
				 gIperm_c_supno, iperm_u, perm_u,
				 bigU, Lsub_buf, Lval_buf, Usub_buf, Uval_buf,
				 grid, LUstruct);
}
int_t sLPanelTrSolve( int_t k,  diagFactBufs_t *dFBuf,
                      factStat_t *factStat,
                      commRequests_t *comReqs,
                      gridinfo_t *grid,
                      LUstruct_t *LUstruct, SCT_t *SCT)
{
    int_t * factored_L = factStat->factored_L;
    double * BlockUFactor =  dFBuf->BlockUFactor;
    dLPanelTrSolve( k, factored_L, BlockUFactor, grid, LUstruct);

    return 0;
}

int_t sUPanelTrSolve( int_t k,
                      int_t ldt,
                      diagFactBufs_t *dFBuf,
                      scuBufs_t* scuBufs,
                      packLUInfo_t* packLUInfo,
                      gridinfo_t *grid,
                      LUstruct_t *LUstruct,
                      SuperLUStat_t *stat, SCT_t *SCT)
{
    double* bigV = scuBufs->bigV;
    Ublock_info_t* Ublock_info = packLUInfo->Ublock_info;
    double * BlockLFactor =  dFBuf->BlockLFactor;

    dUPanelTrSolve( k, BlockLFactor, bigV, ldt, Ublock_info, grid, LUstruct, stat, SCT);
    return 0;
}

#endif /******** End removing extra layer of function calls.  *******/


int_t initCommRequests(commRequests_t* comReqs, gridinfo_t * grid)
{
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    // allocating MPI requests  (for one)
    comReqs->U_diag_blk_recv_req = MPI_REQ_ALLOC( 1 );
    comReqs->L_diag_blk_recv_req = MPI_REQ_ALLOC( 1 );
    comReqs->U_diag_blk_send_req = MPI_REQ_ALLOC( Pr );
    comReqs->L_diag_blk_send_req = MPI_REQ_ALLOC( Pc );
    comReqs->send_req = MPI_REQ_ALLOC(2 * Pc);
    comReqs->recv_req = MPI_REQ_ALLOC(4);
    comReqs->send_requ = MPI_REQ_ALLOC(2 * Pr);
    comReqs->recv_requ =  MPI_REQ_ALLOC(2);
    return 0;
}

commRequests_t** initCommRequestsArr(int_t mxLeafNode, int_t ldt, gridinfo_t* grid)
{
    commRequests_t** comReqss;
    comReqss = (commRequests_t** ) SUPERLU_MALLOC(mxLeafNode * sizeof(commRequests_t*));
    for (int i = 0; i < mxLeafNode; ++i)
	{
	    /* code */
	    comReqss[i] = (commRequests_t* ) SUPERLU_MALLOC(sizeof(commRequests_t));
	    initCommRequests(comReqss[i], grid);
	}/*Minor for loop -2 for (int i = 0; i < mxLeafNode; ++i)*/
    return comReqss;
}

// sherry added
int freeCommRequestsArr(int_t mxLeafNode, commRequests_t** comReqss)
{
    for (int i = 0; i < mxLeafNode; ++i) {
	SUPERLU_FREE(comReqss[i]->U_diag_blk_recv_req);
	SUPERLU_FREE(comReqss[i]->L_diag_blk_recv_req);
	SUPERLU_FREE(comReqss[i]->U_diag_blk_send_req);
	SUPERLU_FREE(comReqss[i]->L_diag_blk_send_req);
	SUPERLU_FREE(comReqss[i]->send_req);
	SUPERLU_FREE(comReqss[i]->recv_req);
	SUPERLU_FREE(comReqss[i]->send_requ);
	SUPERLU_FREE(comReqss[i]->recv_requ);
	SUPERLU_FREE(comReqss[i]);
    }
    SUPERLU_FREE(comReqss);
    return 0;
}

int_t initFactStat(int_t nsupers, factStat_t* factStat)
{
    factStat->IrecvPlcd_D = intMalloc_dist( nsupers);
    factStat->factored_D = intMalloc_dist( nsupers); //INT_T_ALLOC( nsupers);
    factStat->factored_L = intMalloc_dist( nsupers); //INT_T_ALLOC( nsupers);
    factStat->factored_U = intMalloc_dist( nsupers); //INT_T_ALLOC( nsupers);
    factStat->factored = intMalloc_dist( nsupers);   //INT_T_ALLOC( nsupers);
    factStat->IbcastPanel_L = intMalloc_dist(nsupers); //INT_T_ALLOC(nsupers);
    factStat->IbcastPanel_U = intMalloc_dist(nsupers); //INT_T_ALLOC(nsupers);
    factStat->gpuLUreduced = intMalloc_dist(nsupers); //INT_T_ALLOC(nsupers);

    for (int_t i = 0; i < nsupers; ++i)
    {
        /* code */
        factStat->IrecvPlcd_D[i] = 0;
        factStat->factored_D[i] = 0;
        factStat->factored_L[i] = 0;
        factStat->factored_U[i] = 0;
        factStat->IbcastPanel_L[i] = 0;
        factStat->IbcastPanel_U[i] = 0;
        factStat->gpuLUreduced[i] = 0;
    }
    return 0;
}

int freeFactStat(factStat_t* factStat)
{
    SUPERLU_FREE(factStat->IrecvPlcd_D);
    SUPERLU_FREE(factStat->factored_D);
    SUPERLU_FREE(factStat->factored_L);
    SUPERLU_FREE(factStat->factored_U);
    SUPERLU_FREE(factStat->factored);
    SUPERLU_FREE(factStat->IbcastPanel_L);
    SUPERLU_FREE(factStat->IbcastPanel_U);
    SUPERLU_FREE(factStat->gpuLUreduced);
    return 0;
}

int_t initFactNodelists(int_t ldt, int_t num_threads, int_t nsupers,
			factNodelists_t* fNlists)
{
    fNlists->iperm_u = INT_T_ALLOC(nsupers);
    fNlists->perm_u = INT_T_ALLOC(nsupers);
#if 0 // Sherry: change to int type
    fNlists->indirect = INT_T_ALLOC(num_threads * ldt);
    fNlists->indirect2 = INT_T_ALLOC(num_threads * ldt);
#else
    fNlists->indirect = (int*) SUPERLU_MALLOC(num_threads * ldt * sizeof(int));
    fNlists->indirect2 = (int*) SUPERLU_MALLOC(num_threads * ldt * sizeof(int));
#endif    
    return 0;
}

int freeFactNodelists(factNodelists_t* fNlists)
{
    SUPERLU_FREE(fNlists->iperm_u);
    SUPERLU_FREE(fNlists->perm_u);
    SUPERLU_FREE(fNlists->indirect);
    SUPERLU_FREE(fNlists->indirect2);
    return 0;
}

int_t initMsgs(msgs_t* msgs)
{
    msgs->msgcnt = (int *) SUPERLU_MALLOC(4 * sizeof(int));
    msgs->msgcntU = (int *) SUPERLU_MALLOC(4 * sizeof(int));
    return 0;
}

msgs_t** initMsgsArr(int_t numLA)
{
    msgs_t**msgss = (msgs_t**) SUPERLU_MALLOC(numLA * sizeof(msgs_t*));
    for (int_t i = 0; i < numLA; ++i)
	{
	    /* code */
	    msgss[i] = (msgs_t*) SUPERLU_MALLOC(sizeof(msgs_t));
	    initMsgs(msgss[i]);
	} /*minor for loop-3 for (int i = 0; i < numLA; ++i)*/
    return msgss;
}

// sherry added
int freeMsgsArr(int_t numLA, msgs_t **msgss)
{
    for (int i = 0; i < numLA; ++i) {
        SUPERLU_FREE(msgss[i]->msgcnt);
        SUPERLU_FREE(msgss[i]->msgcntU);
	SUPERLU_FREE(msgss[i]);
    }
    SUPERLU_FREE(msgss);
    return 0;
}

int_t initPackLUInfo(int_t nsupers, packLUInfo_t* packLUInfo)
{
    packLUInfo->Ublock_info =  (Ublock_info_t*) SUPERLU_MALLOC (sizeof(Ublock_info_t) * nsupers);
    packLUInfo->Remain_info = (Remain_info_t* ) SUPERLU_MALLOC(sizeof(Remain_info_t) * nsupers);
    packLUInfo->uPanelInfo = (uPanelInfo_t* ) SUPERLU_MALLOC(sizeof(uPanelInfo_t));
    packLUInfo->lPanelInfo = (lPanelInfo_t*) SUPERLU_MALLOC(sizeof(lPanelInfo_t));
    return 0;
}

int freePackLUInfo(packLUInfo_t* packLUInfo)  // sherry added 
{
    SUPERLU_FREE(packLUInfo->Ublock_info);
    SUPERLU_FREE(packLUInfo->Remain_info);
    SUPERLU_FREE(packLUInfo->uPanelInfo);
    SUPERLU_FREE(packLUInfo->lPanelInfo);
    return 0;
}

int_t getNumLookAhead(superlu_dist_options_t *options)
{
    int_t numLA;
    if (getenv("NLULA"))
    {
        numLA = atoi(getenv("NLULA"));
    }
    else
    {
        // printf("NLULA not set using default 2\n");
	// numLA = 2;
	numLA = options->num_lookaheads;
    }
    return numLA;
}

int_t checkRecvUDiag(int_t k, commRequests_t *comReqs,
                     gridinfo_t *grid, SCT_t *SCT)
{

    MPI_Request * U_diag_blk_recv_req  = comReqs->U_diag_blk_recv_req;
    int_t iam = grid->iam;

    int_t mycol = MYCOL (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);

    int_t kcol = PCOL (k, grid);

    if (mycol == kcol  && iam != pkk)
    {
        int_t flag = Test_UDiagBlock_Recv( U_diag_blk_recv_req, SCT);
        return flag;
    }

    return 1;
}

int_t checkRecvLDiag(int_t k,
                     commRequests_t *comReqs,
                     gridinfo_t *grid,
                     SCT_t *SCT)
{
    MPI_Request * L_diag_blk_recv_req  = comReqs->L_diag_blk_recv_req;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);

    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t krow = PROW (k, grid);

    /*factor the U panel*/
    if (myrow == krow  && iam != pkk)
    {
        int_t flag = 0;

        flag = Test_LDiagBlock_Recv( L_diag_blk_recv_req , SCT);

        return flag;
    }
    return 1;
}


