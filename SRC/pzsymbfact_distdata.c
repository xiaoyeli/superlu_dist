/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Redistribute the symbolic structure of L and U from the distribution
 *
 * <pre>
 * -- Parallel symbolic factorization auxialiary routine (version 2.3) --
 * -- Distributes the data from parallel symbolic factorization
 * -- to numeric factorization
 * INRIA France -  July 1, 2004
 * Laura Grigori
 *
 * November 1, 2007
 * Feburary 20, 2008
 * October 15, 2008
 * </pre>
 */

/* limits.h:  the largest positive integer (INT_MAX) */
#include <limits.h>

#include "superlu_zdefs.h"
#include "psymbfact.h"


/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * Redistribute the symbolic structure of L and U from the distribution
 * used in the parallel symbolic factorization step to the distdibution
 * used in the parallel numeric factorization step.  On exit, the L and U
 * structure for the 2D distribution used in the numeric factorization step is
 * stored in p_xlsub, p_lsub, p_xusub, p_usub.  The global supernodal
 * information is also computed and it is stored in Glu_persist->supno
 * and Glu_persist->xsup.
 *
 * This routine allocates memory for storing the structure of L and U
 * and the supernodes information.  This represents the arrays:
 * p_xlsub, p_lsub, p_xusub, p_usub,
 * Glu_persist->supno,  Glu_persist->xsup.
 *
 * This routine also deallocates memory allocated during symbolic
 * factorization routine.  That is, the folloing arrays are freed:
 * Pslu_freeable->xlsub,  Pslu_freeable->lsub,
 * Pslu_freeable->xusub, Pslu_freeable->usub,
 * Pslu_freeable->globToLoc, Pslu_freeable->supno_loc,
 * Pslu_freeable->xsup_beg_loc, Pslu_freeable->xsup_end_loc.
 *
 * Arguments
 * =========
 *
 * n      (Input) int_t
 *        Order of the input matrix
 * Pslu_freeable  (Input) Pslu_freeable_t *
 *        Local L and U structure,
 *        global to local indexing information.
 *
 * Glu_persist (Output) Glu_persist_t *
 *        Stores on output the information on supernodes mapping.
 *
 * p_xlsub (Output) int_t **
 *         Pointer to structure of L distributed on a 2D grid
 *         of processors, stored by columns.
 *
 * p_lsub  (Output) int_t **
 *         Structure of L distributed on a 2D grid of processors,
 *         stored by columns.
 *
 * p_xusub (Output) int_t **
 *         Pointer to structure of U distributed on a 2D grid
 *         of processors, stored by rows.
 *
 * p_usub  (Output) int_t **
 *         Structure of U distributed on a 2D grid of processors,
 *         stored by rows.
 *
 * grid   (Input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Return value
 * ============
 *   < 0, number of bytes allocated on return from the dist_symbLU.
 *   > 0, number of bytes allocated in this routine when out of memory.
 *        (an approximation).
 * </pre>
 */

static float
dist_symbLU (superlu_dist_options_t *options, int_t n,
             Pslu_freeable_t *Pslu_freeable, Glu_persist_t *Glu_persist,
	     int_t **p_xlsub, int_t **p_lsub, int_t **p_xusub, int_t **p_usub,
	     gridinfo_t *grid
	     )
{
  int   iam, nprocs, pc, pr, p, np, p_diag;
  int_t *nnzToSend, *nnzToRecv, *nnzToSend_l, *nnzToSend_u,
    *tmp_ptrToSend, *mem;
  int_t *nnzToRecv_l, *nnzToRecv_u;
  int_t *send_1, *send_2, nsend_1, nsend_2;
  int_t *ptrToSend, *ptrToRecv, sendL, sendU, *snd_luind, *rcv_luind;
  int_t nsupers, nsupers_i, nsupers_j;
  int *nvtcs, *intBuf1, *intBuf2, *intBuf3, *intBuf4, intNvtcs_loc;
  int_t maxszsn, maxNvtcsPProc;
  int_t *xsup_n, *supno_n, *temp, *xsup_beg_s, *xsup_end_s, *supno_s;
  int_t *xlsub_s, *lsub_s, *xusub_s, *usub_s;
  int_t *xlsub_n, *lsub_n, *xusub_n, *usub_n;
  int_t *xsub_s, *sub_s, *xsub_n, *sub_n;
  int_t *globToLoc, nvtcs_loc;
  int_t SendCnt_l, SendCnt_u, nnz_loc_l, nnz_loc_u, nnz_loc,
    RecvCnt_l, RecvCnt_u, ind_loc;
  int_t i, k, j, gb, szsn, gb_n, gb_s, gb_l, fst_s, fst_s_l, lst_s, i_loc;
  int_t nelts, isize;
  float memAux;  /* Memory used during this routine and freed on return */
  float memRet; /* Memory allocated and not freed on return */
  int_t iword, dword;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  iam = grid->iam;
#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Enter dist_symbLU()");
#endif
  nprocs = (int) grid->nprow * grid->npcol;
  xlsub_s = Pslu_freeable->xlsub; lsub_s = Pslu_freeable->lsub;
  xusub_s = Pslu_freeable->xusub; usub_s = Pslu_freeable->usub;
  maxNvtcsPProc = Pslu_freeable->maxNvtcsPProc;
  globToLoc     = Pslu_freeable->globToLoc;
  nvtcs_loc     = Pslu_freeable->nvtcs_loc;
  xsup_beg_s    = Pslu_freeable->xsup_beg_loc;
  xsup_end_s    = Pslu_freeable->xsup_end_loc;
  supno_s       = Pslu_freeable->supno_loc;
  rcv_luind     = NULL;
  iword = sizeof(int_t);
  dword = sizeof(doublecomplex);
  memAux = 0.; memRet = 0.;

  mem           = intCalloc_dist(12 * nprocs);
  if (!mem)
    return (ERROR_RET);
  memAux     = (float) (12 * nprocs * sizeof(int_t));
  nnzToRecv     = mem;
  nnzToSend     = nnzToRecv + 2*nprocs;
  nnzToSend_l   = nnzToSend + 2 * nprocs;
  nnzToSend_u   = nnzToSend_l + nprocs;
  send_1        = nnzToSend_u + nprocs;
  send_2        = send_1 + nprocs;
  tmp_ptrToSend = send_2 + nprocs;
  nnzToRecv_l   = tmp_ptrToSend + nprocs;
  nnzToRecv_u   = nnzToRecv_l + nprocs;

  ptrToSend = nnzToSend;
  ptrToRecv = nnzToSend + nprocs;

  nvtcs = (int *) SUPERLU_MALLOC(5 * nprocs * sizeof(int));
  intBuf1 = nvtcs + nprocs;
  intBuf2 = nvtcs + 2 * nprocs;
  intBuf3 = nvtcs + 3 * nprocs;
  intBuf4 = nvtcs + 4 * nprocs;
  memAux += 5 * nprocs * sizeof(int);

  maxszsn   = sp_ienv_dist(3, options);

  /* Allocate space for storing Glu_persist_n. */
  if ( !(supno_n = intMalloc_dist(n+1)) ) {
    fprintf (stderr, "Malloc fails for supno_n[].");
    return (memAux);
  }
  memRet += (float) ((n+1) * sizeof(int_t));

  /* ------------------------------------------------------------
     DETERMINE SUPERNODES FOR NUMERICAL FACTORIZATION
     ------------------------------------------------------------*/

  if (nvtcs_loc > INT_MAX)
    ABORT("ERROR in dist_symbLU nvtcs_loc > INT_MAX\n");
  intNvtcs_loc = (int) nvtcs_loc;
  MPI_Gather (&intNvtcs_loc, 1, MPI_INT, nvtcs, 1, MPI_INT,
	      0, grid->comm);

  if (!iam) {
    /* set ptrToRecv to point to the beginning of the data for
       each processor */
    for (k = 0, p = 0; p < nprocs; p++) {
      ptrToRecv[p] = k;
      k += nvtcs[p];
    }
  }

  if (nprocs > 1) {
    temp = NULL;
    if (!iam ) {
      if ( !(temp = intMalloc_dist (n+1)) ) {
	fprintf (stderr, "Malloc fails for temp[].");
	return (memAux + memRet);
      }
      memAux += (float) (n+1) * iword;
    }
#if defined (_LONGINT)
    for (p=0; p<nprocs; p++) {
      if (ptrToRecv[p] > INT_MAX)
	ABORT("ERROR in dist_symbLU size to send > INT_MAX\n");
      intBuf1[p] = (int) ptrToRecv[p];
    }
#else  /* Default */
    intBuf1 = ptrToRecv;
#endif
    MPI_Gatherv (supno_s, (int) nvtcs_loc, mpi_int_t,
		 temp, nvtcs, intBuf1, mpi_int_t, 0, grid->comm);
  }
  else
    temp = supno_s;

  if (!iam) {
    nsupers = 0;
    p = (int) OWNER( globToLoc[0] );
    gb = temp[ptrToRecv[p]];
    supno_n[0] = nsupers;
    ptrToRecv[p] ++;
    szsn = 1;
    for (j = 1; j < n; j ++) {
      if (p != (int) OWNER( globToLoc[j] ) || szsn >= maxszsn || gb != temp[ptrToRecv[p]]) {
	nsupers ++;
	p  = (int) OWNER( globToLoc[j] );
	gb = temp[ptrToRecv[p]];
	szsn = 1;
      }
      else {
	szsn ++;
      }
      ptrToRecv[p] ++;
      supno_n[j] = nsupers;
    }
    nsupers++;
    if (nprocs > 1) {
      SUPERLU_FREE (temp);
      memAux -= (float) (n+1) * iword;
    }
    supno_n[n] = nsupers;
  }

  /* reset to 0 nnzToSend */
  for (p = 0; p < 2 *nprocs; p++)
    nnzToSend[p] = 0;

  MPI_Bcast (supno_n, n+1, mpi_int_t, 0, grid->comm);
  nsupers = supno_n[n];
  /* Allocate space for storing Glu_persist_n. */
  if ( !(xsup_n = intMalloc_dist(nsupers+1)) ) {
    fprintf (stderr, "Malloc fails for xsup_n[].");
    return (memAux + memRet);
  }
  memRet += (float) (nsupers+1) * iword;

  /* ------------------------------------------------------------
     COUNT THE NUMBER OF NONZEROS TO BE SENT TO EACH PROCESS,
     THEN ALLOCATE SPACE.
     THIS ACCOUNTS FOR THE FIRST PASS OF L and U.
     ------------------------------------------------------------*/
  gb = EMPTY;
  for (i = 0; i < n; i++) {
    if (gb != supno_n[i]) {
      /* a new supernode starts */
      gb = supno_n[i];
      xsup_n[gb] = i;
    }
  }
  xsup_n[nsupers] = n;

  for (p = 0; p < nprocs; p++) {
    send_1[p] = FALSE;
    send_2[p] = FALSE;
  }
  for (gb_n = 0; gb_n < nsupers; gb_n ++) {
    i = xsup_n[gb_n];
    if (iam == (int) OWNER( globToLoc[i] )) {
      pc = PCOL( gb_n, grid );
      pr = PROW( gb_n, grid );
      p_diag = PNUM( pr, pc, grid);

      i_loc = LOCAL_IND( globToLoc[i] );
      gb_s  = supno_s[i_loc];
      fst_s = xsup_beg_s[gb_s];
      lst_s = xsup_end_s[gb_s];
      fst_s_l = LOCAL_IND( globToLoc[fst_s] );
      for (j = xlsub_s[fst_s_l]; j < xlsub_s[fst_s_l+1]; j++) {
	k = lsub_s[j];
	if (k >= i) {
	  gb = supno_n[k];
	  p = (int) PNUM( PROW(gb, grid), pc, grid );
	  nnzToSend[2*p] ++;
	  send_1[p] = TRUE;
	}
      }
      for (j = xusub_s[fst_s_l]; j < xusub_s[fst_s_l+1]; j++) {
	k = usub_s[j];
	if (k >= i + xsup_n[gb_n+1] - xsup_n[gb_n]) {
	  gb = supno_n[k];
	  p = PNUM( pr, PCOL(gb, grid), grid);
	  nnzToSend[2*p+1] ++;
	  send_2[p] = TRUE;
	}
      }

      nsend_2 = 0;
      for (p = pr * grid->npcol; p < (pr + 1) * grid->npcol; p++) {
	nnzToSend[2*p+1] += 2;
	if (send_2[p])  nsend_2 ++;
      }
      for (p = pr * grid->npcol; p < (pr + 1) * grid->npcol; p++)
	if (send_2[p] || p == p_diag) {
	  if (p == p_diag && !send_2[p])
	    nnzToSend[2*p+1] += nsend_2;
	  else
	    nnzToSend[2*p+1] += nsend_2-1;
	  send_2[p] = FALSE;
	}
      nsend_1 = 0;
      for (p = pc; p < nprocs; p += grid->npcol) {
	nnzToSend[2*p] += 2;
	if (send_1[p]) nsend_1 ++;
      }
      for (p = pc; p < nprocs; p += grid->npcol)
	if (send_1[p]) {
	  nnzToSend[2*p] += nsend_1-1;
	  send_1[p] = FALSE;
	}
	else
	  nnzToSend[2*p] += nsend_1;
    }
  }

  /* All-to-all communication */
  MPI_Alltoall( nnzToSend, 2, mpi_int_t, nnzToRecv, 2, mpi_int_t,
		grid->comm);

  nnz_loc_l = nnz_loc_u = 0;
  SendCnt_l = SendCnt_u = RecvCnt_l = RecvCnt_u = 0;
  for (p = 0; p < nprocs; p++) {
    if ( p != iam ) {
      SendCnt_l += nnzToSend[2*p];   nnzToSend_l[p] = nnzToSend[2*p];
      SendCnt_u += nnzToSend[2*p+1]; nnzToSend_u[p] = nnzToSend[2*p+1];
      RecvCnt_l += nnzToRecv[2*p];   nnzToRecv_l[p] = nnzToRecv[2*p];
      RecvCnt_u += nnzToRecv[2*p+1]; nnzToRecv_u[p] = nnzToRecv[2*p+1];
    } else {
      nnz_loc_l += nnzToRecv[2*p];
      nnz_loc_u += nnzToRecv[2*p+1];
      nnzToSend_l[p] = 0; nnzToSend_u[p] = 0;
      nnzToRecv_l[p] = nnzToRecv[2*p];
      nnzToRecv_u[p] = nnzToRecv[2*p+1];
    }
  }

  /* Allocate space for storing the symbolic structure after redistribution. */
  nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
  nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */
  if ( !(xlsub_n = intCalloc_dist(nsupers_j+1)) ) {
    fprintf (stderr, "Malloc fails for xlsub_n[].");
    return (memAux + memRet);
  }
  memRet += (float) (nsupers_j+1) * iword;

  if ( !(xusub_n = intCalloc_dist(nsupers_i+1)) ) {
    fprintf (stderr, "Malloc fails for xusub_n[].");
    return (memAux + memRet);
  }
  memRet += (float) (nsupers_i+1) * iword;

  /* Allocate temp storage for sending/receiving the L/U symbolic structure. */
  if ( (RecvCnt_l + nnz_loc_l) || (RecvCnt_u + nnz_loc_u) ) {
    if (!(rcv_luind =
	  intMalloc_dist(SUPERLU_MAX(RecvCnt_l+nnz_loc_l, RecvCnt_u+nnz_loc_u))) ) {
      fprintf (stderr, "Malloc fails for rcv_luind[].");
      return (memAux + memRet);
    }
    memAux += (float) SUPERLU_MAX(RecvCnt_l+nnz_loc_l, RecvCnt_u+nnz_loc_u)
      * iword;
  }
  if ( nprocs > 1 && (SendCnt_l || SendCnt_u) ) {
    if (!(snd_luind = intMalloc_dist(SUPERLU_MAX(SendCnt_l, SendCnt_u))) ) {
      fprintf (stderr, "Malloc fails for index[].");
      return (memAux + memRet);
    }
    memAux += (float) SUPERLU_MAX(SendCnt_l, SendCnt_u) * iword;
  }

  /* ------------------------------------------------------------------
     LOAD THE SYMBOLIC STRUCTURE OF L AND U INTO THE STRUCTURES TO SEND.
     THIS ACCOUNTS FOR THE SECOND PASS OF L and U.
     ------------------------------------------------------------------*/
  sendL = TRUE;
  sendU = FALSE;
  while (sendL || sendU) {
    if (sendL) {
      xsub_s = xlsub_s; sub_s = lsub_s; xsub_n = xlsub_n;
      nnzToSend = nnzToSend_l; nnzToRecv = nnzToRecv_l;
    }
    if (sendU) {
      xsub_s = xusub_s; sub_s = usub_s; xsub_n = xusub_n;
      nnzToSend = nnzToSend_u; nnzToRecv = nnzToRecv_u;
    }
    for (i = 0, j = 0, p = 0; p < nprocs; p++) {
      if ( p != iam ) {
	ptrToSend[p] = i;  i += nnzToSend[p];
      }
      ptrToRecv[p] = j;  j += nnzToRecv[p];
    }
    nnzToRecv[iam] = 0;

    ind_loc = ptrToRecv[iam];
    for (gb_n = 0; gb_n < nsupers; gb_n++) {
      nsend_2 = 0;
      i = xsup_n[gb_n];
      if (iam == OWNER( globToLoc[i] )) {
	pc = PCOL( gb_n, grid );
	pr = PROW( gb_n, grid );
	p_diag = PNUM( pr, pc, grid );

	i_loc = LOCAL_IND( globToLoc[i] );
	gb_s  = supno_s[i_loc];
	fst_s = xsup_beg_s[gb_s];
	lst_s = xsup_end_s[gb_s];
	fst_s_l = LOCAL_IND( globToLoc[fst_s] );

	if (sendL) {
	  p = pc;                np = grid->nprow;
	} else {
	  p = pr * grid->npcol;  np = grid->npcol;
	}
	for (j = 0; j < np; j++) {
	  if (p == iam) {
	    rcv_luind[ind_loc] = gb_n;
	    rcv_luind[ind_loc+1] = 0;
	    tmp_ptrToSend[p] = ind_loc + 1;
	    ind_loc += 2;
	  }
	  else {
	    snd_luind[ptrToSend[p]] = gb_n;
	    snd_luind[ptrToSend[p]+1] = 0;
	    tmp_ptrToSend[p] = ptrToSend[p] + 1;
	    ptrToSend[p] += 2;
	  }
	  if (sendL) p += grid->npcol;
	  if (sendU) p++;
	}
	for (j = xsub_s[fst_s_l]; j < xsub_s[fst_s_l+1]; j++) {
	  k = sub_s[j];
	  if ((sendL && k >= i) || (sendU && k >= i + xsup_n[gb_n+1] - xsup_n[gb_n])) {
	    gb = supno_n[k];
	    if (sendL)
	      p = PNUM( PROW(gb, grid), pc, grid );
	    else
	      p = PNUM( pr, PCOL(gb, grid), grid);
	    if (send_1[p] == FALSE) {
	      send_1[p] = TRUE;
	      send_2[nsend_2] = k; nsend_2 ++;
	    }
	    if (p == iam) {
	      rcv_luind[ind_loc] = k;  ind_loc++;
	      if (sendL)
		xsub_n[LBj( gb_n, grid )] ++;
	      else
		xsub_n[LBi( gb_n, grid )] ++;
	    }
	    else {
	      snd_luind[ptrToSend[p]] = k;
	      ptrToSend[p] ++; snd_luind[tmp_ptrToSend[p]] ++;
	    }
	  }
	}
	if (sendL)
	  for (p = pc; p < nprocs; p += grid->npcol) {
	      for (k = 0; k < nsend_2; k++) {
		gb = supno_n[send_2[k]];
		if (PNUM(PROW(gb, grid), pc, grid) != p) {
		  if (p == iam) {
		    rcv_luind[ind_loc] = send_2[k];  ind_loc++;
		    xsub_n[LBj( gb_n, grid )] ++;
		  }
		  else {
		    snd_luind[ptrToSend[p]] = send_2[k];
		    ptrToSend[p] ++; snd_luind[tmp_ptrToSend[p]] ++;
		  }
		}
	      }
	      send_1[p] = FALSE;
	  }
	if (sendU)
	  for (p = pr * grid->npcol; p < (pr + 1) * grid->npcol; p++) {
	    if (send_1[p] || p == p_diag) {
	      for (k = 0; k < nsend_2; k++) {
		gb = supno_n[send_2[k]];
		if(PNUM( pr, PCOL(gb, grid), grid) != p) {
		  if (p == iam) {
		    rcv_luind[ind_loc] = send_2[k];  ind_loc++;
		    xsub_n[LBi( gb_n, grid )] ++;
		  }
		  else {
		    snd_luind[ptrToSend[p]] = send_2[k];
		    ptrToSend[p] ++; snd_luind[tmp_ptrToSend[p]] ++;
		  }
		}
	      }
	      send_1[p] = FALSE;
	    }
	  }
      }
    }

    /* reset ptrToSnd to point to the beginning of the data for
       each processor (structure needed in MPI_Alltoallv) */
    for (i = 0, p = 0; p < nprocs; p++) {
      ptrToSend[p] = i;  i += nnzToSend[p];
    }

    /* ------------------------------------------------------------
       PERFORM REDISTRIBUTION. THIS INVOLVES ALL-TO-ALL COMMUNICATION.
       Note: it uses MPI_Alltoallv.
       ------------------------------------------------------------*/
    if (nprocs > 1) {
#if defined (_LONGINT)
      nnzToSend[iam] = 0;
      for (p=0; p<nprocs; p++) {
	if (nnzToSend[p] > INT_MAX || ptrToSend[p] > INT_MAX ||
	    nnzToRecv[p] > INT_MAX || ptrToRecv[p] > INT_MAX)
	  ABORT("ERROR in dist_symbLU size to send > INT_MAX\n");
	intBuf1[p] = (int) nnzToSend[p];
	intBuf2[p] = (int) ptrToSend[p];
	intBuf3[p] = (int) nnzToRecv[p];
	intBuf4[p] = (int) ptrToRecv[p];
      }
#else  /* Default */
      intBuf1 = nnzToSend;  intBuf2 = ptrToSend;
      intBuf3 = nnzToRecv;  intBuf4 = ptrToRecv;
#endif

      MPI_Alltoallv (snd_luind, intBuf1, intBuf2, mpi_int_t,
		     rcv_luind, intBuf3, intBuf4, mpi_int_t,
		     grid->comm);
    }
    if (sendL)
      nnzToRecv[iam] = nnz_loc_l;
    else
      nnzToRecv[iam] = nnz_loc_u;

    /* ------------------------------------------------------------
       DEALLOCATE TEMPORARY STORAGE.
       -------------------------------------------------------------*/
    if (sendU)
      if ( nprocs > 1 && (SendCnt_l || SendCnt_u) ) {
	SUPERLU_FREE (snd_luind);
	memAux -= (float) SUPERLU_MAX(SendCnt_l, SendCnt_u) * iword;
      }

    /* ------------------------------------------------------------
       CONVERT THE FORMAT.
       ------------------------------------------------------------*/
    /* Initialize the array of column of L/ row of U pointers */
    k = 0;
    for (p = 0; p < nprocs; p ++) {
      if (p != iam) {
	i = k;
	while (i < k + nnzToRecv[p]) {
	  gb = rcv_luind[i];
	  nelts = rcv_luind[i+1];
	  if (sendL)
	    xsub_n[LBj( gb, grid )] = nelts;
	  else
	    xsub_n[LBi( gb, grid )] = nelts;
	  i += nelts + 2;
	}
      }
      k += nnzToRecv[p];
    }

    if (sendL) j = nsupers_j;
    else j = nsupers_i;
    k = 0;
    isize = xsub_n[0];
    xsub_n[0] = 0;
    for (gb_l = 1; gb_l < j; gb_l++) {
      k += isize;
      isize = xsub_n[gb_l];
      xsub_n[gb_l] = k;
    }
    xsub_n[gb_l] = k + isize;
    nnz_loc = xsub_n[gb_l];
    if (sendL) {
      lsub_n = NULL;
      if (nnz_loc) {
	if ( !(lsub_n = intMalloc_dist(nnz_loc)) ) {
	  fprintf (stderr, "Malloc fails for lsub_n[].");
	  return (memAux + memRet);
	}
	memRet += (float) (nnz_loc * iword);
      }
      sub_n = lsub_n;
    }
    if (sendU) {
      usub_n = NULL;
      if (nnz_loc) {
	if ( !(usub_n = intMalloc_dist(nnz_loc)) ) {
	  fprintf (stderr, "Malloc fails for usub_n[].");
	  return (memAux + memRet);
	}
	memRet += (float) (nnz_loc * iword);
      }
      sub_n = usub_n;
    }

    /* Copy the data into the L column / U row oriented storage */
    k = 0;
    for (p = 0; p < nprocs; p++) {
      i = k;
      while (i < k + nnzToRecv[p]) {
	gb = rcv_luind[i];
	if (gb >= nsupers)
	  printf ("Pe[%d] p %d gb %d nsupers %d i " IFMT " i-k " IFMT "\n",
		  iam, p, (int) gb, (int) nsupers, i, i-k);
	i += 2;
	if (sendL) gb_l = LBj( gb, grid );
	if (sendU) gb_l = LBi( gb, grid );
	for (j = xsub_n[gb_l]; j < xsub_n[gb_l+1]; i++, j++) {
	  sub_n[j] = rcv_luind[i];
	}
      }
      k += nnzToRecv[p];
    }
    if (sendL) {
      sendL = FALSE;  sendU = TRUE;
    }
    else
      sendU = FALSE;
  }

  /* deallocate memory allocated during symbolic factorization routine */
  if (rcv_luind != NULL) {
    SUPERLU_FREE (rcv_luind);
    memAux -= (float) SUPERLU_MAX(RecvCnt_l+nnz_loc_l, RecvCnt_u+nnz_loc_u) * iword;
  }
  SUPERLU_FREE (mem);
  memAux -= (float) (12 * nprocs * iword);
  SUPERLU_FREE(nvtcs);
  memAux -= (float) (5 * nprocs * sizeof(int));

  if (xlsub_s != NULL) {
    SUPERLU_FREE (xlsub_s); SUPERLU_FREE (lsub_s);
  }
  if (xusub_s != NULL) {
    SUPERLU_FREE (xusub_s); SUPERLU_FREE (usub_s);
  }
  SUPERLU_FREE (globToLoc);
  if (supno_s != NULL) {
    SUPERLU_FREE (xsup_beg_s); SUPERLU_FREE (xsup_end_s);
    SUPERLU_FREE (supno_s);
  }

  Glu_persist->supno = supno_n;  Glu_persist->xsup  = xsup_n;
  *p_xlsub = xlsub_n; *p_lsub = lsub_n;
  *p_xusub = xusub_n; *p_usub = usub_n;

#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Exit dist_symbLU()");
#endif

  return (-memRet);
}

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Re-distribute A on the 2D process mesh.  The lower part is
 *   stored using a column format and the upper part
 *   is stored using a row format.
 *
 * Arguments
 * =========
 *
 * A      (Input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        The type of A can be: Stype = SLU_NR_loc; Dtype = SLU_Z; Mtype = SLU_GE.
 *
 * ScalePermstruct (Input) zScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_persist  (Input) Glu_persist_t *
 *        Information on supernodes mapping.
 *
 * grid   (Input) gridinfo_t*
 *        The 2D process mesh.
 *
 * p_ainf_colptr (Output) int_t**
 *         Pointer to the lower part of A distributed on a 2D grid
 *         of processors, stored by columns.
 *
 * p_ainf_rowind (Output) int_t**
 *         Structure of of the lower part of A distributed on a
 *         2D grid of processors, stored by columns.
 *
 * p_ainf_val    (Output) doublecomplex**
 *         Numerical values of the lower part of A, distributed on a
 *         2D grid of processors, stored by columns.
 *
 * p_asup_rowptr (Output) int_t**
 *         Pointer to the upper part of A distributed on a 2D grid
 *         of processors, stored by rows.
 *
 * p_asup_colind (Output) int_t**
 *         Structure of of the upper part of A distributed on a
 *         2D grid of processors, stored by rows.
 *
 * p_asup_val    (Output) doublecomplex**
 *         Numerical values of the upper part of A, distributed on a
 *         2D grid of processors, stored by rows.
 *
 * ilsum_i  (Input) int_t *
 *       Starting position of each supernode in
 *       the full array (local, block row wise).
 *
 * ilsum_j  (Input) int_t *
 *       Starting position of each supernode in
 *       the full array (local, block column wise).
 *
 * Return value
 * ============
 *   < 0, number of bytes allocated on return from the dist_symbLU
 *   > 0, number of bytes allocated when out of memory.
 *        (an approximation).
 * </pre>
 */

static float
zdist_A(SuperMatrix *A, zScalePermstruct_t *ScalePermstruct,
	Glu_persist_t *Glu_persist, gridinfo_t *grid,
	int_t **p_ainf_colptr, int_t **p_ainf_rowind, doublecomplex **p_ainf_val,
	int_t **p_asup_rowptr, int_t **p_asup_colind, doublecomplex **p_asup_val,
	int_t *ilsum_i, int_t *ilsum_j
	)
{
  int    iam, p, procs;
  NRformat_loc *Astore;
  int_t  *perm_r; /* row permutation vector */
  int_t  *perm_c; /* column permutation vector */
  int_t  i, it, irow, fst_row, j, jcol, k, gbi, gbj, n, m_loc, jsize, isize;
  int_t  nsupers, nsupers_i, nsupers_j;
  int_t  nnz_loc, nnz_loc_ainf, nnz_loc_asup;    /* number of local nonzeros */
  int_t  SendCnt; /* number of remote nonzeros to be sent */
  int_t  RecvCnt; /* number of remote nonzeros to be sent */
  int_t *ainf_colptr, *ainf_rowind, *asup_rowptr, *asup_colind;
  doublecomplex *asup_val, *ainf_val;
  int_t  *nnzToSend, *nnzToRecv, maxnnzToRecv;
  int_t  *ia, *ja, **ia_send, *index, *itemp;
  int_t  *ptr_to_send;
  doublecomplex *aij, **aij_send, *nzval, *dtemp;
  doublecomplex *nzval_a;
  MPI_Request *send_req;
  MPI_Status  status;
  int_t *xsup = Glu_persist->xsup;    /* supernode and column mapping */
  int_t *supno = Glu_persist->supno;
  float memAux;  /* Memory used during this routine and freed on return */
  float memRet; /* Memory allocated and not freed on return */
  int_t iword, dword, szbuf;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  iam = grid->iam;
#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Enter zdist_A()");
#endif
  iword = sizeof(int_t);
  dword = sizeof(double);

  perm_r = ScalePermstruct->perm_r;
  perm_c = ScalePermstruct->perm_c;
  procs = grid->nprow * grid->npcol;
  Astore = (NRformat_loc *) A->Store;
  n = A->ncol;
  m_loc = Astore->m_loc;
  fst_row = Astore->fst_row;
  if (!(nnzToRecv = intCalloc_dist(2*procs))) {
    fprintf (stderr, "Malloc fails for nnzToRecv[].");
    return (ERROR_RET);
  }
  memAux = (float) (2 * procs * iword);
  memRet = 0.;
  nnzToSend = nnzToRecv + procs;
  nsupers  = supno[n-1] + 1;

  /* ------------------------------------------------------------
     COUNT THE NUMBER OF NONZEROS TO BE SENT TO EACH PROCESS,
     THEN ALLOCATE SPACE.
     THIS ACCOUNTS FOR THE FIRST PASS OF A.
     ------------------------------------------------------------*/
  for (i = 0; i < m_loc; ++i) {
    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
      irow = perm_c[perm_r[i+fst_row]];  /* Row number in Pc*Pr*A */
      jcol = Astore->colind[j];
      gbi = BlockNum( irow );
      gbj = BlockNum( jcol );
      p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );
      ++nnzToSend[p];
    }
  }

  /* All-to-all communication */
  MPI_Alltoall( nnzToSend, 1, mpi_int_t, nnzToRecv, 1, mpi_int_t,
		grid->comm);

  maxnnzToRecv = 0;
  nnz_loc = SendCnt = RecvCnt = 0;

  for (p = 0; p < procs; ++p) {
    if ( p != iam ) {
      SendCnt += nnzToSend[p];
      RecvCnt += nnzToRecv[p];
      maxnnzToRecv = SUPERLU_MAX( nnzToRecv[p], maxnnzToRecv );
    } else {
      nnz_loc += nnzToRecv[p];
      /*assert(nnzToSend[p] == nnzToRecv[p]);*/
    }
  }
  k = nnz_loc + RecvCnt; /* Total nonzeros ended up in my process. */
  szbuf = k;

  /* Allocate space for storing the triplets after redistribution. */
  if ( !(ia = intMalloc_dist(2*k)) ) {
    fprintf (stderr, "Malloc fails for ia[].");
    return (memAux);
  }
  memAux += (float) (2*k*iword);
  ja = ia + k;
  if ( !(aij = doublecomplexMalloc_dist(k)) ) {
    fprintf (stderr, "Malloc fails for aij[].");
    return (memAux);
  }
  memAux += (float) (k*dword);

  /* Allocate temporary storage for sending/receiving the A triplets. */
  if ( procs > 1 ) {
    if ( !(send_req = (MPI_Request *)
	   SUPERLU_MALLOC(2*procs *sizeof(MPI_Request))) ) {
      fprintf (stderr, "Malloc fails for send_req[].");
      return (memAux);
    }
    memAux += (float) (2*procs *sizeof(MPI_Request));
    if ( !(ia_send = (int_t **) SUPERLU_MALLOC(procs*sizeof(int_t*))) ) {
      fprintf(stderr, "Malloc fails for ia_send[].");
      return (memAux);
    }
    memAux += (float) (procs*sizeof(int_t*));
    if ( !(aij_send = (doublecomplex **)SUPERLU_MALLOC(procs*sizeof(doublecomplex*))) ) {
      fprintf(stderr, "Malloc fails for aij_send[].");
      return (memAux);
    }
    memAux += (float) (procs*sizeof(doublecomplex*));
    if ( !(index = intMalloc_dist(2*SendCnt)) ) {
      fprintf(stderr, "Malloc fails for index[].");
      return (memAux);
    }
    memAux += (float) (2*SendCnt*iword);
    if ( !(nzval = doublecomplexMalloc_dist(SendCnt)) ) {
      fprintf(stderr, "Malloc fails for nzval[].");
      return (memAux);
    }
    memAux += (float) (SendCnt * dword);
    if ( !(ptr_to_send = intCalloc_dist(procs)) ) {
      fprintf(stderr, "Malloc fails for ptr_to_send[].");
      return (memAux);
    }
    memAux += (float) (procs * iword);
    if ( !(itemp = intMalloc_dist(2*maxnnzToRecv)) ) {
      fprintf(stderr, "Malloc fails for itemp[].");
      return (memAux);
    }
    memAux += (float) (2*maxnnzToRecv*iword);
    if ( !(dtemp = doublecomplexMalloc_dist(maxnnzToRecv)) ) {
      fprintf(stderr, "Malloc fails for dtemp[].");
      return (memAux);
    }
    memAux += (float) (maxnnzToRecv * dword);

    for (i = 0, j = 0, p = 0; p < procs; ++p) {
      if ( p != iam ) {
	ia_send[p] = &index[i];
	i += 2 * nnzToSend[p]; /* ia/ja indices alternate */
	aij_send[p] = &nzval[j];
	j += nnzToSend[p];
      }
    }
  } /* if procs > 1 */

  nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
  nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */
  if ( !(ainf_colptr = intCalloc_dist(ilsum_j[nsupers_j] + 1)) ) {
    fprintf (stderr, "Malloc fails for *ainf_colptr[].");
    return (memAux);
  }
  memRet += (float) (ilsum_j[nsupers_j] + 1) * iword;
  if ( !(asup_rowptr = intCalloc_dist(ilsum_i[nsupers_i] + 1)) ) {
    fprintf (stderr, "Malloc fails for *asup_rowptr[].");
    return (memAux+memRet);
  }
  memRet += (float) (ilsum_i[nsupers_i] + 1) * iword;

  /* ------------------------------------------------------------
     LOAD THE ENTRIES OF A INTO THE (IA,JA,AIJ) STRUCTURES TO SEND.
     THIS ACCOUNTS FOR THE SECOND PASS OF A.
     ------------------------------------------------------------*/
  nnz_loc = 0; /* Reset the local nonzero count. */
  nnz_loc_ainf = nnz_loc_asup = 0;
  nzval_a = Astore->nzval;
  for (i = 0; i < m_loc; ++i) {
    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
      irow = perm_c[perm_r[i+fst_row]];  /* Row number in Pc*Pr*A */
      jcol = Astore->colind[j];
      gbi = BlockNum( irow );
      gbj = BlockNum( jcol );
      p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );

      if ( p != iam ) { /* remote */
	k = ptr_to_send[p];
	ia_send[p][k] = irow;
	ia_send[p][k + nnzToSend[p]] = jcol;
	aij_send[p][k] = nzval_a[j];
	++ptr_to_send[p];
      } else {          /* local */
	ia[nnz_loc] = irow;
	ja[nnz_loc] = jcol;
	aij[nnz_loc] = nzval_a[j];
	++nnz_loc;
	/* Count nonzeros in each column of L / row of U */
	if (gbi >= gbj) {
	  ainf_colptr[ilsum_j[LBj( gbj, grid )] + jcol - FstBlockC( gbj )] ++;
	  nnz_loc_ainf ++;
	}
	else {
	  asup_rowptr[ilsum_i[LBi( gbi, grid )] + irow - FstBlockC( gbi )] ++;
	  nnz_loc_asup ++;
	}
      }
    }
  }

  /* ------------------------------------------------------------
     PERFORM REDISTRIBUTION. THIS INVOLVES ALL-TO-ALL COMMUNICATION.
     NOTE: Can possibly use MPI_Alltoallv.
     ------------------------------------------------------------*/
  for (p = 0; p < procs; ++p) {
    if ( p != iam ) {
      it = 2*nnzToSend[p];
      MPI_Isend( ia_send[p], it, mpi_int_t,
		 p, iam, grid->comm, &send_req[p] );
      it = nnzToSend[p];
      MPI_Isend( aij_send[p], it, SuperLU_MPI_DOUBLE_COMPLEX,
		 p, iam+procs, grid->comm, &send_req[procs+p] );
    }
  }

  for (p = 0; p < procs; ++p) {
    if ( p != iam ) {
      it = 2*nnzToRecv[p];
      MPI_Recv( itemp, it, mpi_int_t, p, p, grid->comm, &status );
      it = nnzToRecv[p];
      MPI_Recv( dtemp, it, SuperLU_MPI_DOUBLE_COMPLEX, p, p+procs,
		grid->comm, &status );
      for (i = 0; i < nnzToRecv[p]; ++i) {
	ia[nnz_loc] = itemp[i];
	irow = itemp[i];
	jcol = itemp[i + nnzToRecv[p]];
	/* assert(jcol<n); */
	ja[nnz_loc] = jcol;
	aij[nnz_loc] = dtemp[i];
	++nnz_loc;

	gbi = BlockNum( irow );
	gbj = BlockNum( jcol );
	/* Count nonzeros in each column of L / row of U */
	if (gbi >= gbj) {
	  ainf_colptr[ilsum_j[LBj( gbj, grid )] + jcol - FstBlockC( gbj )] ++;
	  nnz_loc_ainf ++;
	}
	else {
	  asup_rowptr[ilsum_i[LBi( gbi, grid )] + irow - FstBlockC( gbi )] ++;
	  nnz_loc_asup ++;
	}
      }
    }
  }

  for (p = 0; p < procs; ++p) {
    if ( p != iam ) {
      MPI_Wait( &send_req[p], &status);
      MPI_Wait( &send_req[procs+p], &status);
    }
  }

  /* ------------------------------------------------------------
     DEALLOCATE TEMPORARY STORAGE
     ------------------------------------------------------------*/

  SUPERLU_FREE(nnzToRecv);
  memAux -= 2 * procs * iword;
  if ( procs > 1 ) {
    SUPERLU_FREE(send_req);
    SUPERLU_FREE(ia_send);
    SUPERLU_FREE(aij_send);
    SUPERLU_FREE(index);
    SUPERLU_FREE(nzval);
    SUPERLU_FREE(ptr_to_send);
    SUPERLU_FREE(itemp);
    SUPERLU_FREE(dtemp);
    memAux -= 2*procs *sizeof(MPI_Request) + procs*sizeof(int_t*) +
      procs*sizeof(doublecomplex*) + 2*SendCnt * iword +
      SendCnt* dword + procs*iword +
      2*maxnnzToRecv*iword + maxnnzToRecv*dword;
  }

  /* ------------------------------------------------------------
     CONVERT THE TRIPLET FORMAT.
     ------------------------------------------------------------*/
  if (nnz_loc_ainf != 0) {
    if ( !(ainf_rowind = intMalloc_dist(nnz_loc_ainf)) ) {
      fprintf (stderr, "Malloc fails for *ainf_rowind[].");
      return (memAux+memRet);
    }
    memRet += (float) (nnz_loc_ainf * iword);
    if ( !(ainf_val = doublecomplexMalloc_dist(nnz_loc_ainf)) ) {
      fprintf (stderr, "Malloc fails for *ainf_val[].");
      return (memAux+memRet);
    }
    memRet += (float) (nnz_loc_ainf * dword);
  }
  else {
    ainf_rowind = NULL;
    ainf_val = NULL;
  }
  if (nnz_loc_asup != 0) {
    if ( !(asup_colind = intMalloc_dist(nnz_loc_asup)) ) {
      fprintf (stderr, "Malloc fails for *asup_colind[].");
      return (memAux + memRet);
    }
    memRet += (float) (nnz_loc_asup * iword);
    if ( !(asup_val = doublecomplexMalloc_dist(nnz_loc_asup)) ) {
      fprintf (stderr, "Malloc fails for *asup_val[].");
      return (memAux  + memRet);
    }
    memRet += (float) (nnz_loc_asup * dword);
  }
  else {
    asup_colind = NULL;
    asup_val = NULL;
  }

  /* Initialize the array of column pointers */
  k = 0;
  jsize = ainf_colptr[0];  ainf_colptr[0] = 0;
  for (j = 1; j < ilsum_j[nsupers_j]; j++) {
    k += jsize;
    jsize = ainf_colptr[j];
    ainf_colptr[j] = k;
  }
  ainf_colptr[ilsum_j[nsupers_j]] = k + jsize;
  i = 0;
  isize = asup_rowptr[0];  asup_rowptr[0] = 0;
  for (j = 1; j < ilsum_i[nsupers_i]; j++) {
    i += isize;
    isize = asup_rowptr[j];
    asup_rowptr[j] = i;
  }
  asup_rowptr[ilsum_i[nsupers_i]] = i + isize;

  /* Copy the triplets into the column oriented storage */
  for (i = 0; i < nnz_loc; ++i) {
    jcol = ja[i];
    irow = ia[i];
    gbi = BlockNum( irow );
    gbj = BlockNum( jcol );
    /* Count nonzeros in each column of L / row of U */
    if (gbi >= gbj) {
      j = ilsum_j[LBj( gbj, grid )] + jcol - FstBlockC( gbj );
      k = ainf_colptr[j];
      ainf_rowind[k] = irow;
      ainf_val[k] = aij[i];
      ainf_colptr[j] ++;
    }
    else {
      j = ilsum_i[LBi( gbi, grid )] + irow - FstBlockC( gbi );
      k = asup_rowptr[j];
      asup_colind[k] = jcol;
      asup_val[k] = aij[i];
      asup_rowptr[j] ++;
    }
  }

  /* Reset the column pointers to the beginning of each column */
  for (j = ilsum_j[nsupers_j]; j > 0; j--)
    ainf_colptr[j] = ainf_colptr[j-1];
  for (j = ilsum_i[nsupers_i]; j > 0; j--)
    asup_rowptr[j] = asup_rowptr[j-1];
  ainf_colptr[0] = 0;
  asup_rowptr[0] = 0;

  SUPERLU_FREE(ia);
  SUPERLU_FREE(aij);
  memAux -= 2*szbuf*iword + szbuf*dword;

  *p_ainf_colptr = ainf_colptr;
  *p_ainf_rowind = ainf_rowind;
  *p_ainf_val    = ainf_val;
  *p_asup_rowptr = asup_rowptr;
  *p_asup_colind = asup_colind;
  *p_asup_val    = asup_val;

#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Exit zdist_A()");
  fprintf (stdout, "Size of allocated memory (MB) %.3f\n", memRet*1e-6);
#endif

  return (-memRet);
} /* dist_A */

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Distribute the input matrix onto the 2D process mesh.
 *
 * Arguments
 * =========
 *
 * fact (input) fact_t
 *        Specifies whether or not the L and U structures will be re-used.
 *        = SamePattern_SameRowPerm: L and U structures are input, and
 *                                   unchanged on exit.
 *          This routine should not be called for this case, an error
 *          is generated.  Instead, pddistribute routine should be called.
 *        = DOFACT or SamePattern: L and U structures are computed and output.
 *
 * n      (Input) int
 *        Dimension of the matrix.
 *
 * A      (Input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        A may be overwritten by diag(R)*A*diag(C)*Pc^T.
 *        The type of A can be: Stype = NR; Dtype = SLU_D; Mtype = GE.
 *
 * ScalePermstruct (Input) zScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_freeable (Input) *Glu_freeable_t
 *        The global structure describing the graph of L and U.
 *
 * LUstruct (Input) zLUstruct_t*
 *        Data structures for L and U factors.
 *
 * grid   (Input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Return value
 * ============
 *   < 0, number of bytes allocated on return from the dist_symbLU
 *   > 0, number of bytes allocated for performing the distribution
 *       of the data, when out of memory.
 *        (an approximation).
 * </pre>
 */

float
zdist_psymbtonum(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
		zScalePermstruct_t *ScalePermstruct,
		Pslu_freeable_t *Pslu_freeable,
		zLUstruct_t *LUstruct, gridinfo_t *grid)
{
  Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
  Glu_freeable_t Glu_freeable_n;
  zLocalLU_t *Llu = LUstruct->Llu;
  int_t bnnz, fsupc, i, irow, istart, j, jb, ib, jj, k, k1,
    len, len1, nsupc, nsupc_gb, ii, nprocs;
  int_t lib;  /* local block row number */
  int_t nlb;  /* local block rows*/
  int_t ljb;  /* local block column number */
  int_t nrbl; /* number of L blocks in current block column */
  int_t nrbu; /* number of U blocks in current block column */
  int_t gb;   /* global block number; 0 < gb <= nsuper */
  int_t lb;   /* local block number; 0 < lb <= ceil(NSUPERS/Pr) */
  int_t ub,gik,iklrow,fnz;
  int iam, jbrow, jbcol, jcol, kcol, krow, mycol, myrow, pc, pr, ljb_i, ljb_j, p;
  int_t mybufmax[NBUFFERS];
  NRformat_loc *Astore;
  doublecomplex *a;
  int_t *asub, *xa;
  int_t *ainf_colptr, *ainf_rowind, *asup_rowptr, *asup_colind;
  doublecomplex *asup_val, *ainf_val;
  int_t *xsup, *supno;    /* supernode and column mapping */
  int_t *lsub, *xlsub, *usub, *usub1, *xusub;
  int_t nsupers, nsupers_i, nsupers_j, nsupers_ij;
  int_t next_ind;      /* next available position in index[*] */
  int_t next_val;      /* next available position in nzval[*] */
  int_t *index;        /* indices consist of headers and row subscripts */
  int   *index1;       /* temporary pointer to array of int */
  doublecomplex *lusup, *uval; /* nonzero values in L and U */
  int *recvBuf;    // 1/16/22 Sherry changed to int, was:  int_t *recvBuf;
  int *ptrToRecv, *nnzToRecv, *ptrToSend, *nnzToSend;
  doublecomplex **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc) */
  doublecomplex **Linv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
  doublecomplex **Uinv_bc_ptr;  /* size ceil(NSUPERS/Pc) */
  int_t  **Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc) */
  int_t   **Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
  int_t *index_srt;         /* indices consist of headers and row subscripts */
  doublecomplex *lusup_srt; /* nonzero values in L and U */
  doublecomplex **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr) */
  int_t  **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr) */
  int_t  *Unnz;  /* size ceil(NSUPERS/Pc) */

  C_Tree  *LBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
  C_Tree  *LRtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
  C_Tree  *UBtree_ptr;       /* size ceil(NSUPERS/Pc)                */
  C_Tree  *URtree_ptr;		  /* size ceil(NSUPERS/Pr)                */
  int msgsize;

  int_t  *Urbs,*Urbs1; /* Number of row blocks in each block column of U. */
  Ucb_indptr_t **Ucb_indptr;/* Vertical linked list pointing to Uindex[] */
  int_t  **Ucb_valptr;      /* Vertical linked list pointing to Unzval[] */


  /*-- Counts to be used in factorization. --*/
  int  *ToRecv, *ToSendD, **ToSendR;

  /*-- Counts to be used in lower triangular solve. --*/
  int  *fmod;          /* Modification count for L-solve.        */
  int  **fsendx_plist; /* Column process list to send down Xk.   */
  int  nfrecvx = 0;    /* Number of Xk I will receive.           */
  int  nfsendx = 0;    /* Number of Xk I will send               */
  int  kseen;

  /*-- Counts to be used in upper triangular solve. --*/
  int  *bmod;          /* Modification count for U-solve.        */
  int  **bsendx_plist; /* Column process list to send down Xk.   */
  int  nbrecvx = 0;    /* Number of Xk I will receive.           */
  int  nbsendx = 0;    /* Number of Xk I will send               */
  int_t  *ilsum;         /* starting position of each supernode in
			    the full array (local)                 */
  int_t  *ilsum_j, ldaspa_j; /* starting position of each supernode in
				the full array (local, block column wise) */
  /*-- Auxiliary arrays; freed on return --*/
  int_t *Urb_marker;  /* block hit marker; size ceil(NSUPERS/Pr)           */
  int_t *LUb_length; /* L,U block length; size nsupers_ij */
  int_t *LUb_indptr; /* pointers to L,U index[]; size nsupers_ij */
  int_t *LUb_number; /* global block number; size nsupers_ij */
  int_t *LUb_valptr; /* pointers to U nzval[]; size ceil(NSUPERS/Pc)      */
  int_t *Lrb_marker;  /* block hit marker; size ceil(NSUPERS/Pr)           */
  int_t *ActiveFlag;
  int_t *ActiveFlagAll;
  int_t Iactive;
  int *ranks;
  int_t *idxs;
  int_t **nzrows;
  double rseed;
  int rank_cnt,rank_cnt_ref,Root;
doublecomplex *dense, *dense_col; /* SPA */
  doublecomplex zero = {0.0, 0.0};
  int_t ldaspa;     /* LDA of SPA */
  int_t iword, dword;
  float mem_use = 0.0;
  int *mod_bit;
  int *frecv, *brecv;
  int_t *lloc;
  double *SeedSTD_BC,*SeedSTD_RD;
  int_t idx_indx,idx_lusup;
  int_t nbrow;
  int_t  ik, il, lk, rel, knsupc, idx_r;
  int_t  lptr1_tmp, idx_i, idx_v,m, uu;
  int_t	nub;

  float memStrLU, memA,
        memDist = 0.; /* memory used for redistributing the data, which does
		         not include the memory for the numerical values
                         of L and U (positive number)*/
  float  memNLU = 0.; /* memory allocated for storing the numerical values of
		         L and U, that will be used in the numeric
                         factorization (positive number) */
  float  memTRS = 0.; /* memory allocated for storing the meta-data for triangular solve (positive number)*/

#if ( PRNTlevel>=1 )
  int_t nLblocks = 0, nUblocks = 0;
#endif
#if ( PROFlevel>=1 )
	double t, t_u, t_l;
	int_t u_blks;
#endif

  /* Initialization. */
  iam = grid->iam;
#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Enter dist_psymbtonum()");
#endif
  myrow = MYROW( iam, grid );
  mycol = MYCOL( iam, grid );
  nprocs = grid->npcol * grid->nprow;
  for (i = 0; i < NBUFFERS; ++i) mybufmax[i] = 0;
  Astore   = (NRformat_loc *) A->Store;

  iword = sizeof(int_t);
  dword = sizeof(doublecomplex);

  if (options->Fact == SamePattern_SameRowPerm) {
    ABORT ("ERROR: call of dist_psymbtonum with fact equals SamePattern_SameRowPerm.");
  }

  if ((memStrLU =
       dist_symbLU (options, n, Pslu_freeable,
		    Glu_persist, &xlsub, &lsub, &xusub, &usub,	grid)) > 0)
    return (memStrLU);
  memDist += (-memStrLU);
  xsup  = Glu_persist->xsup;    /* supernode and column mapping */
  supno = Glu_persist->supno;
  nsupers  = supno[n-1] + 1;
  nsupers_i = CEILING( nsupers, grid->nprow );/* No of local row blocks */
  nsupers_j = CEILING( nsupers, grid->npcol );/* No of local column blocks */
  nsupers_ij = SUPERLU_MAX(nsupers_i, nsupers_j);
  if ( !(ilsum = intMalloc_dist(nsupers_i+1)) ) {
    fprintf (stderr, "Malloc fails for ilsum[].");
    return (memDist + memNLU + memTRS);
  }
  memNLU += (nsupers_i+1) * iword;
  if ( !(ilsum_j = intMalloc_dist(nsupers_j+1)) ) {
    fprintf (stderr, "Malloc fails for ilsum_j[].");
    return (memDist + memNLU + memTRS);
  }
  memDist += (nsupers_j+1) * iword;

  /* Compute ldaspa and ilsum[], ldaspa_j and ilsum_j[]. */
  ilsum[0] = 0;
  ldaspa = 0;
  for (gb = 0; gb < nsupers; gb++)
    if ( myrow == PROW( gb, grid ) ) {
      i = SuperSize( gb );
      ldaspa += i;
      lb = LBi( gb, grid );
      ilsum[lb + 1] = ilsum[lb] + i;
    }
  ilsum[nsupers_i] = ldaspa;

  ldaspa_j = 0; ilsum_j[0] = 0;
  for (gb = 0; gb < nsupers; gb++)
    if (mycol == PCOL( gb, grid )) {
      i = SuperSize( gb );
      ldaspa_j += i;
      lb = LBj( gb, grid );
      ilsum_j[lb + 1] = ilsum_j[lb] + i;
    }
  ilsum_j[nsupers_j] = ldaspa_j;

  if ((memA = zdist_A(A, ScalePermstruct, Glu_persist,
		      grid, &ainf_colptr, &ainf_rowind, &ainf_val,
		      &asup_rowptr, &asup_colind, &asup_val,
		      ilsum, ilsum_j)) > 0)
    return (memDist + memA + memNLU + memTRS);
  memDist += (-memA);

  /* ------------------------------------------------------------
     FIRST TIME CREATING THE L AND U DATA STRUCTURES.
     ------------------------------------------------------------*/

  /* We first need to set up the L and U data structures and then
   * propagate the values of A into them.
   */
  if ( !(ToRecv = SUPERLU_MALLOC(nsupers * sizeof(int))) ) {
    fprintf(stderr, "Calloc fails for ToRecv[].");
    return (memDist + memNLU + memTRS);
  }
  for (i = 0; i < nsupers; ++i) ToRecv[i] = 0;
  memNLU += nsupers * iword;

  k = CEILING( nsupers, grid->npcol ); /* Number of local column blocks */
  if ( !(ToSendR = (int **) SUPERLU_MALLOC(k*sizeof(int*))) ) {
    fprintf(stderr, "Malloc fails for ToSendR[].");
    return (memDist + memNLU + memTRS);
  }
  memNLU += k*sizeof(int_t*);
  j = k * grid->npcol;
  if ( !(index1 = SUPERLU_MALLOC(j * sizeof(int))) ) {
    fprintf(stderr, "Malloc fails for index[].");
    return (memDist + memNLU + memTRS);
  }
  memNLU += j*iword;

  for (i = 0; i < j; ++i) index1[i] = EMPTY;
  for (i = 0,j = 0; i < k; ++i, j += grid->npcol) ToSendR[i] = &index1[j];

  /* Auxiliary arrays used to set up L and U block data structures.
     They are freed on return. */
  if ( !(LUb_length = intCalloc_dist(nsupers_ij)) ) {
    fprintf(stderr, "Calloc fails for LUb_length[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(LUb_indptr = intMalloc_dist(nsupers_ij)) ) {
    fprintf(stderr, "Malloc fails for LUb_indptr[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(LUb_number = intCalloc_dist(nsupers_ij)) ) {
    fprintf(stderr, "Calloc fails for LUb_number[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(LUb_valptr = intCalloc_dist(nsupers_ij)) ) {
    fprintf(stderr, "Calloc fails for LUb_valptr[].");
    return (memDist + memNLU + memTRS);
  }
  memDist += 4 * nsupers_ij * iword;

  k = CEILING( nsupers, grid->nprow );
  /* Pointers to the beginning of each block row of U. */
  if ( !(Unzval_br_ptr =
	 (doublecomplex**)SUPERLU_MALLOC(nsupers_i * sizeof(doublecomplex*))) ) {
    fprintf(stderr, "Malloc fails for Unzval_br_ptr[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(Ufstnz_br_ptr = (int_t**)SUPERLU_MALLOC(nsupers_i * sizeof(int_t*))) ) {
    fprintf(stderr, "Malloc fails for Ufstnz_br_ptr[].");
    return (memDist + memNLU + memTRS);
  }
  memNLU += nsupers_i*sizeof(doublecomplex*) + nsupers_i*sizeof(int_t*);
  Unzval_br_ptr[nsupers_i-1] = NULL;
  Ufstnz_br_ptr[nsupers_i-1] = NULL;

  if ( !(ToSendD = SUPERLU_MALLOC(nsupers_i * sizeof(int))) ) {
    fprintf(stderr, "Malloc fails for ToSendD[].");
    return (memDist + memNLU + memTRS);
  }
  for (i = 0; i < nsupers_i; ++i) ToSendD[i] = NO;

  memNLU += nsupers_i*iword;
  if ( !(Urb_marker = intCalloc_dist(nsupers_j))) {
    fprintf(stderr, "Calloc fails for rb_marker[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(Lrb_marker = intCalloc_dist( nsupers_i ))) {
    fprintf(stderr, "Calloc fails for rb_marker[].");
    return (memDist + memNLU + memTRS);
  }
  memDist += (nsupers_i + nsupers_j)*iword;

  /* Auxiliary arrays used to set up L, U block data structures.
     They are freed on return.
     k is the number of local row blocks.   */
  if ( !(dense = doublecomplexCalloc_dist(SUPERLU_MAX(ldaspa, ldaspa_j)
				   * sp_ienv_dist(3, options))) ) {
    fprintf(stderr, "Calloc fails for SPA dense[].");
    return (memDist + memNLU + memTRS);
  }
  /* These counts will be used for triangular solves. */
  if ( !(fmod = int32Calloc_dist(nsupers_i)) ) {
    fprintf(stderr, "Calloc fails for fmod[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(bmod = int32Calloc_dist(nsupers_i)) ) {
    fprintf(stderr, "Calloc fails for bmod[].");
    return (memDist + memNLU + memTRS);
  }
  /* ------------------------------------------------ */
  memNLU += 2*nsupers_i*iword +
    SUPERLU_MAX(ldaspa, ldaspa_j)*sp_ienv_dist(3, options)*dword;

  /* Pointers to the beginning of each block column of L. */
  if ( !(Lnzval_bc_ptr =
	 (doublecomplex**)SUPERLU_MALLOC(nsupers_j * sizeof(doublecomplex*))) ) {
    fprintf(stderr, "Malloc fails for Lnzval_bc_ptr[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(Lrowind_bc_ptr = (int_t**)SUPERLU_MALLOC(nsupers_j * sizeof(int_t*))) ) {
    fprintf(stderr, "Malloc fails for Lrowind_bc_ptr[].");
    return (memDist + memNLU + memTRS);
  }

  if ( !(Linv_bc_ptr =
			(doublecomplex**)SUPERLU_MALLOC(nsupers_j * sizeof(doublecomplex*))) ) {
	fprintf(stderr, "Malloc fails for Linv_bc_ptr[].");
	return (memDist + memNLU + memTRS);
  }
  if ( !(Uinv_bc_ptr =
			(doublecomplex**)SUPERLU_MALLOC(nsupers_j * sizeof(doublecomplex*))) ) {
	fprintf(stderr, "Malloc fails for Uinv_bc_ptr[].");
	return (memDist + memNLU + memTRS);
  }
  if ( !(Lindval_loc_bc_ptr = (int_t**)SUPERLU_MALLOC(nsupers_j * sizeof(int_t*))) ){
    fprintf(stderr, "Malloc fails for Lindval_loc_bc_ptr[].");
    return (memDist + memNLU + memTRS);
  }

  if ( !(Unnz = (int_t*)SUPERLU_MALLOC(nsupers_j * sizeof(int_t))) ){
    fprintf(stderr, "Malloc fails for Unnz[].");
    return (memDist + memNLU + memTRS);
  }
  memTRS += nsupers_j*sizeof(int_t*) + 2.0*nsupers_j*sizeof(double*) + nsupers_j*iword;  //acount for Lindval_loc_bc_ptr, Unnz, Linv_bc_ptr,Uinv_bc_ptr

  memNLU += nsupers_j * sizeof(double*) + nsupers_j * sizeof(int_t*)+ nsupers_j * sizeof(int_t*);
  Lnzval_bc_ptr[nsupers_j-1] = NULL;
  Lrowind_bc_ptr[nsupers_j-1] = NULL;
  Linv_bc_ptr[nsupers_j-1] = NULL;
  Uinv_bc_ptr[nsupers_j-1] = NULL;
  Lindval_loc_bc_ptr[nsupers_j-1] = NULL;

  /* These lists of processes will be used for triangular solves. */
  if ( !(fsendx_plist = (int **) SUPERLU_MALLOC(nsupers_j*sizeof(int*))) ) {
    fprintf(stderr, "Malloc fails for fsendx_plist[].");
    return (memDist + memNLU + memTRS);
  }
  len = nsupers_j * grid->nprow;
  if ( !(index1 = int32Malloc_dist(len)) ) {
    fprintf(stderr, "Malloc fails for fsendx_plist[0]");
    return (memDist + memNLU + memTRS);
  }
  for (i = 0; i < len; ++i) index1[i] = EMPTY;
  for (i = 0, j = 0; i < nsupers_j; ++i, j += grid->nprow)
    fsendx_plist[i] = &index1[j];
  if ( !(bsendx_plist = (int **) SUPERLU_MALLOC(nsupers_j*sizeof(int*))) ) {
    fprintf(stderr, "Malloc fails for bsendx_plist[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(index1 = int32Malloc_dist(len)) ) {
    fprintf(stderr, "Malloc fails for bsendx_plist[0]");
    return (memDist + memNLU + memTRS);
  }
  for (i = 0; i < len; ++i) index1[i] = EMPTY;
  for (i = 0, j = 0; i < nsupers_j; ++i, j += grid->nprow)
    bsendx_plist[i] = &index1[j];
  /* -------------------------------------------------------------- */
  memNLU += 2*nsupers_j*sizeof(int_t*) + 2*len*iword;

  /*------------------------------------------------------------
    PROPAGATE ROW SUBSCRIPTS AND VALUES OF A INTO L AND U BLOCKS.
    THIS ACCOUNTS FOR ONE-PASS PROCESSING OF A, L AND U.
    ------------------------------------------------------------*/
  for (jb = 0; jb < nsupers; jb++) {
    jbcol = PCOL( jb, grid );
    jbrow = PROW( jb, grid );
    ljb_j = LBj( jb, grid ); /* Local block number column wise */
    ljb_i = LBi( jb, grid);  /* Local block number row wise */
    fsupc = FstBlockC( jb );
    nsupc = SuperSize( jb );

    if ( myrow == jbrow ) { /* Block row jb in my process row */
      /* Scatter A into SPA. */
      for (j = ilsum[ljb_i], dense_col = dense; j < ilsum[ljb_i]+nsupc; j++) {
	for (i = asup_rowptr[j]; i < asup_rowptr[j+1]; i++) {
	  if (i >= asup_rowptr[ilsum[nsupers_i]])
	    printf ("ERR7\n");
	  jcol = asup_colind[i];
	  if (jcol >= n)
	    printf ("Pe[%d] ERR distsn jb %d gb %d j %d jcol %d\n",
		    iam, (int) jb, (int) gb, (int) j, jcol);
	  gb = BlockNum( jcol );
	  lb = LBj( gb, grid );
	  if (gb >= nsupers || lb >= nsupers_j) printf ("ERR8\n");
	  jcol = ilsum_j[lb] + jcol - FstBlockC( gb );
	  if (jcol >= ldaspa_j)
	    printf ("Pe[%d] ERR1 jb %d gb %d j %d jcol %d\n",
		    iam, (int) jb, (int) gb, (int) j, jcol);
	  dense_col[jcol] = asup_val[i];
	}
	dense_col += ldaspa_j;
      }

      /*------------------------------------------------
       * SET UP U BLOCKS.
       *------------------------------------------------*/
      /* Count number of blocks and length of each block. */
      nrbu = 0;
      len = 0; /* Number of column subscripts I own. */
      len1 = 0; /* number of fstnz subscripts */
      for (i = xusub[ljb_i]; i < xusub[ljb_i+1]; i++) {
	if (i >= xusub[nsupers_i]) printf ("ERR10\n");
	jcol = usub[i];
	gb = BlockNum( jcol ); /* Global block number */

	/*if (fsupc <= 146445 && 146445 < fsupc + nsupc && jcol == 397986)
	  printf ("Pe[%d] [%d %d] elt [%d] jbcol %d pc %d\n",
	  iam, jb, gb, jcol, jbcol, pc); */

	lb = LBj( gb, grid );  /* Local block number */
	pc = PCOL( gb, grid ); /* Process col owning this block */
	if (mycol == jbcol) ToSendR[ljb_j][pc] = YES;
	/* if (mycol == jbcol && mycol != pc) ToSendR[ljb_j][pc] = YES; */
	pr = PROW( gb, grid );
	if ( pr != jbrow  && mycol == pc)
	  bsendx_plist[lb][jbrow] = YES;
	if (mycol == pc) {
	  len += nsupc;
	  LUb_length[lb] += nsupc;
	  ToSendD[ljb_i] = YES;
	  if (Urb_marker[lb] <= jb) { /* First see this block */
	    if (Urb_marker[lb] == FALSE && gb != jb && myrow != pr) nbrecvx ++;
	    Urb_marker[lb] = jb + 1;
	    LUb_number[nrbu] = gb;
	    /* if (gb == 391825 && jb == 145361)
	       printf ("Pe[%d] T1 [%d %d] nrbu %d \n",
	       iam, jb, gb, nrbu); */
	    nrbu ++;
	    len1 += SuperSize( gb );
	    if ( gb != jb )/* Exclude diagonal block. */
	      ++bmod[ljb_i];/* Mod. count for back solve */
#if ( PRNTlevel>=1 )
	    ++nUblocks;
#endif
	  }
	}
      } /* for i ... */

      if ( nrbu ) {
	/* Sort the blocks of U in increasing block column index.
	   SuperLU_DIST assumes this is true */
	/* simple insert sort algorithm */
	/* to be transformed in quick sort */
	for (j = 1; j < nrbu; j++) {
	  k = LUb_number[j];
	  for (i=j-1; i>=0 && LUb_number[i] > k; i--) {
	    LUb_number[i+1] = LUb_number[i];
	  }
	  LUb_number[i+1] = k;
	}

	/* Set up the initial pointers for each block in
	   index[] and nzval[]. */
	/* Add room for descriptors */
	len1 += BR_HEADER + nrbu * UB_DESCRIPTOR;
	if ( !(index = intMalloc_dist(len1+1)) ) {
	  fprintf (stderr, "Malloc fails for Uindex[]");
	  return (memDist + memNLU + memTRS);
	}
	Ufstnz_br_ptr[ljb_i] = index;
	if (!(Unzval_br_ptr[ljb_i] =
	      doublecomplexMalloc_dist(len))) {
	  fprintf (stderr, "Malloc fails for Unzval_br_ptr[*][]");
	  return (memDist + memNLU + memTRS);
	}
	memNLU += (len1+1)*iword + len*dword;
	uval = Unzval_br_ptr[ljb_i];
	mybufmax[2] = SUPERLU_MAX( mybufmax[2], len1 );
	mybufmax[3] = SUPERLU_MAX( mybufmax[3], len );
	index[0] = nrbu;  /* Number of column blocks */
	index[1] = len;   /* Total length of nzval[] */
	index[2] = len1;  /* Total length of index */
	index[len1] = -1; /* End marker */
	next_ind = BR_HEADER;
	next_val = 0;
	for (k = 0; k < nrbu; k++) {
	  gb = LUb_number[k];
	  lb = LBj( gb, grid );
	  len = LUb_length[lb];
	  LUb_length[lb] = 0;  /* Reset vector of block length */
	  index[next_ind++] = gb; /* Descriptor */
	  index[next_ind++] = len;
	  LUb_indptr[lb] = next_ind;
	  for (; next_ind < LUb_indptr[lb] + SuperSize( gb ); next_ind++)
	    index[next_ind] = FstBlockC( jb + 1 );
	  LUb_valptr[lb] = next_val;
	  next_val += len;
	}
	/* Propagate the fstnz subscripts to Ufstnz_br_ptr[],
	   and the initial values of A from SPA into Unzval_br_ptr[]. */
	for (i = xusub[ljb_i]; i < xusub[ljb_i+1]; i++) {
	  jcol = usub[i];
	  gb = BlockNum( jcol );

	  if ( mycol == PCOL( gb, grid ) ) {
	    lb = LBj( gb, grid );
	    k = LUb_indptr[lb]; /* Start fstnz in index */
	    index[k + jcol - FstBlockC( gb )] = FstBlockC( jb );
	  }
	}  /* for i ... */

	for (i = 0; i < nrbu; i++) {
	  gb = LUb_number[i];
	  lb = LBj( gb, grid );
	  next_ind = LUb_indptr[lb];
	  k = FstBlockC( jb + 1);
	  jcol = ilsum_j[lb];
	  for (jj = 0; jj < SuperSize( gb ); jj++, jcol++) {
	    dense_col = dense;
	    j = index[next_ind+jj];
	    for (ii = j; ii < k; ii++) {
	      uval[LUb_valptr[lb]++] = dense_col[jcol];
	      dense_col[jcol] = zero;
	      dense_col += ldaspa_j;
	    }
	  }
	}
      } else {
	Ufstnz_br_ptr[ljb_i] = NULL;
	Unzval_br_ptr[ljb_i] = NULL;
      } /* if nrbu ... */
    } /* if myrow == jbrow */

      /*------------------------------------------------
       * SET UP L BLOCKS.
       *------------------------------------------------*/
    if (mycol == jbcol) {  /* Block column jb in my process column */
      /* Scatter A_inf into SPA. */
      for (j = ilsum_j[ljb_j], dense_col = dense; j < ilsum_j[ljb_j] + nsupc; j++) {
	for (i = ainf_colptr[j]; i < ainf_colptr[j+1]; i++) {
	  irow = ainf_rowind[i];
	  if (irow >= n) printf ("Pe[%d] ERR1\n", iam);
	  gb = BlockNum( irow );
	  if (gb >= nsupers) printf ("Pe[%d] ERR5\n", iam);
	  if ( myrow == PROW( gb, grid ) ) {
	    lb = LBi( gb, grid );
	    irow = ilsum[lb] + irow - FstBlockC( gb );
	    if (irow >= ldaspa) printf ("Pe[%d] ERR0\n", iam);
	    dense_col[irow] = ainf_val[i];
	  }
	}
	dense_col += ldaspa;
      }

      /* sort the indices of the diagonal block at the beginning of xlsub */
      if (myrow == jbrow) {
	k = xlsub[ljb_j];
	for (i = xlsub[ljb_j]; i < xlsub[ljb_j+1]; i++) {
	  irow = lsub[i];
	  if (irow < nsupc + fsupc && i != k+irow-fsupc) {
	    lsub[i] = lsub[k + irow - fsupc];
	    lsub[k + irow - fsupc] = irow;
	    i --;
	  }
	}
      }

      /* Count number of blocks and length of each block. */
      nrbl = 0;
      len = 0; /* Number of row subscripts I own. */
      kseen = 0;
      for (i = xlsub[ljb_j]; i < xlsub[ljb_j+1]; i++) {
	irow = lsub[i];
	gb = BlockNum( irow ); /* Global block number */
	pr = PROW( gb, grid ); /* Process row owning this block */
	if ( pr != jbrow && fsendx_plist[ljb_j][pr] == EMPTY &&
	     myrow == jbrow) {
	  fsendx_plist[ljb_j][pr] = YES;
	  ++nfsendx;
	}
	if ( myrow == pr ) {
	  lb = LBi( gb, grid );  /* Local block number */
	  if (Lrb_marker[lb] <= jb) { /* First see this block */
	    Lrb_marker[lb] = jb + 1;
	    LUb_length[lb] = 1;
	    LUb_number[nrbl++] = gb;
	    if ( gb != jb ) /* Exclude diagonal block. */
	      ++fmod[lb]; /* Mod. count for forward solve */
	    if ( kseen == 0 && myrow != jbrow ) {
	      ++nfrecvx;
	      kseen = 1;
	    }
#if ( PRNTlevel>=1 )
	    ++nLblocks;
#endif
	  } else
	    ++LUb_length[lb];
	  ++len;
	}
      } /* for i ... */

      if ( nrbl ) { /* Do not ensure the blocks are sorted! */
	/* Set up the initial pointers for each block in
	   index[] and nzval[]. */
	/* If I am the owner of the diagonal block, order it first in LUb_number.
	   Necessary for SuperLU_DIST routines */
	kseen = EMPTY;
	for (j = 0; j < nrbl; j++) {
	  if (LUb_number[j] == jb)
	    kseen = j;
	}
	if (kseen != EMPTY && kseen != 0) {
	  LUb_number[kseen] = LUb_number[0];
	  LUb_number[0] = jb;
	}

	/* Add room for descriptors */
	len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
	if ( !(index = intMalloc_dist(len1)) ) {
	  fprintf (stderr, "Malloc fails for index[]");
	  return (memDist + memNLU + memTRS);
	}
	Lrowind_bc_ptr[ljb_j] = index;
	if (!(Lnzval_bc_ptr[ljb_j] =
	      doublecomplexMalloc_dist(len*nsupc))) {
	  fprintf(stderr, "Malloc fails for Lnzval_bc_ptr[*][] col block %d\n", (int) jb);
	  return (memDist + memNLU + memTRS);
	}

	if (!(Linv_bc_ptr[ljb_j] = (doublecomplex*)SUPERLU_MALLOC(nsupc*nsupc * sizeof(doublecomplex))))
		ABORT("Malloc fails for Linv_bc_ptr[ljb_j][]");
	if (!(Uinv_bc_ptr[ljb_j] = (doublecomplex*)SUPERLU_MALLOC(nsupc*nsupc * sizeof(doublecomplex))))
		ABORT("Malloc fails for Uinv_bc_ptr[ljb_j][]");

	memNLU += len1*iword + len*nsupc*dword;

	if ( !(Lindval_loc_bc_ptr[ljb_j] = intCalloc_dist(nrbl*3)))
		ABORT("Malloc fails for Lindval_loc_bc_ptr[ljb_j][]");
	memTRS += nrbl*3.0*iword + 2.0*nsupc*nsupc*dword;  //acount for Lindval_loc_bc_ptr[ljb],Linv_bc_ptr[ljb],Uinv_bc_ptr[ljb]

	lusup = Lnzval_bc_ptr[ljb_j];
	mybufmax[0] = SUPERLU_MAX( mybufmax[0], len1 );
	mybufmax[1] = SUPERLU_MAX( mybufmax[1], len*nsupc );
	mybufmax[4] = SUPERLU_MAX( mybufmax[4], len );
	index[0] = nrbl;  /* Number of row blocks */
	index[1] = len;   /* LDA of the nzval[] */
	next_ind = BC_HEADER;
	next_val = 0;
	for (k = 0; k < nrbl; ++k) {
	  gb = LUb_number[k];
	  lb = LBi( gb, grid );
	  len = LUb_length[lb];

	  Lindval_loc_bc_ptr[ljb_j][k] = lb;
	  Lindval_loc_bc_ptr[ljb_j][k+nrbl] = next_ind;
	  Lindval_loc_bc_ptr[ljb_j][k+nrbl*2] = next_val;

	  LUb_length[lb] = 0;
	  index[next_ind++] = gb; /* Descriptor */
	  index[next_ind++] = len;
	  LUb_indptr[lb] = next_ind;
	    LUb_valptr[lb] = next_val;
	    next_ind += len;
	    next_val += len;
	  }
	  /* Propagate the compressed row subscripts to Lindex[],
	     and the initial values of A from SPA into Lnzval[]. */
	  len = index[1];  /* LDA of lusup[] */
	  for (i = xlsub[ljb_j]; i < xlsub[ljb_j+1]; i++) {
	    irow = lsub[i];
	    gb = BlockNum( irow );
	    if ( myrow == PROW( gb, grid ) ) {
	      lb = LBi( gb, grid );
	      k = LUb_indptr[lb]++; /* Random access a block */
	      index[k] = irow;
	      k = LUb_valptr[lb]++;
	      irow = ilsum[lb] + irow - FstBlockC( gb );
	      for (j = 0, dense_col = dense; j < nsupc; ++j) {
		lusup[k] = dense_col[irow];
		dense_col[irow] = zero;
		k += len;
		dense_col += ldaspa;
	      }
	    }
	  } /* for i ... */



		/* sort Lindval_loc_bc_ptr[ljb_j], Lrowind_bc_ptr[ljb_j] and Lnzval_bc_ptr[ljb_j] here*/
		if(nrbl>1){
			krow = PROW( jb, grid );
			if(myrow==krow){ /* skip the diagonal block */
				uu=nrbl-2;
				lloc = &Lindval_loc_bc_ptr[ljb_j][1];
			}else{
				uu=nrbl-1;
				lloc = Lindval_loc_bc_ptr[ljb_j];
			}
			quickSortM(lloc,0,uu,nrbl,0,3);
		}


		if ( !(index_srt = intMalloc_dist(len1)) )
			ABORT("Malloc fails for index_srt[]");
		if (!(lusup_srt = (doublecomplex*)SUPERLU_MALLOC(len*nsupc * sizeof(doublecomplex))))
			ABORT("Malloc fails for lusup_srt[]");

		idx_indx = BC_HEADER;
		idx_lusup = 0;
		for (jj=0;jj<BC_HEADER;jj++)
			index_srt[jj] = index[jj];

		for(i=0;i<nrbl;i++){
			nbrow = index[Lindval_loc_bc_ptr[ljb_j][i+nrbl]+1];
			for (jj=0;jj<LB_DESCRIPTOR+nbrow;jj++){
				index_srt[idx_indx++] = index[Lindval_loc_bc_ptr[ljb_j][i+nrbl]+jj];
			}

			Lindval_loc_bc_ptr[ljb_j][i+nrbl] = idx_indx - LB_DESCRIPTOR - nbrow;

			for (jj=0;jj<nbrow;jj++){
				k=idx_lusup;
				k1=Lindval_loc_bc_ptr[ljb_j][i+nrbl*2]+jj;
				for (j = 0; j < nsupc; ++j) {
					lusup_srt[k] = lusup[k1];
					k += len;
					k1 += len;
				}
				idx_lusup++;
			}
			Lindval_loc_bc_ptr[ljb_j][i+nrbl*2] = idx_lusup - nbrow;
		}

		SUPERLU_FREE(lusup);
		SUPERLU_FREE(index);

		Lrowind_bc_ptr[ljb_j] = index_srt;
		Lnzval_bc_ptr[ljb_j] = lusup_srt;
	} else {
	  Lrowind_bc_ptr[ljb_j] = NULL;
	  Lnzval_bc_ptr[ljb_j] = NULL;
	  Linv_bc_ptr[ljb_j] = NULL;
	  Uinv_bc_ptr[ljb_j] = NULL;
	  Lindval_loc_bc_ptr[ljb_j] = NULL;
	} /* if nrbl ... */
      } /* if mycol == pc */
  } /* for jb ... */

  SUPERLU_FREE(ilsum_j);
  SUPERLU_FREE(Urb_marker);
  SUPERLU_FREE(LUb_length);
  SUPERLU_FREE(LUb_indptr);
  SUPERLU_FREE(LUb_number);
  SUPERLU_FREE(LUb_valptr);
  SUPERLU_FREE(Lrb_marker);
  SUPERLU_FREE(dense);

  /* Free the memory used for storing A */
  SUPERLU_FREE(ainf_colptr);
  if (ainf_rowind != NULL) {
    SUPERLU_FREE(ainf_rowind);
    SUPERLU_FREE(ainf_val);
  }
  SUPERLU_FREE(asup_rowptr);
  if (asup_colind != NULL) {
    SUPERLU_FREE(asup_colind);
    SUPERLU_FREE(asup_val);
  }

  /* exchange information about bsendx_plist in between column of processors */
  k = SUPERLU_MAX( grid->nprow, grid->npcol);
  if ( !(recvBuf = (int *) SUPERLU_MALLOC(nsupers*k* sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for recvBuf[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(nnzToRecv = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for nnzToRecv[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(ptrToRecv = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for ptrToRecv[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(nnzToSend = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for nnzToRecv[].");
    return (memDist + memNLU + memTRS);
  }
  if ( !(ptrToSend = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for ptrToRecv[].");
    return (memDist + memNLU + memTRS);
  }

  if (memDist < (nsupers*k*iword +4*nprocs * sizeof(int)))
    memDist = nsupers*k*iword +4*nprocs * sizeof(int);

  for (p = 0; p < nprocs; p++)
    nnzToRecv[p] = 0;

  for (jb = 0; jb < nsupers; jb++) {
    jbcol = PCOL( jb, grid );
    jbrow = PROW( jb, grid );
    p = PNUM(jbrow, jbcol, grid);
    nnzToRecv[p] += grid->npcol;
  }
  i = 0;
  for (p = 0; p < nprocs; p++) {
    ptrToRecv[p] = i;
    i += nnzToRecv[p];
    ptrToSend[p] = 0;
    if (p != iam)
      nnzToSend[p] = nnzToRecv[iam];
    else
      nnzToSend[p] = 0;
  }
  nnzToRecv[iam] = 0;
  i = ptrToRecv[iam];
  for (jb = 0; jb < nsupers; jb++) {
    jbcol = PCOL( jb, grid );
    jbrow = PROW( jb, grid );
    p = PNUM(jbrow, jbcol, grid);
    if (p == iam) {
      ljb_j = LBj( jb, grid ); /* Local block number column wise */
      for (j = 0; j < grid->npcol; j++, i++)
	recvBuf[i] = ToSendR[ljb_j][j];
    }
  }

#if 0 // Sherry 
  MPI_Alltoallv (&(recvBuf[ptrToRecv[iam]]), nnzToSend, ptrToSend, mpi_int_t,
		 recvBuf, nnzToRecv, ptrToRecv, mpi_int_t, grid->comm);
#else		 
  MPI_Alltoallv (&(recvBuf[ptrToRecv[iam]]), nnzToSend, ptrToSend, MPI_INT,
		 recvBuf, nnzToRecv, ptrToRecv, MPI_INT, grid->comm);
#endif

  for (jb = 0; jb < nsupers; jb++) {
    jbcol = PCOL( jb, grid );
    jbrow = PROW( jb, grid );
    p = PNUM(jbrow, jbcol, grid);
    ljb_j = LBj( jb, grid ); /* Local block number column wise */
    ljb_i = LBi( jb, grid ); /* Local block number row wise */
    /* (myrow == jbrow) {
       if (ToSendD[ljb_i] == YES)
       ToRecv[jb] = 1;
       }
       else {
       if (recvBuf[ptrToRecv[p] + mycol] == YES)
       ToRecv[jb] = 2;
       } */
    if (recvBuf[ptrToRecv[p] + mycol] == YES) {
      if (myrow == jbrow)
	ToRecv[jb] = 1;
      else
	ToRecv[jb] = 2;
    }
    if (mycol == jbcol) {
      for (i = 0, j = ptrToRecv[p]; i < grid->npcol; i++, j++)
	ToSendR[ljb_j][i] = recvBuf[j];
      ToSendR[ljb_j][mycol] = EMPTY;
    }
    ptrToRecv[p] += grid->npcol;
  }

  /* exchange information about bsendx_plist in between column of processors */
#if 0 // Sherry 1/16/2022
  MPI_Allreduce ((*bsendx_plist), recvBuf, nsupers_j * grid->nprow, mpi_int_t,
		 MPI_MAX, grid->cscp.comm);
#else
  MPI_Allreduce ((*bsendx_plist), recvBuf, nsupers_j * grid->nprow, MPI_INT,
		 MPI_MAX, grid->cscp.comm);
#endif

  for (jb = 0; jb < nsupers; jb ++) {
    jbcol = PCOL( jb, grid);
    jbrow = PROW( jb, grid);
    if (mycol == jbcol) {
      ljb_j = LBj( jb, grid ); /* Local block number column wise */
      if (myrow == jbrow ) {
	for (k = ljb_j * grid->nprow; k < (ljb_j+1) * grid->nprow; k++) {
	  (*bsendx_plist)[k] = recvBuf[k];
	  if ((*bsendx_plist)[k] != EMPTY)
	    nbsendx ++;
	}
      }
      else {
	for (k = ljb_j * grid->nprow; k < (ljb_j+1) * grid->nprow; k++)
	  (*bsendx_plist)[k] = EMPTY;
      }
    }
  }

		/////////////////////////////////////////////////////////////////

		/* Set up additional pointers for the index and value arrays of U.
		   nub is the number of local block columns. */
		nub = CEILING( nsupers, grid->npcol); /* Number of local block columns. */
		if ( !(Urbs = (int_t *) intCalloc_dist(2*nub)) )
			ABORT("Malloc fails for Urbs[]"); /* Record number of nonzero
								 blocks in a block column. */
		Urbs1 = Urbs + nub;
		if ( !(Ucb_indptr = SUPERLU_MALLOC(nub * sizeof(Ucb_indptr_t *))) )
			ABORT("Malloc fails for Ucb_indptr[]");
		if ( !(Ucb_valptr = SUPERLU_MALLOC(nub * sizeof(int_t *))) )
			ABORT("Malloc fails for Ucb_valptr[]");
		nlb = CEILING( nsupers, grid->nprow ); /* Number of local block rows. */

		/* Count number of row blocks in a block column.
		   One pass of the skeleton graph of U. */
		for (lk = 0; lk < nlb; ++lk) {
			usub1 = Ufstnz_br_ptr[lk];
			if ( usub1 ) { /* Not an empty block row. */
				/* usub1[0] -- number of column blocks in this block row. */
				i = BR_HEADER; /* Pointer in index array. */
				for (lb = 0; lb < usub1[0]; ++lb) { /* For all column blocks. */
					k = usub1[i];            /* Global block number */
					++Urbs[LBj(k,grid)];
					i += UB_DESCRIPTOR + SuperSize( k );
				}
			}
		}

		/* Set up the vertical linked lists for the row blocks.
		   One pass of the skeleton graph of U. */
		for (lb = 0; lb < nub; ++lb) {
			if ( Urbs[lb] ) { /* Not an empty block column. */
				if ( !(Ucb_indptr[lb]
							= SUPERLU_MALLOC(Urbs[lb] * sizeof(Ucb_indptr_t))) )
					ABORT("Malloc fails for Ucb_indptr[lb][]");
				if ( !(Ucb_valptr[lb] = (int_t *) intMalloc_dist(Urbs[lb])) )
					ABORT("Malloc fails for Ucb_valptr[lb][]");
			}
		}
		for (lk = 0; lk < nlb; ++lk) { /* For each block row. */
			usub1 = Ufstnz_br_ptr[lk];
			if ( usub1 ) { /* Not an empty block row. */
				i = BR_HEADER; /* Pointer in index array. */
				j = 0;         /* Pointer in nzval array. */

				for (lb = 0; lb < usub1[0]; ++lb) { /* For all column blocks. */
					k = usub1[i];          /* Global block number, column-wise. */
					ljb = LBj( k, grid ); /* Local block number, column-wise. */
					Ucb_indptr[ljb][Urbs1[ljb]].lbnum = lk;

					Ucb_indptr[ljb][Urbs1[ljb]].indpos = i;
					Ucb_valptr[ljb][Urbs1[ljb]] = j;

					++Urbs1[ljb];
					j += usub1[i+1];
					i += UB_DESCRIPTOR + SuperSize( k );
				}
			}
		}



/* Count the nnzs per block column */
	for (lb = 0; lb < nub; ++lb) {
		Unnz[lb] = 0;
		k = lb * grid->npcol + mycol;/* Global block number, column-wise. */
		knsupc = SuperSize( k );
		for (ub = 0; ub < Urbs[lb]; ++ub) {
			ik = Ucb_indptr[lb][ub].lbnum; /* Local block number, row-wise. */
			i = Ucb_indptr[lb][ub].indpos; /* Start of the block in usub[]. */
			i += UB_DESCRIPTOR;
			gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
			iklrow = FstBlockC( gik+1 );
			for (jj = 0; jj < knsupc; ++jj) {
				fnz = Ufstnz_br_ptr[ik][i + jj];
				if ( fnz < iklrow ) {
					Unnz[lb] +=iklrow-fnz;
				}
			} /* for jj ... */
		}
	}

		/////////////////////////////////////////////////////////////////

		// if(LSUM<nsupers)ABORT("Need increase LSUM."); /* temporary*/

#if ( PROFlevel>=1 )
			t = SuperLU_timer_();
#endif
		/* construct the Bcast tree for L ... */

		k = CEILING( nsupers, grid->npcol );/* Number of local block columns */
		if ( !(LBtree_ptr = (C_Tree*)SUPERLU_MALLOC(k * sizeof(C_Tree))) )
			ABORT("Malloc fails for LBtree_ptr[].");
		if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
			ABORT("Calloc fails for ActiveFlag[].");
		if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
			ABORT("Malloc fails for ranks[].");
		if ( !(SeedSTD_BC = (double*)SUPERLU_MALLOC(k * sizeof(double))) )
			ABORT("Malloc fails for SeedSTD_BC[].");

		for (i=0;i<k;i++){
			SeedSTD_BC[i]=rand();
		}

		MPI_Allreduce(MPI_IN_PLACE,&SeedSTD_BC[0],k,MPI_DOUBLE,MPI_MAX,grid->cscp.comm);

		for (ljb = 0; ljb <k ; ++ljb) {
			C_BcTree_Nullify(&LBtree_ptr[ljb]);
		}


		if ( !(ActiveFlagAll = intMalloc_dist(grid->nprow*k)) )
			ABORT("Calloc fails for ActiveFlag[].");
		for (j=0;j<grid->nprow*k;++j)ActiveFlagAll[j]=3*nsupers;
		memTRS += k*sizeof(C_Tree) + k*dword + grid->nprow*k*iword;  //acount for LBtree_ptr, SeedSTD_BC, ActiveFlagAll
		for (ljb = 0; ljb < k; ++ljb) { /* for each local block column ... */
			jb = mycol+ljb*grid->npcol;  /* not sure */
			if(jb<nsupers){
			pc = PCOL( jb, grid );

			istart = xlsub[ljb];
			for (i = istart; i < xlsub[ljb+1]; ++i) {
				irow = lsub[i];
				gb = BlockNum( irow );
				pr = PROW( gb, grid );
				ActiveFlagAll[pr+ljb*grid->nprow]=SUPERLU_MIN(ActiveFlagAll[pr+ljb*grid->nprow],gb);
			} /* for j ... */
			}
		}


		MPI_Allreduce(MPI_IN_PLACE,ActiveFlagAll,grid->nprow*k,mpi_int_t,MPI_MIN,grid->cscp.comm);



		for (ljb = 0; ljb < k; ++ljb) { /* for each local block column ... */

			jb = mycol+ljb*grid->npcol;  /* not sure */
			if(jb<nsupers){
			pc = PCOL( jb, grid );

			for (j=0;j<grid->nprow;++j)ActiveFlag[j]=ActiveFlagAll[j+ljb*grid->nprow];
			for (j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
			for (j=0;j<grid->nprow;++j)ranks[j]=-1;

			Root=-1;
			Iactive = 0;
			for (j=0;j<grid->nprow;++j){
				if(ActiveFlag[j]!=3*nsupers){
				gb = ActiveFlag[j];
				pr = PROW( gb, grid );
				if(gb==jb)Root=pr;
				if(myrow==pr)Iactive=1;
				}
			}


			quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,0,2);

			if(Iactive==1){
				// printf("jb %5d damn\n",jb);
				// fflush(stdout);
				assert( Root>-1 );
				rank_cnt = 1;
				ranks[0]=Root;
				for (j = 0; j < grid->nprow; ++j){
					if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
						ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
						++rank_cnt;
					}
				}

				if(rank_cnt>1){

					for (ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
						ranks[ii] = PNUM( ranks[ii], pc, grid );

					// rseed=rand();
					// rseed=1.0;
					msgsize = SuperSize( jb );
				//LBtree_ptr[ljb] = BcTree_Create(grid->comm, ranks, rank_cnt, msgsize,SeedSTD_BC[ljb],'z');
				//BcTree_SetTag(LBtree_ptr[ljb],BC_L,'z');
				C_BcTree_Create(&LBtree_ptr[ljb], grid->comm, ranks, rank_cnt, msgsize, 'z');
				LBtree_ptr[ljb].tag_=BC_L;

					// printf("iam %5d btree rank_cnt %5d \n",iam,rank_cnt);
					// fflush(stdout);

					// if(iam==15 || iam==3){
					// printf("iam %5d btree lk %5d tag %5d root %5d\n",iam, ljb,jb,BcTree_IsRoot(LBtree_ptr[ljb],'z'));
					// fflush(stdout);
					// }

					// #if ( PRNTlevel>=1 )
					if(Root==myrow){
						rank_cnt_ref=1;
						for (j = 0; j < grid->nprow; ++j) {
							if ( fsendx_plist[ljb][j] != EMPTY ) {
								++rank_cnt_ref;
							}
						}
						assert(rank_cnt==rank_cnt_ref);

						// printf("Partial Bcast Procs: col%7d np%4d\n",jb,rank_cnt);

						// // printf("Partial Bcast Procs: %4d %4d: ",iam, rank_cnt);
						// // for(j=0;j<rank_cnt;++j)printf("%4d",ranks[j]);
						// // printf("\n");
					}
					// #endif
				}
			}
			}
		}


		SUPERLU_FREE(ActiveFlag);
		SUPERLU_FREE(ActiveFlagAll);
		SUPERLU_FREE(ranks);
		SUPERLU_FREE(SeedSTD_BC);
		memTRS -= k*dword + grid->nprow*k*iword;  //acount for SeedSTD_BC, ActiveFlagAll

#if ( PROFlevel>=1 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Construct Bcast tree for L: %.2f\t\n", t);
#endif


#if ( PROFlevel>=1 )
			t = SuperLU_timer_();
#endif
		/* construct the Reduce tree for L ... */
		/* the following is used as reference */
		nlb = CEILING( nsupers, grid->nprow );/* Number of local block rows */
		if ( !(mod_bit = int32Malloc_dist(nlb)) )
			ABORT("Malloc fails for mod_bit[].");
		if ( !(frecv = int32Malloc_dist(nlb)) )
			ABORT("Malloc fails for frecv[].");

		for (k = 0; k < nlb; ++k) mod_bit[k] = 0;
		for (k = 0; k < nsupers; ++k) {
			pr = PROW( k, grid );
			if ( myrow == pr ) {
				lib = LBi( k, grid );    /* local block number */
				kcol = PCOL( k, grid );
				if (mycol == kcol || fmod[lib] )
					mod_bit[lib] = 1;  /* contribution from off-diagonal and diagonal*/
			}
		}
		/* Every process receives the count, but it is only useful on the
		   diagonal processes.  */
#if 0 // Sherry
		MPI_Allreduce( mod_bit, frecv, nlb, mpi_int_t, MPI_SUM, grid->rscp.comm);
#else		
		MPI_Allreduce( mod_bit, frecv, nlb, MPI_INT, MPI_SUM, grid->rscp.comm);
#endif

		k = CEILING( nsupers, grid->nprow );/* Number of local block rows */
		if ( !(LRtree_ptr = (C_Tree*)SUPERLU_MALLOC(k * sizeof(C_Tree))) )
			ABORT("Malloc fails for LRtree_ptr[].");
		if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
			ABORT("Calloc fails for ActiveFlag[].");
		if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
			ABORT("Malloc fails for ranks[].");

		// if ( !(idxs = intCalloc_dist(nsupers)) )
			// ABORT("Calloc fails for idxs[].");

		// if ( !(nzrows = (int_t**)SUPERLU_MALLOC(nsupers * sizeof(int_t*))) )
			// ABORT("Malloc fails for nzrows[].");

		if ( !(SeedSTD_RD = (double*)SUPERLU_MALLOC(k * sizeof(double))) )
			ABORT("Malloc fails for SeedSTD_RD[].");

		for (i=0;i<k;i++){
			SeedSTD_RD[i]=rand();
		}

		MPI_Allreduce(MPI_IN_PLACE,&SeedSTD_RD[0],k,MPI_DOUBLE,MPI_MAX,grid->rscp.comm);


		for (lib = 0; lib <k ; ++lib) {
			C_RdTree_Nullify(&LRtree_ptr[lib]);
		}


		if ( !(ActiveFlagAll = intMalloc_dist(grid->npcol*k)) )
			ABORT("Calloc fails for ActiveFlagAll[].");
		for (j=0;j<grid->npcol*k;++j)ActiveFlagAll[j]=-3*nsupers;
		memTRS += k*sizeof(C_Tree) + k*dword + grid->npcol*k*iword;  //acount for LRtree_ptr, SeedSTD_RD, ActiveFlagAll


		for (ljb = 0; ljb < CEILING( nsupers, grid->npcol); ++ljb) { /* for each local block column ... */
			jb = mycol+ljb*grid->npcol;  /* not sure */
			if(jb<nsupers){
				pc = PCOL( jb, grid );
				for(i=xlsub[ljb];i<xlsub[ljb+1];++i){
					irow = lsub[i];
					ib = BlockNum( irow );
					pr = PROW( ib, grid );
					if ( myrow == pr ) { /* Block row ib in my process row */
						lib = LBi( ib, grid ); /* Local block number */
						ActiveFlagAll[pc+lib*grid->npcol]=SUPERLU_MAX(ActiveFlagAll[pc+lib*grid->npcol],jb);
					}
				}
			}
		}

		MPI_Allreduce(MPI_IN_PLACE,ActiveFlagAll,grid->npcol*k,mpi_int_t,MPI_MAX,grid->rscp.comm);

		for (lib=0;lib<k;++lib){
			ib = myrow+lib*grid->nprow;  /* not sure */
			if(ib<nsupers){
				pr = PROW( ib, grid );
				for (j=0;j<grid->npcol;++j)ActiveFlag[j]=ActiveFlagAll[j+lib*grid->npcol];;
				for (j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
				for (j=0;j<grid->npcol;++j)ranks[j]=-1;
				Root=-1;
				Iactive = 0;

				for (j=0;j<grid->npcol;++j){
					if(ActiveFlag[j]!=-3*nsupers){
					jb = ActiveFlag[j];
					pc = PCOL( jb, grid );
					if(jb==ib)Root=pc;
					if(mycol==pc)Iactive=1;
					}
				}


				quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,1,2);

				if(Iactive==1){
					assert( Root>-1 );
					rank_cnt = 1;
					ranks[0]=Root;
					for (j = 0; j < grid->npcol; ++j){
						if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
							ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
							++rank_cnt;
						}
					}
					if(rank_cnt>1){

						for (ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
							ranks[ii] = PNUM( pr, ranks[ii], grid );

						// rseed=rand();
						// rseed=1.0;
						msgsize = SuperSize( ib );

						// if(ib==0){

					//LRtree_ptr[lib] = RdTree_Create(grid->comm, ranks, rank_cnt, msgsize,SeedSTD_RD[lib],'z');
					//RdTree_SetTag(LRtree_ptr[lib], RD_L,'z');
					C_RdTree_Create(&LRtree_ptr[lib], grid->comm, ranks, rank_cnt, msgsize, 'z');
					LRtree_ptr[lib].tag_=RD_L;
						// }

						// printf("iam %5d rtree rank_cnt %5d \n",iam,rank_cnt);
						// fflush(stdout);


						#if ( PRNTlevel>=1 )
						if(Root==mycol){
						assert(rank_cnt==frecv[lib]);
						// printf("Partial Reduce Procs: row%7d np%4d\n",ib,rank_cnt);
						// printf("Partial Reduce Procs: %4d %4d: ",iam, rank_cnt);
						// // for(j=0;j<rank_cnt;++j)printf("%4d",ranks[j]);
						// printf("\n");
						}
						#endif
					}
				}
			}
		}

		SUPERLU_FREE(mod_bit);
		SUPERLU_FREE(frecv);


		SUPERLU_FREE(ActiveFlag);
		SUPERLU_FREE(ActiveFlagAll);
		SUPERLU_FREE(ranks);
		// SUPERLU_FREE(idxs);
		SUPERLU_FREE(SeedSTD_RD);
		// for(i=0;i<nsupers;++i){
			// if(nzrows[i])SUPERLU_FREE(nzrows[i]);
		// }
		// SUPERLU_FREE(nzrows);
		memTRS -= k*dword + grid->nprow*k*iword;  //acount for SeedSTD_RD, ActiveFlagAll
			////////////////////////////////////////////////////////

#if ( PROFlevel>=1 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Construct Reduce tree for L: %.2f\t\n", t);
#endif

#if ( PROFlevel>=1 )
		t = SuperLU_timer_();
#endif

		/* construct the Bcast tree for U ... */

		k = CEILING( nsupers, grid->npcol );/* Number of local block columns */
		if ( !(UBtree_ptr = (C_Tree*)SUPERLU_MALLOC(k * sizeof(C_Tree))) )
			ABORT("Malloc fails for UBtree_ptr[].");
		if ( !(ActiveFlag = intCalloc_dist(grid->nprow*2)) )
			ABORT("Calloc fails for ActiveFlag[].");
		if ( !(ranks = (int*)SUPERLU_MALLOC(grid->nprow * sizeof(int))) )
			ABORT("Malloc fails for ranks[].");
		if ( !(SeedSTD_BC = (double*)SUPERLU_MALLOC(k * sizeof(double))) )
			ABORT("Malloc fails for SeedSTD_BC[].");

		for (i=0;i<k;i++){
			SeedSTD_BC[i]=rand();
		}

		MPI_Allreduce(MPI_IN_PLACE,&SeedSTD_BC[0],k,MPI_DOUBLE,MPI_MAX,grid->cscp.comm);


		for (ljb = 0; ljb <k ; ++ljb) {
			C_BcTree_Nullify(&UBtree_ptr[ljb]);
		}

		if ( !(ActiveFlagAll = intMalloc_dist(grid->nprow*k)) )
			ABORT("Calloc fails for ActiveFlagAll[].");
		for (j=0;j<grid->nprow*k;++j)ActiveFlagAll[j]=-3*nsupers;
		memTRS += k*sizeof(C_Tree) + k*dword + grid->nprow*k*iword;  //acount for UBtree_ptr, SeedSTD_BC, ActiveFlagAll


		for (lib = 0; lib < CEILING( nsupers, grid->nprow); ++lib) { /* for each local block row ... */
			ib = myrow+lib*grid->nprow;  /* not sure */

		// if(ib==0)printf("iam %5d ib %5d\n",iam,ib);
		// fflush(stdout);

			if(ib<nsupers){
				for (i = xusub[lib]; i < xusub[lib+1]; i++) {
				  jcol = usub[i];
				  jb = BlockNum( jcol );
				  ljb = LBj( jb, grid );    /* local block number */
				  pc = PCOL( jb, grid );
				  pr = PROW( ib, grid );
				  if ( mycol == pc ) { /* Block column ib in my process column */
					ActiveFlagAll[pr+ljb*grid->nprow]=SUPERLU_MAX(ActiveFlagAll[pr+ljb*grid->nprow],ib);
				  }
				}  /* for i ... */
				pr = PROW( ib, grid ); // take care of diagonal node stored as L
				pc = PCOL( ib, grid );
				if ( mycol == pc ) { /* Block column ib in my process column */
					ljb = LBj( ib, grid );    /* local block number */
					ActiveFlagAll[pr+ljb*grid->nprow]=SUPERLU_MAX(ActiveFlagAll[pr+ljb*grid->nprow],ib);
					// if(pr+ljb*grid->nprow==0)printf("iam %5d ib %5d ActiveFlagAll %5d pr %5d ljb %5d\n",iam,ib,ActiveFlagAll[pr+ljb*grid->nprow],pr,ljb);
					// fflush(stdout);
				}
			}
		}

		// printf("iam %5d ActiveFlagAll %5d\n",iam,ActiveFlagAll[0]);
		// fflush(stdout);

		MPI_Allreduce(MPI_IN_PLACE,ActiveFlagAll,grid->nprow*k,mpi_int_t,MPI_MAX,grid->cscp.comm);

		for (ljb = 0; ljb < k; ++ljb) { /* for each block column ... */
			jb = mycol+ljb*grid->npcol;  /* not sure */
			if(jb<nsupers){
			pc = PCOL( jb, grid );
			// if ( mycol == pc ) { /* Block column jb in my process column */

			for (j=0;j<grid->nprow;++j)ActiveFlag[j]=ActiveFlagAll[j+ljb*grid->nprow];
			for (j=0;j<grid->nprow;++j)ActiveFlag[j+grid->nprow]=j;
			for (j=0;j<grid->nprow;++j)ranks[j]=-1;

			Root=-1;
			Iactive = 0;
			for (j=0;j<grid->nprow;++j){
				if(ActiveFlag[j]!=-3*nsupers){
				gb = ActiveFlag[j];
				pr = PROW( gb, grid );
				if(gb==jb)Root=pr;
				if(myrow==pr)Iactive=1;
				}
			}

			quickSortM(ActiveFlag,0,grid->nprow-1,grid->nprow,1,2);
		// printf("jb: %5d Iactive %5d\n",jb,Iactive);
		// fflush(stdout);
			if(Iactive==1){
				// if(jb==0)printf("root:%5d jb: %5d ActiveFlag %5d \n",Root,jb,ActiveFlag[0]);
				fflush(stdout);
				assert( Root>-1 );
				rank_cnt = 1;
				ranks[0]=Root;
				for (j = 0; j < grid->nprow; ++j){
					if(ActiveFlag[j]!=-3*nsupers && ActiveFlag[j+grid->nprow]!=Root){
						ranks[rank_cnt]=ActiveFlag[j+grid->nprow];
						++rank_cnt;
					}
				}
		// printf("jb: %5d rank_cnt %5d\n",jb,rank_cnt);
		// fflush(stdout);
				if(rank_cnt>1){
					for (ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
						ranks[ii] = PNUM( ranks[ii], pc, grid );

					// rseed=rand();
					// rseed=1.0;
					msgsize = SuperSize( jb );
				//UBtree_ptr[ljb] = BcTree_Create(grid->comm, ranks, rank_cnt, msgsize,SeedSTD_BC[ljb],'z');
				//BcTree_SetTag(UBtree_ptr[ljb],BC_U,'z');
				C_BcTree_Create(&UBtree_ptr[ljb], grid->comm, ranks, rank_cnt, msgsize, 'z');
				UBtree_ptr[ljb].tag_=BC_U;

					// printf("iam %5d btree rank_cnt %5d \n",iam,rank_cnt);
					// fflush(stdout);

					if(Root==myrow){
					rank_cnt_ref=1;
					for (j = 0; j < grid->nprow; ++j) {
						// printf("ljb %5d j %5d nprow %5d\n",ljb,j,grid->nprow);
						// fflush(stdout);
						if ( bsendx_plist[ljb][j] != EMPTY ) {
							++rank_cnt_ref;
						}
					}
					// printf("ljb %5d rank_cnt %5d rank_cnt_ref %5d\n",ljb,rank_cnt,rank_cnt_ref);
					// fflush(stdout);
					assert(rank_cnt==rank_cnt_ref);
					}
				}
			}
			}
		}
		SUPERLU_FREE(ActiveFlag);
		SUPERLU_FREE(ActiveFlagAll);
		SUPERLU_FREE(ranks);
		SUPERLU_FREE(SeedSTD_BC);
		memTRS -= k*dword + grid->nprow*k*iword;  //acount for SeedSTD_BC, ActiveFlagAll

#if ( PROFlevel>=1 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Construct Bcast tree for U: %.2f\t\n", t);
#endif

#if ( PROFlevel>=1 )
			t = SuperLU_timer_();
#endif
		/* construct the Reduce tree for U ... */
		/* the following is used as reference */
		nlb = CEILING( nsupers, grid->nprow );/* Number of local block rows */
		if ( !(mod_bit = int32Malloc_dist(nlb)) )
			ABORT("Malloc fails for mod_bit[].");
		if ( !(brecv = int32Malloc_dist(nlb)) )
			ABORT("Malloc fails for brecv[].");

		for (k = 0; k < nlb; ++k) mod_bit[k] = 0;
		for (k = 0; k < nsupers; ++k) {
			pr = PROW( k, grid );
			if ( myrow == pr ) {
				lib = LBi( k, grid );    /* local block number */
				kcol = PCOL( k, grid );
				if (mycol == kcol || bmod[lib] )
					mod_bit[lib] = 1;  /* contribution from off-diagonal and diagonal*/
			}
		}
		/* Every process receives the count, but it is only useful on the
		   diagonal processes.  */
#if 0 // Sherry
		MPI_Allreduce( mod_bit, brecv, nlb, mpi_int_t, MPI_SUM, grid->rscp.comm);
#else		
		MPI_Allreduce( mod_bit, brecv, nlb, MPI_INT, MPI_SUM, grid->rscp.comm);
#endif		

		k = CEILING( nsupers, grid->nprow );/* Number of local block rows */
		if ( !(URtree_ptr = (C_Tree*)SUPERLU_MALLOC(k * sizeof(C_Tree))) )
			ABORT("Malloc fails for URtree_ptr[].");
		if ( !(ActiveFlag = intCalloc_dist(grid->npcol*2)) )
			ABORT("Calloc fails for ActiveFlag[].");
		if ( !(ranks = (int*)SUPERLU_MALLOC(grid->npcol * sizeof(int))) )
			ABORT("Malloc fails for ranks[].");

		// if ( !(idxs = intCalloc_dist(nsupers)) )
			// ABORT("Calloc fails for idxs[].");

		// if ( !(nzrows = (int_t**)SUPERLU_MALLOC(nsupers * sizeof(int_t*))) )
			// ABORT("Malloc fails for nzrows[].");

		if ( !(SeedSTD_RD = (double*)SUPERLU_MALLOC(k * sizeof(double))) )
			ABORT("Malloc fails for SeedSTD_RD[].");

		for (i=0;i<k;i++){
			SeedSTD_RD[i]=rand();
		}

		MPI_Allreduce(MPI_IN_PLACE,&SeedSTD_RD[0],k,MPI_DOUBLE,MPI_MAX,grid->rscp.comm);

		for (lib = 0; lib <k ; ++lib) {
			C_RdTree_Nullify(&URtree_ptr[lib]);
		}


		if ( !(ActiveFlagAll = intMalloc_dist(grid->npcol*k)) )
			ABORT("Calloc fails for ActiveFlagAll[].");
		for (j=0;j<grid->npcol*k;++j)ActiveFlagAll[j]=3*nsupers;
		memTRS += k*sizeof(C_Tree) + k*dword + grid->npcol*k*iword;  //acount for URtree_ptr, SeedSTD_RD, ActiveFlagAll

		for (lib = 0; lib < CEILING( nsupers, grid->nprow); ++lib) { /* for each local block row ... */
			ib = myrow+lib*grid->nprow;  /* not sure */
			if(ib<nsupers){
				for (i = xusub[lib]; i < xusub[lib+1]; i++) {
				  jcol = usub[i];
				  jb = BlockNum( jcol );
				  pc = PCOL( jb, grid );
				  if ( mycol == pc ) { /* Block column ib in my process column */
					ActiveFlagAll[pc+lib*grid->npcol]=SUPERLU_MIN(ActiveFlagAll[pc+lib*grid->npcol],jb);
				  }
				}  /* for i ... */
				pc = PCOL( ib, grid );
				if ( mycol == pc ) { /* Block column ib in my process column */
					ActiveFlagAll[pc+lib*grid->npcol]=SUPERLU_MIN(ActiveFlagAll[pc+lib*grid->npcol],ib);
				}
			}
		}

		MPI_Allreduce(MPI_IN_PLACE,ActiveFlagAll,grid->npcol*k,mpi_int_t,MPI_MIN,grid->rscp.comm);

		for (lib=0;lib<k;++lib){
			ib = myrow+lib*grid->nprow;  /* not sure */
			if(ib<nsupers){
				pr = PROW( ib, grid );
				for (j=0;j<grid->npcol;++j)ActiveFlag[j]=ActiveFlagAll[j+lib*grid->npcol];;
				for (j=0;j<grid->npcol;++j)ActiveFlag[j+grid->npcol]=j;
				for (j=0;j<grid->npcol;++j)ranks[j]=-1;
				Root=-1;
				Iactive = 0;

				for (j=0;j<grid->npcol;++j){
					if(ActiveFlag[j]!=3*nsupers){
					jb = ActiveFlag[j];
					pc = PCOL( jb, grid );
					if(jb==ib)Root=pc;
					if(mycol==pc)Iactive=1;
					}
				}

				quickSortM(ActiveFlag,0,grid->npcol-1,grid->npcol,0,2);

				if(Iactive==1){
					assert( Root>-1 );
					rank_cnt = 1;
					ranks[0]=Root;
					for (j = 0; j < grid->npcol; ++j){
						if(ActiveFlag[j]!=3*nsupers && ActiveFlag[j+grid->npcol]!=Root){
							ranks[rank_cnt]=ActiveFlag[j+grid->npcol];
							++rank_cnt;
						}
					}
					if(rank_cnt>1){

						for (ii=0;ii<rank_cnt;ii++)   // use global ranks rather than local ranks
							ranks[ii] = PNUM( pr, ranks[ii], grid );

						// rseed=rand();
						// rseed=1.0;
						msgsize = SuperSize( ib );

						// if(ib==0){

					//URtree_ptr[lib] = RdTree_Create(grid->comm, ranks, rank_cnt, msgsize,SeedSTD_RD[lib],'z');
					//RdTree_SetTag(URtree_ptr[lib], RD_U,'z');
					C_RdTree_Create(&URtree_ptr[lib], grid->comm, ranks, rank_cnt, msgsize, 'z');
					URtree_ptr[lib].tag_=RD_U;
						// }

						// #if ( PRNTlevel>=1 )
						if(Root==mycol){
						// printf("Partial Reduce Procs: %4d %4d %5d \n",iam, rank_cnt,brecv[lib]);
						// fflush(stdout);
						assert(rank_cnt==brecv[lib]);
						// printf("Partial Reduce Procs: row%7d np%4d\n",ib,rank_cnt);
						// printf("Partial Reduce Procs: %4d %4d: ",iam, rank_cnt);
						// // for(j=0;j<rank_cnt;++j)printf("%4d",ranks[j]);
						// printf("\n");
						}
						// #endif
					}
				}
			}
		}

		SUPERLU_FREE(mod_bit);
		SUPERLU_FREE(brecv);


		SUPERLU_FREE(ActiveFlag);
		SUPERLU_FREE(ActiveFlagAll);
		SUPERLU_FREE(ranks);
		// SUPERLU_FREE(idxs);
		SUPERLU_FREE(SeedSTD_RD);
		// for(i=0;i<nsupers;++i){
			// if(nzrows[i])SUPERLU_FREE(nzrows[i]);
		// }
		// SUPERLU_FREE(nzrows);
		memTRS -= k*dword + grid->nprow*k*iword;  //acount for SeedSTD_RD, ActiveFlagAll

#if ( PROFlevel>=1 )
	t = SuperLU_timer_() - t;
	if ( !iam) printf(".. Construct Reduce tree for U: %.2f\t\n", t);
#endif

	////////////////////////////////////////////////////////

  /* Free the memory used for storing L and U */
  SUPERLU_FREE(xlsub); SUPERLU_FREE(xusub);
  if (lsub != NULL)
    SUPERLU_FREE(lsub);
  if (usub != NULL)
    SUPERLU_FREE(usub);


  SUPERLU_FREE(nnzToRecv);
  SUPERLU_FREE(ptrToRecv);
  SUPERLU_FREE(nnzToSend);
  SUPERLU_FREE(ptrToSend);
  SUPERLU_FREE(recvBuf);

  Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
  Llu->Lindval_loc_bc_ptr = Lindval_loc_bc_ptr;
  Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
  Llu->Linv_bc_ptr = Linv_bc_ptr;
  Llu->Uinv_bc_ptr = Uinv_bc_ptr;
  Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
  Llu->Unzval_br_ptr = Unzval_br_ptr;
  Llu->Unnz = Unnz;
  Llu->ToRecv = ToRecv;
  Llu->ToSendD = ToSendD;
  Llu->ToSendR = ToSendR;
  Llu->fmod = fmod;
  Llu->fsendx_plist = fsendx_plist;
  Llu->nfrecvx = nfrecvx;
  Llu->nfsendx = nfsendx;
  Llu->bmod = bmod;
  Llu->bsendx_plist = bsendx_plist;
  Llu->nbrecvx = nbrecvx;
  Llu->nbsendx = nbsendx;
  Llu->ilsum = ilsum;
  Llu->ldalsum = ldaspa;
  LUstruct->Glu_persist = Glu_persist;
  Llu->LRtree_ptr = LRtree_ptr;
  Llu->LBtree_ptr = LBtree_ptr;
  Llu->URtree_ptr = URtree_ptr;
  Llu->UBtree_ptr = UBtree_ptr;
  Llu->Urbs = Urbs;
  Llu->Ucb_indptr = Ucb_indptr;
  Llu->Ucb_valptr = Ucb_valptr;

#if ( PRNTlevel>=1 )
  if ( !iam ) printf(".. # L blocks " IFMT "\t# U blocks " IFMT "\n",
		     nLblocks, nUblocks);
#endif

  k = CEILING( nsupers, grid->nprow );/* Number of local block rows */
  if ( !(Llu->mod_bit = int32Malloc_dist(k)) )
      ABORT("Malloc fails for mod_bit[].");

  /* Find the maximum buffer size. */
  MPI_Allreduce(mybufmax, Llu->bufmax, NBUFFERS, mpi_int_t,
		MPI_MAX, grid->comm);

#if ( DEBUGlevel>=1 )
  /* Memory allocated but not freed:
     ilsum, fmod, fsendx_plist, bmod, bsendx_plist,
     ToRecv, ToSendR, ToSendD, mod_bit
  */
  CHECK_MALLOC(iam, "Exit dist_psymbtonum()");
#endif

  return (- (memDist+memNLU));
} /* zdist_psymbtonum */

