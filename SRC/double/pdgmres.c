/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief GMRES inner-solve kernels used by iterative refinement (pdgsrfs /
 *        pdgsrfs3d) when options->IterRefine == SLU_GMRES.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab.
 *
 * Classical iterative refinement computes the correction by a single
 * triangular solve with the LU factors,
 *
 *     d = U^{-1} L^{-1} r,
 *
 * for the current (true) residual r = b - A x.  These routines instead solve
 * the correction equation
 *
 *     A d = r
 *
 * with RIGHT-preconditioned restarted GMRES, the preconditioner being the same
 * LU factors, M^{-1} = U^{-1} L^{-1} (applied by pdgstrs / pdgstrs3d):
 *
 *     A M^{-1} u = r,      d = M^{-1} u.
 *
 * Right preconditioning is used (rather than left) so that GMRES minimizes the
 * TRUE residual norm ||r - A d||_2 -- the quantity iterative refinement cares
 * about.  The Givens residual produced by the algorithm therefore equals the
 * true residual, the stopping test is on the true residual, and the returned
 * correction d never increases ||r - A d|| (so it is never worse than the
 * classical d = U^{-1}L^{-1} r), even for ill-conditioned A.
 *
 * d is returned (the update x += d is done by the caller).  The OUTER
 * iterative-refinement loop in pdgsrfs/pdgsrfs3d is unchanged: it still forms
 * the true residual r = b - A x and tests the backward error for convergence.
 *
 * pdgmres()   : 2D-grid kernel (uses pdgsmv + pdgstrs).
 * pdgmres3d() : 3D-grid kernel.  All Krylov work is on z-layer 0; each
 *               preconditioner apply is the collective 3D solve
 *               (pdgstrs3d[_newsolve]); the residual norm is broadcast down the
 *               z-axis so every layer performs the same number of collective
 *               solves (same scheme as pdgsrfs3d).
 * </pre>
 */

#include <math.h>
#include "superlu_ddefs.h"

/* ------------------------------------------------------------------------
 * Hand-coded local dense BLAS-1/2 kernels.  These deliberately avoid the
 * superlu_d{scal,axpy,gemv,trsv} wrappers (i.e. the vendor/OpenBLAS BLAS),
 * because on some machines that library is built for an ISA the CPU does not
 * support and any such call raises SIGILL (illegal instruction) -- the same
 * reason local_dtrtri exists in pdgstrs.c.  The GMRES dense work is tiny, so
 * plain loops cost nothing and keep the solver portable.
 * ---------------------------------------------------------------------- */

/*! \brief x[0:n] *= a */
static void gm_scal(int_t n, double a, double *x)
{ int_t i; for (i = 0; i < n; ++i) x[i] *= a; }

/*! \brief y[0:n] += a * x[0:n] */
static void gm_axpy(int_t n, double a, const double *x, double *y)
{ int_t i; for (i = 0; i < n; ++i) y[i] += a * x[i]; }

/*! \brief y := alpha*op(V)*x + beta*y.  V is m-by-k, column-major, ld=m.
 *         op='N': y length m, x length k.   op='T': y length k, x length m. */
static void gm_gemv(char op, int_t m, int k, double alpha, const double *V,
                    const double *x, double beta, double *y)
{
    int_t i; int j;
    if (op == 'N') {
        for (i = 0; i < m; ++i) {
            double s = 0.0;
            for (j = 0; j < k; ++j) s += V[i + (size_t)j*m] * x[j];
            y[i] = alpha*s + (beta != 0.0 ? beta*y[i] : 0.0);
        }
    } else { /* 'T' */
        for (j = 0; j < k; ++j) {
            double s = 0.0;
            for (i = 0; i < m; ++i) s += V[i + (size_t)j*m] * x[i];
            y[j] = alpha*s + (beta != 0.0 ? beta*y[j] : 0.0);
        }
    }
}

/*! \brief Solve U*b = b in place; U is k-by-k upper triangular, ld=ldh. */
static void gm_trsvU(int k, const double *U, int ldh, double *b)
{
    int i, j;
    for (i = k-1; i >= 0; --i) {
        double s = b[i];
        for (j = i+1; j < k; ++j) s -= U[i + (size_t)j*ldh] * b[j];
        b[i] = s / U[i + (size_t)i*ldh];
    }
}

/*! \brief Global (over grid->comm) inner product of two length-n_loc vectors. */
static double
pdgmres_dot(int_t n_loc, const double *x, const double *y, gridinfo_t *grid)
{
    double local = 0.0, global;
    int_t i;
    for (i = 0; i < n_loc; ++i) local += x[i] * y[i];
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, grid->comm);
    return global;
}

/*! \brief Global 2-norm of a length-n_loc local vector. */
static double
pdgmres_norm2(int_t n_loc, const double *x, gridinfo_t *grid)
{
    return sqrt(pdgmres_dot(n_loc, x, x, grid));
}

/*! \brief
 * <pre>
 * PDGMRES (2D inner kernel).  Solve  A d = r  for the correction d, using
 * RIGHT-preconditioned restarted GMRES with M^{-1} = U^{-1} L^{-1} (pdgstrs).
 *
 * On entry  X[0..m_loc-1] holds the right-hand side r.
 * On exit   X[0..m_loc-1] holds the correction d (initial guess d = 0).
 *
 * gsmv_comm   must be initialized (A in the pdgsmv-permuted state).
 * SOLVEstruct must be configured for a single RHS (as pdgsrfs uses it).
 * *totit (if non-NULL) is incremented by the number of GMRES iterations.
 * </pre>
 */
void
pdgmres(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
        dLUstruct_t *LUstruct, dScalePermstruct_t *ScalePermstruct,
        gridinfo_t *grid, pdgsmv_comm_t *gsmv_comm,
        double *X, int_t m_loc, int_t fst_row,
        int restart, int maxit, double rtol, double atol, int gs,
        dSOLVEstruct_t *SOLVEstruct, int *totit,
        SuperLUStat_t *stat, int *info)
{
    int     iam = grid->iam, i, k, it, ldh, nrit, conv, total = 0;
    double  rho, rho0, delta, gamma, tmpd;
    double *work, *V, *hess, *givens_c, *givens_s, *bb, *rhs, *z, *Ax;

    if ( restart <= 0 ) restart = SUPERLU_MIN(50, (int) n);
    if ( restart > maxit && maxit > 0 ) restart = maxit;
    if ( maxit   <= 0 ) maxit   = 50;
    if ( rtol    <= 0.0 ) rtol  = 1e-8;
    if ( atol    <= 0.0 ) atol  = 1e-14;
    ldh = restart + 1;

    {
        size_t lwork = (size_t)2*restart + (restart+1)
                     + (size_t)(restart+1)*restart
                     + (size_t)m_loc*(restart+1) + 3*(size_t)m_loc;
        if ( !(work = doubleMalloc_dist(lwork)) )
            ABORT("Malloc fails for work[] in pdgmres()");
    }
    givens_c = work;
    givens_s = givens_c + restart;
    bb       = givens_s + restart;
    hess     = bb + (restart+1);
    V        = hess + (size_t)(restart+1)*restart;
    rhs      = V + (size_t)m_loc*(restart+1);  /* the RHS r (preserved) */
    z        = rhs + m_loc;                    /* M^{-1} apply / update scratch */
    Ax       = z + m_loc;                      /* A*d for the restart residual */

    for (i = 0; i < m_loc; ++i) rhs[i] = X[i];  /* rhs = r */
    for (i = 0; i < m_loc; ++i) X[i]   = 0.0;   /* d = 0   */

    rho = rho0 = 0.0;
    conv = 0;
    while ( !conv ) {
        /* True residual V(:,0) = r - A d. */
        if ( total > 0 ) {
            pdgsmv(0, A, grid, gsmv_comm, X, Ax);
            for (i = 0; i < m_loc; ++i) V[i] = rhs[i] - Ax[i];
        } else {
            for (i = 0; i < m_loc; ++i) V[i] = rhs[i];
        }

        rho = pdgmres_norm2(m_loc, V, grid);
        if ( total == 0 ) rho0 = rho;
        if ( rho0 == 0.0 ) rho0 = 1.0;
        if ( rho < atol || rho/rho0 < rtol ) break;

        gm_scal(m_loc, 1.0/rho, V);
        bb[0] = rho;
        for (i = 1; i <= restart; ++i) bb[i] = 0.0;
        nrit = restart - 1;

        for (it = 0; it < restart; ++it) {
            double *w   = &V[(size_t)(it+1)*m_loc];
            double *vit = &V[(size_t)it*m_loc];

            ++total;
            if ( totit ) ++(*totit);

            /* w = A M^{-1} V_it  (right preconditioning) */
            for (i = 0; i < m_loc; ++i) z[i] = vit[i];
            pdgstrs(options, n, LUstruct, ScalePermstruct, grid, z,
                    m_loc, fst_row, m_loc, 1, SOLVEstruct, stat, info);
            pdgsmv(0, A, grid, gsmv_comm, z, w);

            if ( gs == 1 ) { /* classical Gram-Schmidt */
                gm_gemv('T', m_loc, it+1, 1.0, V, w, 0.0, &hess[(size_t)it*ldh]);
                MPI_Allreduce(MPI_IN_PLACE, &hess[(size_t)it*ldh], it+1,
                              MPI_DOUBLE, MPI_SUM, grid->comm);
                gm_gemv('N', m_loc, it+1, -1.0, V, &hess[(size_t)it*ldh], 1.0, w);
            } else {         /* modified Gram-Schmidt */
                for (k = 0; k <= it; ++k) {
                    tmpd = pdgmres_dot(m_loc, &V[(size_t)k*m_loc], w, grid);
                    hess[k + (size_t)it*ldh] = tmpd;
                    gm_axpy(m_loc, -tmpd, &V[(size_t)k*m_loc], w);
                }
            }
            hess[it+1 + (size_t)it*ldh] = pdgmres_norm2(m_loc, w, grid);
            gm_scal(m_loc, 1.0/hess[it+1 + (size_t)it*ldh], w);

            for (k = 1; k <= it; ++k) {
                gamma = givens_c[k-1]*hess[k-1 + (size_t)it*ldh]
                      + givens_s[k-1]*hess[k   + (size_t)it*ldh];
                hess[k   + (size_t)it*ldh] =
                    -givens_s[k-1]*hess[k-1 + (size_t)it*ldh]
                    + givens_c[k-1]*hess[k   + (size_t)it*ldh];
                hess[k-1 + (size_t)it*ldh] = gamma;
            }
            delta = sqrt(hess[it   + (size_t)it*ldh]*hess[it   + (size_t)it*ldh]
                       + hess[it+1 + (size_t)it*ldh]*hess[it+1 + (size_t)it*ldh]);
            givens_c[it] = hess[it   + (size_t)it*ldh] / delta;
            givens_s[it] = hess[it+1 + (size_t)it*ldh] / delta;
            hess[it + (size_t)it*ldh] =
                givens_c[it]*hess[it   + (size_t)it*ldh]
              + givens_s[it]*hess[it+1 + (size_t)it*ldh];
            bb[it+1] = -givens_s[it]*bb[it];
            bb[it]   =  givens_c[it]*bb[it];
            rho = fabs(bb[it+1]);   /* true residual norm */
#if ( PRNTlevel>=2 )
            if ( !iam )
                printf("\t.. inner GMRES it. %4d\ttrue.res = %12.6e\n",
                       total, rho/rho0);
#endif
            if ( rho < atol || rho/rho0 < rtol || total >= maxit ) {
                conv = 1;  nrit = it;  break;
            }
        } /* for it */

        /* d += M^{-1} ( V(:,0..nrit) * hess^{-1} bb ). */
        gm_trsvU(nrit+1, hess, ldh, bb);
        gm_gemv('N', m_loc, nrit+1, 1.0, V, bb, 0.0, z);
        pdgstrs(options, n, LUstruct, ScalePermstruct, grid, z,
                m_loc, fst_row, m_loc, 1, SOLVEstruct, stat, info);
        for (i = 0; i < m_loc; ++i) X[i] += z[i];

        if ( total >= maxit ) break;
    } /* while !conv */

    SUPERLU_FREE(work);
    (void)iam;
} /* PDGMRES */


/*! \brief Apply the 3D preconditioner M^{-1} = U^{-1}L^{-1} in place to one
 *         vector v.  Collective over grid3d. */
static void
pdgmres3d_prec(superlu_dist_options_t *options, int_t n, dLUstruct_t *LUstruct,
               dScalePermstruct_t *ScalePermstruct,
               dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d,
               double *v, int_t m_loc, int_t fst_row,
               dSOLVEstruct_t *SOLVEstruct, SuperLUStat_t *stat, int *info)
{
    if ( get_new3dsolve() )
        pdgstrs3d_newsolve(options, n, LUstruct, ScalePermstruct, trf3Dpartition,
                           grid3d, v, m_loc, fst_row, m_loc, 1, SOLVEstruct,
                           stat, info);
    else
        pdgstrs3d(options, n, LUstruct, ScalePermstruct, trf3Dpartition,
                  grid3d, v, m_loc, fst_row, m_loc, 1, SOLVEstruct, stat, info);
}

/*! \brief
 * <pre>
 * PDGMRES3D (3D inner kernel).  Solve  A d = r  for the correction d, using
 * RIGHT-preconditioned restarted GMRES with M^{-1} = U^{-1}L^{-1}
 * (pdgstrs3d[_newsolve]), for the 3D solve layout.
 *
 * On entry  X[0..m_loc-1] holds r on z-layer 0.
 * On exit   X[0..m_loc-1] holds the correction d on z-layer 0.
 *
 * All Krylov work is on z-layer 0; each M^{-1} apply is collective over all
 * z-layers; the residual norm is broadcast over grid3d->zscp.comm so all
 * layers perform the same number of collective solves.
 * </pre>
 */
void
pdgmres3d(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
          dLUstruct_t *LUstruct, dScalePermstruct_t *ScalePermstruct,
          gridinfo3d_t *grid3d, dtrf3Dpartition_t *trf3Dpartition,
          pdgsmv_comm_t *gsmv_comm, double *X, int_t m_loc, int_t fst_row,
          int restart, int maxit, double rtol, double atol, int gs,
          dSOLVEstruct_t *SOLVEstruct, int *totit,
          SuperLUStat_t *stat, int *info)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int     layer0 = (grid3d->zscp.Iam == 0);
    int     iam = grid3d->iam, i, k, it, ldh, nrit, total = 0, done = 0;
    double  rho, rho0, delta, gamma, tmpd, rbuf[2];
    double *work = NULL, *V = NULL, *hess = NULL, *givens_c = NULL,
           *givens_s = NULL, *bb = NULL, *rhs = NULL, *Ax = NULL, *pc;

    if ( restart <= 0 ) restart = SUPERLU_MIN(50, (int) n);
    if ( restart > maxit && maxit > 0 ) restart = maxit;
    if ( maxit   <= 0 ) maxit   = 50;
    if ( rtol    <= 0.0 ) rtol  = 1e-8;
    if ( atol    <= 0.0 ) atol  = 1e-14;
    ldh = restart + 1;

    if ( !(pc = doubleMalloc_dist(SUPERLU_MAX(m_loc,1))) )
        ABORT("Malloc fails for pc[] in pdgmres3d()");
    if ( layer0 ) {
        size_t lwork = (size_t)2*restart + (restart+1)
                     + (size_t)(restart+1)*restart
                     + (size_t)m_loc*(restart+1) + 2*(size_t)m_loc;
        if ( !(work = doubleMalloc_dist(lwork)) )
            ABORT("Malloc fails for work[] in pdgmres3d()");
        givens_c = work;
        givens_s = givens_c + restart;
        bb       = givens_s + restart;
        hess     = bb + (restart+1);
        V        = hess + (size_t)(restart+1)*restart;
        rhs      = V + (size_t)m_loc*(restart+1);
        Ax       = rhs + m_loc;
    }

    if ( layer0 ) for (i = 0; i < m_loc; ++i) rhs[i] = X[i];  /* rhs = r */
    if ( layer0 ) for (i = 0; i < m_loc; ++i) X[i]   = 0.0;   /* d = 0   */

    rho = rho0 = 0.0;
    while ( !done ) {
        /* True residual V(:,0) = r - A d. */
        if ( total > 0 ) {
            if ( layer0 ) {
                pdgsmv(0, A, grid, gsmv_comm, X, Ax);
                for (i = 0; i < m_loc; ++i) V[i] = rhs[i] - Ax[i];
            }
        } else {
            if ( layer0 ) for (i = 0; i < m_loc; ++i) V[i] = rhs[i];
        }
        if ( layer0 ) {
            rho = pdgmres_norm2(m_loc, V, grid);
            if ( total == 0 ) rho0 = rho;
            if ( rho0 == 0.0 ) rho0 = 1.0;
            rbuf[0] = rho;  rbuf[1] = rho0;
        }
        MPI_Bcast(rbuf, 2, MPI_DOUBLE, 0, grid3d->zscp.comm);
        rho = rbuf[0];  rho0 = rbuf[1];
        if ( rho < atol || rho/rho0 < rtol ) break;

        if ( layer0 ) {
            gm_scal(m_loc, 1.0/rho, V);
            bb[0] = rho;
            for (i = 1; i <= restart; ++i) bb[i] = 0.0;
        }
        nrit = restart - 1;

        for (it = 0; it < restart; ++it) {
            double *w   = layer0 ? &V[(size_t)(it+1)*m_loc] : NULL;
            double *vit = layer0 ? &V[(size_t)it*m_loc]     : NULL;

            ++total;
            if ( totit ) ++(*totit);

            /* w = A M^{-1} V_it  (right preconditioning) */
            if ( layer0 ) for (i = 0; i < m_loc; ++i) pc[i] = vit[i];
            pdgmres3d_prec(options, n, LUstruct, ScalePermstruct, trf3Dpartition,
                           grid3d, pc, m_loc, fst_row, SOLVEstruct, stat, info);
            if ( layer0 ) pdgsmv(0, A, grid, gsmv_comm, pc, w);

            if ( layer0 ) {
                if ( gs == 1 ) {
                    gm_gemv('T', m_loc, it+1, 1.0, V, w, 0.0, &hess[(size_t)it*ldh]);
                    MPI_Allreduce(MPI_IN_PLACE, &hess[(size_t)it*ldh], it+1,
                                  MPI_DOUBLE, MPI_SUM, grid->comm);
                    gm_gemv('N', m_loc, it+1, -1.0, V, &hess[(size_t)it*ldh], 1.0, w);
                } else {
                    for (k = 0; k <= it; ++k) {
                        tmpd = pdgmres_dot(m_loc, &V[(size_t)k*m_loc], w, grid);
                        hess[k + (size_t)it*ldh] = tmpd;
                        gm_axpy(m_loc, -tmpd, &V[(size_t)k*m_loc], w);
                    }
                }
                hess[it+1 + (size_t)it*ldh] = pdgmres_norm2(m_loc, w, grid);
                gm_scal(m_loc, 1.0/hess[it+1 + (size_t)it*ldh], w);
                for (k = 1; k <= it; ++k) {
                    gamma = givens_c[k-1]*hess[k-1 + (size_t)it*ldh]
                          + givens_s[k-1]*hess[k   + (size_t)it*ldh];
                    hess[k   + (size_t)it*ldh] =
                        -givens_s[k-1]*hess[k-1 + (size_t)it*ldh]
                        + givens_c[k-1]*hess[k   + (size_t)it*ldh];
                    hess[k-1 + (size_t)it*ldh] = gamma;
                }
                delta = sqrt(hess[it   + (size_t)it*ldh]*hess[it   + (size_t)it*ldh]
                           + hess[it+1 + (size_t)it*ldh]*hess[it+1 + (size_t)it*ldh]);
                givens_c[it] = hess[it   + (size_t)it*ldh] / delta;
                givens_s[it] = hess[it+1 + (size_t)it*ldh] / delta;
                hess[it + (size_t)it*ldh] =
                    givens_c[it]*hess[it   + (size_t)it*ldh]
                  + givens_s[it]*hess[it+1 + (size_t)it*ldh];
                bb[it+1] = -givens_s[it]*bb[it];
                bb[it]   =  givens_c[it]*bb[it];
                rho = fabs(bb[it+1]);
            }
            MPI_Bcast(&rho, 1, MPI_DOUBLE, 0, grid3d->zscp.comm);
            if ( rho < atol || rho/rho0 < rtol || total >= maxit ) {
                done = 1;  nrit = it;  break;
            }
        } /* for it */

        /* d += M^{-1} ( V(:,0..nrit) * hess^{-1} bb ). */
        if ( layer0 ) {
            gm_trsvU(nrit+1, hess, ldh, bb);
            gm_gemv('N', m_loc, nrit+1, 1.0, V, bb, 0.0, pc);
        }
        pdgmres3d_prec(options, n, LUstruct, ScalePermstruct, trf3Dpartition,
                       grid3d, pc, m_loc, fst_row, SOLVEstruct, stat, info);
        if ( layer0 ) for (i = 0; i < m_loc; ++i) X[i] += pc[i];

        if ( total >= maxit ) break;
    } /* while !done */

    SUPERLU_FREE(pc);
    if ( layer0 ) SUPERLU_FREE(work);
    (void)iam;
} /* PDGMRES3D */
