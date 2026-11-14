/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Self-checking driver used by the CI tests (ctest, .ci_tests.sh,
 *        .ci_tests_gpu.sh).
 *
 * <pre>
 * One source, three programs.  It is compiled once per precision:
 *     -DSLU_CI_SINGLE     ->  pstest_ci   (psgssvx, psgssvx3d, ...)
 *     -DSLU_CI_DOUBLE     ->  pdtest_ci   (pdgssvx, pdgssvx3d, ...)
 *     -DSLU_CI_COMPLEX16  ->  pztest_ci   (pzgssvx, pzgssvx3d, ...)
 *
 * Unlike the EXAMPLE drivers, which print an error and exit with status 0
 * whatever the error is, this program verifies every solution it computes
 * and reports the outcome in its exit status:
 *      0   every check passed          (last line "SLU_CI_RESULT: PASS")
 *      1   a check failed              (last line "SLU_CI_RESULT: FAIL")
 *     77   the build cannot run the requested configuration, e.g. a ParMETIS
 *          ordering without ParMETIS   (last line "SLU_CI_RESULT: SKIP")
 *
 * What a run covers is chosen on the command line (see usage()):
 *   - the 2D interface p?gssvx (-d 0) or the 3D interface p?gssvx3d (-d >= 1)
 *     on any process grid,
 *   - a sequence of factorization modes (-f), one letter per call:
 *        D  DOFACT                    new matrix values, new right-hand side
 *        F  FACTORED                  same matrix, new right-hand side
 *        P  SamePattern               new matrix values, new right-hand side
 *        R  SamePattern_SameRowPerm   new matrix values, new right-hand side
 *     Every call is checked against the matrix values and the exact solution
 *     of that call, so a stale factor or a stale solution is detected,
 *   - SolveOnly (-S), level-based ILU (-I), row/column orderings, equilibration,
 *     iterative refinement, GMRES, DiagInv, parallel symbolic factorization,
 *     several right-hand sides, GPU-resident right-hand sides,
 *   - the batched interfaces p?gssvx3d_csc_batch (-b) and
 *     p?gssvx3d_csc_vbatch (-B).
 * CPU or GPU factorization/solve are selected, as everywhere in SuperLU_DIST,
 * with the environment variables SUPERLU_ACC_OFFLOAD, SUPERLU_ACC_SOLVE and
 * GPU3DVERSION; the program prints the values in effect.
 *
 * The matrix is either read from a file (last argument; the format is taken
 * from the suffix, as in the EXAMPLE drivers) or generated: without a file
 * name the program builds a nonsymmetric, diagonally dominant 3/5/7-point
 * convection-diffusion operator on an nx x ny x nz grid (-x -y -z).
 * With -k 1 the rows of the generated matrix are scrambled, so that the
 * solver only succeeds if the row permutation (-p 1) recovers the diagonal.
 *
 * The checks:
 *   - normwise backward error  ||B - A*X|| / (||A||*||X|| + ||B||)   (-t)
 *   - forward error            ||X - Xtrue|| / ||Xtrue||              (-T)
 *   - ILU(0) without refinement: the solution must match the one obtained
 *     from a textbook serial ILU(0) of the scaled and permuted matrix
 *     Pc*Pr*R*A*C*Pc', built here from the returned ScalePermstruct.
 *   - SolveOnly: the input matrix is taken as the factors, A = (L-I) + U,
 *     so the right-hand side is built as L*U*Xtrue.
 * </pre>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#if defined(SLU_CI_COMPLEX16)
#include "superlu_zdefs.h"
typedef doublecomplex scalar_t;
typedef double        real_t;
#define P_(name)   pz##name
#define X_(name)   z##name
#define SMALLOC    doublecomplexMalloc_dist
#define RMALLOC    doubleMalloc_dist
#define SLU_DT     SLU_Z
#define PRECNAME   "pztest_ci"
#define IS_COMPLEX 1
#define TOL_RESID  1e-10
#define TOL_FERR   1e-6
#define TOL_ILU    1e-8
#elif defined(SLU_CI_SINGLE)
#include "superlu_sdefs.h"
typedef float scalar_t;
typedef float real_t;
#define P_(name)   ps##name
#define X_(name)   s##name
#define SMALLOC    floatMalloc_dist
#define RMALLOC    floatMalloc_dist
#define SLU_DT     SLU_S
#define PRECNAME   "pstest_ci"
#define IS_COMPLEX 0
#define TOL_RESID  1e-4
#define TOL_FERR   1e-2
#define TOL_ILU    1e-3
#else
#include "superlu_ddefs.h"
typedef double scalar_t;
typedef double real_t;
#define P_(name)   pd##name
#define X_(name)   d##name
#define SMALLOC    doubleMalloc_dist
#define RMALLOC    doubleMalloc_dist
#define SLU_DT     SLU_D
#define PRECNAME   "pdtest_ci"
#define IS_COMPLEX 0
#define TOL_RESID  1e-10
#define TOL_FERR   1e-6
#define TOL_ILU    1e-8
#endif

#define EXIT_PASS 0
#define EXIT_FAIL 1
#define EXIT_SKIP 77

/* ------------------------------------------------------------------------
   Scalar arithmetic, so that the rest of the file is precision independent.
   ------------------------------------------------------------------------ */
#if IS_COMPLEX
static scalar_t S_make(double re, double im) { scalar_t z; z.r = re; z.i = im; return z; }
static scalar_t S_add(scalar_t a, scalar_t b) { return S_make(a.r + b.r, a.i + b.i); }
static scalar_t S_sub(scalar_t a, scalar_t b) { return S_make(a.r - b.r, a.i - b.i); }
static scalar_t S_mul(scalar_t a, scalar_t b)
{ return S_make(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); }
static scalar_t S_div(scalar_t a, scalar_t b)
{
    double d = b.r * b.r + b.i * b.i;
    return S_make((a.r * b.r + a.i * b.i) / d, (a.i * b.r - a.r * b.i) / d);
}
static scalar_t S_scale(scalar_t a, double s) { return S_make(a.r * s, a.i * s); }
static double   S_abs(scalar_t a) { return hypot(a.r, a.i); }
#else
static scalar_t S_make(double re, double im) { (void) im; return (scalar_t) re; }
static scalar_t S_add(scalar_t a, scalar_t b) { return a + b; }
static scalar_t S_sub(scalar_t a, scalar_t b) { return a - b; }
static scalar_t S_mul(scalar_t a, scalar_t b) { return a * b; }
static scalar_t S_div(scalar_t a, scalar_t b) { return a / b; }
static scalar_t S_scale(scalar_t a, double s) { return (scalar_t) (a * s); }
static double   S_abs(scalar_t a) { return fabs((double) a); }
#endif

/* Deterministic pseudo-random number in [-1,1], a function of (a,b) only, so
   that every process and every run generate the same matrices and vectors. */
static double urand(uint64_t a, uint64_t b)
{
    uint64_t x = (a + 0x9E3779B97F4A7C15ULL) * 0xBF58476D1CE4E5B9ULL
	       ^ (b + 0x632BE59BD9B4E019ULL) * 0x94D049BB133111EBULL;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (double) (x >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

/* ------------------------------------------------------------------------
   Global matrix in compressed row format.  The CI matrices are small, so
   every process keeps a copy; this is what the checks are computed from,
   independently of the distributed matrix handed to (and overwritten by)
   the solver.
   ------------------------------------------------------------------------ */
typedef struct {
    int_t n, nnz;
    int_t *rowptr, *colind;
    scalar_t *val;   /* values as read or generated */
} gmat_t;

/* Run parameters */
typedef struct {
    int nprow, npcol, npdep;     /* npdep == 0: 2D interface */
    int nx, ny, nz;              /* generated matrix */
    int scramble;
    int blk;                     /* unknowns per grid point = user-defined supernode size */
    char steps[64];
    int equil, rowperm, colperm, ir, lookahead, nrhs;
    int ilu, solveonly, batch, vbatch, gpures, gmres, diaginv, parsymb;
    int la_etree, tinypivot;
    int verbose;
    double tol_resid, tol_ferr, tol_ilu, delta;
    char *file;
} params_t;

static void usage(void)
{
    printf("%s: self-checking SuperLU_DIST test driver.  Options:\n", PRECNAME);
    printf("\t-r <int>   process rows                         (default 1)\n");
    printf("\t-c <int>   process columns                      (default 1)\n");
    printf("\t-d <int>   process Z-dimension; 0 = 2D interface (default 0)\n");
    printf("\t-x -y -z <int>  grid of the generated matrix    (default 12 12 1)\n");
    printf("\t-k <0|1>   scramble the rows of the generated matrix (default 0)\n");
    printf("\t-U <int>   unknowns per grid point, coupled by dense blocks; with -I 0 the\n");
    printf("\t           blocks are passed as user-defined supernodes (default 1)\n");
    printf("\t-f <str>   sequence of calls, letters D F P R   (default D)\n");
    printf("\t           D DOFACT, F FACTORED, P SamePattern, R SamePattern_SameRowPerm\n");
    printf("\t-e <int>   equilibration 0/1\n");
    printf("\t-p <int>   row permutation 0 NOROWPERM, 1 LargeDiag_MC64, 2 LargeDiag_HWPM\n");
    printf("\t-q <int>   column permutation 0 NATURAL, 1 MMD_ATA, 2 MMD_AT_PLUS_A,\n");
    printf("\t           3 COLAMD, 4 METIS_AT_PLUS_A, 5 PARMETIS, 6 METIS_ATA,\n");
    printf("\t           8 MY_PERMC: red-black ordering of the generated grid\n");
    printf("\t-i <int>   iterative refinement 0 none, 1 single, 2 double\n");
    printf("\t-s <int>   number of right-hand sides           (default 1)\n");
    printf("\t-l <int>   number of look-ahead levels\n");
    printf("\t-I <int>   ILU level (0 = ILU(0))               (default: complete LU)\n");
    printf("\t-S <0|1>   SolveOnly: the matrix holds the L and U factors (3D only)\n");
    printf("\t-b <int>   batch count, fixed-size batched interface (1 process)\n");
    printf("\t-B <int>   batch count, variable-size batched interface (1 process)\n");
    printf("\t-g <0|1>   right-hand side resident on the GPU\n");
    printf("\t-G <0|1>   solve with GMRES preconditioned by the factors (2D only)\n");
    printf("\t-D <0|1>   invert the diagonal blocks (DiagInv)\n");
    printf("\t-P <0|1>   parallel symbolic factorization (needs -q 5)\n");
    printf("\t-L <0|1>   schedule with the etree of the symbolic factorization (lookahead_etree)\n");
    printf("\t-R <0|1>   replace tiny pivots (ReplaceTinyPivot)\n");
    printf("\t-t <real>  tolerance on the backward error\n");
    printf("\t-T <real>  tolerance on the forward error\n");
    printf("\t-u <real>  tolerance on the difference with the reference ILU(0)\n");
    printf("\t-w <real>  relative size of the value changes between calls (default 0.01)\n");
    printf("\t-v <0|1>   print the solver statistics\n");
    printf("\t[file]     matrix file (.rua .cua .mtx .rb .dat .datnh .bin); generated if absent\n");
}

/* ------------------------------------------------------------------------
   Matrix construction
   ------------------------------------------------------------------------ */

/* Nonsymmetric, strictly row diagonally dominant convection-diffusion
   stencil on an nx x ny x nz grid, in natural ordering.  With blk > 1 every
   grid point carries blk unknowns, numbered consecutively, and every stencil
   entry is a dense blk x blk block. */
static void generate_matrix(int nx, int ny, int nz, int blk, int scramble, gmat_t *G)
{
    int_t ng = (int_t) nx * ny * nz, n = ng * blk;
    int_t i, j, k, g, row, col, p = 0, q;
    int d, a, c;

    G->n = n;
    G->rowptr = intMalloc_dist(n + 1);
    G->colind = intMalloc_dist(7 * n * blk);
    G->val = SMALLOC(7 * n * blk);

    for (k = 0; k < nz; ++k) for (j = 0; j < ny; ++j) for (i = 0; i < nx; ++i) {
	int_t nbr[7]; int nn = 0;
	g = i + nx * (j + (int_t) ny * k);
	/* neighbours in increasing column order */
	if (k > 0)      nbr[nn++] = g - (int_t) nx * ny;
	if (j > 0)      nbr[nn++] = g - nx;
	if (i > 0)      nbr[nn++] = g - 1;
	nbr[nn++] = g;
	if (i < nx - 1) nbr[nn++] = g + 1;
	if (j < ny - 1) nbr[nn++] = g + nx;
	if (k < nz - 1) nbr[nn++] = g + (int_t) nx * ny;
	for (a = 0; a < blk; ++a) {
	    int_t pdiag = -1;
	    double sum = 0.0;
	    row = g * blk + a;
	    G->rowptr[row] = p;
	    for (d = 0; d < nn; ++d) for (c = 0; c < blk; ++c, ++p) {
		col = nbr[d] * blk + c;
		G->colind[p] = col;
		if (col == row) { pdiag = p; continue; }
		G->val[p] = S_make(-(1.0 + 0.3 * urand(row, col)),
				   IS_COMPLEX ? 0.3 * urand(col, row + n) : 0.0);
		sum += S_abs(G->val[p]);
	    }
	    G->val[pdiag] = S_make(sum + 0.2, IS_COMPLEX ? 0.3 : 0.0);
	}
    }
    G->rowptr[n] = p;
    G->nnz = p;

    if (scramble) { /* new row r holds the old row sigma(r) */
	int_t *sigma = intMalloc_dist(n);
	int_t *rowptr = intMalloc_dist(n + 1), *colind = intMalloc_dist(G->nnz);
	scalar_t *val = SMALLOC(G->nnz);
	for (row = 0; row < n; ++row) sigma[row] = row;
	for (row = n - 1; row > 0; --row) { /* Fisher-Yates */
	    q = (int_t) ((urand(row, 12345) + 1.0) * 0.5 * (row + 1));
	    if (q > row) q = row;
	    col = sigma[row]; sigma[row] = sigma[q]; sigma[q] = col;
	}
	p = 0;
	for (row = 0; row < n; ++row) {
	    rowptr[row] = p;
	    for (q = G->rowptr[sigma[row]]; q < G->rowptr[sigma[row] + 1]; ++q, ++p) {
		colind[p] = G->colind[q];
		val[p] = G->val[q];
	    }
	}
	rowptr[n] = p;
	SUPERLU_FREE(G->rowptr); SUPERLU_FREE(G->colind); SUPERLU_FREE(G->val);
	SUPERLU_FREE(sigma);
	G->rowptr = rowptr; G->colind = colind; G->val = val;
    }
}

/* Column permutation for ColPerm = MY_PERMC: red-black ordering of the grid
   points, the unknowns of a point kept together.  The red points are
   independent of each other, so the elimination tree is bushy. */
static void redblack_perm_c(int nx, int ny, int nz, int blk, int *perm_c)
{
    int_t g, next = 0;
    int i, j, k, a, color;

    for (color = 0; color < 2; ++color)
	for (k = 0; k < nz; ++k) for (j = 0; j < ny; ++j) for (i = 0; i < nx; ++i) {
	    if ((i + j + k) % 2 != color) continue;
	    g = i + nx * (j + (int_t) ny * k);
	    for (a = 0; a < blk; ++a) perm_c[g * blk + a] = (int) (next * blk + a);
	    ++next;
	}
}

/* Read a matrix file on the calling process; the format follows the suffix. */
static void read_matrix(const char *file, gmat_t *G)
{
    FILE *fp;
    const char *suffix = strrchr(file, '.');
    int_t m, n, nnz, *rowind, *colptr;
    scalar_t *nzval;

    if (!(fp = fopen(file, "r"))) ABORT("File does not exist");
    suffix = suffix ? suffix + 1 : "";
    if (!strcmp(suffix, "rua") || !strcmp(suffix, "cua"))
	X_(readhb_dist)(0, fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
    else if (!strcmp(suffix, "mtx") || !strcmp(suffix, "mm"))
	X_(readMM_dist)(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
    else if (!strcmp(suffix, "rb"))
	X_(readrb_dist)(0, fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
    else if (!strcmp(suffix, "dat"))
	X_(readtriple_dist)(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
    else if (!strcmp(suffix, "datnh"))
	X_(readtriple_noheader)(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
    else if (!strcmp(suffix, "bin"))
	X_(read_binary)(fp, &m, &n, &nnz, &nzval, &rowind, &colptr);
    else
	ABORT("File format not known");
    fclose(fp);
    if (m != n) ABORT("The matrix must be square");

    G->n = n;
    G->nnz = nnz;
    X_(CompCol_to_CompRow_dist)(m, n, nnz, nzval, colptr, rowind,
				&G->val, &G->rowptr, &G->colind);
    SUPERLU_FREE(nzval); SUPERLU_FREE(rowind); SUPERLU_FREE(colptr);
}

static void bcast_matrix(gmat_t *G, int iam, MPI_Comm comm)
{
    MPI_Bcast(&G->n, 1, mpi_int_t, 0, comm);
    MPI_Bcast(&G->nnz, 1, mpi_int_t, 0, comm);
    if (iam) {
	G->rowptr = intMalloc_dist(G->n + 1);
	G->colind = intMalloc_dist(G->nnz);
	G->val = SMALLOC(G->nnz);
    }
    MPI_Bcast(G->rowptr, (int) (G->n + 1), mpi_int_t, 0, comm);
    MPI_Bcast(G->colind, (int) G->nnz, mpi_int_t, 0, comm);
    MPI_Bcast(G->val, (int) (G->nnz * sizeof(scalar_t)), MPI_BYTE, 0, comm);
}

static void free_matrix(gmat_t *G)
{
    SUPERLU_FREE(G->rowptr); SUPERLU_FREE(G->colind); SUPERLU_FREE(G->val);
}

/* Values of the matrix used by call number 'ver' (0 = as read/generated).
   'salt' distinguishes the systems of a batch.  With 'factors' the matrix is
   meant to be read as A = (L-I) + U: the strict lower triangle is scaled down
   so that L, hence L*U, is well conditioned. */
static void matrix_values(const gmat_t *G, int ver, int salt, double delta, int factors,
			  scalar_t *val)
{
    int_t i, p;
    for (i = 0; i < G->nnz; ++i)
	val[i] = (ver == 0 && salt == 0) ? G->val[i] :
	    S_scale(G->val[i], 1.0 + delta * urand(i, 1000003 * (uint64_t) ver + salt));
    if (factors)
	for (i = 0; i < G->n; ++i)
	    for (p = G->rowptr[i]; p < G->rowptr[i + 1]; ++p)
		if (G->colind[p] < i) val[p] = S_scale(val[p], 0.125);
}

/* Exact solution of call number 'ver': n x nrhs, column major. */
static void exact_solution(int_t n, int nrhs, int ver, int salt, scalar_t *x)
{
    int_t i; int j;
    for (j = 0; j < nrhs; ++j)
	for (i = 0; i < n; ++i) {
	    uint64_t key = 7919 * (uint64_t) ver + 104729 * (uint64_t) salt + j;
	    x[i + j * n] = S_make(1.0 + 0.5 * urand(i, key),
				  IS_COMPLEX ? 0.5 * urand(i + n, key) : 0.0);
	}
}

/* y(lo:hi-1) = op(A) * x for one vector, where op selects all the entries
   (part = 0), the upper triangle with the diagonal (part = 1), or the strict
   lower triangle plus a unit diagonal (part = 2). */
static void matvec_rows(const gmat_t *G, const scalar_t *val, int part,
			int_t lo, int_t hi, const scalar_t *x, scalar_t *y)
{
    int_t i, p;
    for (i = lo; i < hi; ++i) {
	scalar_t s = (part == 2) ? x[i] : S_make(0.0, 0.0);
	for (p = G->rowptr[i]; p < G->rowptr[i + 1]; ++p) {
	    int_t j = G->colind[p];
	    if ((part == 1 && j < i) || (part == 2 && j >= i)) continue;
	    s = S_add(s, S_mul(val[p], x[j]));
	}
	y[i - lo] = s;
    }
}

/* ------------------------------------------------------------------------
   Reference ILU(0) solve, serial, of the system the solver factorized:
       B = Pc*Pr*R*A*C*Pc',   B*y = Pc*Pr*R*b,   x = C*Pc'*y.
   R and perm_r are indexed by the row numbering the solver works in, which
   'rowmap' gives for each row of A (see solver_row_order()).
   Returns 0 on success, 1 if a zero pivot is met.
   ------------------------------------------------------------------------ */
typedef struct { int_t col; scalar_t v; } ent_t;
static int cmp_ent(const void *a, const void *b)
{
    int_t ca = ((const ent_t *) a)->col, cb = ((const ent_t *) b)->col;
    return (ca > cb) - (ca < cb);
}

static int ilu0_reference(const gmat_t *G, const scalar_t *val, DiagScale_t ds,
			  const real_t *R, const real_t *C,
			  const int *perm_r, const int *perm_c, const int_t *rowmap,
			  int nrhs, const scalar_t *b, scalar_t *x)
{
    int_t n = G->n, nnz = G->nnz, i, j, k, p, q;
    int_t *rowptr = intCalloc_dist(n + 1), *diag = intMalloc_dist(n);
    int_t *iw = intMalloc_dist(n), *next = intMalloc_dist(n);
    ent_t *E = (ent_t *) SUPERLU_MALLOC((nnz > 0 ? nnz : 1) * sizeof(ent_t));
    scalar_t *y = SMALLOC(n);
    int useR = (ds == ROW || ds == BOTH), useC = (ds == COL || ds == BOTH);
    int irhs, fail = 0;

    /* B in row format, rows sorted by column */
    for (i = 0; i < n; ++i)
	rowptr[perm_c[perm_r[rowmap[i]]] + 1] += G->rowptr[i + 1] - G->rowptr[i];
    for (i = 0; i < n; ++i) rowptr[i + 1] += rowptr[i];
    for (i = 0; i < n; ++i) next[i] = rowptr[i];
    for (i = 0; i < n; ++i) {
	int_t bi = perm_c[perm_r[rowmap[i]]];
	for (p = G->rowptr[i]; p < G->rowptr[i + 1]; ++p) {
	    j = G->colind[p];
	    q = next[bi]++;
	    E[q].col = perm_c[j];
	    E[q].v = S_scale(val[p], (useR ? R[rowmap[i]] : 1.0) * (useC ? C[j] : 1.0));
	}
    }
    for (i = 0; i < n; ++i) {
	qsort(E + rowptr[i], rowptr[i + 1] - rowptr[i], sizeof(ent_t), cmp_ent);
	diag[i] = -1;
	for (p = rowptr[i]; p < rowptr[i + 1]; ++p)
	    if (E[p].col == i) diag[i] = p;
	if (diag[i] < 0) fail = 1;
	iw[i] = -1;
    }

    /* IKJ ILU(0): fill outside the pattern of B is dropped */
    for (i = 0; i < n && !fail; ++i) {
	for (p = rowptr[i]; p < rowptr[i + 1]; ++p) iw[E[p].col] = p;
	for (p = rowptr[i]; p < diag[i]; ++p) {
	    k = E[p].col;
	    E[p].v = S_div(E[p].v, E[diag[k]].v);
	    for (q = diag[k] + 1; q < rowptr[k + 1]; ++q)
		if (iw[E[q].col] >= 0)
		    E[iw[E[q].col]].v = S_sub(E[iw[E[q].col]].v, S_mul(E[p].v, E[q].v));
	}
	if (S_abs(E[diag[i]].v) == 0.0) fail = 1;
	for (p = rowptr[i]; p < rowptr[i + 1]; ++p) iw[E[p].col] = -1;
    }

    for (irhs = 0; irhs < nrhs && !fail; ++irhs) {
	const scalar_t *bb = b + (size_t) irhs * n;
	scalar_t *xx = x + (size_t) irhs * n;
	for (i = 0; i < n; ++i)
	    y[perm_c[perm_r[rowmap[i]]]] = S_scale(bb[i], useR ? R[rowmap[i]] : 1.0);
	for (i = 0; i < n; ++i)           /* L y = y */
	    for (p = rowptr[i]; p < diag[i]; ++p)
		y[i] = S_sub(y[i], S_mul(E[p].v, y[E[p].col]));
	for (i = n - 1; i >= 0; --i) {    /* U y = y */
	    for (p = diag[i] + 1; p < rowptr[i + 1]; ++p)
		y[i] = S_sub(y[i], S_mul(E[p].v, y[E[p].col]));
	    y[i] = S_div(y[i], E[diag[i]].v);
	}
	for (j = 0; j < n; ++j)
	    xx[j] = S_scale(y[perm_c[j]], useC ? C[j] : 1.0);
    }

    SUPERLU_FREE(rowptr); SUPERLU_FREE(diag); SUPERLU_FREE(iw); SUPERLU_FREE(next);
    SUPERLU_FREE(E); SUPERLU_FREE(y);
    return fail;
}

/* Row numbering the solver works in.  The 2D interface keeps the numbering
   of A.  The 3D interface gathers the rows of the processes that share a 2D
   grid position onto layer 0; in the default Z-major rank order, and unless
   RowPerm = NOROWPERM, the gathered rows are numbered 2D process by 2D
   process, which is a row permutation of A (the row pivoting then finds
   the diagonal again).  displs[] is the block row distribution of A. */
static void solver_row_order(int_t n, int np, const int *displs, int hidden,
			     int pxy, int_t *rowmap)
{
    int_t i, next = 0;
    int p, q;
    if (!hidden) {
	for (i = 0; i < n; ++i) rowmap[i] = i;
	return;
    }
    for (p = 0; p < pxy; ++p)
	for (q = p; q < np; q += pxy) { /* the Z-column of 2D process p */
	    int_t last = (q == np - 1) ? n : displs[q + 1];
	    for (i = displs[q]; i < last; ++i) rowmap[i] = next++;
	}
}

/* ------------------------------------------------------------------------
   Checks on a full (global) solution; pure serial arithmetic.
   ------------------------------------------------------------------------ */
static void check_solution(const gmat_t *G, const scalar_t *val, int solveonly,
			   int nrhs, const scalar_t *x, const scalar_t *b,
			   const scalar_t *xtrue, double *resid, double *ferr)
{
    int_t n = G->n, i, p;
    int j;
    double anorm = 0.0;
    scalar_t *r = SMALLOC(n), *t = SMALLOC(n);

    for (i = 0; i < n; ++i) {
	double s = 0.0;
	for (p = G->rowptr[i]; p < G->rowptr[i + 1]; ++p) s += S_abs(val[p]);
	if (s > anorm) anorm = s;
    }
    *resid = *ferr = 0.0;
    for (j = 0; j < nrhs; ++j) {
	const scalar_t *xj = x + (size_t) j * n, *bj = b + (size_t) j * n;
	const scalar_t *xt = xtrue + (size_t) j * n;
	double rn = 0.0, xn = 0.0, bn = 0.0, en = 0.0, tn = 0.0, v;
	if (solveonly) { /* A holds the factors: the operator is L*U */
	    matvec_rows(G, val, 1, 0, n, xj, t);
	    matvec_rows(G, val, 2, 0, n, t, r);
	} else
	    matvec_rows(G, val, 0, 0, n, xj, r);
	for (i = 0; i < n; ++i) {
	    v = S_abs(S_sub(bj[i], r[i]));  if (!(v <= rn)) rn = v;
	    v = S_abs(xj[i]);               if (!(v <= xn)) xn = v;
	    v = S_abs(bj[i]);               if (v > bn) bn = v;
	    v = S_abs(S_sub(xj[i], xt[i])); if (!(v <= en)) en = v;
	    v = S_abs(xt[i]);               if (v > tn) tn = v;
	}
	/* written so that a NaN anywhere fails the test */
	v = rn / (anorm * xn + bn);  if (!(v <= *resid)) *resid = v;
	v = en / tn;                 if (!(v <= *ferr)) *ferr = v;
    }
    SUPERLU_FREE(r); SUPERLU_FREE(t);
}

static int report(int iam, const char *what, double value, double tol)
{
    int ok = (value <= tol); /* false for NaN */
    if (!iam) {
	printf("    %-34s %10.3e  (tol %8.1e)  %s\n", what, value, tol, ok ? "ok" : "FAILED");
	fflush(stdout);
    }
    return !ok;
}

/* The build cannot run what was asked for: tell why, return the reason. */
static const char *unsupported(const params_t *par, superlu_dist_options_t *options)
{
    (void) options;
#ifndef GPU_ACC
    {
	char *e;
	if ((e = getenv("SUPERLU_ACC_OFFLOAD")) && atoi(e))
	    return "SUPERLU_ACC_OFFLOAD=1 needs a GPU build";
	if ((e = getenv("SUPERLU_ACC_SOLVE")) && atoi(e))
	    return "SUPERLU_ACC_SOLVE=1 needs a GPU build";
	if (par->gpures == 1) return "a GPU-resident right-hand side needs a GPU build";
    }
#else
    if (get_acc_solve() && par->nprow * par->npcol > 1) {
#if !defined(HAVE_NVSHMEM) && !defined(HAVE_ROCSHMEM)
	return "the GPU triangular solve on more than one process per 2D grid needs NVSHMEM/ROCSHMEM";
#endif
    }
#endif
#if ( DEBUGlevel>=1 )
    /* Known defect: with a NATURAL ordering the upper Z-layers of the 3D
       factorization request empty arrays, on which the debug allocator of
       DEBUGlevel>=1 aborts ("superlu_malloc: nonpositive size"). */
    if (par->npdep > 1 && (par->colperm == NATURAL || par->solveonly))
	return "DEBUGlevel>=1: the debug allocator aborts on the empty allocations "
	       "of a NATURAL ordering on several Z-layers";
#endif
#ifndef HAVE_PARMETIS
    if (par->colperm == PARMETIS || par->parsymb == 1)
	return "ColPerm = PARMETIS / ParSymbFact need ParMETIS";
#endif
#if !defined(SLU_CI_DOUBLE)
    /* Only pdgssvx3d has the 3D parallel symbolic factorization, and ParMETIS
       on several Z-layers. */
    if (par->npdep >= 1 && (par->parsymb == 1 ||
			    (par->colperm == PARMETIS && par->npdep > 1)))
	return "ParMETIS on several Z-layers and ParSymbFact in the 3D interface "
	       "are implemented in double precision only";
#endif
#ifndef HAVE_METIS
    if (par->colperm == METIS_AT_PLUS_A || par->colperm == METIS_ATA)
	return "the METIS orderings need METIS";
#endif
#ifndef HAVE_COLAMD
    if (par->colperm == COLAMD) return "ColPerm = COLAMD needs COLAMD";
#endif
#ifndef HAVE_COMBBLAS
    if (par->rowperm == LargeDiag_HWPM) return "RowPerm = LargeDiag_HWPM needs CombBLAS";
#endif
#ifndef SLU_HAVE_LAPACK
    if (par->diaginv == 1) return "DiagInv needs LAPACK";
#endif
    if (par->batch > 0 || par->vbatch > 0) {
#ifdef GPU_ACC
#ifndef HAVE_MAGMA
	if (sp_ienv_dist(12, options) == 1)
	    return "the batched interface with GPU3DVERSION=1 needs MAGMA";
#endif
#endif
    }
    return NULL;
}

/* ------------------------------------------------------------------------
   Batched interfaces (one process).  System d of the batch has its own
   values and its own exact solution; with -B the systems also differ in
   size when the matrix is generated.
   ------------------------------------------------------------------------ */
static int run_batch(params_t *par, superlu_dist_options_t *options,
		     gmat_t *G0, gridinfo3d_t *grid3d)
{
    int nb = par->vbatch > 0 ? par->vbatch : par->batch;
    int variable = par->vbatch > 0, nrhs = par->nrhs;
    int d, info, nfail = 0, ver = 0, istep, nsteps = (int) strlen(par->steps);
    gmat_t *G = (gmat_t *) SUPERLU_MALLOC(nb * sizeof(gmat_t));
    handle_t *handles = (handle_t *) SUPERLU_MALLOC(nb * sizeof(handle_t));
    handle_t Fstate = 0, *F = variable ? &Fstate : NULL;
    scalar_t **val = (scalar_t **) SUPERLU_MALLOC(nb * sizeof(scalar_t *));
    scalar_t **RHSptr = (scalar_t **) SUPERLU_MALLOC(nb * sizeof(scalar_t *));
    scalar_t **bsave = (scalar_t **) SUPERLU_MALLOC(nb * sizeof(scalar_t *));
    scalar_t **xtrue = (scalar_t **) SUPERLU_MALLOC(nb * sizeof(scalar_t *));
    scalar_t **Xptr = (scalar_t **) SUPERLU_MALLOC(nb * sizeof(scalar_t *));
    real_t **ReqPtr = (real_t **) SUPERLU_MALLOC(nb * sizeof(real_t *));
    real_t **CeqPtr = (real_t **) SUPERLU_MALLOC(nb * sizeof(real_t *));
    real_t **Berrs = (real_t **) SUPERLU_MALLOC(nb * sizeof(real_t *));
    int **RpivPtr = (int **) SUPERLU_MALLOC(nb * sizeof(int *));
    int **CpivPtr = (int **) SUPERLU_MALLOC(nb * sizeof(int *));
    DiagScale_t *DiagScale = (DiagScale_t *) SUPERLU_MALLOC(nb * sizeof(DiagScale_t));
    int *md = int32Malloc_dist(nb), *nnzd = int32Malloc_dist(nb), *ld = int32Malloc_dist(nb);
    SuperMatrix *Amat = (SuperMatrix *) SUPERLU_MALLOC(nb * sizeof(SuperMatrix));
    SuperLUStat_t stat;

    for (d = 0; d < nb; ++d) {
	if (variable && !par->file && d > 0)
	    generate_matrix(par->nx + d, par->ny, par->nz, par->blk, par->scramble, &G[d]);
	else
	    G[d] = *G0;
	md[d] = (int) G[d].n; nnzd[d] = (int) G[d].nnz; ld[d] = md[d];
	val[d] = SMALLOC(G[d].nnz);
	RHSptr[d] = SMALLOC((size_t) md[d] * nrhs);
	bsave[d] = SMALLOC((size_t) md[d] * nrhs);
	xtrue[d] = SMALLOC((size_t) md[d] * nrhs);
	Xptr[d] = SMALLOC((size_t) md[d] * nrhs);
	Berrs[d] = RMALLOC(nrhs);
	RpivPtr[d] = int32Malloc_dist(md[d]);
	CpivPtr[d] = int32Malloc_dist(md[d]);
	DiagScale[d] = NOEQUIL;
	ReqPtr[d] = CeqPtr[d] = NULL;
    }

    for (istep = 0; istep < nsteps; ++istep, ++ver) {
	char step = par->steps[istep];
	double worst_resid = 0.0, worst_ferr = 0.0;

	if (step == 'D') options->Fact = DOFACT;
	else if (step == 'R' && variable && istep > 0) options->Fact = SamePattern_SameRowPerm;
	else {
	    printf("  the batched interface supports -f D%s\n", variable ? ", then R" : "");
	    nfail++;
	    break;
	}

	/* Fresh matrices in the original CSC structure, fresh right-hand sides */
	for (d = 0; d < nb; ++d) {
	    int_t n = G[d].n, *rowind, *colptr;
	    scalar_t *nzval;
	    int j;
	    matrix_values(&G[d], ver, d, par->delta, 0, val[d]);
	    /* the row format of A is the column format of A' */
	    X_(CompCol_to_CompRow_dist)(n, n, G[d].nnz, val[d], G[d].rowptr, G[d].colind,
					&nzval, &colptr, &rowind);
	    X_(Create_CompCol_Matrix_dist)(&Amat[d], n, n, G[d].nnz, nzval, rowind, colptr,
					   SLU_NC, SLU_DT, SLU_GE);
	    handles[d] = (handle_t) &Amat[d];
	    exact_solution(n, nrhs, ver, d, xtrue[d]);
	    for (j = 0; j < nrhs; ++j)
		matvec_rows(&G[d], val[d], 0, 0, n, xtrue[d] + (size_t) j * n,
			    bsave[d] + (size_t) j * n);
	    memcpy(RHSptr[d], bsave[d], (size_t) n * nrhs * sizeof(scalar_t));
	}

	PStatInit(&stat);
	info = 0;
	if (variable)
	    P_(gssvx3d_csc_vbatch)(options, nb, md, md, nnzd, nrhs, handles, RHSptr, ld,
				   ReqPtr, CeqPtr, RpivPtr, CpivPtr, DiagScale, F,
				   Xptr, ld, Berrs, grid3d, &stat, &info);
	else
	    P_(gssvx3d_csc_batch)(options, nb, md[0], md[0], nnzd[0], nrhs, handles,
				  RHSptr, ld, ReqPtr, CeqPtr, RpivPtr, CpivPtr, DiagScale,
				  F, Xptr, ld, Berrs, grid3d, &stat, &info);
	PStatFree(&stat);

	printf("  call %d: %s, batch of %d systems, Fact = %s\n", istep + 1,
	       variable ? "variable-size" : "fixed-size", nb,
	       step == 'D' ? "DOFACT" : "SamePattern_SameRowPerm");
	if (info) {
	    printf("    ERROR: info = %d returned from the batched solver\n", info);
	    nfail++;
	} else {
	    for (d = 0; d < nb; ++d) {
		double resid, ferr;
		check_solution(&G[d], val[d], 0, nrhs, Xptr[d], bsave[d], xtrue[d],
			       &resid, &ferr);
		if (!(resid <= worst_resid)) worst_resid = resid;
		if (!(ferr <= worst_ferr)) worst_ferr = ferr;
	    }
	    nfail += report(0, "backward error (worst system)", worst_resid, par->tol_resid);
	    nfail += report(0, "forward error (worst system)", worst_ferr, par->tol_ferr);
	}
	for (d = 0; d < nb; ++d) Destroy_CompCol_Matrix_dist(&Amat[d]);
	if (info) break;
    }

    if (variable) X_(vbatch_free)(F);
    for (d = 0; d < nb; ++d) {
	if (DiagScale[d] == ROW || DiagScale[d] == BOTH) SUPERLU_FREE(ReqPtr[d]);
	if (DiagScale[d] == COL || DiagScale[d] == BOTH) SUPERLU_FREE(CeqPtr[d]);
	if (variable && !par->file && d > 0) free_matrix(&G[d]);
	SUPERLU_FREE(val[d]); SUPERLU_FREE(RHSptr[d]); SUPERLU_FREE(bsave[d]);
	SUPERLU_FREE(xtrue[d]); SUPERLU_FREE(Xptr[d]); SUPERLU_FREE(Berrs[d]);
	SUPERLU_FREE(RpivPtr[d]); SUPERLU_FREE(CpivPtr[d]);
    }
    SUPERLU_FREE(G); SUPERLU_FREE(handles); SUPERLU_FREE(val); SUPERLU_FREE(RHSptr);
    SUPERLU_FREE(bsave); SUPERLU_FREE(xtrue); SUPERLU_FREE(Xptr); SUPERLU_FREE(ReqPtr);
    SUPERLU_FREE(CeqPtr); SUPERLU_FREE(Berrs); SUPERLU_FREE(RpivPtr); SUPERLU_FREE(CpivPtr);
    SUPERLU_FREE(DiagScale); SUPERLU_FREE(md); SUPERLU_FREE(nnzd); SUPERLU_FREE(ld);
    SUPERLU_FREE(Amat);
    return nfail;
}

/* ------------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    X_(ScalePermstruct_t) ScalePermstruct;
    X_(LUstruct_t) LUstruct;
    X_(SOLVEstruct_t) SOLVEstruct;
    gridinfo3d_t grid3d;
    gridinfo_t grid2d_only, *grid;
    MPI_Comm comm;
    MPI_Datatype scalar_type;
    params_t par;
    gmat_t G;
    scalar_t *val = NULL, *b = NULL, *bglob = NULL, *xtrue = NULL, *xglob = NULL, *work = NULL;
    real_t *berr = NULL;
    int_t n, m_loc, fst_row, nnz_loc, i;
    int iam, np, info, nfail = 0, result, provided, use3d, p, j;
    int have_A = 0, have_LU = 0, ver = 0, istep, nsteps;
    int *counts = NULL, *displs = NULL;
    const char *why;
    char **cpp, c;

    memset(&par, 0, sizeof(par));
    par.nprow = par.npcol = 1; par.npdep = 0;
    par.nx = par.ny = 12; par.nz = 1;
    strcpy(par.steps, "D");
    par.blk = 1;
    par.equil = par.rowperm = par.colperm = par.ir = par.lookahead = -1;
    par.ilu = -1; par.gpures = -1; par.diaginv = -1; par.parsymb = -1;
    par.la_etree = par.tinypivot = -1;
    par.nrhs = 1;
    par.tol_resid = TOL_RESID; par.tol_ferr = TOL_FERR; par.tol_ilu = TOL_ILU;
    par.delta = 0.01;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    for (cpp = argv + 1; *cpp; ++cpp) {
	if (**cpp == '-') {
	    c = *(*cpp + 1);
	    if (c == 'h') { usage(); MPI_Finalize(); return 0; }
	    ++cpp;
	    if (!*cpp) { usage(); MPI_Abort(MPI_COMM_WORLD, 2); }
	    switch (c) {
	    case 'r': par.nprow = atoi(*cpp); break;
	    case 'c': par.npcol = atoi(*cpp); break;
	    case 'd': par.npdep = atoi(*cpp); break;
	    case 'x': par.nx = atoi(*cpp); break;
	    case 'y': par.ny = atoi(*cpp); break;
	    case 'z': par.nz = atoi(*cpp); break;
	    case 'k': par.scramble = atoi(*cpp); break;
	    case 'U': par.blk = atoi(*cpp); break;
	    case 'f': strncpy(par.steps, *cpp, sizeof(par.steps) - 1); break;
	    case 'e': par.equil = atoi(*cpp); break;
	    case 'p': par.rowperm = atoi(*cpp); break;
	    case 'q': par.colperm = atoi(*cpp); break;
	    case 'i': par.ir = atoi(*cpp); break;
	    case 's': par.nrhs = atoi(*cpp); break;
	    case 'l': par.lookahead = atoi(*cpp); break;
	    case 'I': par.ilu = atoi(*cpp); break;
	    case 'S': par.solveonly = atoi(*cpp); break;
	    case 'b': par.batch = atoi(*cpp); break;
	    case 'B': par.vbatch = atoi(*cpp); break;
	    case 'g': par.gpures = atoi(*cpp); break;
	    case 'G': par.gmres = atoi(*cpp); break;
	    case 'D': par.diaginv = atoi(*cpp); break;
	    case 'P': par.parsymb = atoi(*cpp); break;
	    case 'L': par.la_etree = atoi(*cpp); break;
	    case 'R': par.tinypivot = atoi(*cpp); break;
	    case 't': par.tol_resid = atof(*cpp); break;
	    case 'T': par.tol_ferr = atof(*cpp); break;
	    case 'u': par.tol_ilu = atof(*cpp); break;
	    case 'w': par.delta = atof(*cpp); break;
	    case 'v': par.verbose = atoi(*cpp); break;
	    default: usage(); MPI_Abort(MPI_COMM_WORLD, 2);
	    }
	} else {
	    par.file = *cpp;
	    break;
	}
    }
    use3d = (par.npdep >= 1);
    nsteps = (int) strlen(par.steps);

    /* ------------------------------------------------------------
       Options.  A CI run must not depend on the defaults of the day,
       so the refinement is off unless asked for.
       ------------------------------------------------------------ */
    set_default_options_dist(&options);
    options.IterRefine = NOREFINE;
    options.PrintStat = par.verbose ? YES : NO;
    if (par.equil != -1) options.Equil = par.equil;
    if (par.rowperm != -1) options.RowPerm = par.rowperm;
    if (par.colperm != -1) options.ColPerm = par.colperm;
    if (par.ir != -1) options.IterRefine = par.ir;
    if (par.lookahead != -1) options.num_lookaheads = par.lookahead;
    if (par.gpures != -1) options.GPURES = par.gpures;
    if (par.diaginv != -1) options.DiagInv = par.diaginv;
    if (par.parsymb != -1) options.ParSymbFact = par.parsymb;
    if (par.ilu != -1) options.ILU_level = par.ilu;
    if (par.la_etree != -1) options.lookahead_etree = par.la_etree;
    if (par.tinypivot != -1) options.ReplaceTinyPivot = par.tinypivot;
    if (par.solveonly) options.SolveOnly = YES;
    /* The dense blocks of a generated matrix are the supernodes of its ILU */
    if (par.blk > 1 && par.ilu != -1) options.UserDefineSupernode = YES;
    options.UseGMRES = par.gmres;
    if (par.batch > 0) options.batchCount = par.batch;
    if (par.vbatch > 0) { options.batchCount = par.vbatch; options.Algo3d = YES; }
#ifndef HAVE_METIS
    /* The default ordering needs METIS; fall back as the library would. */
    if (par.colperm == -1) options.ColPerm = MMD_AT_PLUS_A;
#endif

    /* ------------------------------------------------------------
       Process grid
       ------------------------------------------------------------ */
    if (use3d || par.batch > 0 || par.vbatch > 0) {
	if (par.npdep < 1) par.npdep = 1;
	use3d = 1;
	superlu_gridinit3d(MPI_COMM_WORLD, par.nprow, par.npcol, par.npdep, &grid3d);
	grid = &grid3d.grid2d;
	comm = grid3d.comm;
	iam = grid3d.iam;
    } else {
	superlu_gridinit(MPI_COMM_WORLD, par.nprow, par.npcol, &grid2d_only);
	grid = &grid2d_only;
	comm = grid->comm;
	iam = grid->iam;
    }
    if (iam == -1 || iam >= par.nprow * par.npcol * (use3d ? par.npdep : 1)) {
	/* not in the grid: nothing to do, and nothing to report */
	if (use3d) superlu_gridexit3d(&grid3d); else superlu_gridexit(grid);
	MPI_Finalize();
	return EXIT_PASS;
    }
    MPI_Comm_size(comm, &np);

    if (!iam) {
	int v_major, v_minor, v_bugfix;
	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("%s: SuperLU_DIST %d.%d.%d, int_t %d bytes, %s interface, process grid %d x %d",
	       PRECNAME, v_major, v_minor, v_bugfix, (int) sizeof(int_t),
	       use3d ? "3D" : "2D", par.nprow, par.npcol);
	if (use3d) printf(" x %d", par.npdep);
	printf("\n  GPU factorization (SUPERLU_ACC_OFFLOAD) %d, GPU solve (SUPERLU_ACC_SOLVE) %d, "
	       "GPU3DVERSION %d\n", get_acc_offload(&options), get_acc_solve(),
	       (int) sp_ienv_dist(12, &options));
	fflush(stdout);
    }

    why = unsupported(&par, &options);
#if defined(GPU_ACC) && defined(HAVE_CUDA) && defined(HAVE_NVSHMEM)
    if (get_acc_solve()) {
	/* The library aborts on a 2D grid with one process on one GPU and
	   several on another, where NVSHMEM would hang.  Each Z-layer has its
	   own 2D grid: skip if any of them is laid out that way. */
	int mixed = superlu_gpu_sharing_is_mixed(grid->comm, NULL, NULL), any;
	MPI_Allreduce(&mixed, &any, 1, MPI_INT, MPI_MAX, comm);
	if (!why && any)
	    why = "NVSHMEM hangs with one process on one GPU and several on another: "
		  "use a multiple of the number of GPUs or SUPERLU_MPI_PROCESS_PER_GPU";
    }
#endif
    if (!why && par.solveonly && !use3d) why = "SolveOnly is implemented in the 3D interface only";
    if (!why && par.gmres && use3d) why = "UseGMRES is implemented in the 2D interface only";
    if (!why && (par.blk < 1 || (par.blk > 1 && par.file)))
	why = "-U needs a generated matrix and a block size >= 1";
    if (!why && options.ColPerm == MY_PERMC && par.file)
	why = "-q 8 (MY_PERMC) orders the grid of a generated matrix";
    if (!why && options.UserDefineSupernode == YES &&
	(par.scramble || options.RowPerm != NOROWPERM ||
	 (options.ColPerm != NATURAL && options.ColPerm != MY_PERMC) ||
	 options.ParSymbFact == YES || par.batch > 0 || par.vbatch > 0))
	why = "user-defined supernodes need the blocks kept together: "
	      "-k 0, -p 0, -q 0 or 8, serial symbolic factorization, no batch";
    if (!why && (par.batch > 0 || par.vbatch > 0) && np != 1)
	why = "the batched interfaces run on one process";
    if (why) {
	if (!iam) printf("  cannot run: %s\nSLU_CI_RESULT: SKIP\n", why);
	if (use3d) superlu_gridexit3d(&grid3d); else superlu_gridexit(grid);
	MPI_Finalize();
	return EXIT_SKIP;
    }

    /* ------------------------------------------------------------
       Matrix: built on process 0, copied to every process
       ------------------------------------------------------------ */
    if (!iam) {
	if (par.file) read_matrix(par.file, &G);
	else generate_matrix(par.nx, par.ny, par.nz, par.blk, par.scramble, &G);
	if (par.file) printf("  matrix %s", par.file);
	else {
	    printf("  generated %d x %d x %d grid%s", par.nx, par.ny, par.nz,
		   par.scramble ? ", rows scrambled" : "");
	    if (par.blk > 1)
		printf(", %d unknowns per point%s", par.blk,
		       options.UserDefineSupernode == YES ? " = user-defined supernodes" : "");
	}
	printf(": n %ld, nnz %ld, nrhs %d\n", (long) G.n, (long) G.nnz, par.nrhs);
	if (options.PrintStat) {
	    print_sp_ienv_dist(&options);
	    print_options_dist(&options);
	}
	fflush(stdout);
    }
    bcast_matrix(&G, iam, comm);
    n = G.n;

    if (par.batch > 0 || par.vbatch > 0) {
	nfail = run_batch(&par, &options, &G, &grid3d);
	free_matrix(&G);
	goto finish;
    }

    /* Block row distribution over the processes of the grid */
    m_loc = n / np;
    fst_row = iam * m_loc;
    if (iam == np - 1) m_loc = n - fst_row;
    counts = int32Malloc_dist(np);
    displs = int32Malloc_dist(np);
    for (p = 0; p < np; ++p) {
	displs[p] = (int) (p * (n / np));
	counts[p] = (p == np - 1) ? (int) (n - displs[p]) : (int) (n / np);
    }
    MPI_Type_contiguous((int) sizeof(scalar_t), MPI_BYTE, &scalar_type);
    MPI_Type_commit(&scalar_type);

    val = SMALLOC(G.nnz);
    xtrue = SMALLOC((size_t) n * par.nrhs);
    xglob = SMALLOC((size_t) n * par.nrhs);
    bglob = SMALLOC((size_t) n * par.nrhs);
    work = SMALLOC(n);
    b = SMALLOC((size_t) (m_loc > 0 ? m_loc : 1) * par.nrhs);
    if (!(berr = RMALLOC(par.nrhs))) ABORT("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       The sequence of calls
       ------------------------------------------------------------ */
    for (istep = 0; istep < nsteps; ++istep) {
	char step = par.steps[istep];
	const char *name;
	double resid, ferr;
	int ldb = (int) (m_loc > 0 ? m_loc : 1);

	if (step != 'D' && !have_LU) {
	    if (!iam) printf("  -f %s: the sequence must start with D\n", par.steps);
	    nfail++;
	    break;
	}

	switch (step) {
	case 'D':
	    name = "DOFACT";
	    if (have_LU) { /* start over: release everything of the previous calls */
		X_(SolveFinalize)(&options, &SOLVEstruct);
#ifdef GPU_ACC
		if (get_acc_solve()) P_(gstrs_delete_device_lsum_x)(&SOLVEstruct);
#endif
		X_(Destroy_LU)(n, grid, &LUstruct);
		if (use3d) X_(Destroy_A3d_gathered_on_2d)(&SOLVEstruct, &grid3d);
		X_(ScalePermstructFree)(&ScalePermstruct);
		X_(LUstructFree)(&LUstruct);
	    }
	    X_(ScalePermstructInit)(n, n, &ScalePermstruct);
	    X_(LUstructInit)(n, &LUstruct);
	    options.Fact = DOFACT;
	    if (options.ColPerm == MY_PERMC)
		redblack_perm_c(par.nx, par.ny, par.nz, par.blk, ScalePermstruct.perm_c);
	    if (istep > 0) ++ver;
	    break;
	case 'F':
	    name = "FACTORED";
	    options.Fact = FACTORED;
	    break;
	case 'P':
	    name = "SamePattern";
	    X_(Destroy_LU)(n, grid, &LUstruct);
	    options.Fact = SamePattern;
	    ++ver;
	    break;
	case 'R':
	    name = "SamePattern_SameRowPerm";
	    if (!use3d || grid3d.zscp.Iam == 0) { /* the factors live on layer 0 */
		X_(ZeroLblocks)(iam, n, grid, &LUstruct);
		X_(ZeroUblocks)(iam, n, grid, &LUstruct);
	    }
	    options.Fact = SamePattern_SameRowPerm;
	    ++ver;
	    break;
	default:
	    if (!iam) printf("  -f %s: unknown letter '%c'\n", par.steps, step);
	    nfail++;
	    goto done;
	}

	/* User-defined supernodes: one per grid point.  The partition is in
	   the final column order, and holds for any ordering that keeps the
	   unknowns of a point together because the blocks have one size (the
	   postordering of the etree keeps them together too: inside a dense
	   block the etree is a chain).  The symbolic factorization runs on
	   every process in 2D and on Z-layer 0 in 3D, which then sends the
	   partition to the other layers.  Destroy_LU() frees the arrays. */
	if (options.UserDefineSupernode == YES && (step == 'D' || step == 'P') &&
	    (!use3d || grid3d.zscp.Iam == 0)) {
	    int_t *supno = intMalloc_dist(n), *xsup = intMalloc_dist(n + 1);
	    for (i = 0; i < n; ++i) supno[i] = i / par.blk;
	    for (i = 0; i <= n / par.blk; ++i) xsup[i] = i * par.blk;
	    LUstruct.Glu_persist->supno = supno;
	    LUstruct.Glu_persist->xsup = xsup;
	}

	/* The distributed matrix.  The solver overwrites it (equilibration),
	   and FACTORED expects that overwritten matrix back. */
	if (step != 'F') {
	    int_t *rowptr, *colind;
	    scalar_t *nzval;
	    if (have_A) Destroy_CompRowLoc_Matrix_dist(&A);
	    matrix_values(&G, ver, 0, par.delta, par.solveonly, val);
	    nnz_loc = G.rowptr[fst_row + m_loc] - G.rowptr[fst_row];
	    rowptr = intMalloc_dist(m_loc + 1);
	    colind = intMalloc_dist(nnz_loc > 0 ? nnz_loc : 1);
	    nzval = SMALLOC(nnz_loc > 0 ? nnz_loc : 1);
	    for (i = 0; i <= m_loc; ++i)
		rowptr[i] = G.rowptr[fst_row + i] - G.rowptr[fst_row];
	    for (i = 0; i < nnz_loc; ++i) {
		colind[i] = G.colind[G.rowptr[fst_row] + i];
		nzval[i] = val[G.rowptr[fst_row] + i];
	    }
	    X_(Create_CompRowLoc_Matrix_dist)(&A, n, n, nnz_loc, m_loc, fst_row,
					      nzval, colind, rowptr,
					      SLU_NR_loc, SLU_DT, SLU_GE);
	    have_A = 1;
	}

	/* Exact solution and right-hand side of this call */
	exact_solution(n, par.nrhs, 7 * ver + istep, 0, xtrue);
	for (j = 0; j < par.nrhs; ++j) {
	    scalar_t *bj = bglob + (size_t) j * n;
	    if (par.solveonly) {
		matvec_rows(&G, val, 1, 0, n, xtrue + (size_t) j * n, work);
		matvec_rows(&G, val, 2, 0, n, work, bj);
	    } else
		matvec_rows(&G, val, 0, 0, n, xtrue + (size_t) j * n, bj);
	    memcpy(b + (size_t) j * ldb, bj + fst_row, (size_t) m_loc * sizeof(scalar_t));
	}

	PStatInit(&stat);
	info = 0;
	{
	    scalar_t *rhs = b;
#ifdef GPU_ACC
	    scalar_t *d_b = NULL;
	    size_t nbytes = sizeof(scalar_t) * (size_t) ldb * (size_t) par.nrhs;
	    if (options.GPURES == YES) {
		checkGPU(gpuMalloc((void **) &d_b, nbytes));
		checkGPU(gpuMemcpy(d_b, b, nbytes, gpuMemcpyHostToDevice));
		rhs = d_b;
	    }
#endif
	    if (use3d)
		P_(gssvx3d)(&options, &A, &ScalePermstruct, rhs, ldb, par.nrhs, &grid3d,
			    &LUstruct, &SOLVEstruct, berr, &stat, &info);
	    else
		P_(gssvx)(&options, &A, &ScalePermstruct, rhs, ldb, par.nrhs, grid,
			  &LUstruct, &SOLVEstruct, berr, &stat, &info);
#ifdef GPU_ACC
	    if (options.GPURES == YES) {
		checkGPU(gpuMemcpy(b, d_b, nbytes, gpuMemcpyDeviceToHost));
		checkGPU(gpuFree(d_b));
	    }
#endif
	}
	if (options.PrintStat && (!use3d || grid3d.zscp.Iam == 0))
	    PStatPrint(&options, &stat, grid);
	PStatFree(&stat);
	have_LU = 1;

	if (!iam) {
	    printf("  call %d: Fact = %s\n", istep + 1, name);
	    fflush(stdout);
	}
	MPI_Allreduce(MPI_IN_PLACE, &info, 1, MPI_INT, MPI_MAX, comm);
	if (info) {
	    if (!iam) printf("    ERROR: info = %d returned from the solver\n", info);
	    nfail++;
	    break;
	}

	/* Collect the solution and check it against this call's system */
	for (j = 0; j < par.nrhs; ++j)
	    MPI_Allgatherv(b + (size_t) j * ldb, (int) m_loc, scalar_type,
			   xglob + (size_t) j * n, counts, displs, scalar_type, comm);
	if (!iam) {
	    int refine = (options.IterRefine != NOREFINE || options.UseGMRES);
	    check_solution(&G, val, par.solveonly, par.nrhs, xglob, bglob, xtrue,
			   &resid, &ferr);
	    if (par.ilu == -1 || refine) {
		nfail += report(iam, "backward error", resid, par.tol_resid);
		nfail += report(iam, "forward error", ferr, par.tol_ferr);
	    } else {
		/* An incomplete factorization does not solve the system: the
		   solution must be the one of the reference ILU(0). */
		scalar_t *xref = SMALLOC((size_t) n * par.nrhs);
		int_t *rowmap = intMalloc_dist(n);
		double diff = 0.0, xn = 0.0, v;
		int_t k;
		solver_row_order(n, np, displs, use3d && par.npdep > 1 &&
				 grid3d.rankorder != 1 && options.RowPerm != NOROWPERM,
				 par.nprow * par.npcol, rowmap);
		printf("    %-34s %10.3e\n", "backward error (ILU, informative)", resid);
		if (ilu0_reference(&G, val, ScalePermstruct.DiagScale, ScalePermstruct.R,
				   ScalePermstruct.C, ScalePermstruct.perm_r,
				   ScalePermstruct.perm_c, rowmap, par.nrhs, bglob, xref)) {
		    printf("    the reference ILU(0) met a zero pivot\n");
		    nfail++;
		} else {
		    for (k = 0; k < n * par.nrhs; ++k) {
			v = S_abs(S_sub(xglob[k], xref[k])); if (!(v <= diff)) diff = v;
			v = S_abs(xref[k]);                  if (v > xn) xn = v;
		    }
		    nfail += report(iam, "difference with reference ILU(0)", diff / xn,
				    par.tol_ilu);
		}
		SUPERLU_FREE(xref); SUPERLU_FREE(rowmap);
	    }
	}
	MPI_Bcast(&nfail, 1, MPI_INT, 0, comm);
    }
done:

    /* ------------------------------------------------------------
       Release everything
       ------------------------------------------------------------ */
    if (have_LU) {
	X_(SolveFinalize)(&options, &SOLVEstruct);
#ifdef GPU_ACC
	if (get_acc_solve()) P_(gstrs_delete_device_lsum_x)(&SOLVEstruct);
#endif
	X_(Destroy_LU)(n, grid, &LUstruct);
	if (use3d) X_(Destroy_A3d_gathered_on_2d)(&SOLVEstruct, &grid3d);
	X_(ScalePermstructFree)(&ScalePermstruct);
	X_(LUstructFree)(&LUstruct);
    }
    if (have_A) Destroy_CompRowLoc_Matrix_dist(&A);
    MPI_Type_free(&scalar_type);
    SUPERLU_FREE(counts); SUPERLU_FREE(displs);
    SUPERLU_FREE(val); SUPERLU_FREE(xtrue); SUPERLU_FREE(xglob); SUPERLU_FREE(bglob);
    SUPERLU_FREE(work); SUPERLU_FREE(b); SUPERLU_FREE(berr);
    free_matrix(&G);

finish:
    MPI_Bcast(&nfail, 1, MPI_INT, 0, comm);
    result = nfail ? EXIT_FAIL : EXIT_PASS;
    if (!iam) {
	printf("SLU_CI_RESULT: %s\n", nfail ? "FAIL" : "PASS");
	fflush(stdout);
    }
    if (use3d) superlu_gridexit3d(&grid3d); else superlu_gridexit(grid);
    MPI_Finalize();
    return result;
}
