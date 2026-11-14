/* What SuperLU_DIST does with singular matrices. Every case is n = 20 in double precision, and
   the structurally singular ones should return info > 0, as pdgssvx documents.

   mpiexec -n 2 ./singular {ok|numeric|zero_row|zero_col|structural} {2d|abglobal|3d}

   Produced from SuperLUDIST.jl testing:
   https://github.com/xiaoyeli/superlu_dist/issues/246
*/
#include <string.h>
#include "superlu_ddefs.h"

#define N 20
static const char *which;

/* The value of A(i,j), 0-based, or 0 if there is no entry. */
static double entry(int i, int j) {
    double v = i == j ? 4.0 : (abs(i - j) == 1 ? -1.0 : 0.0);
    if (!strcmp(which, "zero_row") && i == 5) return 0.0;
    if (!strcmp(which, "zero_col") && j == 5) return 0.0;
    /* Rows 5 and 6 both have their only entry in column 5. No row or column is empty,
       but the structural rank is n - 1. */
    if (!strcmp(which, "structural") && (i == 5 || i == 6)) return j == 5 ? 1.0 : 0.0;
    /* Rows 5 and 6 are identical, with a full structural rank: the control for info > 0. */
    if (!strcmp(which, "numeric")) {
	if (i == 5 || i == 6) return (j == 5 || j == 6) ? 1.0 : 0.0;
	if (j == 5 || j == 6) return 0.0;
    }
    return v;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    which = argv[1];
    const char *driver = argv[2];
    int rank, size, info = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    superlu_dist_options_t options;
    set_default_options_dist(&options);
    options.PrintStat = NO;
    SuperLUStat_t stat;
    PStatInit(&stat);
    SuperMatrix A;
    dScalePermstruct_t perm;
    dScalePermstructInit(N, N, &perm);
    dLUstruct_t lu;
    dLUstructInit(N, &lu);
    dSOLVEstruct_t solve;
    double berr[1];
    double *b;

    if (!strcmp(driver, "abglobal")) {
	gridinfo_t grid;
	superlu_gridinit(MPI_COMM_WORLD, 1, size, &grid);
	int_t nnz = 0;
	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) nnz += entry(i, j) != 0.0;
	double *a = doubleMalloc_dist(nnz);
	int_t *ind = intMalloc_dist(nnz), *ptr = intMalloc_dist(N + 1);
	nnz = 0;
	for (int j = 0; j < N; j++) {
	    ptr[j] = nnz;
	    for (int i = 0; i < N; i++) if (entry(i, j) != 0.0) { a[nnz] = entry(i, j); ind[nnz++] = i; }
	}
	ptr[N] = nnz;
	dCreate_CompCol_Matrix_dist(&A, N, N, nnz, a, ind, ptr, SLU_NC, SLU_D, SLU_GE);
	b = doubleMalloc_dist(N);
	for (int i = 0; i < N; i++) b[i] = 1.0;
	pdgssvx_ABglobal(&options, &A, &perm, b, N, 1, &grid, &lu, berr, &stat, &info);
    } else {
	/* A block of rows on each rank. */
	int first = rank * N / size, last = (rank + 1) * N / size;
	int m_loc = last - first;
	int_t nnz = 0;
	for (int i = first; i < last; i++) for (int j = 0; j < N; j++) nnz += entry(i, j) != 0.0;
	double *a = doubleMalloc_dist(nnz + 1);
	int_t *ind = intMalloc_dist(nnz + 1), *ptr = intMalloc_dist(m_loc + 1);
	nnz = 0;
	for (int i = first; i < last; i++) {
	    ptr[i - first] = nnz;
	    for (int j = 0; j < N; j++) if (entry(i, j) != 0.0) { a[nnz] = entry(i, j); ind[nnz++] = j; }
	}
	ptr[m_loc] = nnz;
	dCreate_CompRowLoc_Matrix_dist(&A, N, N, nnz, m_loc, first, a, ind, ptr, SLU_NR_loc, SLU_D, SLU_GE);
	b = doubleMalloc_dist(m_loc);
	for (int i = 0; i < m_loc; i++) b[i] = 1.0;
	if (!strcmp(driver, "3d")) {
	    gridinfo3d_t grid;
	    superlu_gridinit3d(MPI_COMM_WORLD, 1, 1, size, &grid);
	    pdgssvx3d(&options, &A, &perm, b, m_loc, 1, &grid, &lu, &solve, berr, &stat, &info);
	} else {
	    gridinfo_t grid;
	    superlu_gridinit(MPI_COMM_WORLD, 1, size, &grid);
	    pdgssvx(&options, &A, &perm, b, m_loc, 1, &grid, &lu, &solve, berr, &stat, &info);
	}
    }
    printf("rank %d returned, info = %d\n", rank, info);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
