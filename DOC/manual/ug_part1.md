## Chapter 2

# Sequential SuperLU (Version 4.2)


## 2.1 About SuperLU

In this chapter, SuperLU will always mean Sequential SuperLU. package
contains a set of subroutines to solve sparse linear systems $AX=B$.
Here $A$ is a square, nonsingular, $n\times n$ sparse matrix, and $X$
and $B$ are dense $n\times nrhs$ matrices, where $nrhs$ is the number of
right-hand sides and solution vectors. Matrix $A$ need not be symmetric
or definite; indeed, is particularly appropriate for matrices with very
unsymmetric structure.

The package uses $LU$ decomposition with partial (or threshold)
pivoting, and forward/back substitutions. The columns of $A$ may be
preordered before factorization (either by the user or by ); this
preordering for sparsity is completely separate from the factorization.
To improve backward stability, we provide working precision iterative
refinement subroutines [@arioli89]. Routines are also available to
equilibrate the system, estimate the condition number, calculate the
relative backward error, and estimate error bounds for the refined
solutions. We also include a Matlab MEX-file interface, so that our
factor and solve routines can be called as alternatives to those built
into Matlab. The $LU$ factorization routines can handle non-square
matrices, but the triangular solves are performed only for square
matrices.

Starting from Version 4.0, we provide the incomplete factorization (ILU)
routines which can be used as preconditioners for iterative
solvers [@lishao10].

The factorization algorithm uses a graph reduction technique to reduce
graph traversal time in the symbolic analysis. We exploit dense
submatrices in the numerical kernel, and organize computational loops in
a way that reduces data movement between levels of the memory hierarchy.
The resulting algorithm is highly efficient on modern architectures. The
performance gains are particularly evident for large problems. There are
"tuning parameters" to optimize the peak performance as a function of
cache size. For a detailed description of the algorithm, see
reference [@superlu99].

 is implemented in ANSI C, and must be compiled with a standard ANSI C
compiler. It includes versions for both real and complex matrices, in
both single and double precision. The file names for the
single-precision real version start with letter "s" (such as
`sgstrf.c`); the file names for the double-precision real version start
with letter "d" (such as `dgstrf.c`); the file names for the
single-precision complex version start with letter "c" (such as
`cgstrf.c`); the file names for the double-precision complex version
start with letter "z" (such as `zgstrf.c`).

$$
\begin{aligned}
&\left(
\begin{array}{ccccc}
s &   & u & u &   \\
l & u &   &   &   \\
  & l & p &   &   \\
  &   &   & e & u \\
l & l &   &   & r 
\end{array}
\right)
&&
\left(
\begin{array}{ccccc}
19.00 &       & 21.00 & 21.00 &       \\
0.63  & 21.00 & -13.26& -13.26&       \\
      & 0.57  & 23.58 & 7.58  &       \\
      &       &       & 5.00  & 21.00 \\
0.63  & 0.57  & -0.24 & -0.77 & 34.20
\end{array}
\right)
\\[10pt]
& &nbsp; &nbsp; &nbsp; &nbsp;\text{Original matrix } A &&&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; \text{Factors } F = L + U - I \\[6pt]
&s = 19,\; u = 21,\; p = 16,\; e = 5,\; r = 18,\; l = 12
\end{aligned}
$$

<div style="text-align:center;">
Figure 2.1: A 5 x 5 matrix and its L and U factors.
</div>


(sec:ex5x5)=
# 2.2 How to call a SuperLU routine

As a simple example, let us consider how to solve a $5\times 5$ sparse
linear system $AX=B$, by calling a driver routine `dgssv()`.
Figure [\[5x5\]](#5x5) shows matrix $A$, and its $L$ and $U$ factors. This sample program is located
in `SuperLU/EXAMPLE/superlu.c.`

The program first initializes the three arrays, `a[], asub[]` and
`xa[]`, which store the nonzero coefficients of matrix $A$, their row
indices, and the indices indicating the beginning of each column in the
coefficient and row index arrays. This storage format is called
compressed column format, also known as Harwell-Boeing
format [@duffgrimes92]. Next, the two utility routines
`dCreate_CompCol_Matrix()` and `dCreate_Dense_Matrix()` are called to
set up the matrix structures for $A$ and $B$, respectively. The routine
`set_default_options()` sets the default values to the input `options`
argument. This controls how the matrix will be factorized and how the
system will be solved. After calling the SuperLU routine `dgssv()`, the $B$
matrix is overwritten by the solution matrix $X$. In the end, all the
dynamically allocated data structures are de-allocated by calling
various utility routines.
SuperLU can perform more general tasks, which will be explained later.

    #include "slu_ddefs.h"

    main(int argc, char *argv[])
    {
    /*
     * Purpose
     * =======
     * 
     * This is the small 5x5 example used in the Sections 2 and 3 of the 
     * Users' Guide to illustrate how to call a SuperLU routine, and the
     * matrix data structures used by SuperLU.
     *
     */
        SuperMatrix A, L, U, B;
        double   *a, *rhs;
        double   s, u, p, e, r, l;
        int      *asub, *xa;
        int      *perm_r; /* row permutations from partial pivoting */
        int      *perm_c; /* column permutation vector */
        int      nrhs, info, i, m, n, nnz, permc_spec;
        superlu_options_t options;
        SuperLUStat_t stat;

        /* Initialize matrix A. */
        m = n = 5;
        nnz = 12;
        if ( !(a = doubleMalloc(nnz)) ) ABORT("Malloc fails for a[].");
        if ( !(asub = intMalloc(nnz)) ) ABORT("Malloc fails for asub[].");
        if ( !(xa = intMalloc(n+1)) ) ABORT("Malloc fails for xa[].");
        s = 19.0; u = 21.0; p = 16.0; e = 5.0; r = 18.0; l = 12.0;
        a[0] = s; a[1] = l; a[2] = l; a[3] = u; a[4] = l; a[5] = l;
        a[6] = u; a[7] = p; a[8] = u; a[9] = e; a[10]= u; a[11]= r;
        asub[0] = 0; asub[1] = 1; asub[2] = 4; asub[3] = 1;
        asub[4] = 2; asub[5] = 4; asub[6] = 0; asub[7] = 2;
        asub[8] = 0; asub[9] = 3; asub[10]= 3; asub[11]= 4;
        xa[0] = 0; xa[1] = 3; xa[2] = 6; xa[3] = 8; xa[4] = 10; xa[5] = 12;

        /* Create matrix A in the format expected by SuperLU. */
        dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
        
        /* Create right-hand side matrix B. */
        nrhs = 1;
        if ( !(rhs = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhs[].");
        for (i = 0; i < m; ++i) rhs[i] = 1.0;
        dCreate_Dense_Matrix(&B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);

        if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");
        if ( !(perm_c = intMalloc(n)) ) ABORT("Malloc fails for perm_c[].");

        /* Set the default input options. */
        set_default_options(&options);
        options.ColPerm = NATURAL;

        /* Initialize the statistics variables. */
        StatInit(&stat);

        /* Solve the linear system. */
        dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
        
        dPrint_CompCol_Matrix("A", &A);
        dPrint_CompCol_Matrix("U", &U);
        dPrint_SuperNode_Matrix("L", &L);
        print_int_vec("\nperm_r", m, perm_r);

        /* De-allocate storage */
        SUPERLU_FREE (rhs);
        SUPERLU_FREE (perm_r);
        SUPERLU_FREE (perm_c);
        Destroy_CompCol_Matrix(&A);
        Destroy_SuperMatrix_Store(&B);
        Destroy_SuperNode_Matrix(&L);
        Destroy_CompCol_Matrix(&U);
        StatFree(&stat);
    }

(sec:rep)=
# 2.3 Matrix data structures

 uses a principal data structure `SuperMatrix` (defined in
`SRC/supermatrix.h`) to represent a general matrix, sparse or dense.
Figure [\[fig:struct\]](#fig:struct) gives the specification of the `SuperMatrix`
structure. The `SuperMatrix` structure contains two levels of fields.
The first level defines all the properties of a matrix which are
independent of how it is stored in memory. In particular, it specifies
the following three orthogonal properties: storage type (`Stype`)
indicates the type of the storage scheme in `Store`; data type (`Dtype`)
encodes the four precisions; mathematical type (`Mtype`) specifies some
mathematical properties. The second level (`Store`) points to the actual
storage used to store the matrix. We associate with each `Stype XX` a
storage format called `XXformat`, such as `NCformat`, `SCformat`, etc.


    typedef struct {
        Stype_t Stype; /* Storage type: indicates the storage format of *Store. */
        Dtype_t Dtype; /* Data type. */
        Mtype_t Mtype; /* Mathematical type */
        int  nrow;     /* number of rows */
        int  ncol;     /* number of columns */
        void *Store;   /* pointer to the actual storage of the matrix */
    } SuperMatrix;

    typedef enum {
        SLU_NC,        /* column-wise, not supernodal */
        SLU_NR,        /* row-wise, not supernodal */
        SLU_SC,        /* column-wise, supernodal */
        SLU_SR,        /* row-wise, supernodal */
        SLU_NCP,       /* column-wise, not supernodal, permuted by columns
                         (After column permutation, the consecutive columns of 
                          nonzeros may not be stored contiguously. */
        SLU_DN,        /* Fortran style column-wise storage for dense matrix */
        SLU_NR_loc     /* distributed compressed row format */ 
    } Stype_t;

    typedef enum {
        SLU_S,         /* single */
        SLU_D,         /* double */
        SLU_C,         /* single-complex */
        SLU_Z          /* double-complex */
    } Dtype_t;

    typedef enum {
        SLU_GE,        /* general */
        SLU_TRLU,      /* lower triangular, unit diagonal */
        SLU_TRUU,      /* upper triangular, unit diagonal */
        SLU_TRL,       /* lower triangular */
        SLU_TRU,       /* upper triangular */
        SLU_SYL,       /* symmetric, store lower half */
        SLU_SYU,       /* symmetric, store upper half */
        SLU_HEL,       /* Hermitian, store lower half */
        SLU_HEU        /* Hermitian, store upper half */
    } Mtype_t;

The `SuperMatrix` type so defined can accommodate various types of
matrix structures and appropriate operations to be applied on them,
although currently  implements only a subset of this collection.
Specifically, matrices $A$, $L$, $U$, $B$, and $X$ can have the
following types:

::: {.center}
             $A$   $L$   $U$   $B$   $X$
  --------- ----- ----- ----- ----- -----
  `Stype`     or                    
  `Dtype`    any   any   any   any   any
  `Mtype`                           
:::

In what follows, we illustrate the storage schemes defined by `Stype`.
Following C's convention, all array indices and locations below are
zero-based.

-   $A$ may have storage type or . The format is the same as the
    Harwell-Boeing sparse matrix format [@duffgrimes92], that is, the
    compressed column storage.

            typedef struct {
                int  nnz;     /* number of nonzeros in the matrix */
                void *nzval;  /* array of nonzero values packed by column */
                int  *rowind; /* array of row indices of the nonzeros */
                int  *colptr; /* colptr[j] stores the location in nzval[] and rowind[]
                                 which starts column j. It has ncol+1 entries, 
                                 and colptr[ncol] = nnz. */
            } NCformat;

    The format is the compressed row storage defined below.

            typedef struct {
                int  nnz;     /* number of nonzeros in the matrix */
                void *nzval;  /* array of nonzero values packed by row */
                int  *colind; /* array of column indices of the nonzeros */
                int  *rowptr; /* rowptr[j] stores the location in nzval[] and colind[]
                                 which starts row j. It has nrow+1 entries,
                                 and rowptr[nrow] = nnz. */
            } NRformat;

    The factorization and solve routines in are designed to handle
    column-wise storage only. If the input matrix $A$ is in row-oriented
    storage, i.e., in format, then the driver routines (`dgssv()` and
    `dgssvx()`) actually perform the $LU$ decomposition on $A^T$, which
    is column-wise, and solve the system using the $L^T$ and $U^T$
    factors. The data structures holding $L$ and $U$ on output are
    different (swapped) from the data structures you get from
    column-wise input. For more detailed descriptions about this
    process, please refer to the leading comments of the routines
    `dgssv()` and `dgssvx()`.

    Alternatively, the users may call a utility routine
    `dCompRow_to_CompCol()` to convert the input matrix in format to
    another matrix in format, before calling SuperLU. The definition of
    this routine is

            void dCompRow_to_CompCol(int m, int n, int nnz,
                                     double *a, int *colind, int *rowptr,
                                     double **at, int **rowind, int **colptr);

    This conversion takes time proportional to the number of nonzeros in
    $A$. However, it requires storage for a separate copy of matrix $A$.

-   $L$ is a supernodal matrix with the storage type . Due to the
    supernodal structure, $L$ is in fact stored as a sparse block lower
    triangular matrix [@superlu99].

            typedef struct {
                int  nnz;           /* number of nonzeros in the matrix */
                int  nsuper;        /* index of the last supernode */
                void *nzval;        /* array of nonzero values packed by column */
                int  *nzval_colptr; /* nzval_colptr[j] stores the location in
                                       nzval[] which starts column j */
                int  *rowind;       /* array of compressed row indices of 
                                       rectangular supernodes */
                int  *rowind_colptr;/* rowind_colptr[j] stores the location in
                                       rowind[] which starts column j */
                int  *col_to_sup;   /* col_to_sup[j] is the supernode number to 
                                       which column j belongs */
                int  *sup_to_col;   /* sup_to_col[s] points to the starting column
                                       of the s-th supernode */
            } SCformat;

-   Both $B$ and $X$ are stored as conventional two-dimensional arrays
    in column-major order, with the storage type .

            typedef struct {
                int lda;     /* leading dimension */
                void *nzval; /* array of size lda-by-ncol to represent 
                                a dense matrix */
            } DNformat;

Figure [\[matrixeg\]](#fig:matrixeg) shows the data structures for the example
matrices in Figure [\[5x5\]](#5x5).

For a description of `NCPformat`, see
section [5.1](#sec:permX).

# 2.4 `Options` argument

The `options` argument is the input argument to control the behaviour of
the library. The user can tell the solver how the linear systems should
be solved based on some known characteristics of the system. For
example, for diagonally dominant matrices, choosing the diagonal pivots
ensures stability; there is no need for numerical pivoting (i.e., $P_r$
can be an Identity matrix). In another situation where a sequence of
matrices with the same sparsity pattern need be factorized, the column
permutation $P_c$ (and also the row permutation $P_r$, if the numerical
values are similar) need be computed only once, and reused thereafter.
In these cases, the solvers' performance can be much improved over using
the default settings. `Options` is implemented as a C structure
containing the following fields:

-   `Fact`\
    Specifies whether or not the factored form of the matrix $A$ is
    supplied on entry, and if not, how the matrix $A$ will be factorized
    base on the previous history, such as factor from scratch, reuse
    $P_c$ and/or $P_r$, or reuse the data structures of $L$ and $U$.
    `fact` can be one of:

    -   `DOFACT`: the matrix $A$ will be factorized from scratch.

    -   `SamePattern`: the matrix $A$ will be factorized assuming that a
        factorization of a matrix with the same sparsity pattern was
        performed prior to this one. Therefore, this factorization will
        reuse column permutation vector `perm_c`.

    -   `SampPattern_SameRowPerm`: the matrix $A$ will be factorized
        assuming that a factorization of a matrix with the same sparsity
        pattern and similar numerical values was performed prior to this
        one. Therefore, this factorization will reuse both row and
        column permutation vectors `perm_r` and `perm_c`, both row and
        column scaling factors $D_r$ and $D_c$, and the distributed data
        structure set up from the previous symbolic factorization.

    -   `FACTORED`: the factored form of $A$ is input.

-   `Equil` { `YES` $|$ `NO` }\
    Specifies whether to equilibrate the system (scale $A$'s rows and
    columns to have unit norm).

-   `ColPerm`\
    Specifies how to permute the columns of the matrix for sparsity
    preservation.

    -   `NATURAL`: natural ordering.

    -   `MMD_ATA`: minimum degree ordering on the structure of $A^TA$.

    -   `MMD_AT_PLUS_A`: minimum degree ordering on the structure of
        $A^T+A$.

    -   `COLAMD`: approximate minimum degree column ordering

    -   `MY_PERMC`: use the ordering given in `perm_c` input by the
        user.

-   `Trans` { `NOTRANS` $|$ `TRANS` $|$ `CONJ` }\
    Specifies whether to solve the transposed system.

-   `IterRefine`\
    Specifies whether to perform iterative refinement, and in what
    precision to compute the residual.

    -   `NO`: no iterative refinement

    -   `SINGLE`: perform iterative refinement in single precision

    -   `DOUBLE`: perform iterative refinement in double precision

    -   `EXTRA`: perform iterative refinement in extra precision

-   `DiagPivotThresh` $[0.0, 1.0]$\
    Specifies the threshold used for a diagonal entry to be an
    acceptable pivot.

-   `SymmetricMode` { `YES` $|$ `NO` }\
    Specifies whether to use the symmetric mode. Symmetric mode gives
    preference to diagonal pivots, and uses an $(A^T + A)$-based column
    permutation algorithm.

-   `PivotGrowth` { `YES` $|$ `NO` }\
    Specifies whether to compute the reciprocal pivot growth.

-   `ConditionNumber` { `YES` $|$ `NO` }\
    Specifies whether to compute the reciprocal condition number.

-   `RowPerm` (only for ILU or SuperLU_DIST)\
    Specifies whether to permute the rows of the original matrix.

    -   `NO`: not to permute the rows

    -   `LargeDiag_MC64`: use a serial, weighted bipartite matching
        algorithm implemented in MC64 to permute the rows to make the
        diagonal large relative to the off-diagonal [@duffkoster01].

    -   `LargeDiag_AWPM`: use a parallel, approximate weighted bipartite
        matching algorithm implemented in CombBLAS to permute the rows
        to make the diagonal large relative to the off-diagonal [@awpm].

    -   `MY_PERMR`: use the permutation given by the user

-   `ILU_DropRule`\
    Specifies the dropping rule for ILU: ( Default:
    `DROP_BASIC | DROP_AREA` )

    -   `DROP_BASIC:` Basic dropping rule, supernodal based
        ILUTP($\tau$).

    -   `DROP_PROWS:` Supernodal based ILUTP($p,\tau$),
        $p = \gamma\cdot nnz(A)/n$.

    -   `DROP_COLUMN:` Variant of ILUTP($p,\tau$), for j-th column,
        $p = \gamma \cdot nnz(A(:,j))$.

    -   `DROP_AREA:` Variation of ILUTP, for j-th column, use
        $nnz(F(:,1:j)) / nnz(A(:,1:j))$ to control memory.

    -   `DROP_DYNAMIC:` Dynamically adjust the threshold $\tau$ during
        factorizaion:\
        If $nnz(L(:,1:j)) / nnz(A(:,1:j)) > \gamma$,
        $\tau_L(j) := \min(\tau_0, \tau_L(j-1)\cdot 2)$; Otherwise
        $\tau_L(j) := \max(\tau_0, \tau_L(j-1) / 2)$. $\tau_U(j)$ uses
        the similar rule.

    -   `DROP_INTERP:` Compute the second dropping threshold by
        interpolation instead of quick select (default). In this case,
        the actual fill ratio is not guaranteed to be smaller than
        gamma.

-   `ILU_DropTol` $[0.0, 1.0]$\
    Specifies the numerical dropping threshold for ILU.

-   `ILU_FillFactor` ($\ge 1.0$)\
    Specifies the expected fill ratio upper bound, $\gamma$, for ILU.

-   `ILU_MILU` { `SILU` $|$ `SMILU_1` $|$ ` SMILU_2` $|$ `SMILU_3` }\
    Specifies which version of modified ILU to use.

-   `PrintStat` { `YES` $|$ `NO` }\
    Specifies whether to print the solver's statistics.

The routine `set_default_options()` sets the following default values:

        Fact              = DOFACT       /* factor from scratch */
        Equil             = YES             
        ColPerm           = COLAMD
        Trans             = NOTRANS
        IterRefine        = NOREFINE
        DiagPivotThresh   = 1.0          /* partial pivoting */
        SymmetricMode     = NO
        PivotGrowth       = NO;
        ConditionNumber   = NO;
        PrintStat         = YES

To use the ILU routines, such as `dgsitrf()`, the user should call
`ilu_set_default_options()` to set the default values
(`set_default_options()` is first called in this routine prior to the
following):

        DiagPivotThresh   = 0.1          /* partial pivoting */
        RowPerm           = LargeDiag
        ILU_DropRule      = DROP_BASIC | DROP_AREA;
        ILU_DropTol       = 1e-4;
        ILU_FillFactor    = 10.0;
        ILU_Norm          = INF_NORM;
        ILU_MILU          = SILU;        /* not to use MILU */   
        ILU_FillTol       = 1e-2;

The other possible values for each field are documented in the source
code `SRC/slu_util.h`. The users can reset each default value according
to their needs.

(sec:perm)=
# 2.5 Permutations

Two permutation matrices are involved in the solution process. In fact,
the actual factorization we perform is $P_rAP_c^T=LU$, where $P_r$ is
determined from partial pivoting (with a threshold pivoting option), and
$P_c$ is a column permutation chosen either by the user or , usually to
make the $L$ and $U$ factors as sparse as possible. $P_r$ and $P_c$ are
represented by two integer vectors `perm_r[]` and `perm_c[]`, which are
the permutations of the integers $(0:m-1)$ and $(0:n-1)$, respectively.

(sec:permX)=
## 2.5.1 Ordering for sparsity 

Column reordering for sparsity is completely separate from the $LU$
factorization. The column permutation $P_c$ should be applied before
calling the factorization routine `dgstrf()`. In principle, any ordering
heuristic used for symmetric matrices can be applied to $A^TA$ (or
$A+A^T$ if the matrix is nearly structurally symmetric) to obtain $P_c$.
Currently, we provide the following ordering options through `options`
argument. The `options.ColPerm` field can take the following values:

-   `NATURAL`: use natural ordefring (i.e., $P_c = I$).

-   `MMD_AT_PLUS_A`: use minimum degree ordering on the structure of
    $A^T+A$.

-   `MMD_ATA`: use minimum degree ordering on the structure of $A^TA$.

-   `COLAMD`: use approximate minimum degree column ordering.

-   `MY_PERMC`: use the ordering given in the permutation vector
    `perm_c[]`, which is input by the user.

If `options.ColPerm` is set to the last value, the library will use the
permutation vector `perm_c[]` as an input, which may be obtained from
any other ordering algorithm. For example, the nested-dissection type of
ordering codes include Metis [@kaku:98a], Chaco [@hele:95] and
Scotch [@scotch].

Alternatively, the users can provide their own column permutation
vector. For example, it may be an ordering suitable for the underlying
physical problem. Both driver routines `dgssv` and `dgssvx` take
`perm_c[]` as an input argument.

After permutation $P_c$ is applied to $A$, we use format to represent
the permuted matrix $AP_c^T$, in which the consecutive columns of
nonzeros may not be stored contiguously in memory. Therefore, we need
two separate arrays of pointers, `colbeg[]` and `colend[]`, to indicate
the beginning and end of each column in `nzval[]` and `rowind[]`.

        typedef struct {
            int  nnz;     /* number of nonzeros in the matrix */
            void *nzval;  /* array of nonzero values, packed by column */
            int  *rowind; /* array of row indices of the nonzeros */
            int  *colbeg; /* colbeg[j] points to the location in nzval[] and rowind[]
                             which starts column j */
            int  *colend; /* colend[j] points to one past the location in nzval[]
                             and rowind[] which ends column j */
        } NCPformat;

## 2.5.2 Partial pivoting with threshold

We have included a threshold pivoting parameter $u\in [0,1]$ to control
numerical stability. The user can choose to use a row permutation
obtained from a previous factorization. (The argument
`options.Fact = SamePattern_SameRowPerm` should be passed to the
factorization routine `dgstrf()`.) The pivoting subroutine `dpivotL()`
checks whether this choice of pivot satisfies the threshold; if not, it
will try the diagonal element. If neither of the above satisfies the
threshold, the maximum magnitude element in the column will be used as
the pivot. The pseudo-code of the pivoting policy for column $j$ is
given below.

::: {.tabbing}
junk j̄unk j̄unk ̄ (1) compute $thresh = u~|a_{mj}|$, where
$|a_{mj}|=\max_{i\ge j}|a_{ij}|$;\
\
(2) **if** user specifies pivot row $k$ **and** $|a_{kj}|\ge thresh$
**and** $a_{kj}\ne 0$ **then**\
pivot row $= k$;\
**else if** $|a_{jj}| \ge thresh$ **and** $a_{jj}\ne 0$ **then**\
pivot row $= j$;\
**else**\
pivot row $= m$;\
**endif**;
:::

Two special values of $u$ result in the following two strategies:

-   $u=0.0$: either use user-specified pivot order if available, or else
    use diagonal pivot;

-   $u=1.0$: classical partial pivoting.

# 2.6 Symmetric Mode

In many applications, matrix $A$ may be diagonally dominant or nearly
so. In this case, pivoting on the diagonal is sufficient for stability
and is preferable for sparsity to off-diagonal pivoting. To do this, the
user can set a small (less-than-one) diagonal pivot threshold (e.g.,
0.0, 0.01) and choose an ($A^T + A$)--based column permutation
algorithm. We call this setting *symmetric mode*. In this case, the
`options.SymmetricMode = YES` must be set.

Note that, when a diagonal entry is smaller than the threshold, the code
will still choose an off-diagonal pivot. That is, the row permutation
$P_r$ may not be Identity. Please refer to [@li05] for more discussion
on the symmetric mode.

# 2.7 Incomplete LU factorization (ILU) preconditioner

Starting from SuperLU version 4.0, we provide the ILU routines to be
used as preconditioners for iterative solvers. Our ILU method can be
considered to be a variant of the ILUTP method originally proposed by
Saad [@saad94], which combines a dual dropping strategy with numerical
pivoting ("T" stands for threshold, and "P" stands for pivoting). We
adapted the classic dropping strategies of ILUTP in order to incorporate
supernode structures and to accommodate dynamic supernodes due to
partial pivoting. For the secondary dropping strategy, we proposed an
area-based fill control method, which is more flexible and numerically
robust than the traditional column-based scheme. Furthermore, we
incorporated several heuristics for adaptively modifying various
threshold parameters as the factorization proceeds, which improves the
robustness of the algorithm. The details can be found in [@lishao10].

# 2.8 Memory management for $L$ and $U$ {#sec:mem}

In the sparse $LU$ algorithm, the amount of space needed to hold the
data structures of $L$ and $U$ cannot be accurately predicted prior to
the factorization. The dynamically growing arrays include those for the
nonzero values (`nzval[]`) and the compressed row indices (`rowind[]`)
of $L$, and for the nonzero values (`nzval[]`) and the row indices
(`rowind[]`) of $U$.

Two alternative memory models are presented to the user:

-   system-level -- based on C's dynamic allocation capability
    (`malloc/free`);

-   user-level -- based on a user-supplied `work[]` array of size
    `lwork` (in bytes). This is similar to Fortran-style handling of
    work space. `Work[]` is organized as a two-ended stack, one end
    holding the $L$ and $U$ data structures, the other end holding the
    auxiliary arrays of known size.

Except for the different ways to allocate/deallocate space, the logical
view of the memory organization is the same for both schemes. Now we
describe the policies in the memory module.

At the outset of the factorization, we guess there will be `FILL*nnz(A)`
fills in the factors and allocate corresponding storage for the above
four arrays, where `nnz(A)` is the number of nonzeros in original matrix
$A$, and `FILL` is an integer, say 20. (The value of `FILL` can be set
in an inquiry function `sp_ienv()`, see
section [11.3](#sec:parameters){reference-type="ref"
reference="sec:parameters"}.) If this initial request exceeds the
physical memory constraint, the `FILL` factor is repeatedly reduced, and
attempts are made to allocate smaller arrays, until the initial
allocation succeeds.

During the factorization, if any array size exceeds the allocated bound,
we expand it as follows. We first allocate a chunk of new memory of size
`EXPAND` times the old size, then copy the existing data into the new
memory, and then free the old storage. The extra copying is necessary,
because the factorization algorithm requires that each of the
aforementioned four data structures be *contiguous* in memory. The
values of `FILL` and `EXPAND` are normally set to 20 and 1.5,
respectively. See `xmemory.c` for details.

After factorization, we do not garbage-collect the extra space that may
have been allocated. Thus, there will be external fragmentation in the
$L$ and $U$ data structures. The settings of `FILL` and `EXPAND` should
take into account the trade-off between the number of expansions and the
amount of fragmentation.

Arrays of known size, such as various column pointers and working
arrays, are allocated just once. All dynamically-allocated working
arrays are freed after factorization.

(sec:routine)=
# 2.9 User-callable routines 

The naming conventions, calling sequences and functionality of these
routines mimic the corresponding  software [@lapackmanual2]. In the
routine names, such as `dgstrf`, we use the two letters `GS` to denote
*general sparse* matrices. The leading letter`x` stands for `S, D, C`,
or `Z`, specifying the data type.

## 2.9.1 Driver routines

We provide two types of driver routines for solving systems of linear
equations. The driver routines can handle both column- and row-oriented
storage schemes.

-   A simple driver `dgssv()`, which solves the system $AX=B$ by
    factorizing $A$ and overwriting $B$ with the solution $X$.

-   An expert driver `dgssvx()`, which, in addition to the above, also
    performs the following functions (some of them optionally):

    -   solve $A^TX=B$;

    -   equilibrate the system (scale $A$'s rows and columns to have
        unit norm) if $A$ is poorly scaled;

    -   estimate the condition number of $A$, check for
        near-singularity, and check for pivot growth;

    -   refine the solution and compute forward and backward error
        bounds.

-   An expert driver `dgsisx()`, which gives the approximate solutions
    of linear equations $AX=B$ or $A^TX=B$, using the ILU factorization
    from `dgsitrf()`. An estimation of the condition number is provide,
    and the pivot growth is computed.

These driver routines cover all the functionality of the computational
routines. We expect that most users can simply use these driver routines
to fulfill their tasks with no need to bother with the computational
routines.

## 2.9.2 Computational routines

The users can invoke the following computational routines, instead of
the driver routines, to directly control the behavior of . The
computational routines can only handle column-oriented storage.

-   `dgstrf()`: Factorize.

    This implements the first-time factorization, or later
    re-factorization with the same nonzero pattern. In
    re-factorizations, the code has the ability to use the same column
    permutation $P_c$ and row permutation $P_r$ obtained from a previous
    factorization. The input argument `options` contains several scalar
    arguments to control how the $LU$ decomposition and the numerical
    pivoting should be performed. `dgstrf()` can handle non-square
    matrices.

-   `dgsitrf()`: ILU.

    This implements the incomplete LU factorization The input argument
    `options` contains several scalar arguments to control how the
    incomplete facotirzation and the numerical pivoting should be
    performed.

-   `dgstrs()`: Triangular solve.

    This takes the $L$ and $U$ triangular factors, the row and column
    permutation vectors, and the right-hand side to compute a solution
    matrix $X$ of $AX=B$ or $A^TX=B$.

-   `dgscon()`: Estimate condition number.

    Given the matrix $A$ and its factors $L$ and $U$, this estimates the
    condition number in the one-norm or infinity-norm. The algorithm is
    due to Hager and Higham [@higham96], and is the same as `CONDEST` in
    sparse Matlab.

-   `dgsequ()/dlaqgs()`: Equilibrate.

    `dgsequ` first computes the row and column scalings $D_r$ and $D_c$
    which would make each row and each column of the scaled matrix
    $D_rAD_c$ have equal norm. `dlaqgs` then applies them to the
    original matrix $A$ if it is indeed badly scaled. The equilibrated
    $A$ overwrites the original $A$.

-   `dgsrfs()`: Refine solution.

    Given $A$, its factors $L$ and $U$, and an initial solution $X$,
    this does iterative refinement, using the same precision as the
    input data. It also computes forward and backward error bounds for
    the refined solution.

(sec:slu_utility)=
## 2.9.3 Utility routines 

The utility routines can help users create and destroy the matrices
easily. These routines reside in two places: `SRC/util.c` contains the
routines that are precision-independent;\
`SRC/{s,d,c,z}util.c` contains the routines dependent on precision.
Here, we list the prototypes of these routines.

        /* Create a supermatrix in compressed column format. A is the output. */
        dCreate_CompCol_Matrix(SuperMatrix *A, int m, int n, int nnz, 
                               double *nzval, int *rowind, int *colptr,
                               Stype_t stype, Dtype_t dtype, Mtype_t mtype);

        /* Create a supermatrix in compressed row format. A is the output. */
        dCreate_CompRow_Matrix(SuperMatrix *A, int m, int n, int nnz, 
                               double *nzval, int *colind, int *rowptr,
                               Stype_t stype, Dtype_t dtype, Mtype_t mtype);

        /* Copy matrix A into matrix B, both in compressed column format. */
        dCopy_CompCol_Matrix(SuperMatrix *A, SuperMatrix *B);

        /* Create a supermatrix in dense format. X is the output.*/
        dCreate_Dense_Matrix(SuperMatrix *X, int m, int n, double *x, int ldx,
                             Stype_t stype, Dtype_t dtype, Mtype_t mtype);

        /* Create a supermatrix in supernodal format. L is the output. */
        dCreate_SuperNode_Matrix(SuperMatrix *L, int m, int n, int nnz, 
                                 double *nzval, int *nzval_colptr, int *rowind,
                                 int *rowind_colptr, int *col_to_sup, int *sup_to_col,
                                 Stype_t stype, Dtype_t dtype, Mtype_t mtype);

        /* Convert the compressed row fromat to the compressed column format. */
        dCompRow_to_CompCol(int m, int n, int nnz, 
                            double *a, int *colind, int *rowptr,
                            double **at, int **rowind, int **colptr);

        /* Print a supermatrix in compressed column format. */
        dPrint_CompCol_Matrix(char *what, SuperMatrix *A);

        /* Print a supermatrix in supernodal format. */
        dPrint_SuperNode_Matrix(char *what, SuperMatrix *A);

        /* Print a supermatrix in dense format. */
        dPrint_Dense_Matrix(char *what, SuperMatrix *A);

        /* Deallocate the storage structure *Store. */
        Destroy_SuperMatrix_Store(SuperMatrix *A);

        /* Deallocate the supermatrix structure in compressed column format. */
        Destroy_CompCol_Matrix(SuperMatrix *A)

        /* Deallocate the supermatrix structure in supernodal format. */
        Destroy_SuperNode_Matrix(SuperMatrix *A)

        /* Deallocate the supermatrix structure in permuted compressed column format. */
        Destroy_CompCol_Permuted(SuperMatrix *A)

        /* Deallocate the supermatrix structure in dense format. */
        Destroy_Dense_Matrix(SuperMatrix *A)

(sec:MatlabInterface)=
# 2.10 Matlab interface 

In the `/MATLAB` subdirectory, we have developed a set of MEX-files
interface to Matlab. Typing `make` in this directory produces
executables to be invoked in Matlab. The current `Makefile` is set up so
that the MEX-files are compatible with Matlab Version 5. The user should
edit `Makefile` for Matlab Version 4 compatibility. Right now, only the
factor routine `dgstrf()` and the simple driver routine `dgssv()` are
callable by invoking `superlu` and `lusolve` in Matlab, respectively.
`Superlu` and `lusolve` correspond to the two Matlab built-in functions
`lu` and `\backslash`$\;$. In Matlab, when you type

you will find the following description about `superlu`'s functionality
and how to use it.

      SUPERLU : Supernodal LU factorization
     
      Executive summary:

      [L,U,p] = superlu(A)          is like [L,U,P] = lu(A), but faster.
      [L,U,prow,pcol] = superlu(A)  preorders the columns of A by min degree,
                                        yielding A(prow,pcol) = L*U.

      Details and options:

      With one input and two or three outputs, SUPERLU has the same effect as LU,
      except that the pivoting permutation is returned as a vector, not a matrix:

      [L,U,p] = superlu(A) returns unit lower triangular L, upper triangular U,
                and permutation vector p with A(p,:) = L*U.
      [L,U] = superlu(A) returns permuted triangular L and upper triangular U
                with A = L*U.

      With a second input, the columns of A are permuted before factoring:

      [L,U,prow] = superlu(A,psparse) returns triangular L and U and permutation 
                prow with A(prow,psparse) = L*U.
      [L,U] = superlu(A,psparse) returns permuted triangular L and triangular U 
                with A(:,psparse) = L*U.
      Here psparse will normally be a user-supplied permutation matrix or vector
      to be applied to the columns of A for sparsity.  COLMMD is one way to get
      such a permutation; see below to make SUPERLU compute it automatically.
      (If psparse is a permutation matrix, the matrix factored is A*psparse'.)

      With a fourth output, a column permutation is computed and applied:

      [L,U,prow,pcol] = superlu(A,psparse)  returns triangular L and U and
                permutations prow and pcol with A(prow,pcol) = L*U.
                Here psparse is a user-supplied column permutation for sparsity,
                and the matrix factored is A(:,psparse) (or A*psparse' if the
                input is a permutation matrix).  Output pcol is a permutation
                that first performs psparse, then postorders the etree of the 
                column intersection graph of A.  The postorder does not affect 
                sparsity, but makes supernodes in L consecutive.
      [L,U,prow,pcol] = superlu(A,0) is the same as ... = superlu(A,I); it does
                not permute for sparsity but it does postorder the etree.
      [L,U,prow,pcol] = superlu(A) is the same as ... = superlu(A,colmmd(A));
                it uses column minimum degree to permute columns for sparsity,
                then postorders the etree and factors.

For a description about `lusolve`'s functionality and how to use it, you
can type

      LUSOLVE : Solve linear systems by supernodal LU factorization.
     
      x = lusolve(A, b) returns the solution to the linear system A*x = b,
          using a supernodal LU factorization that is faster than Matlab's 
          builtin LU.  This m-file just calls a mex routine to do the work.

      By default, A is preordered by column minimum degree before factorization.
      Optionally, the user can supply a desired column ordering:

      x = lusolve(A, b, pcol) uses pcol as a column permutation.  
          It still returns x = A\b, but it factors A(:,pcol) (if pcol is a 
          permutation vector) or A*Pcol (if Pcol is a permutation matrix).
           
      x = lusolve(A, b, 0) suppresses the default minimum degree ordering;
          that is, it forces the identity permutation on columns.

Two M-files `trysuperlu.m` and `trylusolve.m` are written to test the
correctness of `superlu` and `lusolve`. In addition to testing the
residual norms, they also test the function invocations with various
number of input/output arguments.

(sec:install)=
# 2.11 Installation 

## 2.11.1 File structure

The top level SuperLU/ directory is structured as follows:

        SuperLU/README    instructions on installation
        SuperLU/CBLAS/    needed BLAS routines in C, not necessarily fast
        SuperLU/EXAMPLE/  example programs
        SuperLU/INSTALL/  test machine dependent parameters; this Users' Guide
        SuperLU/MAKE_INC/ sample machine-specific make.inc files
        SuperLU/MATLAB/   Matlab mex-file interface
        SuperLU/SRC/      C source code, to be compiled into the superlu.a library
        SuperLU/TESTING/  driver routines to test correctness
        SuperLU/Makefile  top level Makefile that does installation and testing
        SuperLU/make.inc  compiler, compile flags, library definitions and C
                          preprocessor definitions, included in all Makefiles

Before installing the package, you may need to edit `SuperLU/make.inc`
for your system. This make include file is referenced inside each of the
`Makefiles` in the various subdirectories. As a result, there is no need
to edit the `Makefiles` in the subdirectories. All information that is
machine specific has been defined in `make.inc`.

Sample machine-specific `make.inc` are provided in the `MAKE_INC/`
subdirectory for several systems, including IBM RS/6000, DEC Alpha,
SunOS 4.x, SunOS 5.x (Solaris), HP-PA and SGI Iris 4.x. When you have
selected the machine on which you wish to install SuperLU, you may copy
the appropriate sample include file (if one is present) into `make.inc`.
For example, if you wish to run SuperLU on an IBM RS/6000, you can do:

For systems other than those listed above, slight modifications to the
`make.inc` file will need to be made. In particular, the following three
items should be examined:

1.  The BLAS library.\
    If there is a BLAS library available on your machine, you may define
    the following in `make.inc`:

    The `CBLAS/` subdirectory contains the part of the C BLAS needed by
    the package. However, these codes are intended for use only if there
    is no faster implementation of the BLAS already available on your
    machine. In this case, you should do the following:

    -   In `make.inc`, undefine (comment out) BLASDEF, define:

    -   In the / directory, type:

        to make the BLAS library from the routines in the `CBLAS/`
        subdirectory.

2.  C preprocessor definition `CDEFS`.\
    In the header file `SRC/Cnames.h`, we use macros to determine how C
    routines should be named so that they are callable by Fortran. [^1]
    The possible options for `CDEFS` are:

    -   `-DAdd_`: Fortran expects a C routine to have an underscore
        postfixed to the name;

    -   `-DNoChange`: Fortran expects a C routine name to be identical
        to that compiled by C;

    -   `-DUpCase`: Fortran expects a C routine name to be all
        uppercase.

3.  The Matlab MEX-file interface.\
    The `MATLAB/` subdirectory includes Matlab C MEX-files, so that our
    factor and solve routines can be called as alternatives to those
    built into Matlab. In the file `SuperLU/make.inc`, define MATLAB to
    be the directory in which Matlab is installed on your system, for
    example:

    At the SuperLU/ directory, type:

    `make matlabmex`

    to build the MEX-file interface. After you have built the interface,
    you may go to the `MATLAB/` subdirectory to test the correctness by
    typing (in Matlab):

A `Makefile` is provided in each subdirectory. The installation can be
done completely automatically by simply typing `make` at the top level.

## 2.11.2 Testing

The test programs in `/INSTALL` subdirectory test two routines:

-   `slamch()/dlamch()` determines properties of the floating-point
    arithmetic at run-time (both single and double precision), such as
    the machine epsilon, underflow threshold, overflow threshold, and
    related parameters;

-   `SuperLU_timer_()` returns the time in seconds used by the process.
    This function may need to be modified to run on your machine.

The test programs in the `/TESTING` subdirectory are designed to test
all the functions of the driver routines, especially the expert drivers.
The Unix shell script files `xtest.csh` are used to invoke tests with
varying parameter settings. The input matrices include an actual sparse
matrix `/EXAMPLE/g10` of dimension $100\times 100$, [^2] and numerous
matrices with special properties from the  test suite.
Table [\[tab:testmats\]](#tab:testmats){reference-type="ref"
reference="tab:testmats"} describes the properties of the test matrices.

::: {.center}
::: {#tab:tests}
  Matrix type   Description
  ------------- -----------------------------------------
  0             sparse matrix `g10`
  1             diagonal
  2             upper triangular
  3             lower triangular
  4             random, $\kappa=2$
  5             first column zero
  6             last column zero
  7             last $n/2$ columns zero
  8             random, $\kappa=\sqrt{0.1/\varepsilon}$
  9             random, $\kappa=0.1/\varepsilon$
  10            scaled near underflow
  11            scaled near overflow

  : Types of tests. $x^*$ is the true solution, $FERR$ is the error
  bound, and $BERR$ is the backward error.
:::
:::

::: {.center}
::: {#tab:tests}
  Test Type   Test ratio                               Routines
  ----------- ---------------------------------------- -------------------
  0           $||LU-A||/(n||A||\varepsilon)$           `dgstrf`
  1           $||b-Ax|| / (||A||\;||x||\varepsilon)$   `dgssv`, `dgssvx`
  2           $||x-x^*||/(||x^*||\kappa\varepsilon)$   `dgssvx`
  3           $||x-x^*|| / (||x^*||\; FERR)$           `dgssvx`
  4           $BERR / \varepsilon$                     `dgssvx`

  : Types of tests. $x^*$ is the true solution, $FERR$ is the error
  bound, and $BERR$ is the backward error.
:::
:::

For each command line option specified in `dtest.csh`, the test program
`ddrive` reads in or generates an appropriate matrix, calls the driver
routines, and computes a number of test ratios to verify that each
operation has performed correctly. If the test ratio is smaller than a
preset threshold, the operation is considered to be correct. Each test
matrix is subject to the tests listed in
Table [2](#tab:tests){reference-type="ref" reference="tab:tests"}.

Let $r$ be the residual $r=b-Ax$, and let $m_i$ be the number of
nonzeros in row $i$ of $A$. Then the componentwise backward error $BERR$
and forward error $FERR$ [@lapackmanual2] are calculated by:
$$BERR = \max_i\frac{|r|_i}{(|A|~|x|+|b|)_i}\ .$$
$$FERR = \frac{||~|A^{-1}|~f~||_\infty}{||x||_\infty}\ .$$ Here, $f$ is
a nonnegative vector whose components are computed as
$f_i=|r|_i + m_i~\varepsilon~(|A|~|x|+|b|)_i$, and the norm in the
numerator is estimated using the same subroutine used for estimating the
condition number. $BERR$ measures the smallest relative perturbation one
can make to each entry of A and of b so that the computed solution is an
exact solution of the perturbed problem. $FERR$ is an estimated bound on
the error $\| x^* - x \|_{\infty} / \| x \|_{\infty}$, where $x^*$ is
the true solution. For further details on error analysis and error
bounds estimation, see [@lapackmanual2 Chapter 4] and  [@arioli89].

(sec:parameters)=
## 2.11.3 Performance-tuning parameters

chooses such machine-dependent parameters as block size by calling an
inquiry function `sp_ienv()`, which may be set to return different
values on different machines. The declaration of this function is

`int sp_ienv(int ispec);`

`Ispec` specifies the parameter to be returned, (See
reference [@superlu99] for their definitions.)

::: {.tabbing}
xxxxxx x̄xxx j̄unk ̄ ispec= 1: the panel size ($w$)\
= 2: the relaxation parameter to control supernode amalgamation
($relax$)\
= 3: the maximum allowable size for a supernode ($maxsup$)\
= 4: the minimum row dimension for 2D blocking to be used ($rowblk$)\
= 5: the minimum column dimension for 2D blocking to be used ($colblk$)\
= 6: the estimated fills factor for L and U, compared with A
:::

Users are encouraged to modify this subroutine to set the tuning
parameters for their own local environment. The optimal values depend
mainly on the cache size and the  speed. If your system has a very small
cache, or if you want to efficiently utilize the closest cache in a
multilevel cache organization, you should pay special attention to these
parameter settings. In our technical paper [@superlu99], we described a
detailed methodology for setting these parameters for high performance.

The $relax$ parameter is usually set between 4 and 8. The other
parameter values which give good performance on several machines are
listed in Table [3](#tab:block_params){reference-type="ref"
reference="tab:block_params"}. In a supernode-panel update, if the
updating supernode is too large to fit in cache, then a 2D block
partitioning of the supernode is used, in which $rowblk$ and $colblk$
determine that a block of size $rowblk\times colblk$ is used to update
current panel.

If $colblk$ is set greater than $maxsup$, then the program will never
use 2D blocking. For example, for the Cray J90 (which does not have
cache), $w=1$ and 1D blocking give good performance; more levels of
blocking only increase overhead.

::: {.small}
::: {.center}
::: {#tab:block_params}
  -------------- ---------- ---------- ----- ---------- ---------- ----------
                    On-chip   External                             
  Machine             Cache      Cache   $w$   $maxsup$   $rowblk$   $colblk$
  RS/6000-590        256 KB         --     8        100        200         40
  MIPS R8000          16 KB       4 MB    20        100        800        100
  Alpha 21064          8 KB     512 KB     8        100        400         40
  Alpha 21164       8 KB-L1       4 MB    16         50        100         40
                   96 KB-L2                                        
  Sparc 20            16 KB       1 MB     8        100        400         50
  UltraSparc-I        16 KB     512 KB     8        100        400         40
  Cray J90               --         --     1        100       1000        100
  -------------- ---------- ---------- ----- ---------- ---------- ----------

  : Typical blocking parameter values for several machines.
:::
:::
:::

(sec:example)=
# 2.12 Example programs 

In the `SuperLU/EXAMPLE/` subdirectory, we present a few sample programs
to illustrate how to use various functions provded in . The users can
modify these examples to suit their applications. Here are the brief
descriptions of the double precision version of the examples:

-   `dlinsol`: use simple driver `dgssv()` to solve a linear system one
    time.

-   `dlinsol1`: use simple driver `dgssv()` in the symmetric mode.

-   `dlinsolx`: use `dgssvx()` with the full (default) set of options to
    solve a linear system.

-   `dlinsolx1`: use `dgssvx()` to factorize $A$ first, then solve the
    system later.

-   `dlinsolx2`: use `dgssvx()` to solve systems repeatedly with the
    same sparsity pattern of matrix A.

-   `superlu`: the small 5x5 sample program in
    Section [2](#sec:ex5x5){reference-type="ref" reference="sec:ex5x5"}.

In this directory, a `Makefile` is provided to generate the executables,
and a `README` file describes how to run these examples.

# 2.13 Calling from Fortran

The `SuperLU/FORTRAN/` subdirectory contains an example of using from a
Fortran program. The General rules for mixing Fortran and C programs are
as follows.

-   Arguments in C are passed by value, while in Fortran are passed by
    reference. So we always pass the address (as a pointer) in the C
    calling routine. (You cannot make a call with numbers directly in
    the parameters.)

-   Fortran uses 1-based array addressing, while C uses 0-based.
    Therefore, the row indices (`rowind[]`) and the integer pointers to
    arrays (`colptr[]`) should be adjusted before they are passed into a
    C routine.

Because of the above language differences, in order to embed in a
Fortran environment, users are required to use "wrapper" routines (in C)
for all the routines that will be called from Fortran programs. The
example `c_fortran_dgssv.c` in the `FORTRAN/` directory shows how a
wrapper program should be written. This program is listed below.

    #include "dsp_defs.h"

    #define HANDLE_SIZE  8

    typedef struct {
        SuperMatrix *L;
        SuperMatrix *U;
        int *perm_c;
        int *perm_r;
    } factors_t;

    int
    c_fortran_dgssv_(int *iopt, int *n, int *nnz, int *nrhs, double *values,
                     int *rowind, int *colptr, double *b, int *ldb,
                     int factors[HANDLE_SIZE], /* a handle containing the pointer
                                                  to the factored matrices */
                     int *info)

    {
    /* 
     * This routine can be called from Fortran.
     *
     * iopt (input) int
     *      Specifies the operation:
     *      = 1, performs LU decomposition for the first time
     *      = 2, performs triangular solve
     *      = 3, free all the storage in the end
     *
     * factors (input/output) integer array of size 8
     *      If iopt == 1, it is an output and contains the pointer pointing to
     *                    the structure of the factored matrices.
     *      Otherwise, it it an input.
     *
     */
        SuperMatrix A, AC, B;
        SuperMatrix *L, *U;
        int *perm_r; /* row permutations from partial pivoting */
        int *perm_c; /* column permutation vector */
        int *etree;  /* column elimination tree */
        SCformat *Lstore;
        NCformat *Ustore;
        int      i, panel_size, permc_spec, relax;
        trans_t  trans;
        double   drop_tol = 0.0;
        mem_usage_t   mem_usage;
        superlu_options_t options;
        SuperLUStat_t stat;
        factors_t *LUfactors;

        trans = NOTRANS;

        if ( *iopt == 1 ) { /* LU decomposition */

            /* Set the default input options. */
            set_default_options(&options);

            /* Initialize the statistics variables. */
            StatInit(&stat);

            /* Adjust to 0-based indexing */
            for (i = 0; i < *nnz; ++i) --rowind[i];
            for (i = 0; i <= *n; ++i) --colptr[i];

            dCreate_CompCol_Matrix(&A, *n, *n, *nnz, values, rowind, colptr,
                                   SLU_NC, SLU_D, SLU_GE);
            L = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
            U = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
            if ( !(perm_r = intMalloc(*n)) ) ABORT("Malloc fails for perm_r[].");
            if ( !(perm_c = intMalloc(*n)) ) ABORT("Malloc fails for perm_c[].");
            if ( !(etree = intMalloc(*n)) ) ABORT("Malloc fails for etree[].");

            /*
             * Get column permutation vector perm_c[], according to permc_spec:
             *   permc_spec = 0: natural ordering 
             *   permc_spec = 1: minimum degree on structure of A'*A
             *   permc_spec = 2: minimum degree on structure of A'+A
             *   permc_spec = 3: approximate minimum degree for unsymmetric matrices
             */     
            permc_spec = 3;
            get_perm_c(permc_spec, &A, perm_c);
        
            sp_preorder(&options, &A, perm_c, etree, &AC);

            panel_size = sp_ienv(1);
            relax = sp_ienv(2);

            dgstrf(&options, &AC, drop_tol, relax, panel_size, 
                   etree, NULL, 0, perm_c, perm_r, L, U, &stat, info);

            if ( *info == 0 ) {
                Lstore = (SCformat *) L->Store;
                Ustore = (NCformat *) U->Store;
                printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
                printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
                printf("No of nonzeros in L+U = %d\n", Lstore->nnz + Ustore->nnz);
                dQuerySpace(L, U, &mem_usage);
                printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
                       mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
                       mem_usage.expansions);
            } else {
                printf("dgstrf() error returns INFO= %d\n", *info);
                if ( *info <= *n ) { /* factorization completes */
                   dQuerySpace(L, U, &mem_usage);
                   printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
                         mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
                         mem_usage.expansions);
                }
            }
        
            /* Restore to 1-based indexing */
            for (i = 0; i < *nnz; ++i) ++rowind[i];
            for (i = 0; i <= *n; ++i) ++colptr[i];

            /* Save the LU factors in the factors handle */
            LUfactors = (factors_t*) SUPERLU_MALLOC(sizeof(factors_t));
            LUfactors->L = L;
            LUfactors->U = U;
            LUfactors->perm_c = perm_c;
            LUfactors->perm_r = perm_r;
            factors[0] = (int) LUfactors;

            /* Free un-wanted storage */
            SUPERLU_FREE(etree);
            Destroy_SuperMatrix_Store(&A);
            Destroy_CompCol_Permuted(&AC);
            StatFree(&stat);

        } else if ( *iopt == 2 ) { /* Triangular solve */
            /* Initialize the statistics variables. */
            StatInit(&stat);

            /* Extract the LU factors in the factors handle */
            LUfactors = (factors_t*) factors[0];
            L = LUfactors->L;
            U = LUfactors->U;
            perm_c = LUfactors->perm_c;
            perm_r = LUfactors->perm_r;

            dCreate_Dense_Matrix(&B, *n, *nrhs, b, *ldb, SLU_DN, SLU_D, SLU_GE);

            /* Solve the system A*X=B, overwriting B with X. */
            dgstrs (trans, L, U, perm_c, perm_r, &B, &stat, info);

            Destroy_SuperMatrix_Store(&B);
            StatFree(&stat);

        } else if ( *iopt == 3 ) { /* Free storage */
            /* Free the LU factors in the factors handle */
            LUfactors = (factors_t*) factors[0];
            SUPERLU_FREE (LUfactors->perm_r);
            SUPERLU_FREE (LUfactors->perm_c);
            Destroy_SuperNode_Matrix(LUfactors->L);
            Destroy_CompCol_Matrix(LUfactors->U);
            SUPERLU_FREE (LUfactors->L);
            SUPERLU_FREE (LUfactors->U);
            SUPERLU_FREE (LUfactors);
        } else {
            fprintf(stderr, "Invalid iopt=%d passed to c_fortran_dgssv()\n");
            exit(-1);
        }
    }

Since the matrix structures in C cannot be directly returned to Fortran,
we use a handle named `factors` to access those structures. The handle
is essentially an integer pointer pointing to the factored matrices
obtained from . So the factored matrices are opaque objects to the
Fortran program, but can only be manipulated from the C wrapper program.

The Fortran program `FORTRAN/f77_main.f` shows how a Fortran program may
call\
`c_fortran_dgssv()`, and is listed below. A `README` file in this
directory describes how to compile and run this program.

          program f77_main
          integer maxn, maxnz
          parameter ( maxn = 10000, maxnz = 100000 )
          integer rowind(maxnz), colptr(maxn)
          real*8  values(maxnz), b(maxn)
          integer n, nnz, nrhs, ldb, info
          integer factors(8), iopt
    *
    *     Read the matrix file in Harwell-Boeing format
          call hbcode1(n, n, nnz, values, rowind, colptr)
    *
          nrhs = 1
          ldb = n
          do i = 1, n
             b(i) = 1
          enddo
    *
    * First, factorize the matrix. The factors are stored in factor() handle.
          iopt = 1
          call c_fortran_dgssv( iopt, n, nnz, nrhs, values, rowind, colptr, 
         $                      b, ldb, factors, info )
    *
          if (info .eq. 0) then
             write (*,*) 'Factorization succeeded'
          else
             write(*,*) 'INFO from factorization = ', info
          endif
    *
    * Second, solve the system using the existing factors.
          iopt = 2
          call c_fortran_dgssv( iopt, n, nnz, nrhs, values, rowind, colptr, 
         $                      b, ldb, factors, info )
    *
          if (info .eq. 0) then
             write (*,*) 'Solve succeeded'
             write (*,*) (b(i), i=1, 10)
          else
             write(*,*) 'INFO from triangular solve = ', info
          endif

    * Last, free the storage allocated inside SuperLU
          iopt = 3
          call c_fortran_dgssv( iopt, n, nnz, nrhs, values, rowind, colptr, 
         $                      b, ldb, factors, info )
    *
          stop
          end

[^1]: Some vendor-supplied libraries do not have C interfaces. So the
    re-naming is needed in order for the calls (in C) to interface with
    the Fortran-style .

[^2]: Matrix `g10` is first generated with the structure of the 10-by-10
    five-point grid, and random numerical values. The columns are then
    permuted by COLMMD ordering from Matlab.
