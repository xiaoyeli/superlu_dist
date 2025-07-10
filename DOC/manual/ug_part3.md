## Chapter 4

# Distributed-memory SuperLU on manycore nodes (Version 4.0)


## 4.1 About SuperLU_DIST

In this part, we describe the library designed for distributed-memory
pearallel computers using SPMD parallel programming model, together with
multithreading for manycore node architectures. The library is
implemented in ANSI C, using MPI [@mpi-forum] for communication, OpenMP
for multithreading and CUDA for GPU. We have tested the code on a number
of platforms, including IBM, Cray XE6, Cray XT7, and numerous Linux
clusters. The library includes routines to handle both real and complex
matrices in double precision. The parallel routine names for the
double-precision real version start with letters "pd" (such as
`pdgstrf`); the parallel routine names for double-precision complex
version start with letters "pz" (such as `pzgstrf`).

(sec:InputFormat)=
# 4.2 Formats of the input matrices $A$ and $B$ 

We provide two input interfaces for matrices $A$ and $B$---one is
global, and the other is entirely distributed.

(sec:GlobalInput)=
## 4.2.1 Global input 

The input matrices $A$ and $B$ are globally available (replicated) on
all the processes. The storage type for $A$ is (*compressed column*), as
in sequential case (see
Section [\[sec:rep\]](#sec:rep)). The user-callable routines with this interface
all have the names "xxxxxxx_ABglobal". If there is sufficient memory,
this interface is faster than the distributed input interface described
in the next section, because the latter requires more data
re-distribution at different stages of the algorithm.

(sec:DistInput)=
## 4.2.2 Distributed input 

Both input matrices $A$ and $B$ are distributed among all the processes.
They use the same distribution based on block rows. That is, each
process owns a block of consecutive rows of $A$ and $B$. Each local part
of sparse matrix $A$ is stored in a *compressed row* format, called
storage type, which is defined below.

```c
        typedef struct {
            int nnz_loc;  /* number of nonzeros in the local submatrix */
            int m_loc;    /* number of rows local to this process */
            int fst_row;  /* row number of the first row in the local submatrix */
            void *nzval;  /* pointer to array of nonzero values, packed by row */
            int *rowptr;  /* pointer to array of beginning of rows in nzval[] 
                             and colind[]  */
            int *colind;  /* pointer to array of column indices of the nonzeros */
        } NRformat_loc;
```

Let $m_i$ be the number of rows owned by the $i$th process. Then the
global row dimension for $A$ is $nrow = \sum_{i=0}^{P-1}m_i$. The global
column dimension is $ncol$. Both $nrow$ and $ncol$ are recorded in the
higher level `SuperMatrix` data structure, see
Figure [\[fig:struct\]](#fig:struct). The utility routine\
`dCreate_CompRowLoc_Matrix_dist` can help the user to create the
structure for $A$. The definition of this routine is

```c
      void dCreate_CompRowLoc_Matrix_dist(SuperMatrix *A, int m, int n,
                                          int nnz_loc, int m_loc, int fst_row,
                                          double *nzval, int *colind, int *rowptr,
                                          Stype_t stype, Dtype_t dtype, Mtype_t mtype);
```

where, the first argument is output and the rest are inputs.

The local full matrix $B$ is stored in the standard Fortran-style column
major format, with dimension $m\_loc\times nrhs$, and $ldb$ refers to
the local leading dimension in the local storage.

(sec:datastruct)=
# 4.3 Distributed data structures for $L$ and $U$ 

We distribute both $L$ and $U$ matrices in a two-dimensional
block-cyclic fashion. We first identify the supernode boundary based on
the nonzero structure of $L$. This supernode partition is then used as
the block partition in both row and column dimensions for both $L$ and
$U$. The size of each block is matrix dependent. It should be clear that
all the diagonal blocks are square and full (we store zeros from $U$ in
the upper triangle of the diagonal block), whereas the off-diagonal
blocks may be rectangular and may not be full. The matrix in 
illustrates such a partition. By block-cyclic mapping we mean block
$(I,J)$ ($0\le I, J\le N-1$) is mapped into the process at coordinate
{$I\ mod\ {\tt nprow}, J\ mod\ {\tt npcol}$} of the
${\tt nprow}\times {\tt npcol}$ 2D process grid. Using this mapping, a
block $L(I,J)$ in the factorization is only needed by the row of
processes that own blocks in row $I$. Similarly, a block $U(I,J)$ is
only needed by the column of processes that own blocks in column $J$.

In this 2D mapping, each block column of $L$ resides on more than one
process, namely, a column of processes. For example in , the second
block column of $L$ resides on the column processes {1, 4}. Process 4
only owns two nonzero blocks, which are not contiguous in the global
matrix. The schema on the right of  depicts the data structure to store
the nonzero blocks on a process. Besides the numerical values stored in
a Fortran-style array `nzval[]` in column major order, we need the
information to interpret the location and row subscript of each nonzero.
This is stored in an integer array `index[]`, which includes the
information for the whole block column and for each individual block in
it. Note that many off-diagonal blocks are zero and hence not stored.
Neither do we store the zeros in a nonzero block. Both lower and upper
triangles of the diagonal block are stored in the $L$ data structure. A
process owns $\lceil{N/{\tt npcol}}\rceil$ block columns of $L$, so it
needs $\lceil{N/{\tt nprow}}\rceil$ pairs of `index/nzval` arrays.

For $U$, we use a row oriented storage for the block rows owned by a
process, although for the numerical values within each block we still
use column major order. Similar to $L$, we also use a pair of
`index/nzval` arrays to store a block row of $U$. Due to asymmetry, each
nonzero block in $U$ has the skyline structure as shown in 
(see [@superlu99] for details on the skyline structure). Therefore, the
organization of the `index[]` array is different from that for $L$,
which we omit showing in the figure.

(sec:grid)=
# 4.4 Process grid and MPI communicator 

All MPI applications begin with a default communication domain that
includes all processes, say $N_p$, of this parallel job. The default
communicator `MPI_COMM_WORLD` represents this communication domain. The
$N_p$ processes are identified as a linear array of process IDs in the
range $0\;\ldots\;N_p-1$.

## 4.4.1 2D process grid

For library, we create a new process group derived from an existing
group using $N_g$ processes. There is a good reason to use a new group
rather than `MPI_COMM_WORLD`, that is, the message passing calls of the
SuperLU library will be isolated from those in other libraries or in the
user's code. For better scalability of the $LU$ factorization, we map
the 1D array of $N_g$ processes into a logical 2D process grid. This
grid will have `nprow` process rows and `npcol` process columns, such
that ${\tt nprow} * {\tt npcol} = N_g$. A process can be referenced
either by its rank in the new group or by its coordinates within the
grid. The routine `superlu_gridinit` maps already-existing processes to
a 2D process grid.

```c
        superlu_gridinit(MPI_Comm Bcomm, int nprow, int npcol, gridinfo_t *grid);
```

This process grid will use the first ${\tt nprow} * {\tt npcol}$
processes from the base MPI communicator `Bcomm`, and assign them to the
grid in a row-major ordering. The input argument `Bcomm` is an MPI
communicator representing the existing base group upon which the new
group will be formed. For example, it can be `MPI_COMM_WORLD`. The
output argument `grid` represents the derived group to be used in .
`Grid` is a structure containing the following fields:

```c
       struct {
           MPI_Comm comm;        /* MPI communicator for this group */
           int iam;              /* my process rank in this group   */
           int nprow;            /* number of process rows          */
           int npcol;            /* number of process columns       */
           superlu_scope_t rscp; /* process row scope               */
           superlu_scope_t cscp; /* process column scope            */
       } grid;
```

In the $LU$ factorization, some communications occur only among the
processes in a row (column), not among all processes. For this purpose,
we introduce two process subgroups, namely `rscp` (row scope) and `cscp`
(column scope). For `rscp` (`cscp`) subgroup, all processes in a row
(column) participate in the communication.

The macros `MYROW(iam, grid)` and `MYCOL(iam, grid)` give the row and
column coordinates in the 2D grid of the process who has rank `iam`.

*NOTE: All processes in the base group, including those not in the new
group, must call this grid creation routine. This is required by the MPI
routine `MPI_Comm_create` to create a new communicator.*

## 4.4.2 Arbitrary grouping of processes

It is sometimes desirable to divide up the processes into several
subgroups, each of which performs independent work of a single
application. In this situation, we cannot simply use the first
${\tt nprow} * {\tt npcol}$ processes to define the grid. A more
sophisticated process-to-grid mapping routine `superlu_gridmap` is
designed to create a grid with processes of arbitrary ranks.

```c
        superlu_gridmap(MPI_Comm Bcomm, int nprow, int npcol,
                        int usermap[], int ldumap, gridinfo_t *grid);
```

The array `usermap[]` contains the processes to be used in the newly
created grid. `usermap[]` is indexed like a Fortran-style 2D array with
`ldumap` as the leading dimension. So `usermap[i+j*ldumap]` (i.e.,
`usermap(i,j)` in Fortran notation) holds the process rank to be placed
in {i, j} position of the 2D process grid. After grid creation, this
subset of processes is logically numbered in a consistent manner with
the initial set of processes; that is, they have the ranks in the range
$0\;\ldots\;{\tt nprow} * {\tt npcol}-1$ in the new grid. For example,
if we want to map 6 processes with ranks $11\;\ldots\;16$ into a
$2\times 3$ grid, we define ${\tt usermap} = \{11, 14, 12, 15, 13, 16\}$
and ${\tt ldumap}=2$. Such a mapping is shown below



<table style="margin-left:auto; margin-right:auto; text-align:center; border-collapse: collapse;">
  <tr>
    <td style="border-bottom: 1px solid white; border-right: 1px solid white;"></td>
    <td style="border-bottom: 1px solid white; border-right: 1px solid white;">0</td>
    <td style="border-bottom: 1px solid white; border-right: 1px solid white;">1</td>
    <td style="border-bottom: 1px solid white;">2</td>
  </tr>
  <tr>
    <td style="border-right: 1px solid white;">0</td>
    <td style="border: 1px solid white;">11</td>
    <td style="border: 1px solid white;">12</td>
    <td style="border: 1px solid white;">13</td>
  </tr>
  <tr>
    <td style="border-right: 1px solid white;">1</td>
    <td style="border: 1px solid white;">14</td>
    <td style="border: 1px solid white;">15</td>
    <td style="border: 1px solid white;">16</td>
  </tr>
</table>

*NOTE: All processes in the base group, including those not in the new
group, must call this routine.*

`Superlu_gridinit` simply calls `superlu_gridmap` with `usermap[]`
holding the first ${\tt nprow} * {\tt npcol}$ process ranks.

# 4.5 Algorithmic background

Although partial pivoting is used in both sequential and shared-memory
parallel factorization algorithms, it is not used in the
distributed-memory parallel algorithm, because it requires dynamic
adaptation of data structures and load balancing, and so is hard to make
it scalable. We use alternative techniques to stabilize the algorithm,
which include statically pivot large elements to the diagonal,
single-precision diagonal adjustment to avoid small pivots, and
iterative refinement.
Figure [\[fig:GESP_alg\]](#fig:GESP_alg) sketches our GESP algorithm (Gaussian
elimination with Static Pivoting). Numerical experiments show that for a
wide range of problems, GESP is as stable as GEPP [@lidemmel03].

(fig:GESP_alg)=
$$
\begin{aligned}
&(1)\quad\text{Perform row/column equilibration and row permutation: } A \leftarrow P_r \cdot D_r \cdot A \cdot D_c,\\
&\quad\quad\text{where } D_r \text{ and } D_c \text{ are diagonal matrices and } P_r \text{ is a row permutation chosen}\\
&\quad\quad\text{to make the diagonal large compared to the off-diagonal.}\\[6pt]
&(2)\quad\text{Find a column permutation } P_c \text{ to preserve sparsity: } A \leftarrow P_c\cdot A\cdot P_c^T\\[6pt]
&(3)\quad\text{Perform symbolic analysis to determine the nonzero structures of } L \text{ and } U.\\[6pt]
&(4)\quad\text{Factorize } A = L\cdot U \text{ with control of diagonal magnitude:}\\
&\quad\quad\mathbf{if}\;(|a_{ii}| < \sqrt{\varepsilon}\cdot\|A\|_1)\;\mathbf{then}\\
&\quad\quad\quad a_{ii} = \sqrt{\varepsilon}\cdot\|A\|_1\\
&\quad\quad\mathbf{endif}\\[6pt]
&(5)\quad\text{Perform triangular solutions using } L \text{ and } U.\\[6pt]
&(6)\quad\text{If needed, use an iterative solver like GMRES or iterative refinement (shown below)}\\
&\quad\quad r = b - A\cdot x \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{(sparse matrix-vector multiply)}\\
&\quad\quad\text{Solve } A\cdot dx = r \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{(triangular solution)}\\
&\quad\quad berr = \max_i\frac{|r|_i}{(|A|\cdot|x|+|b|)_i}\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{(componentwise backward error)}\\
&\quad\quad\mathbf{if}\;(berr > \varepsilon\;\text{and}\;berr \leq \frac{1}{2}\cdot lastberr)\;\mathbf{then}\\
&\quad\quad\quad x = x + dx\\
&\quad\quad\quad lastberr = berr\\
&\quad\quad\quad\mathbf{goto}\;iterate\\
&\quad\quad\mathbf{endif}\\[6pt]
&(7)\quad\text{If desired, estimate the condition number of } A
\end{aligned}
$$

<div style="text-align:center;">
Figure 4.2: The outline of the GESP algorithm.
</div>

Step (1) is accomplished by a weighted bipartite matching algorithm due
to Duff and Koster [@duffkoster99]. Currently, process 0 computes $P_r$
and then broadcasts it to all the other processes. If the distributed
input interface is used
(Section [2.2](#sec:DistInput)), we first gather the distributed matrix $A$
onto processor 0. Work is underway to remove this sequential bottleneck.

In Step (2), we provide several ordering options, such as multiple
minimum degree ordering [@liu85] on the graphs of $A+A^T$, or the
 [@kaku:98a] ordering on the graphs of $A+A^T$. The user can use any
other ordering in place of the ones provided in the library. (*Note,
since we will pivot on the diagonal in step (4), an ordering based on
the structure of $A+A^T$ almost always yields sparser factors than that
based on the structure of $A^TA$. This is different from SuperLU and
SuperLU_MT, where we allow to pivot off-diagonal.*) In this step, when a
sequential ordering algorithm is used, every process runs the same
algorithm independently.

Step (3) can be done either sequentially or in parallel depending on how
the ***options*** argument is set (see
Section [8.1](#sec:drivers) for details.) The parallel symbolic
factorization was a newly added feature since the ***v2.1***
release. It is designed tightly around the separator tree returned from
a graph partitioning type of ordering (presently we use  [@kaku:03]),
and works only on power-of-two processors. We first re-distribute the
graph of $A$ onto the largest $2^q$ number of processors which is
smaller than the total $N_p$ processors, then perform parallel symbolic
factorization, and finally re-populate the $\{L \backslash U\}$
structure to all $N_p$ processors. The algorithm and performance was
studied in [@grigoridemmelli07]. To invoke parallel symbolic
factorization, the user needs to set the two fields of the
***options*** argument as follows:

        options.ParSymbFact       = YES
        options.ColPerm           = PARMETIS;

Note that, even if the user sets ***options.ColPerm*** to use
an ordering algorithm other than , the driver routine overrides it with
when it sees 'options.ParSymbFact = ***YES***.

Steps (4) to (7) are the most time-consuming steps and were parallelized
a while ago, see the papers [@lidemmel03; @li05].

## 4.5.1 Multicore and GPU enhancements

Given that the node architecture trend is to have many simpler cores on
a NUMA node, possibly with GPU-like accelerators, and the amount of
memory per-core becomes smaller, the pure MPI model does not match the
light-weight processor architecture. We need to resort to other forms of
parallelism at node level. In sparse factorization (step (4) in
Figure [\[fig:GESP_alg\]](#fig:GESP_alg)), the Schur complement update after each panel
factorization step exhibits ample fine-grid parallelism. We have
designed the OpenMP + CUDA code to exploit the on-node parallelism. For
each MPI task of the Schur complement update, we aggegrate the small $L$
and $U$ matrix blocks into a larger block, divide the GEMM work between
CPU and GPU using some heuristic performance model, offload the larger
block of GEMM to GPU. The key to success is to overlap the CPU work
(multithreaded Scatter/Gather, some portion of GEMM) with the GPU work
(GEMM with multiple CUDA streams). This way, we are able to hide the
latency time due to PCI bus transfer between CPU and GPU. The detailed
algorithm description and performance data were given in the paper by
Sao et al. [@sao2014].

To use Nvidia GPU, you must set the following Linux shell environment
variable before compilation:

`setenv ACC GPU`

Hybrid MPI+OpenMP setting may outperform MPI-only configurations in some
cases and in most cases hybrid MPI+OpenMP would require less memory. The
environment variable `OMP_NUM_THREADS` needs to be set appropriately.
Hybrid configuration obtains threaded parallelism from both, explicit
OpenMP pragmas and multithreaded BLAS. Thus for good performance it is
better if OpenMP threading library and BLAS threading are synergistic.
For example, when using Intel MKL libray, just setting `OMP_NUM_THREADS`
would set the number of threads for both MKL and OpenMP. However, it is
possible to have different number of threads for MKL, in which case
`MKL_NUM_THREADS` controls the number of threads used by MKL. In our
case, just setting `OMP_NUM_THREADS` is sufficient.

Triangular solve phase does not use multithreading yet. The MPI-only
configuration may be more suitable in case of many right hand sides or
in other cases, where solve phase seems to be a performance bottleneck.

(sec:options)=
# 4.6 *Options* argument 

One important input argument is `options`, which controls how the linear
system will be solved. Although the algorithm presented in  consists of
seven steps, for some matrices not all steps are needed to get accurate
solution. For example, for diagonally dominant matrices, choosing the
diagonal pivots ensures the stability; there is no need for row pivoting
in step (1). In another situation where a sequence of matrices with the
same sparsity pattern need be factorized, the column permutation $P_c$
(and also the row permutation $P_r$, if the numerical values are
similar) need be computed only once, and reused thereafter. ($P_r$ and
$P_c$ are implemented as permutation vectors `perm_r` and `perm_c`.) For
the above examples, performing all seven steps does more work than
necessary. `Options` is used to accommodate the various requirements of
applications; it contains the following fields:

-   `Fact`\
    Specifies whether or not the factored form of the matrix $A$ is
    supplied on entry, and if not, how the matrix $A$ will be factored
    base on some assumptions of the previous history. *fact* can be one
    of:

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

-   `Equil` { *YES* $|$ *NO* }\
    Specifies whether to equilibrate the system.

-   `ParSymbFact` { *YES* $|$ *NO* }\
    Specifies whether to perform parallel symbolic factorization. If it
    is set to ***YES***, the ***ColPerm*** field
    should be set to ***PARMETIS***. Otherwise, the driver
    routine `pdgssvx` will use anyway, ignoring the other setting in
    ***ColPerm***.

-   `ColPerm`\
    Specifies the column ordering method for fill reduction.

    -   `NATURAL`: natural ordering.

    -   `MMD_AT_PLUS_A`: minimum degree ordering on the structure of
        $A^T+A$.

    -   `MMD_ATA`: minimum degree ordering on the structure of $A^TA$.

    -   `METIS_AT_PLUS_A`: ordering on the structure of $A^T+A$.

    -   `PARMETIS`: ordering on the structure of $A^T+A$.

    -   `MY_PERMC`: use the ordering given in `perm_c` input by the
        user.

-   `RowPerm`\
    Specifies how to permute rows of the original matrix.

    -   `NATURAL`: use the natural ordering.

    -   `LargeDiag_MC64`: use a serial, weighted bipartite matching
        algorithm implemented in MC64 to permute the rows to make the
        diagonal large relative to the off-diagonal [@duffkoster01].

    -   `LargeDiag_AWPM`: use a parallel, approximate weighted bipartite
        matching algorithm implemented in CombBLAS to permute the rows
        to make the diagonal large relative to the off-diagonal [@awpm].

    -   `MY_PERMR`: use the ordering given in `perm_r` input by the
        user.

-   `ReplaceTinyPivot` { *YES* $|$ *NO* }\
    Specifies whether to replace the tiny diagonals by
    $\sqrt\varepsilon\cdot||A||$ during $LU$ factorization.

-   `IterRefine`\
    Specifies how to perform iterative refinement.

    -   `NO`: no iterative refinement.

    -   `DOUBLE`: accumulate residual in double precision.

    -   `EXTRA`: accumulate residual in extra precision. (*not yet
        implemented.*)

-   `Trans` { *NOTRANS* $|$ *TRANS* $|$ *CONJ* }\
    Specifies whether to solve the transposed system.

-   `SolveInitialized` { *YES* $|$ *NO* }\
    Specifies whether the initialization has been performed to the
    triangular solve.\
    (used only by the distributed input interface)

-   `RefineInitialized` { *YES* $|$ *NO* }\
    Specifies whether the initialization has been performed to the
    sparse matrix-vector multiplication routine needed in the iterative
    refinement.\
    (used only by the distributed input interface)

-   `num_lookaheads` { integer }\
    Specifies the number of levels in the look-ahead factorization

-   `lookahead_etree` { *YES* $|$ *NO* }\
    Specifies whether to use the elimination tree computed from the
    serial symbolic factorization to perform static scheduling.

-   `SymPattern` { *YES* $|$ *NO* }\
    Gives the scheduling algorithm a hint whether the matrix has the
    symmetric pattern.

-   `PrintStat` { *YES* $|$ *NO* }\
    Specifies whether to print the solver's statistics.

There is a routine named `set_default_options_dist()` that sets the
default values of these options, which are:

        fact              = DOFACT           /* factor from scratch */
        equil             = YES
        ParSymbFact       = NO
        colperm           = MMD_AT_PLUS_A
        rowperm           = LargeDiag        /* use MC64 */
        ReplaceTinyPivot  = YES
        IterRefine        = DOUBLE
        Trans             = NOTRANS
        SolveInitialized  = NO
        RefineInitialized = NO
        num_lookaheads    = 10;
        lookahead_etree   = NO;
        SymPattern        = NO;
        PrintStat         = YES

# 4.7 Basic steps to solve a linear system

In this section, we use a complete sample program to illustrate the
basic steps required to use . This program is listed below, and is also
available as `EXAMPLE/pddrive.c` in the source code distribution. All
the routines must include the header file `superlu_ddefs.h` (or
`superlu_zdefs.h`, the complex counterpart) which contains the
definitions of the data types, the macros and the function prototypes.

```c
    #include <math.h>
    #include "superlu_ddefs.h"

    main(int argc, char *argv[])
    /*
     * Purpose
     * =======
     *
     * The driver program PDDRIVE.
     *
     * This example illustrates how to use PDGSSVX with the full
     * (default) options to solve a linear system.
     * 
     * Five basic steps are required:
     *   1. Initialize the MPI environment and the SuperLU process grid
     *   2. Set up the input matrix and the right-hand side
     *   3. Set the options argument
     *   4. Call pdgssvx
     *   5. Release the process grid and terminate the MPI environment
     *
     * On the Cray T3E, the program may be run by typing
     *    mpprun -n <procs> pddrive -r <proc rows> -c <proc columns> <input_file>
     *
     */
   {
        superlu_options_t options;
        SuperLUStat_t stat;
        SuperMatrix A;
        ScalePermstruct_t ScalePermstruct;
        LUstruct_t LUstruct;
        SOLVEstruct_t SOLVEstruct;
        gridinfo_t grid;
        double   *berr;
        double   *b, *xtrue;
        int_t    m, n, nnz;
        int_t    nprow, npcol;
        int      iam, info, ldb, ldx, nrhs;
        char     trans[1];
        char     **cpp, c;
        FILE *fp, *fopen();

        nprow = 1;  /* Default process rows.      */
        npcol = 1;  /* Default process columns.   */
        nrhs = 1;   /* Number of right-hand side. */

        /* ------------------------------------------------------------
           INITIALIZE MPI ENVIRONMENT. 
           ------------------------------------------------------------*/
        MPI_Init( &argc, &argv );

        /* Parse command line argv[]. */
        for (cpp = argv+1; *cpp; ++cpp) {
            if ( **cpp == '-' ) {
                c = *(*cpp+1);
                ++cpp;
                switch (c) {
                  case 'h':
                      printf("Options:\n");
                      printf("\t-r <int>: process rows    (default %d)\n", nprow);
                      printf("\t-c <int>: process columns (default %d)\n", npcol);
                      exit(0);
                      break;
                  case 'r': nprow = atoi(*cpp);
                      break;
                  case 'c': npcol = atoi(*cpp);
                      break;
                }
            } else { /* Last arg is considered a filename */
                if ( !(fp = fopen(*cpp, "r")) ) {
                    ABORT("File does not exist");
                }
                break;
            }
        }

        /* ------------------------------------------------------------
           INITIALIZE THE SUPERLU PROCESS GRID. 
           ------------------------------------------------------------*/
        superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);

        /* Bail out if I do not belong in the grid. */
        iam = grid.iam;
        if ( iam >= nprow * npcol ) goto out;

        /* ------------------------------------------------------------
           GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
           ------------------------------------------------------------*/
        dcreate_matrix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, &grid);

        if ( !(berr = doubleMalloc_dist(nrhs)) )
            ABORT("Malloc fails for berr[].");

        /* ------------------------------------------------------------
           NOW WE SOLVE THE LINEAR SYSTEM.
           ------------------------------------------------------------*/

        /* Set the default input options. */
        set_default_options_dist(&options);

        m = A.nrow;
        n = A.ncol;

        /* Initialize ScalePermstruct and LUstruct. */
        ScalePermstructInit(m, n, &ScalePermstruct);
        LUstructInit(n, &LUstruct);

        /* Initialize the statistics variables. */
        PStatInit(&stat);

        /* Call the linear equation solver. */
        pdgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
                &LUstruct, &SOLVEstruct, berr, &stat, &info);


        /* Check the accuracy of the solution. */
        pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
                         nrhs, b, ldb, xtrue, ldx, &grid);

        PStatPrint(&options, &stat, &grid);        /* Print the statistics. */

        /* ------------------------------------------------------------
           DEALLOCATE STORAGE.
           ------------------------------------------------------------*/

        PStatFree(&stat);
        Destroy_CompRowLoc_Matrix_dist(&A);
        ScalePermstructFree(&ScalePermstruct);
        Destroy_LU(n, &grid, &LUstruct);
        LUstructFree(&LUstruct);
        if ( options.SolveInitialized ) {
            dSolveFinalize(&options, &SOLVEstruct);
        }
        SUPERLU_FREE(b);
        SUPERLU_FREE(xtrue);
        SUPERLU_FREE(berr);

        /* ------------------------------------------------------------
           RELEASE THE SUPERLU PROCESS GRID.
           ------------------------------------------------------------*/
    out:
        superlu_gridexit(&grid);

        /* ------------------------------------------------------------
           TERMINATES THE MPI EXECUTION ENVIRONMENT.
           ------------------------------------------------------------*/
        MPI_Finalize();

    }
```

Five basic steps are required to call a SuperLU routine:

1.  Initialize the MPI environment and the SuperLU process grid.\
    This is achieved by the calls to the MPI routine `MPI_Init()` and
    the SuperLU routine\
    `superlu_gridinit()`. In this example, the communication domain for
    SuperLU is built upon the MPI default communicator `MPI_COMM_WORLD`.
    In general, it can be built upon any MPI communicator. contains the
    details about this step.

2.  Set up the input matrix and the right-hand side.\
    This example uses the interface with the distributed input matrices,
    see Section [2.2](#sec:DistInput). In most practical applications, the
    matrices can be generated on each process without the need to have a
    centralized place to hold them. But for this example, we let process
    0 read the input matrix stored on disk in Harwell-Boeing
    format [@duffgrimes92] (a.k.a. compressed column storage), and
    distribute it to all the other processes, so that each process only
    owns a block of rows of matrix. The right-hand side matrix is
    generated so that the exact solution matrix consists of all ones.
    The subroutine `dcreate_matrix()` accomplishes this task.

3.  Initialize the input arguments:
    `options, ScalePermstruct, LUstruct, stat`.\
    The input argument `options` controls how the linear system would be
    solved---use equilibration or not, how to order the rows and the
    columns of the matrix, use iterative refinement or not. The
    subroutine `set_default_options_dist()` sets the `options` argument
    so that the solver performs all the functionality. You can also set
    it up according to your own needs, see
    section [6](#sec:options) for the fields of this structure.
    `ScalePermstruct` is the data structure that stores the several
    vectors describing the transformations done to $A$. `LUstruct` is
    the data structure in which the distributed $L$ and $U$ factors are
    stored. `Stat` is a structure collecting the statistics about
    runtime and flop count, etc.

4.  Call the SuperLU routine `pdgssvx()`.

5.  Release the process grid and terminate the MPI environment.\
    After the computation on a process grid has been completed, the
    process grid should be released by a call to the SuperLU routine
    `superlu_gridexit()`. When all computations have been completed, the
    MPI routine `MPI_Finalize()` should be called.

# 4.8 User-callable routines

(sec:drivers)=
## 4.8.1 Driver routines 

There are two driver routines to solve systems of linear equations,
which are named `pdgssvx_ABglobal` for the global input interface, and
`pdgssvx` for the distributed interface. We recommend that the general
users, especially the beginners, use a driver routine rather than the
computational routines, because correctly using the driver routine does
not require thorough understanding of the underlying data structures.
Although the interface of these routines are simple, we expect their
rich functionality can meet the requirements of most applications.
`Pdgssvx_ABglobal`/`pdgssvx` perform the following functions:

-   Equilibrate the system (scale $A$'s rows and columns to have unit
    norm) if $A$ is poorly scaled;

-   Find a row permutation that makes diagonal of $A$ large relative to
    the off-diagonal;

-   Find a column permutation that preserves the sparsity of the $L$ and
    $U$ factors;

-   Solve the system $AX=B$ for $X$ by factoring $A$ followed by forward
    and back substitutions;

-   Refine the solution $X$.

## 4.8.2 Computational routines

The experienced users can invoke the following computational routines to
directly control the behavior of in order to meet their requirements.

-   `pdgstrf()`: Factorize in parallel.\
    This routine factorizes the input matrix $A$ (or the scaled and
    permuted $A$). It assumes that the distributed data structures for
    $L$ and $U$ factors are already set up, and the initial values of
    $A$ are loaded into the data structures. If not, the routine
    `symbfact()` should be called to determine the nonzero patterns of
    the factors, and the routine `pddistribute()` should be called to
    distribute the matrix. `Pdgstrf()` can factor non-square matrices.

-   `pdgstrs()/pdgstrs_Bglobal()`: Triangular solve in parallel.\
    This routine solves the system by forward and back substitutions
    using the the $L$ and $U$ factors computed by `pdgstrf()`.
    `Pdgstrs()` takes distributed $B$. For `pdgstrs_Bglobal()`, $B$ must
    be globally available on all processes.

-   `pdgsrfs()/pdgsrfs_ABXglobal()`: Refine solution in parallel.\
    Given $A$, its factors $L$ and $U$, and an initial solution $X$,
    this routine performs iterative refinement. `Pdgsrfs()` takes
    distributed $A$, $B$ and $X$. For `pdgsrfs_ABXglobal()`, $A$, $B$
    and $X$ must be globally available on all processes.

(sec:slud_utility)=
## 4.8.3 Utility routines 

The following utility routines can help users create and destroy the
matrices. These routines reside in three places: `SRC/util.c`,
`SRC/{d,z}util.c`, and `SRC/p{d,z}util.c`. Most of the utility routines
in sequential can also be used in for the local data, see
Section [\[sec:slu_utility\]](#sec:slu_utility). Here, we only list those new routines
specific to . Note that in order to avoid name clash between and , we
append "`_dist`" to each routine name in .

```c
        /* Create a supermatrix in distributed compressed row format. A is output. */
        dCreate_CompRowLoc_Matrix_dist(SuperMatrix *A, int_t m, int_t n,
                                       int_t nnz_loc, int_t m_loc, int_t fst_row,
                                       double *nzval, int_t *colind, int_t *rowptr,
                                       Stype_t stype, Dtype_t dtype, Mtype_t mtype);

        /* Deallocate the supermatrix in distributed compressed row format. */
        Destroy_CompRowLoc_Matrix_dist(SuperMatrix *A);

        /* Allocate storage in ScalePermstruct. */
        ScalePermstructInit(const int_t m, const int_t n,
                            ScalePermstruct_t *ScalePermstruct);

        /* Deallocate ScalePermstruct */
        ScalePermstructFree(ScalePermstruct_t *ScalePermstruct);

        /* Allocate storage in LUstruct. */
        LUstructInit(const int_t n, LUstruct_t *LUstruct);

        /* Deallocate the distributed L & U factors in LUstruct. */
        Destroy_LU(int_t n, gridinfo_t *grid, LUstruct_t *LUstruct);

        /* Deallocate LUstruct. */
        LUstructFree(LUstruct_t *LUstruct);

        /* Initialize the statistics variable. */
        PStatInit(SuperLUStat_t *stat);

        /* Print the statistics. */
        PStatPrint(superlu_options_t *options, SuperLUStat_t *stat,
                   gridinfo_t *grid);

        /* Deallocate the statistics variable. */
        PStatFree(SuperLUStat_t *stat);
```

(sec:dist_install)=
# 4.9 Installation 

## 4.9.1 File structure and complilation

The top level SuperLU_DIST/ directory is structured as follows:

        SuperLU_DIST/README.md instructions on installation
        SuperLU_DIST/CBLAS/    BLAS routines in C, functional but not fast
        SuperLU_DIST/DOC/      Users' Guide
        SuperLU_DIST/EXAMPLE/  example programs
        SuperLU_DIST/INSTALL/  test machine dependent parameters
        SuperLU_DIST/SRC/      C source code, to be compiled into libsuperlu_dist.a
        SuperLU_DIST/lib/      contains library archive libsuperlu_dist.a
        SuperLU_DIST/Makefile  top level Makefile that does installation and testing
        SuperLU_DIST/make.inc  compiler, compiler flags, library definitions and C
                               preprocessor definitions, included in all Makefiles.
                               (You may need to edit it to suit for your system
                                before compiling the whole package.)
        SuperLU_DIST/MAKE_INC/ sample machine-specific make.inc files

You can use CMake automic build system to install the package. Please
see README.md for instruction. The following describes how to install
manually by editing a Makefile.

Before installing the package, you may need to edit
`SuperLU_DIST/make.inc` for your system. This make include file is
referenced inside all the Makefiles in the various subdirectories. As a
result, there is no need to edit the Makefiles in the subdirectories.
All information that is machine specific has been defined in this
include file.

Sample machine-specific `make.inc` are provided in the `MAKE_INC/`
directory for several platforms, such as Cray XE6 and IBM SP. When you
have selected the machine to which you wish to install , you may copy
the appropriate sample include file (if one is present) into `make.inc`.
For example, if you wish to run on a Cray XE6, you can do:

For the systems other than those listed above, slight modifications to
the `make.inc` file will need to be made. In particular, the following
items should be examined:

1.  The BLAS library.\
    If there is a BLAS library available on your machine, you may define
    the following in `make.inc`:

    The `CBLAS/` subdirectory contains the part of the BLAS (in C)
    needed by `SuperLU_DIST` package. However, these routines are
    intended for use only if there is no faster implementation of the
    BLAS already available on your machine. In this case, you should do
    the following:

    -   In make.inc, undefine (comment out) BLASDEF, define:

    -   At the top level SuperLU_DIST directory, type:

        to create the BLAS library from the routines in `CBLAS/`
        subdirectory.

2.  External libraries: and .\
    If you will use or ordering, or parallel symbolic factorization
    (which depends on ), you will need to install them yourself. Since
    package already contains the source code for the library, you can
    just download at:

    <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download>

    After you have installed it, you should define the following in
    `make.inc`:

3.  C preprocessor definition `CDEFS`.\
    In the header file `SRC/Cnames.h`, we use macros to determine how C
    routines should be named so that they are callable by Fortran. [^1]
    The possible options for `CDEFS` are:

    -   `-DAdd_`: Fortran expects a C routine to have an underscore
        postfixed to the name;

    -   `-DNoChange`: Fortran expects a C routine name to be identical
        to that compiled by C;

    -   `-DUpCase`: Fortran expects a C routine name to be all
        uppercase.

4.  (optional) Enable Nvidia GPU access.

    1.  Set the following Linux environment variable: `setenv ACC GPU`

    2.  Add the CUDA library location in `make.inc`:

                ifeq "${ACC}" "GPU"
                CFLAGS += -DGPU_ACC
                INCS += -I<CUDA directory>/include
                LIBS += -L<CUDA directory>/lib64 -lcublas -lcudart 
                endif

A `Makefile` is provided in each subdirectory. The installation can be
done automatically by simply typing "`make`" at the top level.

Hybrid MPI+OpenMP setting is implemented in the factorization routines.
It may outperform MPI-only configurations in some cases and requires
less memory. To use OpenMP parallelism, you need to compile the code
with the following CPP definition:

      -D_OPENMP

and set the number of threads to be used in the environment variable:

      setenv OMP_NUM_THREADS <##>

needs to be set to enable this feature.

(sec:SuperLU_DIST_sp_ienv)=
## 4.9.2 Performance-tuning parameters 

Similar to sequential SuperLU, several performance related parameters
are set in the inquiry function `sp_ienv()`. The declaration of this
function is

`int sp_ienv(int ispec);`

`Ispec` specifies the parameter to be returned [^2]:

ispec = 

      = 2: the relaxation parameter to control supernode amalgamation \((relax)\)

      = 3: the maximum allowable size for a supernode \((maxsup)\)
    
      = 6: size of the array to store the values of the L supernodes \((nzval)\)

<br>

The values to be returned may be set differently on different machines.
The setting of maximum block size (parameter 3) should take into account
the local Level 3 BLAS speed, the load balance and the degree of
parallelism. Small block size may result in better load balance and more
parallelism, but poor individual node performance, and vice versa for
large block size.

These parameters can also be set as Linux environment variables, so that
the routine `sp_ienv()` does not need to be recompiled every time when
you change the settings.

      setenv NREL <##>    /* parameter #2: maximum size of the relaxed supernode */
      setenv NSUP <##>    /* parameter #3: maximum supernode size */

The following parameters are related to GPU usage:

      setenv CUBLAS_NB <##>        /* middle dimension of CUDA GEMM (default 64) */
      setenv MAX_BUFFER_SIZE <##>  /* maximum buffer size on GPU (default 5M words) */

These parameters are described in detail in various algorithm papers,
see [@li05; @sao2014].

# 4.10 Example programs

In the `SuperLU_DIST/EXAMPLE/` directory, we present a few sample
programs to illustrate the complete calling sequences to use the expert
driver to solve systems of equations. These include how to set up the
process grid and the input matrix, how to obtain a fill-reducing
ordering. A `Makefile` is provided to generate the executables. A
`README` file in this directory shows how to run these examples. The
leading comment in each routine describes the functionality of the
example. The two basic examples are `pddrive_ABglobal()` and
`pddrive()`. The first shows how to use the global input interface, and
the second shows how to use the distributed input interface.

(sec:slud_fortran)=
# 4.11 Fortran 90 Interface 

We developed a complete Fortran 90 interface for . All the interface
files and an example driver program are located in the
`SuperLU_DIST/FORTRAN/` directory.
Table [1](#tab:f90_files) lists all the files.


`f_pddrive.f90`  
: An example Fortran driver routine.

`superlu_mod.f90`  
: Fortran 90 module that defines the interface functions to access **SuperLU_DIST**'s data structures.

`superlupara.f90`  
: It contains parameters that correspond to **SuperLU_DIST**'s enums.

`hbcode1.f90`  
: Fortran function for reading a sparse Harwell-Boeing matrix from the file.

`superlu_c2f_wrap.c`  
: C wrapper functions, callable from Fortran. The functions fall into three classes: 1) Those that allocate a structure and return a handle, or deallocate the memory of a structure. 2) Those that get or set the value of a component of a struct. 3) Those that are wrappers for **SuperLU_DIST** functions.

`dcreate_dist_matrix.c`  
: C function for distributing the matrix in a distributed compressed row format.

<div style="text-align:center;">
Table 4.1: The Fortran 90 interface files and an example driver routine.
</div>
<br>

Note that in this interface, all objects (such as ***grid***,
***options***, etc.) in are *opaque*, meaning their size and
structure are not visible to the Fortran user. These opaque objects are
allocated, deallocated and operated in the C side and not directly
accessible from Fortran side. They can only be accessed via *handles*
that exist in Fortran's user space. In Fortran, all handles have type
***INTEGER***. Specifically, in our interface, the size of
Fortran handle is defined by ***superlu_ptr*** in
***superlupara.f90***. For different systems, the size might
need to be changed. Then using these handles, Fortran user can call C
wrapper routines to manipulate the opaque objects. For example, you can
call ***f_create_gridinfo(grid_handle)*** to allocate memory
for structure ***grid***, and return a handle
***grid_handle***.

The sample program illustrates the basic steps required to use in
Fortran to solve systems of equations. These include how to set up the
processor grid and the input matrix, how to call the linear equation
solver. This program is listed below, and is also available as
***f_pddrive.f90*** in the subdirectory. Note that the routine
must include the moudle ***superlu_mod*** which contains the
definitions of all parameters and the Fortran wrapper functions. A
***Makefile*** is provided to generate the executable. A
***README*** file in this directory shows how to run the
example.
```c
          program f_pddrive
    ! 
    ! Purpose
    ! =======
    !
    ! The driver program F_PDDRIVE.
    !
    ! This example illustrates how to use F_PDGSSVX with the full
    ! (default) options to solve a linear system.
    ! 
    ! Seven basic steps are required:
    !   1. Create C structures used in SuperLU
    !   2. Initialize the MPI environment and the SuperLU process grid
    !   3. Set up the input matrix and the right-hand side
    !   4. Set the options argument
    !   5. Call f_pdgssvx
    !   6. Release the process grid and terminate the MPI environment
    !   7. Release all structures
    !
          use superlu_mod
          include 'mpif.h'
          implicit none
          integer maxn, maxnz, maxnrhs
          parameter ( maxn = 10000, maxnz = 100000, maxnrhs = 10 )
          integer rowind(maxnz), colptr(maxn)
          real*8  values(maxnz), b(maxn), berr(maxnrhs)
          integer n, m, nnz, nrhs, ldb, i, ierr, info, iam
          integer nprow, npcol
          integer init

          integer(superlu_ptr) :: grid
          integer(superlu_ptr) :: options
          integer(superlu_ptr) :: ScalePermstruct
          integer(superlu_ptr) :: LUstruct
          integer(superlu_ptr) :: SOLVEstruct
          integer(superlu_ptr) :: A
          integer(superlu_ptr) :: stat


    ! Create Fortran handles for the C structures used in SuperLU_DIST
          call f_create_gridinfo(grid)
          call f_create_options(options)
          call f_create_ScalePermstruct(ScalePermstruct)
          call f_create_LUstruct(LUstruct)
          call f_create_SOLVEstruct(SOLVEstruct)
          call f_create_SuperMatrix(A)
          call f_create_SuperLUStat(stat)

    ! Initialize MPI environment 
          call mpi_init(ierr)

    ! Initialize the SuperLU_DIST process grid
          nprow = 2
          npcol = 2
          call f_superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, grid)

    ! Bail out if I do not belong in the grid. 
          call get_GridInfo(grid, iam=iam)
          if ( iam >= nprow * npcol ) then 
             go to 100
          endif
          if ( iam == 0 ) then 
             write(*,*) ' Process grid ', nprow, ' X ', npcol
          endif

    ! Read Harwell-Boeing matrix, and adjust the pointers and indices
    ! to 0-based indexing, as required by C routines.
          if ( iam == 0 ) then 
             open(file = "g20.rua", status = "old", unit = 5)
             call hbcode1(m, n, nnz, values, rowind, colptr)
             close(unit = 5)
    !
             do i = 1, n+1
                colptr(i) = colptr(i) - 1
             enddo
             do i = 1, nnz
                rowind(i) = rowind(i) - 1
             enddo
          endif

    ! Distribute the matrix to the gird
          call  f_dcreate_matrix_dist(A, m, n, nnz, values, rowind, colptr, grid)

    ! Setup the right hand side
          nrhs = 1
          call  get_CompRowLoc_Matrix(A, nrow_loc=ldb)
          do i = 1, ldb
             b(i) = 1.0
          enddo

    ! Set the default input options
          call f_set_default_options(options)

    ! Change one or more options
    !      call set_superlu_options(options,Fact=FACTORED)

    ! Initialize ScalePermstruct and LUstruct
          call get_SuperMatrix(A,nrow=m,ncol=n)
          call f_ScalePermstructInit(m, n, ScalePermstruct)
          call f_LUstructInit(m, n, LUstruct)

    ! Initialize the statistics variables
          call f_PStatInit(stat)

    ! Call the linear equation solver
          call f_pdgssvx(options, A, ScalePermstruct, b, ldb, nrhs, &
                         grid, LUstruct, SOLVEstruct, berr, stat, info)

          if (info == 0) then
             write (*,*) 'Backward error: ', (berr(i), i = 1, nrhs)
          else
             write(*,*) 'INFO from f_pdgssvx = ', info
          endif

    ! Deallocate SuperLU allocated storage
          call f_PStatFree(stat)
          call f_Destroy_CompRowLoc_Matrix_dist(A)
          call f_ScalePermstructFree(ScalePermstruct)
          call f_Destroy_LU(n, grid, LUstruct)
          call f_LUstructFree(LUstruct)
          call get_superlu_options(options, SolveInitialized=init)
          if (init == YES) then
             call f_dSolveFinalize(options, SOLVEstruct)
          endif

    ! Release the SuperLU process grid
    100   call f_superlu_gridexit(grid)

    ! Terminate the MPI execution environment
          call mpi_finalize(ierr)

    ! Destroy all C structures
          call f_destroy_gridinfo(grid)
          call f_destroy_options(options)
          call f_destroy_ScalePermstruct(ScalePermstruct)
          call f_destroy_LUstruct(LUstruct)
          call f_destroy_SOLVEstruct(SOLVEstruct)
          call f_destroy_SuperMatrix(A)
          call f_destroy_SuperLUStat(stat)

          stop
          end
```
Similar to the driver routine ***pddrive.c*** in C, seven basic
steps are required to call a routine in Fortran:

1.  Create C structures used in SuperLU: ***grid***,
    ***options***, ***ScalePermstruct***,
    ***LUstruct***, ***SOLVEstruct***,
    ***A*** and ***stat***. This is achieved by the
    calls to the C wrapper *"create"* routines
    ***f_create_XXX()***, where ***XXX*** is the name
    of the corresponding structure.

2.  Initialize the MPI environment and the SuperLU process grid. This is
    achieved by the calls to ***mpi_init()*** and the C wrapper
    routine ***f_superlu_gridinit()***. Note that
    ***f_superlu_gridinit()*** requires the numbers of row and
    column of the process grid. In this example, we set them to be $2$,
    respectively.

3.  Set up the input matrix and the right-hand side. This example uses
    the distributed input interface, so we need to convert the input
    matrix to the distributed compressed row format. Process $0$ first
    reads the input matrix stored on disk in Harwell-Boeing format by
    calling Fortran routine ***hbcode1()***. The file name in
    this example is `g20.rua`. Then all processes call a C wrapper
    routine ***f_dcreate_dist_matrix()*** to distribute the
    matrix to all the processes distributed by block rows. The
    right-hand side matrix in this example is a column vector of all
    ones. Note that, before setting the right-hand side, we use
    ***get_CompRowLoc_Matrix()*** to get the number of local
    rows in the distributed matrix $A$.

    *One important note is that all the C routines use 0-based indexing
    scheme. Therefore, after process 0 reads the matrix in compressed
    column format, we decrement its column pointers (`colptr`) and row
    indices (`rowind`) by 1 so they become 0-based indexing.*

4.  Set the input arguments: ***options***,
    ***ScalePermstruct***, ***LUstruct***, and
    ***stat***. The input argument ***options***
    controls how the linear system would be sloved. The routine
    ***f_set_default_options_dist()*** sets the
    ***options*** argument so that the slover performs all the
    functionalities. You can also set it according to your own needs,
    using a call to the Fortran routine
    ***set_superlu_options()***. ***LUstruct*** is the
    data struture in which the distributed $L$ and $U$ factors are
    stored. ***ScalePermstruct*** is the data struture in which
    several vectors describing the transformations done to matrix $A$
    are stored. ***stat*** is a structure collecting the
    statistcs about runtime and flop count. These three structures can
    be set by calling the C wrapper *"init"* routines
    ***f_XXXInit***.

5.  Call the C wrapper routine ***f_pdgssvx()*** to solve the
    equation.

6.  Release the process grid and terminate the MPI environment. After
    the computation on a process grid has been completed, the process
    grid should be released by a call to
    ***f_spuerlu_gridexit()***. When all computations have been
    completed, the C wrapper routine ***mpi_finalize()***
    should be called.

7.  Deallocate all the structures. First we need to deallocate the
    storage allocated by by a set of *"free"* calls. Note that this
    should be called before ***f_spuerlu_gridexit()***, since
    some of the *"free"* calls use the grid. Then we call the C wrapper
    *"destroy"* routines ***f_destroy_XXX()*** to destroy all
    the Fortran handles. Note that ***f_destroy_gridinfo()***
    should be called after ***f_spuerlu_gridexit()***.

## Callable functions in the Fortran 90 module file ***spuerlu_mod.f90***

The Fortran 90 module `superlu_mod` contains the interface routines that
can manipulate a object from Fortran. The object is pointed to by the
corresponding handle input to these routines. The routines are divided
into two sets. One set is to get the properties of an object, with the
routine names "`get_XXX()`". Another set is to set some properties for
an object, with the routine names "`set_XXX()`". These functions have
optional arguments, so the users do not have to provide the full set of
parameters. `Superlu_mod` module uses `superluparam_mod` module that
defines all the integer constants corresponding to the enumeration
constants in . Below are the calling sequences of all the routines.
```c
    subroutine get_GridInfo(grid, iam, nprow, npcol)
      integer(superlu_ptr) :: grid
      integer, optional :: iam, nprow, npcol

    subroutine get_SuperMatrix(A, nrow, ncol)
      integer(superlu_ptr) :: A
      integer, optional :: nrow, ncol

    subroutine set_SuperMatrix(A, nrow, ncol)
      integer(superlu_ptr) :: A
      integer, optional :: nrow, ncol

    subroutine get_CompRowLoc_Matrix(A, nrow, ncol, nnz_loc, nrow_loc, fst_row)
      integer(superlu_ptr) :: A
      integer, optional :: nrow, ncol, nnz_loc, nrow_loc, fst_row

    subroutine set_CompRowLoc_Matrix(A, nrow, ncol, nnz_loc, nrow_loc, fst_row)
      integer(superlu_ptr) :: A
      integer, optional :: nrow, ncol, nnz_loc, nrow_loc, fst_row

    subroutine get_superlu_options(opt, Fact, Trans, Equil, RowPerm, &
                                   ColPerm, ReplaceTinyPivot, IterRefine, &
                                   SolveInitialized, RefineInitialized)
    integer(superlu_ptr) :: opt
      integer, optional :: Fact, Trans, Equil, RowPerm, ColPerm, &
                           ReplaceTinyPivot, IterRefine, SolveInitialized, &
                           RefineInitialized

    subroutine set_superlu_options(opt, Fact, Trans, Equil, RowPerm, &
                                   ColPerm, ReplaceTinyPivot, IterRefine, &
                                   SolveInitialized, RefineInitialized)
      integer(superlu_ptr) :: opt
      integer, optional :: Fact, Trans, Equil, RowPerm, ColPerm, &
                           ReplaceTinyPivot, IterRefine, SolveInitialized, &
                           RefineInitialized
```
## C wrapper functions callable by Fortran in ***file spuerlu_c2f_wrap.c***

This file contains the Fortran-callable C functions which wraps around
the user-callable C routines in . The functions are divided into three
classes: 1) allocate a C structure and return a handle to Fortran, or
deallocate the memory of of a C structure given its Fortran handle; 2)
get or set the value of certain fields of a C structure given its
Fortran handle; 3) wrapper functions for the C functions. Below are the
calling sequences of these routines.
```c
    /* functions that allocate memory for a structure and return a handle */
    void f_create_gridinfo(fptr *handle)
    void f_create_options(fptr *handle)
    void f_create_ScalePermstruct(fptr *handle)
    void f_create_LUstruct(fptr *handle)
    void f_create_SOLVEstruct(fptr *handle)
    void f_create_SuperMatrix(fptr *handle)
    void f_create_SuperLUStat(fptr *handle)

    /* functions that free the memory allocated by the above functions */
    void f_destroy_gridinfo(fptr *handle)
    void f_destroy_options(fptr *handle)
    void f_destroy_ScalePermstruct(fptr *handle)
    void f_destroy_LUstruct(fptr *handle)
    void f_destroy_SOLVEstruct(fptr *handle)
    void f_destroy_SuperMatrix(fptr *handle)
    void f_destroy_SuperLUStat(fptr *handle)

    /* functions that get or set certain fields in a C structure. */
    void f_get_gridinfo(fptr *grid, int *iam, int *nprow, int *npcol)
    void f_get_SuperMatrix(fptr *A, int *nrow, int *ncol)
    void f_set_SuperMatrix(fptr *A, int *nrow, int *ncol)
    void f_get_CompRowLoc_Matrix(fptr *A, int *m, int *n, int *nnz_loc,
                                          int *m_loc, int *fst_row)
    void f_set_CompRowLoc_Matrix(fptr *A, int *m, int *n, int *nnz_loc,
                                          int *m_loc, int *fst_row)
    void f_get_superlu_options(fptr *opt, int *Fact, int *Trans, int *Equil,
                               int *RowPerm, int *ColPerm, int *ReplaceTinyPivot,
                               int *IterRefine, int *SolveInitialized,
                               int *RefineInitialized)
    void f_set_superlu_options(fptr *opt, int *Fact, int *Trans, int *Equil,
                               int *RowPerm, int *ColPerm, int *ReplaceTinyPivot,
                               int *IterRefine, int *SolveInitialized,
                               int *RefineInitialized)

    /* wrappers for SuperLU_DIST routines */
    void f_dCreate_CompRowLoc_Matrix_dist(fptr *A, int *m, int *n, int *nnz_loc,
                                          int *m_loc, int *fst_row, double *nzval,
                                          int *colind, int *rowptr, int *stype,
                                          int *dtype, int *mtype)
    void f_set_default_options(fptr *options)
    void f_superlu_gridinit(int *Bcomm, int *nprow, int *npcol, fptr *grid)
    void f_superlu_gridexit(fptr *grid)
    void f_ScalePermstructInit(int *m, int *n, fptr *ScalePermstruct)
    void f_ScalePermstructFree(fptr *ScalePermstruct)
    void f_PStatInit(fptr *stat)
    void f_PStatFree(fptr *stat)
    void f_LUstructInit(int *m, int *n, fptr *LUstruct)
    void f_LUstructFree(fptr *LUstruct)
    void f_Destroy_LU(int *n, fptr *grid, fptr *LUstruct)
    void f_Destroy_CompRowLoc_Matrix_dist(fptr *A)
    void f_dSolveFinalize(fptr *options, fptr *SOLVEstruct)
    void f_pdgssvx(fptr *options, fptr *A, fptr *ScalePermstruct, double *B,
                   int *ldb, int *nrhs, fptr *grid, fptr *LUstruct,
                   fptr *SOLVEstruct, double *berr, fptr *stat, int *info)
    void f_check_malloc(int *iam)

[^1]: Some vendor-supplied BLAS libraries do not have C interfaces. So
    the re-naming is needed in order for the SuperLU BLAS calls (in C)
    to interface with the Fortran-style BLAS.

[^2]: The numbering of 2, 3 and 6 is consistent with that used in
    SuperLU and SuperLU_MT.
```