## Chapter 3

# Multithreaded SuperLU (Version 2.0)


## 3.1 About SuperLU_MT

Among the various steps of the solution process in the sequential
SuperLU, the $LU$ factorization dominates the computation; it usually
takes more than 95% of the sequential runtime for large sparse linear
systems. We have designed and implemented an algorithm to perform the
factorization in parallel on machines with a shared address space and
multithreading. The parallel algorithm is based on the efficient
sequential algorithm implemented in SuperLU. Although we attempted to
minimize the amount of changes to the sequential code, there are still a
number of non-trivial modifications to the serial SuperLU, mostly
related to the matrix data structures and memory organization. All these
changes are summarized in
Table [1](#tab:diff_superlu) and their impacts on performance are
studied thoroughly in [@superlu_smp99; @li96]. In this part of the
Users' Guide, we describe only the changes that the user should be aware
of. Other than these differences, most of the material in
chapter [\[chap:superlu\]](#chap:superlu) is still applicable.

::: {.center}
::: {#tab:diff_superlu}
  Construct           Parallel algorithm
  ------------------- ----------------------------------------------------------------------------------------------------------------------
  panel               restricted so it does not contain branchings in the elimination tree
  supernode           restricted to be a fundamental supernode in the elimination tree
  supernode storage   use either static or dynamic upper bound (section [5.2.1](#sec:mt_mem))
  pruning & DFS       use both $G(L^T)$ and pruned $G(L^T)$ to avoid locking

  : The differences between the parallel and the sequential algorithms.
:::
:::

(sec:mt_datastructure)=
# 3.2 Storage types for $L$ and $U$ 

As in the sequential code, the type for the factored matrices $L$ and
$U$ is `SuperMatrix`
Figure [\[fig:struct\]](#fig:struct), however, their storage formats (stored in
`Store`) are changed. In the parallel algorithm, the adjacent panels of
the columns may be assigned to different processes, and they may be
finished and put in global memory out of order. That is, the consecutive
columns or supernodes may not be stored contiguously in memory. Thus, in
addition to the pointers to the beginning of each column or supernode,
we need pointers to the end of the column or supernode. In particular,
the storage type for $L$ is `SCP` (Supernode, Column-wise and Permuted),
defined as:

        typedef struct {
            int  nnz;          /* number of nonzeros in the matrix */
            int  nsuper;       /* number of supernodes */
            void *nzval;       /* pointer to array of nonzero values, 
                                  packed by column */
            int *nzval_colbeg; /* nzval_colbeg[j] points to beginning of column j
                                  in nzval[] */
            int *nzval_colend; /* nzval_colend[j] points to one past the last 
                                  element of column j in nzval[] */
            int *rowind;       /* pointer to array of compressed row indices of 
                                  the supernodes */
            int *rowind_colbeg;/* rowind_colbeg[j] points to beginning of column j
                                  in rowind[] */
            int *rowind_colend;/* rowind_colend[j] points to one past the last
                                  element of column j in rowind[] */
            int *col_to_sup;   /* col_to_sup[j] is the supernode number to which
                                  column j belongs */
            int *sup_to_colbeg;/* sup_to_colbeg[s] points to the first column 
                                  of the s-th supernode /
            int *sup_to_colend;/* sup_to_colend[s] points to one past the last
                                  column of the s-th supernode */
        } SCPformat;

The storage type for $U$ is `NCP`, defined as:

        typedef struct {
            int  nnz;     /* number of nonzeros in the matrix */
            void *nzval;  /* pointer to array of nonzero values, packed by column */
            int  *rowind; /* pointer to array of row indices of the nonzeros */
            int  *colbeg; /* colbeg[j] points to the location in nzval[] and rowind[]
                             which starts column j */
            int  *colend; /* colend[j] points to one past the location in nzval[]
                             and rowind[] which ends column j */
        } NCPformat;

The table below summarizes the data and storage types of all the
matrices involved in the parallel routines:

::: {.center}
             $A$   $L$   $U$   $B$   $X$
  --------- ----- ----- ----- ----- -----
  `Stype`     or                    
  `Dtype`    any   any   any   any   any
  `Mtype`                           
:::

# 3.3 Options argument

The `options` argument is the input argument to control the behaviour of
the libraries. `Options` is implemented as a C structure containing the
following fields:

-   `nprocs`\
    Specifies the number of threads to be spawned.

-   `Fact`\
    Specifies whether or not the factored form of the matrix $A$ is
    supplied on entry, and if not, how the matrix $A$ will be factorized
    base on the previous history, such as factor from scratch, reuse
    $P_c$ and/or $P_r$, or reuse the data structures of $L$ and $U$.
    `fact` can be one of:

    -   `DOFACT`: the matrix $A$ will be factorized from scratch.

    -   `EQUILIBRATE`: the matrix A will be equilibrated, then factored
        into L and U.

    -   `FACTORED`: the factored form of $A$ is input.

-   `Trans` { `NOTRANS` $|$ `TRANS` $|$ `CONJ` }\
    Specifies whether to solve the transposed system.

-   `panel_size`\
    Specifies the number of consecutive columns to be treated as a unit
    of task.

-   `relax`\
    Specifies the number of columns to be grouped as a relaxed
    supernode.

-   `refact` { `YES` $|$ `NO` }\
    Specifies whether this is first time or subsequent factorization.

-   `diag_pivot_thresh` $[0.0, 1.0]$\
    Specifies the threshold used for a diagonal entry to be an
    acceptable pivot.

-   `SymmetricMode` { `YES` $|$ `NO` }\
    Specifies whether to use the symmetric mode.

-   `PrintStat` { `YES` $|$ `NO` }\
    Specifies whether to print the solver's statistics.

# 3.4 User-callable routines

As in the sequential SuperLU, we provide both computational routines and
driver routines. To name those routines that involve parallelization in
the call-graph, we prepend a letter `p` to the names of their sequential
counterparts, for example `pdgstrf`. For the purely sequential routines,
we use the same names as before. Here, we only list the routines that
are different from the sequential ones.

## 3.4.1 Driver routines

We provide two types of driver routines for solving systems of linear
equations. The driver routines can handle both column- and row-oriented
storage schemes.

-   A simple driver `pdgssv`, which solves the system $AX=B$ by
    factorizing $A$ and overwriting $B$ with the solution $X$.

-   An expert driver `pdgssvx`, which, in addition to the above, also
    performs the following functions (some of them optionally):

    -   solve $A^TX=B$;

    -   equilibrate the system (scale $A$'s rows and columns to have
        unit norm) if $A$ is poorly scaled;

    -   estimate the condition number of $A$, check for
        near-singularity, and check for pivot growth;

    -   refine the solution and compute forward and backward error
        bounds.

## 3.4.2 Computational routines

The user can invoke the following computational routines to directly
control the behavior of SuperLU. The computational routines can only
handle column-oriented storage. Except for the parallel factorization
routine `pdgstrf`, all the other routines are identical to those
appeared in the sequential superlu.

-   `pdgstrf`: Factorize (in parallel).

    This implements the first-time factorization, or later
    re-factorization with the same nonzero pattern. In
    re-factorizations, the code has the ability to use the same column
    permutation $P_c$ and row permutation $P_r$ obtained from a previous
    factorization. Several scalar arguments control how the $LU$
    decomposition and the numerical pivoting should be performed.
    `pdgstrf` can handle non-square matrices.

-   `dgstrs`: Triangular solve.

    This takes the $L$ and $U$ triangular factors, the row and column
    permutation vectors, and the right-hand side to compute a solution
    matrix $X$ of $AX=B$ or $A^TX=B$.

-   `dgscon`: Estimate condition number.

    Given the matrix $A$ and its factors $L$ and $U$, this estimates the
    condition number in the one-norm or infinity-norm. The algorithm is
    due to Hager and Higham [@higham96], and is the same as `condest` in
    sparse Matlab.

-   `dgsequ/dlaqgs`: Equilibrate.

    `dgsequ` first computes the row and column scalings $D_r$ and $D_c$
    which would make each row and each column of the scaled matrix
    $D_rAD_c$ have equal norm. `dlaqgs` then applies them to the
    original matrix $A$ if it is indeed badly scaled. The equilibrated
    $A$ overwrites the original $A$.

-   `dgsrfs`: Refine solution.

    Given $A$, its factors $L$ and $U$, and an initial solution $X$,
    this does iterative refinement, using the same precision as the
    input data. It also computes forward and backward error bounds for
    the refined solution.

(sec:mt_install)=
# 3.5 Installation 

## 3.5.1 File structure

The top level SuperLU_MT/ directory is structured as follows:

       SuperLU_MT_2.0/README    instructions on installation
       SuperLU_MT_2.0/CBLAS/    BLAS routines in C, functional but not fast
       SuperLU_MT_2.0/DOC/      Users' Guide
       SuperLU_MT_2.0/EXAMPLE/  example programs
       SuperLU_MT_2.0/INSTALL/  test machine dependent parameters
       SuperLU_MT_2.0/SRC/      C source code, to be compiled into libsuperlu_mt.a
       SuperLU_MT_2.0/TESTING/  driver routines to test correctness
       SuperLU_MT_2.0/lib/      SuperLU_MT library archive libsuperlu_mt.a
       SuperLU_MT_2.0/Makefile  top level Makefile that does installation and testing
       SuperLU_MT_2.0/MAKE_INC  sample machine-specific make.inc files
       SuperLU_MT_2.0/make.inc  compiler, compiler flags, library definitions and C
                                preprocessor definitions, included in all Makefiles.
                                (You may need to edit it to suit for your system 
                                 before compiling the whole package.)

We have ported the parallel programs to a number of platforms, which are
reflected in the make include files provided in the top level directory,
for example,
`make.pthreads, make.openmp, make.ibm, make.sun, make.sgi, make.cray`.
If you are using one of these machines, such as an IBM, you can simply
copy `make.sun` into `make.inc` before compiling. If you are not using
any of the machines to which we have ported, you will need to read
section [7](#sec:mt_port)
about the porting instructions.

The rest of the installation and testing procedure is similar to that
described in
section [\[sec:install\]](#sec:install) for the serial SuperLU. Then, you can type
`make` at the top level directory to finish installation. In the
`SuperLU_MT/TESTING` subdirectory, you can type `pdtest.csh` to perform
testings.

(sec:superlu_mt_perf)=
## 3.5.2 Performance issues 

(sec:mt_mem)=
### Memory management for $L$ and $U$ 

In the sequential SuperLU, four data arrays associated with the $L$ and
$U$ factors can be expanded dynamically, as described in
section [\[sec:mem\]](#sec:mem). In the parallel code, the expansion is hard and
costly to implement, because when a process detects that an array bound
is exceeded, it has to send a signal to and suspend the execution of the
other processes. Then the detecting process can proceed with the array
expansion. After the expansion, this process must wake up all the
suspended processes.

In this release of the parallel code, we have not yet implemented the
above expansion mechanism. For now, the user must pre-determine an
estimated size for each of the four arrays through the inquiry function
`sp_ienv()`. There are two interpretations for each integer value `FILL`
returned by calling this function with `ispec = 6, 7, or 8`. A negative
number is interpreted as the fills growth factor, that is, the program
will allocate `(-FILL)*nnz(A)` elements for the corresponding array. A
positive number is interpreted as the true amount the user wants to
allocate, that is, the program will allocate `FILL` elements for the
corresponding array. In both cases, if the initial request exceeds the
physical memory constraint, the sizes of the arrays are repeatedly
reduced until the initial allocation succeeds.

`int sp_ienv(int ispec);`

`Ispec` specifies the parameter to be returned:

ispec =

      = 6: size of the array to store the values of the L supernodes \((nzval)\)

      = 7: size of the array to store the columns in U \((nzval/rowind)\)

      = 8: size of the array to store the subscripts of the L supernodes \((rowind)\)

If the actual fill exceeds any array size, the program will abort with a
message showing the current column when failure occurs, and indicating
how many elements are needed up to the current column. The user may
reset a larger fill parameter for this array and then restart the
program.

To make the storage allocation more efficient for the supernodes in $L$,
we devised a special storage scheme. The need for this special treatment
and how we implement it are fully explained and studied
in [@superlu_smp99; @li96]. Here, we only sketch the main idea. Recall
that the parallel algorithm assigns one panel of columns to one process.
Two consecutive panels may be assigned to two different processes, even
though they may belong to the same supernode discovered later. Moreover,
a third panel may be finished by a third process and put in memory
between these two panels, resulting in the columns of a supernode being
noncontiguous in memory. This is undesirable, because then we cannot
directly call BLAS routines using this supernode unless we pay the cost
of copying the columns into contiguous memory first. To overcome this
problem, we exploited the observation that the nonzero structure for $L$
is contained in that of the Householder matrix $H$ from the Householder
sparse $QR$ transformation [@georgeliung88; @georgeng87]. Furthermore,
it can be shown that a fundamental supernode of $L$ is always contained
in a fundamental supernode of $H$. This containment property is true for
any row permutation $P_r$ in $P_rA = LU$. Therefore, we can pre-allocate
storage for the $L$ supernodes based on the size of $H$ supernodes.
Fortunately, there exists a fast algorithm (almost linear in the number
of nonzeros of $A$) to compute the size of $H$ and the supernodes
partition in $H$ [@glnp:01].

In practice, the above static prediction is fairly tight for most
problems. However, for some others, the number of nonzeros in $H$
greatly exceeds the number of nonzeros in $L$. To handle this situation,
we implemented an algorithm that still uses the supernodes partition in
$H$, but dynamically searches the supernodal graph of $L$ to obtain a
much tighter bound for the storage. Table 6 in [@superlu_smp99]
demonstrates the storage efficiency achieved by both static and dynamic
approach.

In summary, our program tries to use the static prediction first for the
$L$ supernodes. In this case, we ignore the integer value given in the
function `sp_ienv(6)`, and simply use the nonzero count of $H$. If the
user finds that the size of $H$ is too large, he can invoke the dynamic
algorithm at runtime by setting the following Linux shell environment
variable:

`setenv SuperLU_DYNAMIC_SNODE_STORE 1`

The dynamic algorithm incurs runtime overhead. For example, this
overhead is usually between 2% and 15% on a single processor RS/6000-590
for a range of test matrices.

### Symmetric structure pruning

In both serial and parallel algorithms, we have implemented Eisenstat
and Liu's symmetric pruning idea of representing the graph $G(L^T)$ by a
reduced graph $G'$, and thereby reducing the DFS traversal time. A
subtle difficulty arises in the parallel implementation.

When the owning process of a panel starts DFS (depth-first search) on
$G'$ built so far, it only sees the partial graph, because the part of
$G'$ corresponding to the busy panels down the elimination tree is not
yet complete. So the structural prediction at this stage can miss some
nonzeros. After performing the updates from the finished supernodes, the
process will wait for all the busy descendant panels to finish and
perform more updates from them. Now, we make a conservative assumption
that all these busy panels will update the current panel so that their
nonzero structures are included in the current panel.

This approximate scheme works fine for most problems. However, we found
that this conservatism may sometimes cause a large number of structural
zeros (they are related to the supernode amalgamation performed at the
bottom of the elimination tree) to be included and they in turn are
propagated through the rest of the factorization.

We have implemented an exact structural prediction scheme to overcome
this problem. In this scheme, when each numerical nonzero is scattered
into the sparse accumulator array, we set the occupied flag as well.
Later when we accumulate the updates from the busy descendant panels, we
check the occupied flags to determine the exact nonzero structure. This
scheme avoids unnecessary zero propagation at the expense of runtime
overhead, because setting the occupied flags must be done in the inner
loop of the numeric updates.

We recommend that the user use the approximate scheme (by default)
first. If the user finds that the amount of fill from the parallel
factorization is substantially greater than that from the sequential
factorization, he can then use the accurate scheme. To invoke the second
scheme, the user should recompile the code by defining the macro:

`-D SCATTER_FOUND`

for the C preprocessor.

(sec:SuperLU_MT_sp_ienv)=
### The inquiry function `sp_ienv()` 

For some user controllable constants, such as the blocking parameters
and the size of the global storage for $L$ and $U$, SuperLU_MT calls the
inquiry function `sp_ienv()` to retrieve their values. The declaration
of this function is

`int sp_ienv(int ispec).`

The full meanings of the returned values are as follows:

ispec = ...

      = 1: the panel size \(w\)

      = 2: the relaxation parameter to control supernode amalgamation \((relax)\)

      = 3: the maximum allowable size for a supernode \((maxsup)\)

      = 4: the minimum row dimension for 2D blocking to be used \((rowblk)\)

      = 5: the minimum column dimension for 2D blocking to be used \((colblk)\)
    
      = 6: size of the array to store the values of the L supernodes \((nzval)\)

      = 7: size of the array to store the columns in U \((nzval/rowind)\)

      = 8: size of the array to store the subscripts of the L supernodes \((rowind)\)
<br>
<br>


| make.inc      | Platforms                   | Programming Model | Environment Variable |
|---------------|-----------------------------|-------------------|----------------------|
| make.pthreads | Machines with POSIX threads | pthreads          |                      |
| make.openmp   | Machines with OpenMP        | OpenMP            | `OMP_NUM_THREADS`    |
| make.alpha    | DEC Alpha Servers           | DECthreads        |                      |
| make.cray     | Cray C90/J90                | microtasking      | `NCPUS`              |
| make.ibm      | IBM Power series            | pthreads          |                      |
| make.origin   | SGI/Cray Origin2000         | parallel C        | `MP_SET_NUMTHREADS`  |
| make.sgi      | SGI Power Challenge         | parallel C        | `MPC_NUM_THREADS`    |
| make.sun      | Sun Ultra Enterprise        | Solaris threads   |                      |
<div style="text-align:center;">
Figure 3.2: Platforms on which SuperLU_MT was tested.
</div>
<br>
<br>

We should take into account the trade-off between cache reuse and amount
of parallelism in order to set the appropriate $w$ and $maxsup$. Since
the parallel algorithm assigns one panel factorization to one process,
large values may constrain concurrency, even though they may be good for
uniprocessor performance. We recommend that $w$ and $maxsup$ be set a
bit smaller than the best values used in the sequential code.

The settings for parameters 2, 4 and 5 are the same as those described
in section [\[sec:parameters\]](#sec:parameters). The settings for parameters 6, 7 and 8 are
discussed in section [5.2.1](#sec:mt_mem).

In the file `SRC/sp_ienv.c`, we provide sample settings of these
parameters for several machines.

# 3.6 Example programs

In the `SuperLU_MT/EXAMPLE/` subdirectory, we present a few sample
programs to illustrate the complete calling sequences to use the simple
and expert drivers to solve systems of equations. Examples are also
given to illustrate how to perform a sequence of factorizations for the
matrices with the same sparsity pattern, and how SuperLU_MT can be
integrated into the other multithreaded application such that threads
are created only once. A `Makefile` is provided to generate the
executables. A `README` file in this directory shows how to run these
examples. The leading comment in each routine describes the
functionality of the example.

(sec:mt_port)=
# 3.7 Porting to other platforms 

We have provided the parallel interfaces for a number of shared-memory
machines. Table [2](#tab:mt_machines) lists the platforms on which we have tested
the library, and the respective `make.inc` files. The most portable
interface for shared memory programming is POSIX threads [@posix], since
nowadays many commercial UNIX operating systems have support for it. We
call our POSIX threads interface the `Pthreads` interface. To use this
interface, you can copy `make.pthreads` into `make.inc` and then compile
the library. In the last column of
Table [2](#tab:mt_machines), we list the runtime environment variable
to be set in order to use multiple CPUs. For example, to use 4 CPUs on
the Origin2000, you need to set the following before running the
program:

`setenv MP_SET_NUMTHREADS 4`

| Mutex         | Critical region                                      |
|---------------|------------------------------------------------------|
| `ULOCK`       | allocate storage for a column of matrix $U$          |
| `LLOCK`       | allocate storage for row subscripts of matrix $L$    |
| `LULOCK`      | allocate storage for the values of the supernodes    |
| `NSUPER_LOCK` | increment supernode number `nsuper`                  |
| `SCHED_LOCK`  | invoke `Scheduler()` which may update global task queue |

<div style="text-align:center;">
Table 3.3: Five mutex variables.
</div>
<br>
<br>

In the source code, all the platform specific constructs are enclosed in
the C `#ifdef` preprocessor statement. If your platform is different
from any one listed in Table [2](#tab:mt_machines), you need to go to these places and create
the parallel constructs suitable for your machine. The two constructs,
concurrency and synchronization, are explained in the following two
subsections, respectively.

## 3.7.1 Creating multiple threads

Right now, only the factorization routine `pdgstrf` is parallelized,
since this is the most time-consuming part in the whole solution
process. There is one single thread of control on entering and exiting
`pdgstrf`. Inside this routine, more than one thread may be created. All
the newly created threads begin by calling the thread function
`pdgstrf_thread` and they are concurrently executed on multiple
processors. The thread function `pdgstrf_thread` expects a single
argument of type `void*`, which is a pointer to the structure containing
all the shared data objects.

## 3.7.2 Use of mutexes

Although the threads `pdgstrf_thread` execute independently of each
other, they share the same address space and can communicate efficiently
through shared variables. Problems may arise if two threads try to
access (at least one is to modify) the shared data at the same time.
Therefore, we must ensure that all memory accesses to the same data are
mutually exclusive. There are five critical regions in the program that
must be protected by mutual exclusion. Since we want to allow different
processors to enter different critical regions simultaneously, we use
five mutex variables as listed in
Table [3](#tab:mutexes).
The user should properly initialize them in routine `ParallelInit`, and
destroy them in routine `ParallelFinalize`. Both these routines are in
file `pxgstrf_synch.c`.
