# Introduction

(sec:PurposeofSuperLU)=

## 1.1 Purpose of SuperLU

This document describes a collection of three related ANSI C subroutine
libraries for solving sparse linear systems of equations $AX=B$. Here
$A$ is a square, nonsingular, $n\times n$ sparse matrix, and $X$ and $B$
are dense $n\times nrhs$ matrices, where $nrhs$ is the number of
right-hand sides and solution vectors. The LU factorization routines can
handle non-square matrices. Matrix $A$ need not be symmetric or
definite; indeed, SuperLU is particularly appropriate for matrices with
very unsymmetric structure. All three libraries use variations of
Gaussian elimination optimized to take advantage of both sparsity and
the computer architecture, in particular memory hierarchies (caches) and
parallelism.

In this introduction we refer to all three libraries collectively as
SuperLU. The three libraries within SuperLU are as follows. Detailed
references are also given (see also {cite}`li96`).

- **Sequential SuperLU** is designed for sequential processors with one
  or more layers of memory hierarchy (caches) {cite}`superlu99`,{cite}`lidemmel03`.

- **Multithreaded SuperLU** (***SuperLU_MT***) is designed for shared memory
  multiprocessors (SMPs), and can effectively use up to 16 or 32
  parallel processors on sufficiently large matrices in order to speed
  up the computation {cite}`superlu99`.

- **Distributed SuperLU** (***SuperLU_DIST***) is designed for distributed memory parallel
  processors, using MPI {cite}`liu85` for interprocess communication. It
  can effectively use hundreds of parallel processors on sufficiently
  large matrices  {cite}`li03`.

The rest of the Introduction is organized as follows.
Section [1.2](#sec:OverallAlgorithm) describes the high-level algorithm
used by all three libraries, pointing out some common features and
differences. Section [1.3](#sec:Commonalities) describes the detailed algorithms, data
structures, and interface issues common to all three routines.
Section [1.4](#sec:Differences) describes how the three routines differ,
emphasizing the differences that most affect the user.
Section [1.6](#sec:SoftwareStatus) describes the software status, including
planned developments, bug reporting, and licensing.
**Table 1.1** summarizes the current status of the software. All the routines are implemented in C, with parallel extensions using Pthreads or OpenMP for shared-memory programming, or MPI for distributed-memory programming. We provide a Fortran interface for all three libraries.

|                          | Sequential **SuperLU** | **SuperLU_MT**             | **SuperLU_DIST**            |
|--------------------------|------------------------|----------------------------|-----------------------------|
| **Platform**             | serial                 | shared-memory              | distributed-memory          |
| **Language**<br>(with Fortran interface) | C                      | C + Pthreads (or OpenMP)   | C + MPI                     |
| **Data type**            | real/complex,<br>single/double | real/complex,<br>single/double | real/complex,<br>double |

:::{div align="center"}

**Table 1.1**: SuperLU software status

:::

(sec:OverallAlgorithm)=
# 1.2 Overall Algorithm

A simple description of the algorithm for solving linear equations by
sparse Gaussian elimination is as follows:

1.  Compute a *triangular factorization* $P_r D_r A D_c P_c = L U$. Here
    $D_r$ and $D_c$ are diagonal matrices to equilibrate the system,
    $P_r$ and $P_c$ are *permutation matrices*. Premultiplying $A$ by
    $P_r$ reorders the rows of $A$, and postmultiplying $A$ by $P_c$
    reorders the columns of $A$. $P_r$ and $P_c$ are chosen to enhance
    sparsity, numerical stability, and parallelism. $L$ is a unit lower
    triangular matrix ($L_{ii}=1$) and $U$ is an upper triangular
    matrix. The factorization can also be applied to non-square
    matrices.

2.  Solve $AX=B$ by evaluating
    $X = A^{-1}B = (D_r^{-1}P_r^{-1}LUP_c^{-1}D_c^{-1})^{-1} B
       = D_c (P_c(U^{-1}(L^{-1}(P_r (D_r B)))))$. This is done
    efficiently by multiplying from right to left in the last
    expression: Scale the rows of $B$ by $D_r$. Multiplying $P_rB$ means
    permuting the rows of $D_r B$. Multiplying $L^{-1}(P_r D_r B)$ means
    solving $nrhs$ triangular systems of equations with matrix $L$ by
    substitution. Similarly, multiplying $U^{-1}(L^{-1}(P_r D_r B))$
    means solving triangular systems with $U$.

In addition to complete factorization, we also have limited support for
incomplete factorization (ILU) preconditioner.

The simplest implementation, used by the "simple driver" routines in
SuperLU and SuperLU_MT, is as follows:

## Simple Driver Algorithm

1.  *Choose $P_c$ to order the columns of $A$* to increase the sparsity
    of the computed $L$ and $U$ factors, and hopefully increase
    parallelism (for SuperLU_MT).

2.  *Compute the LU factorization of $AP_c$.* SuperLU and SuperLU_MT can
    perform dynamic pivoting with row interchanges for numerical
    stability, computing $P_r$, $L$ and $U$ at the same time.

3.  *Solve the system* using $P_r$, $P_c$, $L$ and $U$ as described
    above. ($D_r = D_c = I$)

The simple driver subroutines for double precision real data are called
`dgssv` and `pdgssv` for SuperLU and SuperLU_MT, respectively. The
letter `d` in the subroutine names means double precision real; other
options are `s` for single precision real, `c` for single precision
complex, and `z` for double precision complex. The subroutine naming
scheme is analogous to the one used in LAPACK {cite}`lapackmanual2`.
SuperLU_DIST does not include this simple driver.

There is also an "expert driver" routine that can provide more accurate
solutions, compute error bounds, and solve a sequence of related linear
systems more economically. It is available in all three libraries.

**Expert Driver Algorithm**

1.  *Equilibrate* the matrix $A$, i.e. compute diagonal matrices $D_r$
    and $D_c$ so that $\hat{A} = D_r A D_c$ is "better conditioned" than
    $A$, i.e. $\hat{A}^{-1}$ is less sensitive to perturbations in
    $\hat{A}$ than $A^{-1}$ is to perturbations in $A$.

2.  *Preorder the rows of $\hat{A}$ (SuperLU_DIST only)*, i.e. replace
    $\hat{A}$ by $P_r \hat{A}$ where $P_r$ is a permutation matrix. We
    call this step "static pivoting", and it is only done in the
    distributed-mmemory algorithm.

3.  *Order the columns of $\hat{A}$* to increase the sparsity of the
    computed $L$ and $U$ factors, and hopefully increase parallelism
    (for SuperLU_MT and SuperLU_DIST). In other words, replace $\hat{A}$
    by $\hat{A} P_c^T$ in SuperLU and SuperLU_MT, or replace $\hat{A}$
    by $P_c\hat{A} P_c^T$ in SuperLU_DIST, where $P_c$ is a permutation
    matrix.

4.  *Compute the LU factorization of $\hat{A}$.* SuperLU and SuperLU_MT
    can perform dynamic pivoting with row interchanges for numerical
    stability. In contrast, SuperLU_DIST uses the order computed by the
    preordering step but replaces tiny pivots by larger values for
    stability.

5.  *Solve the system* using the computed triangular factors.

6.  *Iteratively refine the solution*, again using the computed
    triangular factors. This is equivalent to Newton's method.

7.  *Compute error bounds.* Both forward and backward error bounds are
    computed, as described below.

The expert driver subroutines for double precision real data are called
`dgssvx`, `pdgssvx` and `pdgssvx` for SuperLU, SuperLU_MT and
SuperLU_DIST, respectively.

The driver routines are composed of several lower level computational
routines for computing permutations, computing LU factorization, solving
triangular systems, and so on. For large matrices, the LU factorization
steps takes most of the time, although choosing $P_c$ to order the
columns can also be time-consuming.

(sec:Commonalities)=
# 1.3 What the three libraries have in common

## 1.3.1 Input and Output Data Formats

Sequential SuperLU and accept $A$ and $B$ as single precision real,
double precision real, and both single and double precision complex.
accepts double precision real or complex.

$A$ is stored in a sparse data structure according to the struct
`SuperMatrix`, which is described in
section [3.2](#sec:mt_datastructure). In particular, $A$ may be supplied in
either column-compressed format ("Harwell-Boeing format"), or
row-compressed format (i.e. $A^T$ stored in column-compressed format).
$B$, which is overwritten by the solution $X$, is stored as a dense
matrix in column-major order. In SuperLU_DIST, $A$ and $B$ can be either
replicated or distributed across all processes.

(The storage of $L$ and $U$ differs among the three libraries, as
discussed in section [4](#sec:Differences))

## 1.3.2 Tuning Parameters for BLAS

All three libraries depend on having high performance BLAS (Basic Linear
Algebra Subroutine) libraries {cite}`blas1`,{cite}`blas3`,{cite}`superlu_smp99` in order to get high performance. In particular, they depend on matrix-vector multiplication or matrix-matrix multiplication of relatively small dense matrices. The sizes of these small dense matrices can be tuned to match the "sweet spot" of the BLAS by setting certain tuning parameters described in
section [2.11.3](#sec:parameters) for SuperLU, in
section [3.5.2](#sec:SuperLU_MT_sp_ienv) for SuperLU_MT, and in
section [4.9.2](#sec:SuperLU_DIST_sp_ienv) for SuperLU_DIST.

(In addition, SuperLU_MT and SuperLU_DIST let one control the number of
parallel processes to be used, as described in
section [1.4](#sec:Differences)

## 1.3.3 Performance Statistics

Most of the computational routines use a struct to record certain kinds
of performance data, namely the time and number of floating point
operations in each phase of the computation, and data about the sizes of
the matrices $L$ and $U$. These statistics are collected during the
computation. A statistic variable is declared with the following type:

```c
        typedef struct {
            int     *panel_histo; /* histogram of panel size distribution */
            double  *utime;       /* time spent in various phases */
            float   *ops;         /* floating-point operations at various phases */
            int     TinyPivots;   /* number of tiny pivots */
            int     RefineSteps;  /* number of iterative refinement steps */
        } SuperLUStat_t;
```

For both SuperLU and SuperLU_MT, there is only one copy of these
statistics variable. But for SuperLU_DIST, each process keeps a local
copy of this variable, and records its local statistics. We need to use
MPI reduction routines to find any global information, such as the sum
of the floating-point operation count on all processes.

Before the computation, routine `StatInit()` should be called to malloc
storage and perform initialization for the fields `panel_histo`,
`utime`, and `ops`. The algorithmic phases are defined by the
enumeration type `PhaseType` in `SRC/util.h`. In the end, routine
`StatFree()` should be called to free storage of the above statistics
fields. After deallocation, the statistics are no longer accessible.
Therefore, users should extract the information they need before calling
`StatFree()`, which can be accomplished by calling `(P)StatPrint()`.

An inquiry function `dQuerySpace()` is provided to compute memory usage
statistics. This routine should be called after the $LU$ factorization.
It calculates the storage requirement based on the size of the $L$ and
$U$ data structures and working arrays.

(sec:SuperLU_ErrorHandling)=
## 1.3.4 Error Handling

### Invalid arguments and (P)XERBLA

Similar to LAPACK, for all the SuperLU routines, we check the validity
of the input arguments to each routine. If an illegal value is supplied
to one of the input arguments, the error handler XERBLA is called, and a
message is written to the standard output, indicating which argument has
an illegal value. The program returns immediately from the routine, with
a negative value of INFO.

### Computational failures with $\text{INFO} > 0$

A positive value of INFO on return from a routine indicates a failure in
the course of the computation, such as a matrix being singular, or the
amount of memory (in bytes) already allocated when malloc fails.

(sec:abort)=
### ABORT on unrecoverable errors

A macro `ABORT` is defined in `SRC/util.h` to handle unrecoverable
errors that occur in the middle of the computation, such as `malloc`
failure. The default action of `ABORT` is to call

`superlu_abort_and_exit(char *msg)`

which prints an error message, the line number and the file name at
which the error occurs, and calls the `exit` function to terminate the
program.

If this type of termination is not appropriate in some environment,
users can alter the behavior of the abort function. When compiling the
 library, users may choose the C preprocessor definition

`-DUSER_ABORT = my_abort`

At the same time, users would supply the following `my_abort` function

`my_abort(char *msg)`

which overrides the behavior of `superlu_abort_and_exit`.

## 1.3.5 Ordering the Columns of $A$ for Sparse Factors

There is a choice of orderings for the columns of $A$ both in the simple
or expert driver, in
section [1.2](#sec:OverallAlgorithm):

- Natural ordering,

- Multiple Minimum Degree (MMD) {cite}`lishao10` applied to the structure of
  $A^TA$,

- Multiple Minimum Degree (MMD) {cite}`lishao10` applied to the structure of
  $A^T+A$,

- Column Approximate Minimum Degree (COLAMD) {cite}`davis96`, and

- Use a $P_c$ supplied by the user as input.

COLAMD is designed particularly for unsymmetric matrices when partial
pivoting is needed, and does not require explicit formation of $A^TA$.
It usually gives comparable orderings as MMD on $A^TA$, and is faster.

The orderings based on graph partitioning heuristics are also popular,
as exemplified in the **MeTiS** package {cite}`higham96`. The user can simply input this ordering in the permutation vector for $P_c$. Note that many graph partitioning algorithms are designed for symmetric matrices. The user may still apply them to the structures of $A^TA$ or $A^T+A$. Our routines `getata()` and `at_plus_a()` in the file `get_perm_c.c` can be used to form $A^TA$ or $A^T+A$.

## 1.3.6 Iterative Refinement

Step 6 of the expert driver algorithm, iterative refinement, serves to
increase accuracy of the computed solution. Given the initial
approximate solution $x$ from step 5, the algorithm for step 6 is as
follows (where $x$ and $b$ are single columns of $X$ and $B$,
respectively):

&nbsp;&nbsp;&nbsp;&nbsp;Compute residual $r = Ax - b$  
&nbsp;&nbsp;&nbsp;&nbsp;While residual too large  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Solve $Ad = r$ for correction $d$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update solution $x = x - d$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update residual $r = Ax - b$  
&nbsp;&nbsp;&nbsp;&nbsp;End while

If $r$ and then $d$ were computed exactly, the updated solution $x-d$
would be the exact solution. Roundoff prevents immediate convergence.

The criterion "residual too large" in the iterative refinement algorithm above is essentially that

(eqn:eqn_defBERR)=

<p align="center">
\(
\displaystyle BERR \equiv \max_i \frac{|r_i|}{s_i}
\)
</p>

exceeds the machine roundoff level, or is continuing to decrease quickly enough. Here $s_i$ is the scale factor:

<p align="center">
\(
\displaystyle s_i = (|A|\cdot|x| + |b|)_i = \sum_j |A_{ij}|\cdot|x_j| + |b_i|
\)
</p>


In this expression, $|A|$ is the $n \times n$ matrix with entries $|A|_{ij} = |A_{ij}|$, and $|b|$ and $|x|$ are similarly column vectors of absolute entries of $b$ and $x$, respectively. The operation $|A|\cdot|x|$ represents conventional matrix-vector multiplication.
    The purpose of this stopping criterion is explained in the next section.

## 1.3.7 Error Bounds

Step 7 of the expert driver algorithm computes error bounds.

It is shown in {cite}`arioli89`,{cite}`oettliprager` that $BERR$ defined in Equation [1.1](#eqn:eqn_defBERR) measures the *componentwise relative backward error* of the computed solution. This means that the computed $x$ satisfies a slightly perturbed linear system of equations $(A+E)x=b+f$, where $|E_{ij}| \leq BERR \cdot |A_{ij}|$ and $|f_{i}| \leq BERR \cdot |b_{i}|$ for all $i$ and $j$. It is shown in {cite}`arioli89`,{cite}`skeel80` that one step of iterative refinement usually reduces $BERR$ to near machine epsilon. For example, if $BERR$ is 4 times machine epsilon, then the computed solution $x$ is identical to the solution one would get by changing each nonzero entry of $A$ and $b$ by at most 4 units in their last places, and then solving this perturbed system **exactly**. If the nonzero entries of $A$ and $b$ are uncertain in their bottom 2 bits, then one should generally not expect a more accurate solution. Thus $BERR$ is a measure of backward error specifically suited to solving sparse linear systems of equations. Despite roundoff, $BERR$ itself is always computed to within about $\pm n$ times machine epsilon (and usually much more accurately) and so $BERR$ is quite accurate.

In addition to backward error, the expert driver computes a *forward error bound*

<p align="center">
\(
\displaystyle FERR \geq \frac{\|x_{\mathrm{true}} - x\|_{\infty}}{\|x\|_{\infty}}
\)
</p>

Here, the infinity norm is defined as:

<p align="center">
\(
\displaystyle \|x\|_{\infty} \equiv \max_i |x_i|
\)
</p>

Thus, if $FERR = 10^{-6}$, then each component of $x$ has an error bounded by about $10^{-6}$ times the largest component of $x$. The algorithm used to compute $FERR$ is an approximation; see {cite}`arioli89`; {cite}`higham96` for a discussion. Generally, $FERR$ is accurate to within a factor of 10 or better, which is adequate to determine how many digits of the largest entries of $x$ are correct.

(SuperLU_DIST's algorithm for $FERR$ is slightly less reliable {cite}`lidemmel03`)

(sec:SolvingRelatedSystems)=
## 1.3.8 Solving a Sequence of Related Linear Systems

It is very common to solve a sequence of related linear systems
$A^{(1)} X^{(1)} = B^{(1)}$, $A^{(2)} X^{(2)} = B^{(2)}$, \... rather
than just one. When $A^{(1)}$ and $A^{(2)}$ are similar enough in
sparsity pattern and/or numerical entries, it is possible to save some
of the work done when solving with $A^{(1)}$ to solve with $A^{(2)}$.
This can result in significant savings. Here are the options, in
increasing order of "reuse of prior information":

1.  *Factor from scratch.* No previous information is used. If one were
    solving just one linear system, or a sequence of unrelated linear
    systems, this is the option to use.

2.  *Reuse $P_c$, the column permutation.* The user may save the column
    permutation and reuse it. This is most useful when $A^{(2)}$ has the
    same sparsity structure as $A^{(1)}$, but not necessarily the same
    (or similar) numerical entries. Reusing $P_c$ saves the sometimes
    quite expensive operation of computing it.

3.  *Reuse $P_c$, $P_r$ and data structures allocated for $L$ and $U$.*
    If $P_r$ and $P_c$ do not change, then the work of building the data
    structures associated with $L$ and $U$ (including the elimination
    tree {cite}`GilbertNg-IMA`) can be avoided. This is most useful when
    $A^{(2)}$ has the same sparsity structure and similar numerical
    entries as $A^{(1)}$. When the numerical entries are not similar,
    one can still use this option, but at a higher risk of numerical
    instability ($BERR$ will always report whether or not the solution
    was computed stably, so one cannot get an unstable answer without
    warning).

4.  *Reuse $P_c$, $P_r$, $L$ and $U$.* In other words, we reuse
    essentially everything. This is most commonly used when
    $A^{(2)} = A^{(1)}$, but $B^{(2)} \neq B^{(1)}$, i.e. when only the
    right-hand sides differ. It could also be used when $A^{(2)}$ and
    $A^{(1)}$ differed just slightly in numerical values, in the hopes
    that iterative refinement converges (using $A^{(2)}$ to compute
    residuals but the triangular factorization of $A^{(1)}$ to solve).

Because of the different ways $L$ and $U$ are computed and stored in the
three libraries, these 4 options are specified slightly differently; see
Chapters [2](#sec:ch2)
through [4](#sec:ch4) for details.

## 1.3.9 Interfacing to other languages

It is possible to call all the drivers and the computational routines
from Fortran. However, currently the Fortran wrapper functions are not
complete. The users are expected to look at the Fortran example programs
in the FORTRAN/ directory, together with the C "bridge" routine, and
learn how to call SuperLU from a Fortran program. The users can modify
the C bridge routine to fit their needs.

(sec:Differences)=
# 1.4 How the three libraries differ

## 1.4.1 Input and Output Data Formats

All Sequential SuperLU and routines are available in single and double
precision (real or complex), but routines are available only in double
precision (real or complex).

$L$ and $U$ are stored in different formats in the three libraries:

- *$L$ and $U$ in Sequential SuperLU.* $L$ is a "column-supernodal"
  matrix, in storage type `SCformat`. This means it is stored sparsely,
  with supernodes (consecutive columns with identical structures) stored
  as dense blocks. $U$ is stored in column-compressed format `NCformat`.
  See section [2.3](#sec:rep) for details.

- *$L$ and $U$ in SuperLU_MT.* Because of parallelism, the columns of
  $L$ and $U$ may not be computed in consecutive order, so they may be
  allocated and stored out of order. This means we use the
  "column-supernodal-permuted" format `SCPformat` for $L$ and
  "column-permuted" format `NCPformat` for $U$. See
  section [3.2](#sec:mt_datastructure) for details.

- *$L$ and $U$ in SuperLU_DIST.* Now $L$ and $U$ are distributed across
  multiple processors. As described in detail in
  Sections [4.3](#sec:datastruct)
  and [4.4](#sec:grid), we use a 2D block-cyclic format, which has been
  used for dense matrices in libraries like ScaLAPACK
  {cite}`scalapackmanual`. But for sparse matrices, the blocks are no longer
  identical in size, and vary depending on the sparsity structure of $L$
  and $U$. The detailed storage format is discussed in
  section [4.3](#sec:datastruct) and illustrated in
  Figure [4.1](#sec:grid).

## 1.4.2 Parallelism

Sequential SuperLU has no explicit parallelism. Some parallelism may
still be exploited on an SMP by using a multithreaded BLAS library if
available. But it is likely to be more effective to use SuperLU_MT on an
SMP, described next.

SuperLU_MT lets the user choose the number of parallel threads to use.
The mechanism varies from platform to platform and is described in
section [3.7](#sec:mt_port).

SuperLU_DIST not only lets the user specify the number of processors,
but how they are arranged into a 2D grid. Furthermore, MPI permits any
subset of the processors allocated to the user may be used for
SuperLU_DIST, not just consecutively numbered processors (say 0 through
P-1). See section [4.4](#sec:grid) for details.

## 1.4.3 Pivoting Strategies for Stability

Sequential SuperLU and SuperLU_MT use the same pivoting strategy, called
*threshold pivoting*, to determine the row permutation $P_r$. Suppose we
have factored the first $i-1$ columns of $A$, and are seeking the pivot
for column $i$. Let $a_{mi}$ be a largest entry in magnitude on or below
the diagonal of the partially factored $A$:
$|a_{mi}| = \max_{j \geq i} |a_{ji}|$. Depending on a threshold
$0 < u \leq 1$ input by the user, the code will use the diagonal entry
$a_{ii}$ as the pivot in column $i$ as long as
$|a_{ii}| \geq u \cdot |a_{mi}|$, and otherwise use $a_{mi}$. So if the
user sets $u=1$, $a_{mi}$ (or an equally large entry) will be selected
as the pivot; this corresponds to the classical *partial pivoting
strategy*. If the user has ordered the matrix so that choosing diagonal
pivots is particularly good for sparsity or parallelism, then smaller
values of $u$ will tend to choose those diagonal pivots, at the risk of
less numerical stability. Using $u=0$ guarantees that the pivots on the
diagonal will be chosen, unless they are zero. The error bound $BERR$
measure how much stability is actually lost.

Threshold pivoting turns out to be hard to parallelize on distributed
memory machines, because of the fine-grain communication and dynamic
data structures required. So SuperLU_DIST uses a new scheme called
*static pivoting* instead. In static pivoting the pivot order ($P_r$) is
chosen before numerical factorization, using a weighted perfect matching
algorithm {cite}`duffkoster99`, and kept fixed during factorization. Since
both row and column orders ($P_r$ and $P_c$) are fixed before numerical
factorization, we can extensively optimize the data layout, load
balance, and communication schedule. The price is a higher risk of
numeric instability, which is mitigated by diagonal scaling, setting
very tiny pivots to larger values, and iterative refinement
{cite}`lidemmel03`. Again, error bound $BERR$ measure how much stability is
actually lost.

## 1.4.4 Memory Management

Because of fill-in of entries during Gaussian elimination, $L$ and $U$
typically have many more nonzero entries than $A$. If $P_r$ and $P_c$
are not already known, we cannot determine the number and locations of
these nonzeros before performing the numerical factorization. This means
that some kind of dynamic memory allocation is needed.

Sequential SuperLU lets the user either supply a preallocated space
`work[]` of length `lwork`, or depend on malloc/free. The variable
`FILL` can be used to help the code predict the amount of fill, which
can reduce both fragmentation and the number of calls to malloc/free. If
the initial estimate of the size of $L$ and $U$ from `FILL` is too
small, the routine allocates more space and copies the current $L$ and
$U$ factors to the new space and frees the old space. If the routine
cannot allocate enough space, it calls a user-specifiable routine ABORT.
See sections [1.3.4](#sec:abort) for details.

SuperLU_MT is similar, except that the current alpha version cannot
reallocate more space for $L$ and $U$ if the initial size estimate from
`FILL` is too small. Instead, the program calls ABORT and the user must
start over with a larger value of `FILL`. See
section [3.5.2](#sec:mt_mem).

SuperLU_DIST actually has a simpler memory management chore, because
once $P_r$ and $P_c$ are determined, the structures of $L$ and $U$ can
be determined efficiently and just the right amount of memory allocated
using malloc and later free. So it will call ABORT only if there is
really not enough memory available to solve the problem.

## 1.4.5 Interfacing to other languages

Sequential SuperLU has a Matlab interface to the driver via a MEX file.
See
section [2.10](#sec:MatlabInterface) for details.

(sec:perf)=
# 1.5 Performance

SuperLU library incorporates a number of novel algorithmic ideas
developed recently. These algorithms also exploit the features of modern
computer architectures, in particular, the multi-level cache
organization and parallelism. We have conducted extensive experiments on
various platforms, with a large collection of test matrices. The
Sequential SuperLU achieved up to 40% of the theoretical floating-point
rate on a number of processors, see {cite}`superlu99`; {cite}`li96`. The megaflop
rate usually increases with increasing ratio of floating-point
operations count over the number of nonzeros in the $L$ and $U$ factors.
The parallel LU factorization in SuperLU_MT demonstrated 5--10 fold
speedups on a range of commercially popular SMPs, and up to 2.5
Gigaflops factorization rate, see {cite}`superlu_smp99`; {cite}`li96`. The parallel
LU factorization in SuperLU_DIST achieved up to 100 fold speedup on a
512-processor Cray T3E, and 10.2 Gigaflops factorization rate,
see {cite}`lidemmel98`.

(sec:SoftwareStatus)=
# 1.6 Software Status and Availability

All three libraries are freely available for all uses, commercial or
noncommercial, subject to the following caveats. No warranty is
expressed or implied by the authors, although we will gladly answer
questions and try to fix all reported bugs. We ask that proper credit be
given to the authors and that a notice be included if any modifications
are made.

The following Copyright applies to the whole SuperLU software.

> Copyright (c) 2003, The Regents of the University of California,
> through Lawrence Berkeley National Laboratory (subject to receipt of
> any required approvals from U.S. Dept. of Energy)
>
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
>
> \(1\) Redistributions of source code must retain the above copyright
> notice, this list of conditions and the following disclaimer. (2)
> Redistributions in binary form must reproduce the above copyright
> notice, this list of conditions and the following disclaimer in the
> documentation and/or other materials provided with the distribution.
> (3) Neither the name of Lawrence Berkeley National Laboratory, U.S.
> Dept. of Energy nor the names of its contributors may be used to
> endorse or promote products derived from this software without
> specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
> A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Some routines carry the additional notices as follows.

1.  Some subroutines carry the following notice:

    > Copyright (c) 1994 by Xerox Corporation. All rights reserved.
    >
    > THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
    > EXPRESSED OR IMPLIED. ANY USE IS AT YOUR OWN RISK.
    >
    > Permission is hereby granted to use or copy this program for any
    > purpose, provided the above notices are retained on all copies.
    > Permission to modify the code and to distribute modified code is
    > granted, provided the above notices are retained, and a notice
    > that the code was modified is included with the above copyright
    > notice.

2.  The MC64 routine (**only used in SuperLU_DIST**) carries the
    following notice:

    > COPYRIGHT (c) 1999 Council for the Central Laboratory of the
    > Research Councils. All rights reserved. PACKAGE MC64A/AD AUTHORS
    > Iain Duff (i.duff@rl.ac.uk) and Jacko Koster (jak@ii.uib.no) LAST
    > UPDATE 20/09/99
    >
    > \*\*\* Conditions on external use \*\*\*
    >
    > The user shall acknowledge the contribution of this package in any
    > publication of material dependent upon the use of the package. The
    > user shall use reasonable endeavours to notify the authors of the
    > package of this publication.
    >
    > The user can modify this code but, at no time shall the right or
    > title to all or any part of this package pass to the user. The
    > user shall make available free of charge to the authors for any
    > purpose all information relating to any alteration or addition
    > made to this package for the purposes of extending the
    > capabilities or enhancing the performance of this package.
    >
    > The user shall not pass this code directly to a third party
    > without the express prior consent of the authors. Users wanting to
    > licence their own copy of these routines should send email to
    > hsl@aeat.co.uk
    >
    > None of the comments from the Copyright notice up to and including
    > this one shall be removed or altered in any way.

All three libraries can be obtained from the following URLs:

            http://crd.lbl.gov/~xiaoye/SuperLU/
            http://www.netlib.org/scalapack/prototype/

In the future, we will add more functionality in the software, such as
sequential and parallel incomplete LU factorizations, as well as
parallel symbolic and ordering algorithms for SuperLU_DIST; these latter
routines would replace MC64 and have no restrictions on external use.

All bugs reports and queries can be e-mailed to `xsli@lbl.gov` and
`demmel@cs.berkeley.edu`.

# 1.7 Acknowledgement

With great gratitude, we acknowledge Stan Eisenstat and Joesph Liu for
their significant contributions to the development of Sequential
SuperLU. Meiyue Shao helped the development of the incomplete
factorization ILU routines in sequential SuperLU.

We would like to thank Jinqchong Teo for helping generate the code in
Sequential SuperLU to work with four floating-point data types, and
Daniel Schreiber for doing this with SuperLU_MT.

Yu Wang and William F. Mitchell developed the Fortran 90 interface for
SuperLU_DIST. Laura Grigori developed the parallel symbolic
factorization code for SuperLU_DIST.

We thank Tim Davis for his contribution of some subroutines related to
column ordering and suggestions on improving the routines' interfaces.
We thank Ed Rothberg of Silicon Graphics for discussions and providing
us access to the SGI Power Challenge during the SuperLU_MT development.

We acknowledge the following organizations that provided the computer
resources during our code development: NERSC at Lawrence Berkeley
National Laboratory, Livermore Computing at Lawrence Livermore National
Laboratory, NCSA at University of Illinois at Urbana-Champaign, Silicon
Graphics, and Xerox Palo Alto Research Center. We thank UC Berkeley and
NSF Infrastructure grant CDA-9401156 for providing Berkeley NOW.

