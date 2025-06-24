# SuperLU_DIST

SuperLU_DIST contains a set of subroutines to solve a sparse linear system A*X=B. It uses Gaussian elimination with static pivoting (GESP). Static pivoting is a technique that combines the numerical stability of partial pivoting with the scalability of Cholesky (no pivoting), to run accurately and efficiently on large numbers of processors. SuperLU_DIST is a parallel extension to the serial SuperLU library. It is targeted for the distributed memory parallel machines. SuperLU_DIST is implemented in ANSI C, with OpenMP for on-node parallelism and MPI for off-node communications. Numerical LU factorization and triangular solvers can be performed on multiple GPU nodes for Nvidia, AMD, and Intel GPUs.


## Main Topics

```{toctree}
:maxdepth: 1

manual/index
```

<a href="./ug.pdf">SuperLU_DIST Users Manual in PDF</a>


## Toolkits/libraries that use SuperLU_DIST

- [NIMROD](https://nimrodteam.org/) NIMROD solves the fully 3D, extended-magnetohydrodynamic equations. NIMROD has simulated aspects of a wide range of fusion experiments (tokamaks, stellarators, spheromaks, reversed-field pinches, z-pinches, field-reversed configurations, etc.), laboratory plasma-physics experiments and astrophysical plasmas.

## Citing SuperLU_DIST

For general citations on SuperLU_DIST please use the following:

```{literalinclude} /superlu.bib
:append: '}'
:end-at: year
:language: none
:start-at: superlu_v8
```
