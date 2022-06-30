# SuperLU_DIST (version 8.0.0)   <img align=center width="55" alt="superlu" src="https://user-images.githubusercontent.com/11741943/103982988-5a9a9d00-5139-11eb-9ac4-a55e80a79f8d.png">

[![Build Status](https://travis-ci.org/xiaoyeli/superlu_dist.svg?branch=master)](https://travis-ci.org/xiaoyeli/superlu_dist) 
[Nightly tests](http://my.cdash.org/index.php?project=superlu_dist)

SuperLU_DIST contains a set of subroutines to solve a sparse linear system 
A*X=B. It uses Gaussian elimination with static pivoting (GESP). 
Static pivoting is a technique that combines the numerical stability of
partial pivoting with the scalability of Cholesky (no pivoting),
to run accurately and efficiently on large numbers of processors. 

SuperLU_DIST is a parallel extension to the serial SuperLU library.
It is targeted for the distributed memory parallel machines.
SuperLU_DIST is implemented in ANSI C, with OpenMP for on-node parallelism
and MPI for off-node communications. We are actively developing GPU
acceleration capabilities.
<!-- Currently, the LU factorization and triangular solution routines, -->
<!-- which are the most time-consuming part of the solution process,-->
<!-- are parallelized. The other routines, such as static pivoting and -->
<!-- column preordering for sparsity are performed sequentially. -->
<!-- This "alpha" release contains double-precision real and-->
<!-- double-precision complex data types.-->

Table of Contents
=================

* [SuperLU_DIST (version 8.0.0)   <a href="https://user-images.githubusercontent.com/11741943/103982988-5a9a9d00-5139-11eb-9ac4-a55e80a79f8d.png" target="_blank" rel="nofollow"><img align="center" width="55" alt="superlu" src="https://user-images.githubusercontent.com/11741943/103982988-5a9a9d00-5139-11eb-9ac4-a55e80a79f8d.png" style="max-width:100%;"></a>](#superlu_dist-version-80---)
* [Directory structure of the source code](#directory-structure-of-the-source-code)
* [Installation](#installation)
   * [Installation option 1: Using CMake build system.](#installation-option-1-using-cmake-build-system)
      * [Dependent external libraries: BLAS and ParMETIS](#dependent-external-libraries-blas-and-parmetis)
      * [Optional external libraries: CombBLAS, LAPACK](#optional-external-libraries-combblas-lapack)
      * [Use GPU](#use-gpu)
      * [Summary of the CMake definitions.](#summary-of-the-cmake-definitions)
   * [Installation option 2: Manual installation with makefile.](#installation-option-2-manual-installation-with-makefile)
      * [2.1 Edit the make.inc include file.](#21-edit-the-makeinc-include-file)
      * [2.2. The BLAS library.](#22-the-blas-library)
      * [2.3. External libraries.](#23-external-libraries)
         * [2.3.1 Metis and ParMetis.](#231-metis-and-parmetis)
         * [2.3.2 LAPACK.](#232-lapack)
         * [2.3.3 CombBLAS.](#233-combblas)
      * [2.4. C preprocessor definition CDEFS. (Replaced by cmake module FortranCInterface.)](#24-c-preprocessor-definition-cdefs-replaced-by-cmake-module-fortrancinterface)
      * [2.5. Multicore and GPU.](#25-multicore-and-gpu)
* [Summary of the environment variables.](#summary-of-the-environment-variables)
* [Windows Usage](#windows-usage)
* [Reading sparse matrix files](#reading-sparse-matrix-files)
* [REFERENCES](#references)
* [RELEASE VERSIONS](#release-versions)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)

# SuperLU_DIST (version 8.0)   <img align=center width="55" alt="superlu" src="https://user-images.githubusercontent.com/11741943/103982988-5a9a9d00-5139-11eb-9ac4-a55e80a79f8d.png">

[![Build Status](https://travis-ci.org/xiaoyeli/superlu_dist.svg?branch=master)](https://travis-ci.org/xiaoyeli/superlu_dist) 
[Nightly tests](http://my.cdash.org/index.php?project=superlu_dist)

SuperLU_DIST contains a set of subroutines to solve a sparse linear system 
A*X=B. It uses Gaussian elimination with static pivoting (GESP). 
Static pivoting is a technique that combines the numerical stability of
partial pivoting with the scalability of Cholesky (no pivoting),
to run accurately and efficiently on large numbers of processors. 

SuperLU_DIST is a parallel extension to the serial SuperLU library.
It is targeted for the distributed memory parallel machines.
SuperLU_DIST is implemented in ANSI C, with OpenMP for on-node parallelism
and MPI for off-node communications. We are actively developing GPU
acceleration capabilities.
<!-- Currently, the LU factorization and triangular solution routines, -->
<!-- which are the most time-consuming part of the solution process,-->
<!-- are parallelized. The other routines, such as static pivoting and -->
<!-- column preordering for sparsity are performed sequentially. -->
<!-- This "alpha" release contains double-precision real and-->
<!-- double-precision complex data types.-->


# Directory structure of the source code

```
SuperLU_DIST/README    instructions on installation
SuperLU_DIST/CBLAS/    needed BLAS routines in C, not necessarily fast
	 	       (NOTE: this version is single threaded. If you use the
		       library with multiple OpenMP threads, performance
		       relies on a good multithreaded BLAS implementation.)
SuperLU_DIST/DOC/      the Users' Guide
SuperLU_DIST/FORTRAN/  Fortran90 wrapper functions
SuperLU_DIST/EXAMPLE/  example programs
SuperLU_DIST/INSTALL/  test machine dependent parameters
SuperLU_DIST/SRC/      C source code, to be compiled into libsuperlu_dist.a
SuperLU_DIST/TEST/     testing code
SuperLU_DIST/lib/      contains library archive libsuperlu_dist.a
SuperLU_DIST/Makefile  top-level Makefile that does installation and testing
SuperLU_DIST/make.inc  compiler, compiler flags, library definitions and C
	               preprocessor definitions, included in all Makefiles.
	               (You may need to edit it to suit your system
	               before compiling the whole package.)
SuperLU_DIST/MAKE_INC/ sample machine-specific make.inc files
```

# Installation

There are two ways to install the package. The first method is to use
CMake automatic build system. The other method requires users to 
The procedures are described below.

## Installation option 1: Using CMake build system.
You will need to create a build tree from which to invoke CMake.

### Dependent external libraries: BLAS and ParMETIS
If you have a BLAS library on your machine, you can link with it
with the following cmake definition:
```
-DTPL_BLAS_LIBRARIES="<BLAS library name>"
```
Otherwise, the CBLAS/ subdirectory contains the part of the C BLAS
(single threaded) needed by SuperLU_DIST, but they are not optimized.
You can compile and use it with the following cmake definition:
```
-DTPL_ENABLE_INTERNAL_BLASLIB=ON
```

The default sparsity ordering is METIS. But, in order to use parallel
symbolic factorization function, you
need to install ParMETIS parallel ordering package and define the
two environment variables: PARMETIS_ROOT and PARMETIS_BUILD_DIR

(Note: ParMETIS library also contains serial METIS library.)

```
export PARMETIS_ROOT=<Prefix directory of the ParMETIS installation>
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
```

### Optional external libraries: CombBLAS, LAPACK

In order to use parallel weighted matching HWPM (Heavy Weight
Perfect Matching) for numerical pre-pivoting, you need to install 
CombBLAS and define the environment variable:

```
export COMBBLAS_ROOT=<Prefix directory of the CombBLAS installation>
export COMBBLAS_BUILD_DIR=${COMBBLAS_ROOT}/_build
```
Then, install with cmake option:
```
-DTPL_ENABLE_COMBBLASLIB=ON
```

By default, LAPACK is not needed. Only in triangular solve routine, we
may use LAPACK to explicitly invert the dense diagonal block to improve
speed. You can use it with the following cmake option:
```
-DTPL_ENABLE_LAPACKLIB=ON
```

### Use GPU
You can enable (NVIDIA) GPU with CUDA with the following cmake option:
```
-DTPL_ENABLE_CUDALIB=TRUE
```
You can enable (AMD) GPU with HIP with the following cmake option:
```
-DTPL_ENABLE_HIPLIB=TRUE
```

Once these needed third-party libraries are in place, the installation
can be done as follows at the top level directory:

For a simple installation with default setting, do:
(ParMETIS is needed, i.e., TPL_ENABLE_PARMETISLIB=ON)
```
mkdir build ; cd build;
cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
```
For a more sophisticated installation including third-party libraries, do:
```
cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DTPL_ENABLE_COMBBLASLIB=ON \
    -DTPL_COMBBLAS_INCLUDE_DIRS="${COMBBLAS_ROOT}/_install/include;${COMBBLAS_R\
OOT}/Applications/BipartiteMatchings" \
    -DTPL_COMBBLAS_LIBRARIES="${COMBBLAS_BUILD_DIR}/libCombBLAS.a" \
    -DCMAKE_C_FLAGS="-std=c99 -g -DPRNTlevel=0 -DDEBUGlevel=0" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-std=c++11" \
    -DTPL_ENABLE_BLASLIB=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=.

( see example cmake script: run_cmake_build.sh )
```

You can disable CombBLAS or LAPACK with the following cmake options:
```
-DTPL_ENABLE_LAPACKLIB=FALSE
-DTPL_ENABLE_COMBBLASLIB=FALSE
```

To actually build (compile), type:
`make`

To install the libraries, type:
`make install`

To run the installation test, type:
`ctest`
(The outputs are in file: `build/Testing/Temporary/LastTest.log`)
or,
`ctest -D Experimental`
or,
`ctest -D Nightly`

**NOTE:**
The parallel execution in ctest is invoked by "mpiexec" command which is
from MPICH environment. If your MPI is not MPICH/mpiexec based, the test
execution may fail. You can pass the definition option "-DMPIEXEC_EXECUTABLE"
to cmake. For example on Cori at NERSC, you will need the following:
`-DMPIEXEC_EXECUTABLE=/usr/bin/srun`

Or, you can always go to TEST/ directory to perform testing manually.

### Summary of the CMake definitions.
The following list summarize the commonly used CMake definitions. In each case,
the first choice is the default setting. After running 'cmake' installation,
a configuration header file is generated in SRC/superlu_dist_config.h, which
contains the key CPP definitions used throughout the code.
```
    -TPL_ENABLE_PARMETISLIB=ON | OFF
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF | ON
    -DTPL_ENABLE_LAPACKLIB=OFF | ON
    -TPL_ENABLE_COMBBLASLIB=OFF | ON
    -DTPL_ENABLE_CUDALIB=OFF | ON
    -DTPL_ENABLE_HIPLIB=OFF | ON
    -Denable_complex16=OFF | ON
    -DXSDK_INDEX_SIZE=32 | 64

    -DBUILD_SHARED_LIBS= OFF | ON
    -DCMAKE_INSTALL_PREFIX=<...>.
    -DCMAKE_C_COMPILER=<MPI C compiler>
    -DCMAKE_C_FLAGS="..." 
    -DCMAKE_CXX_COMPILER=<MPI C++ compiler>
    -DMAKE_CXX_FLAGS="..."
    -DCMAKE_CUDA_FLAGS="..." 
    -DHIP_HIPCC_FLAGS="..." 
    -DXSDK_ENABLE_Fortran=OFF | ON
    -DCMAKE_Fortran_COMPILER=<MPI F90 compiler>
```

## Installation option 2: Manual installation with makefile.
Before installing the package, please examine the three things dependent 
on your system setup:

### 2.1 Edit the make.inc include file.

This make include file is referenced inside each of the Makefiles
in the various subdirectories. As a result, there is no need to 
edit the Makefiles in the subdirectories. All information that is
machine specific has been defined in this include file. 

Sample machine-specific make.inc are provided in the MAKE_INC/
directory for several platforms, such as Cray XT5, Linux, Mac-OS, and CUDA.
When you have selected the machine to which you wish to install
SuperLU_DIST, copy the appropriate sample include file 
(if one is present) into make.inc.

For example, if you wish to run SuperLU_DIST on a Cray XT5,  you can do
`cp MAKE_INC/make.xt5  make.inc`

For the systems other than listed above, some porting effort is needed
for parallel factorization routines. Please refer to the Users' Guide 
for detailed instructions on porting.

The following CPP definitions can be set in CFLAGS.
```
-DXSDK_INDEX_SIZE=64
use 64-bit integers for indexing sparse matrices. (default 32 bit)

-DPRNTlevel=[0,1,2,...]
printing level to show solver's execution details. (default 0)

-DDEBUGlevel=[0,1,2,...]
diagnostic printing level for debugging purpose. (default 0)
```      

### 2.2. The BLAS library.

The parallel routines in SuperLU_DIST use some BLAS routines on each MPI
process. Moreover, if you enable OpenMP with multiple threads, you need to
link with a multithreaded BLAS library. Otherwise performance will be poor.
A good public domain BLAS library is OpenBLAS (http://www.openblas.net),
which has OpenMP support.

If you have a BLAS library your machine, you may define the following in
the file make.inc:
```
BLASDEF = -DUSE_VENDOR_BLAS
BLASLIB = <BLAS library you wish to link with>
```
The CBLAS/ subdirectory contains the part of the C BLAS (single threaded) 
needed by SuperLU_DIST package. However, these codes are intended for use
only if there is no faster implementation of the BLAS already
available on your machine. In this case, you should go to the
top-level SuperLU_DIST/ directory and do the following:

1) In make.inc, undefine (comment out) BLASDEF, and define:
` BLASLIB = ../lib/libblas$(PLAT).a`

2) Type: `make blaslib`
to make the BLAS library from the routines in the
` CBLAS/ subdirectory.`

### 2.3. External libraries. 

  #### 2.3.1 Metis and ParMetis.

If you will use Metis or ParMetis for sparsity ordering, you will
need to install them yourself. Since ParMetis package already
contains the source code for the Metis library, you can just
download and compile ParMetis from:
[http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download)

After you have installed it, you should define the following in make.inc:
```
HAVE_PARMETIS = TRUE
METISLIB = -L<metis directory> -lmetis
PARMETISLIB = -L<parmetis directory> -lparmetis
I_PARMETIS = -I<parmetis directory>/include -I<parmetis directory>/metis/include
```
You can disable ParMetis with the following line in SRC/superlu_dist_config.h:
```
#undef HAVE_PARMETIS
```
  #### 2.3.2 LAPACK.
  Starting Version 6.0, the triangular solve routine can perform explicit
  inversion on the diagonal blocks, using LAPACK's xTRTRI inversion routine.
  To use this feature, you should define the following in make.inc:
```
SLU_HAVE_LAPACK = TRUE
LAPACKLIB = <lapack library you wish to link with>
```
You can disable LAPACK with the following line in SRC/superlu_dist_config.h:
```
#undef SLU_HAVE_LAPACK
```

 #### 2.3.3 CombBLAS.

You can use parallel approximate weight perfect matching (AWPM) algorithm
to perform numerical pre-pivoting for stability. The default pre-pivoting
is to use MC64 provided internally, which is an exact algorithm, but serial.
In order to use AWPM, you will need to install CombBLAS yourself, at the
download site:
[https://people.eecs.berkeley.edu/~aydin/CombBLAS/html/index.html](https://people.eecs.berkeley.edu/~aydin/CombBLAS/html/index.html)

After you have installed it, you should define the following in make.inc:
```
HAVE_COMBBLAS = TRUE
COMBBLASLIB = <combblas root>/_build/libCombBLAS.a
I_COMBBLAS=-I<combblas root>/_install/include -I<combblas root>/Applications/BipartiteMatchings
```
You can disable CombBLAS with the following line in SRC/superlu_dist_config.h:
```
#undef HAVE_COMBBLAS
```

### 2.4. C preprocessor definition CDEFS. (Replaced by cmake module FortranCInterface.)

In the header file SRC/superlu_Cnames.h, we use macros to determine how
C routines should be named so that they are callable by Fortran.
(Some vendor-supplied BLAS libraries do not have C interfaces. So the 
re-naming is needed in order for the SuperLU BLAS calls (in C) to 
interface with the Fortran-style BLAS.)
The possible options for CDEFS are:
```
-DAdd_: Fortran expects a C routine to have an underscore
  postfixed to the name;
  (This is set as the default)
-DNoChange: Fortran expects a C routine name to be identical to
      that compiled by C;
-DUpCase: Fortran expects a C routine name to be all uppercase.
```

### 2.5. Multicore and GPU.

To use OpenMP parallelism, need to link with an OpenMP library, and
set the number of threads you wish to use as follows (bash):

`export OMP_NUM_THREADS=<##>`

To enable NVIDIA GPU access, need to take the following step:
Add the CUDA library location in make.inc:
```
HAVE_CUDA=TRUE
INCS += -I<CUDA directory>/include
LIBS += -L<CUDA directory>/lib64 -lcublas -lcudart 
endif
```
A Makefile is provided in each subdirectory. The installation can be done
completely automatically by simply typing "make" at the top level.


# Summary of the environment variables.
A couple of environment variables affect parallel execution.
```
    export OMP_NUM_THREADS=<...>
    export SUPERLU_ACC_OFFLOAD=1  // this enables use of GPU. Default is 1.
```
Several integer blocking parameters may affect performance. Most of them can be
set by the user through environment variables. Oherwise the default values
are provided. Various SuperLU routines call an environment inquiry function
to obtain these parameters. This function is provided in the file SRC/sp_ienv.c.
Please consult that file for detailed description of the meanings.
```
    export NREL=<...>   // supernode relaxation parameter
    export NSUP=<...>   // maximum allowable supernode size, not to exceed 512
    export FILL=<...>   // estimated fill ratio of nonzeros(L+U)/nonzeros(A)
    export MAX_BUFFER_SIZE=<...>   // maximum buffer size on GPU for GEMM
```

# Windows Usage
Prerequisites: CMake, Visual Studio, Microsoft HPC Pack
This has been tested with Visual Studio 2017, without Parmetis,
without Fortran, and with OpenMP disabled. 

The cmake configuration line used was
```
'/winsame/contrib-vs2017/cmake-3.9.4-ser/bin/cmake' \
  -DCMAKE_INSTALL_PREFIX:PATH=C:/winsame/volatile-vs2017/superlu_dist-master.r147-parcomm \
  -DCMAKE_BUILD_TYPE:STRING=Release \
  -DCMAKE_COLOR_MAKEFILE:BOOL=FALSE \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE \
  -Denable_openmp:BOOL=FALSE \
  -DCMAKE_C_COMPILER:FILEPATH='C:/Program Files (x86)/Microsoft Visual Studio/2017/Professional/VC/Tools/MSVC/14.11.25503/bin/HostX64/x64/cl.exe' \
  -DCMAKE_C_FLAGS:STRING='/DWIN32 /D_WINDOWS /W3' \
  -DTPL_ENABLE_PARMETISLIB:BOOL=FALSE \
  -DXSDK_ENABLE_Fortran=OFF \
  -G 'NMake Makefiles JOM' \
  C:/path/to/superlu_dist
```

After configuring, simply do
```
  jom # or nmake
  jom install  # or nmake install
```

Libraries will be installed under
C:/winsame/volatile-vs2017/superlu_dist-master.r147-parcomm/lib
for the above configuration.

If you wish to test:
  `ctest`

# Reading sparse matrix files

The SRC/ directory contains the following routines to read different file 
formats, they all have the similar calling sequence.
```
$ ls -l dread*.c
dreadMM.c              : Matrix Market, files with suffix .mtx
dreadhb.c              : Harrell-Boeing, files with suffix .rua
dreadrb.c              : Rutherford-Boeing, files with suffix .rb
dreadtriple.c          : triplet, with header
dreadtriple_noheader.c : triplet, no header, which is also readable in Matlab
```

# REFERENCES

**[1]** X.S. Li and J.W. Demmel, "SuperLU_DIST: A Scalable Distributed-Memory
 Sparse Direct Solver for Unsymmetric Linear Systems", ACM Trans. on Math.
 Software, Vol. 29, No. 2, June 2003, pp. 110-140.  
**[2]** L. Grigori, J. Demmel and X.S. Li, "Parallel Symbolic Factorization
 for Sparse LU with Static Pivoting", SIAM J. Sci. Comp., Vol. 29, Issue 3,
 1289-1314, 2007.  
**[3]** P. Sao, R. Vuduc and X.S. Li, "A distributed CPU-GPU sparse direct
 solver", Proc. of EuroPar-2014 Parallel Processing, August 25-29, 2014.
 Porto, Portugal.  
**[4]** P. Sao, X.S. Li, R. Vuduc, “A Communication-Avoiding 3D Factorization
 for Sparse Matrices”, Proc. of IPDPS, May 21–25, 2018, Vancouver.   
**[5]** P. Sao, R. Vuduc, X. Li, "Communication-avoiding 3D algorithm for
 sparse LU factorization on heterogeneous systems", J. Parallel and
 Distributed Computing (JPDC), September 2019.     
**[6]** Y. Liu, M. Jacquelin, P. Ghysels and X.S. Li, “Highly scalable
 distributed-memory sparse triangular solution algorithms”, Proc. of
 SIAM workshop on Combinatorial Scientific Computing, June 6-8, 2018,
 Bergen, Norway.   
**[7]** N. Ding, S. Williams, Y. Liu, X.S. Li, "Leveraging One-Sided
 Communication for Sparse Triangular Solvers", Proc. of SIAM Conf. on
 Parallel Processing for Scientific Computing. Feb. 12-15, 2020.   
**[8]** A. Azad, A. Buluc, X.S. Li, X. Wang, and J. Langguth,
"A distributed-memory algorithm for computing a heavy-weight perfect matching 
on bipartite graphs", SIAM J. Sci. Comput., Vol. 42, No. 4, pp. C143-C168, 2020.   


**Xiaoye S. Li**, Lawrence Berkeley National Lab, [xsli@lbl.gov](xsli@lbl.gov)   
**Gustavo Chavez**, Lawrence Berkeley National Lab, [gichavez@lbl.gov](gichavez@lbl.gov)   
**Nan Ding**, Lawrence Berkeley National Lab, [nanding@lbl.gov](nanding@lbl.gov)  
**Laura Grigori**, INRIA, France, [laura.grigori@inria.fr](laura.grigori@inria.fr)  
**Yang Liu**, Lawrence Berkeley National Lab, [liuyangzhuan@lbl.gov](liuyangzhuan@lbl.gov)   
**Piyush Sao**, Georgia Institute of Technology, [piyush.feynman@gmail.com](piyush.feynman@gmail.com)  
**Meiyue Shao**, Lawrence Berkeley National Lab, [myshao@lbl.gov](myshao@lbl.gov)   
**Ichitaro Yamazaki**, Univ. of Tennessee, [ic.yamazaki@gmail.com](ic.yamazaki@gmail.com)  
**Jim Demmel**, UC Berkeley, [demmel@cs.berkeley.edu](demmel@cs.berkeley.edu)   
**John Gilbert**, UC Santa Barbara, [gilbert@cs.ucsb.edu](gilbert@cs.ucsb.edu)


# RELEASE VERSIONS
```
October 15, 2003    Version 2.0  
October 1,  2007    Version 2.1  
Feburary 20, 2008   Version 2.2  
October 15, 2008    Version 2.3  
June 9, 2010        Version 2.4  
November 23, 2010   Version 2.5  
March 31, 2013      Version 3.3  
October 1, 2014     Version 4.0  
July 15, 2014       Version 4.1  
September 25, 2015  Version 4.2  
December 31, 2015   Version 4.3  
April 8, 2016       Version 5.0.0  
May 15, 2016        Version 5.1.0  
October 4, 2016     Version 5.1.1  
December 31, 2016   Version 5.1.3  
September 30, 2017  Version 5.2.0  
January 28, 2018    Version 5.3.0
June 1, 2018        Version 5.4.0
September 22, 2018  Version 6.0.0
December 9, 2018    Version 6.1.0
February 8, 2019    Version 6.1.1
November 12, 2019   Version 6.2.0
February 23, 2020   Version 6.3.0
October 23, 2020    Version 6.4.0
May 10, 2021        Version 7.0.0
October 5, 2021     Version 7.1.0
October 18, 2021    Version 7.1.1
December 12, 2021   Version 7.2.0
May 22, 2022        Version 8.0.0
```
