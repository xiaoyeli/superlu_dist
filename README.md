# SuperLU_DIST (version 5.3)

[![Build Status](https://travis-ci.org/xiaoyeli/superlu_dist.svg?branch=master)](https://travis-ci.org/xiaoyeli/superlu_dist) 
[Nightly tests](http://my.cdash.org/index.php?project=superlu_dist)

SuperLU_DIST contains a set of subroutines to solve a sparse linear system 
A*X=B. It uses Gaussian elimination with static pivoting (GESP). 
Static pivoting is a technique that combines the numerical stability of
partial pivoting with the scalability of Cholesky (no pivoting),
to run accurately and efficiently on large numbers of processors. 

SuperLU_DIST is a parallel extension to the serial SuperLU library.
It is targeted for the distributed memory parallel machines.
SuperLU_DIST is implemented in ANSI C, and MPI for communications.
Currently, the LU factorization and triangular solution routines,
which are the most time-consuming part of the solution process,
are parallelized. The other routines, such as static pivoting and 
column preordering for sparsity are performed sequentially. 
This "alpha" release contains double-precision real and double-precision
complex data types.

### The distribution contains the following directory structure:

```
SuperLU_DIST/README    instructions on installation
SuperLU_DIST/CBLAS/    needed BLAS routines in C, not necessarily fast
	 	       (NOTE: this version is single threaded. If you use the
		       library with multiple OpenMP threads, performance
		       relies on a good multithreaded BLAS implementation.)
SuperLU_DIST/DOC/      the Users' Guide
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

## INSTALLATION

There are two ways to install the package. One requires users to 
edit makefile manually, the other uses CMake build system.
The procedures are described below.

### Installation option 1: Manual installation with makefile.
Before installing the package, please examine the three things dependent 
on your system setup:

1.1 Edit the make.inc include file.

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

1.2. The BLAS library.
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

1.3. External libraries: Metis and ParMetis.

If you will use Metis or ParMetis ordering, you will
need to install them yourself. Since ParMetis package already
contains the source code for the Metis library, you can just
download and compile ParMetis from:
[http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download)

After you have installed it, you should define the following in make.inc:
```
METISLIB = -L<metis directory> -lmetis
PARMETISLIB = -L<parmetis directory> -lparmetis
I_PARMETIS = -I<parmetis directory>/include -I<parmetis directory>/metis/include
```
You can disable ParMetis with the following line in SRC/superlu_dist_config.h:
```
#undef HAVE_PARMETIS
```
1.4. C preprocessor definition CDEFS.
In the header file SRC/Cnames.h, we use macros to determine how
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
1.5. Multicore and GPU (optional).

To use OpenMP parallelism, need to link with an OpenMP library, and
set the number of threads you wish to use as follows (bash):

`export OMP_NUM_THREADS=<##>`

To enable NVIDIA GPU access, need to take the following 2 step:
1) Set the following Linux environment variable:
`export ACC=GPU`

2) Add the CUDA library location in make.inc:
```
ifeq "${ACC}" "GPU"
CFLAGS += -DGPU_ACC
INCS += -I<CUDA directory>/include
LIBS += -L<CUDA directory>/lib64 -lcublas -lcudart 
endif
```
A Makefile is provided in each subdirectory. The installation can be done
completely automatically by simply typing "make" at the top level.

### Installation option 2: Using CMake build system.
You will need to create a build tree from which to invoke CMake.

First, in order to use parallel symbolic factorization function, you
need to install ParMETIS parallel ordering package and define the
two environment variables: PARMETIS_ROOT and PARMETIS_BUILD_DIR

```
export PARMETIS_ROOT=<Prefix directory of the ParMETIS installation>
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
```
Then, the installation procedure is the following.

From the top level directory, do:
```
mkdir build ; cd build
cmake .. \
-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include"

( Example cmake script: see run_cmake_build.sh

export PARMETIS_ROOT=~/lib/dynamic/parmetis-4.0.3 
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64 
cmake .. \
-DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
-DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
-DCMAKE_C_FLAGS="-std=c99 -g" \
-Denable_blaslib=OFF \
-DBUILD_SHARED_LIBS=OFF \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_INSTALL_PREFIX=.

)
```
To actually build, type:
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
execution may fail. You can always go to TEST/ directory to perform
testing manually.

**Note on the C-Fortran name mangling handled by C preprocessor definition:**  
In the default setting, we assume that Fortran expects a C routine
to have an underscore postfixed to the name. Depending on the
compiler, you may need to define one of the following flags in
during the cmake build to overwrite default setting:
```
cmake .. -DCMAKE_C_FLAGS="-DNoChange" 
cmake .. -DCMAKE_C_FLAGS="-DUpCase"
```
## Windows Usage
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
  -Denable_parmetislib:BOOL=FALSE \
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

## READING SPARSE MATRIX FILES

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

## REFERENCES

**[1]** SuperLU_DIST: A Scalable Distributed-Memory Sparse Direct Solver for Unsymmetric Linear Systems. Xiaoye S. Li and James W. Demmel. ACM Trans. on Math. Software, Vol. 29, No. 2, June 2003, pp. 110-140.  
**[2]** Parallel Symbolic Factorization for Sparse LU with Static Pivoting. L. Grigori, J. Demmel and X.S. Li. SIAM J. Sci. Comp., Vol. 29, Issue 3, 1289-1314, 2007.  
**[3]** A distributed CPU-GPU sparse direct solver. P. Sao, R. Vuduc and X.S. Li, Proc. of EuroPar-2014 Parallel Processing, August 25-29, 2014. Porto, Portugal.  

**Xiaoye S. Li**, Lawrence Berkeley National Lab, [xsli@lbl.gov](xsli@lbl.gov)  
**Laura Grigori**, INRIA, France, [laura.grigori@inria.fr](laura.grigori@inria.fr)  
**Piyush Sao**, Georgia Institute of Technology, [piyush.feynman@gmail.com](piyush.feynman@gmail.com)  
**Ichitaro Yamazaki**, Univ. of Tennessee, [ic.yamazaki@gmail.com](ic.yamazaki@gmail.com)  

## RELEASE VERSIONS
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
```
