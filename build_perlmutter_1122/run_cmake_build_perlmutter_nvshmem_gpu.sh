#!/bin/bash
module load PrgEnv-gnu
module load gcc/11.2.0
module load cmake/3.22.0
module load craype-accel-nvidia80
module load cudatoolkit/11.7
module load cray-libsci/22.06.1.3

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.5\/compat:/}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-11.7\/compat:/}
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
##export FI_LOG_LEVEL Warn

export NVSHMEM_HOME=/global/cfs/cdirs/m2956/nanding/software/nvshmem_src_2.7.0-6/build
export NVSHMEM_PREFIX=/global/cfs/cdirs/m2956/nanding/software/nvshmem_src_2.7.0-6/build
export NVSHMEM_USE_GDRCOPY=1
export GDRCOPY_HOME=/usr
export NVSHMEM_MPI_SUPPORT=1
export MPI_HOME=${MPICH_DIR}
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
export NVSHMEM_DEFAULT_PMI2=1
export NVCUFLAGS=--allow-unsupported-compiler
export MPICC=CC
export CC=cc
export CXX=CC

export NVSHMEM_LIBFABRIC_SUPPORT=1
export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.0.0
export LD_LIBRARY_PATH=/global/cfs/cdirs/m2956/nanding/software/MPI_Bootstrap_For_Nan/:$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
export NVSHMEM_BOOTSTRAP=plugin
export NVSHMEM_BOOTSTRAP_PLUGIN=/global/cfs/cdirs/m2956/nanding/software/MPI_Bootstrap_For_Nan/nvshmem_bootstrap_mpich.so

alias ls "ls --color"
export NVSHMEM_DISABLE_CUDA_VMM=1
export FI_CXI_OPTIMIZED_MRS=false
export NVSHMEM_REMOTE_TRANSPORT=libfabric

export CRAYPE_LINK_TYPE=dynamic
export PARMETIS_ROOT=/global/cfs/cdirs/m2956/nanding/software/parmetis-4.0.3-perlmutter-32bit
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
export ACC=GPU
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7
export CUDA_DRV=${CUDA_HOME}/lib64/stubs
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf DartConfiguration.tcl

cmake .. \
  -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_ -DGPU_ACC -I${NVSHMEM_HOME}/include/" \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_Fortran_COMPILER=ftn \
  -DXSDK_ENABLE_Fortran=ON \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DTPL_ENABLE_CUDALIB=ON \
  -DCMAKE_CUDA_FLAGS="-ccbin /opt/cray/pe/craype/2.7.16/bin/CC -std=c++11 -gencode=arch=compute_80,code=sm_80 -I${MPICH_DIR}/include -I${NVSHMEM_HOME}/include/ --disable-warnings" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_INSTALL_LIBDIR=./lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPL_BLAS_LIBRARIES=/opt/cray/pe/libsci/22.06.1.3/GNU/9.1/x86_64/lib/libsci_gnu_81_mp.so \
  -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
  -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.so;${PARMETIS_BUILD_DIR}/libmetis/libmetis.so;${LIB_VTUNE}" \
  -DTPL_ENABLE_COMBBLASLIB=OFF \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DMPIEXEC_NUMPROC_FLAG=-n \
  -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
  -DMPIEXEC_MAX_NUMPROCS=16\
  -DCMAKE_SHARED_LINKER_FLAGS="-lcuda -L${CUDA_HOME}/lib64 -lcudart -L${CUDA_DRV}/ -lnvidia-ml -L${NVSHMEM_HOME}/lib -lnvshmem -L/usr/lib64 -lgdrapi"\
  -DCMAKE_EXE_LINKER_FLAGS="-lcuda -L${CUDA_HOME}/lib64 -lcudart -L${CUDA_DRV}/ -lnvidia-ml -L${NVSHMEM_HOME}/lib -lnvshmem -L/usr/lib64 -lgdrapi -lrt -L/usr/lib64 -lstdc++ -L/opt/cray/libfabric/1.15.0.0/lib64/ -lfabric"
make pddrive		
