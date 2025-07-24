# cmake build for SuperLU_dist for ALCF Polaris GPU compute nodes
#
# Last updated: 07/23/2025

# export PATH=/eagle/projects/ATPESC2025/usr/MathPackages/cmake-3.29.1/bin/:$PATH
export MAGMA_ROOT=/eagle/projects/ATPESC2025/usr/MathPackages/magma-master/install/
# export NVSHMEM_HOME=/eagle/projects/ATPESC2025/usr/MathPackages/nvshmem_src_2.8.0-3/build/
export PARMETIS_ROOT=/eagle/projects/ATPESC2025/usr/MathPackages/parmetis-4.0.3/install

module use /soft/modulefiles
module use /eagle/ATPESC2025/usr/modulefiles
module load track-5-numerical
# module load cudatoolkit-standalone
# module load nvhpc-mixed craype-accel-nvidia80

module list

# avoid bug in cudatoolkit
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda-12.2\/compat:/}
cd ..
rm -fr gcc-build; mkdir gcc-build; cd gcc-build


cmake .. \
      -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=1 -DDEBUGlevel=0" \
      -DCMAKE_C_COMPILER=cc \
      -DCMAKE_CXX_COMPILER=CC \
      -DCMAKE_Fortran_COMPILER=ftn \
      -Denable_single=ON \
      -Denable_complex16=ON \
      -DXSDK_ENABLE_Fortran=ON \
      -DTPL_ENABLE_PARMETISLIB=ON \
      -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include"\
      -DTPL_PARMETIS_LIBRARIES="${PARMETIS_ROOT}/lib/libparmetis.so;${PARMETIS_ROOT}/lib/libmetis.so" \
      -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
      -DTPL_BLAS_LIBRARIES=/opt/cray/pe/libsci/23.12.5/GNU/12.3/x86_64/lib/libsci_gnu.so \
      -DTPL_ENABLE_LAPACKLIB=ON \
      -DTPL_LAPACK_LIBRARIES=/opt/cray/pe/libsci/23.12.5/GNU/12.3/x86_64/lib/libsci_gnu.so \
      -DTPL_ENABLE_CUDALIB=ON \
      -DCMAKE_CUDA_FLAGS="-I${NVSHMEM_HOME}/include -I${MPICH_DIR}/include -ccbin=/opt/cray/pe/craype/2.7.30/bin/CC" \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DTPL_ENABLE_NVSHMEM=OFF \
      -DTPL_NVSHMEM_LIBRARIES="-L${CUDA_HOME}/lib64/stubs/ -lnvidia-ml -L/usr/lib64 -lgdrapi -lstdc++ -L/opt/cray/libfabric/1.15.2.0/lib64 -lfabric -L${NVSHMEM_HOME}/lib -lnvshmem" \
      -DTPL_ENABLE_MAGMALIB=ON \
      -DTPL_MAGMA_INCLUDE_DIRS="${MAGMA_ROOT}/include" \
      -DTPL_MAGMA_LIBRARIES="${MAGMA_ROOT}/lib/libmagma.so" \
      -DTPL_ENABLE_COMBBLASLIB=OFF \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=. \
      -DCMAKE_INSTALL_LIBDIR=./lib \
      -DMPIEXEC_NUMPROC_FLAG=-n \
      -DBUILD_SHARED_LIBS=ON 
make install -j

#      -DMPIEXEC_EXECUTABLE=/usr/bin/srun \
#      -DMPIEXEC_MAX_NUMPROCS=16 \
     #     -DXSDK_INDEX_SIZE=64
