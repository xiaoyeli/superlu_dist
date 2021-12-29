module load cmake
module swap xl gcc
module load cuda
module load essl
module load spectrum-mpi
module load netlib-lapack
module load netlib-scalapack
export PARMETIS_ROOT=/gpfs/wolf/gen170/scratch/liuyangzhuan/parmetis-4.0.3
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-ppc64le
cmake .. \
     -DTPL_PARMETIS_LIBRARIES=ON \
     -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include;${OLCF_CUDA_ROOT}/include" \
     -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
     -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DDEBUGlevel=0 -DAdd_" \
     -DCMAKE_C_COMPILER=mpicc \
     -DCMAKE_CXX_COMPILER=mpiCC \
     -DXSDK_ENABLE_Fortran=ON \
     -DTPL_ENABLE_CUDALIB=TRUE \
     -DTPL_CUDA_LIBRARIES="${OLCF_CUDA_ROOT}/lib65/libcublas.so;${OLCF_CUDA_ROOT}/lib64/libcusparse.so;${OLCF_CUDA_ROOT}/lib64/libcudart.so" \
     -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
     -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
     -DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/libblas.so" \
     -DBUILD_SHARED_LIBS=OFF \
     -DCMAKE_INSTALL_PREFIX=. 
make pddrive


#     -DTPL_CUDA_LIBRARIES="/global/common/software/nersc/cos1.3/cuda/11.3.0/targets/x86_64-linux/lib/libcublas.so;/global/common/software/nersc/cos1.3/cuda/11.3.0/targets/x86_64-linux/lib/libcudart.so" \
