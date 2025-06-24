module purge
module load gcc/13.1.0
module load python/gcc-13.1.0/3.12.4
module load openmpi/gcc-13.1.0/4.0.1
module load cmake/3.19.2



cmake .. \
  -DCMAKE_C_FLAGS="  -std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_" \
  -DCMAKE_CXX_COMPILER=$MPICXX \
  -DCMAKE_C_COMPILER=$MPICC \
  -DCMAKE_Fortran_COMPILER=$MPIF90 \
  -DXSDK_ENABLE_Fortran=ON \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DTPL_ENABLE_LAPACKLIB=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DTPL_ENABLE_CUDALIB=OFF \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_INSTALL_LIBDIR=./lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPL_ENABLE_MAGMALIB=OFF \
  -DTPL_BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libblas.so" \
  -DTPL_LAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.so" \
  -DTPL_PARMETIS_INCLUDE_DIRS="/home/administrator/Desktop/Software/parmetis-4.0.3/install/include" \
  -DTPL_PARMETIS_LIBRARIES="/home/administrator/Desktop/Software/parmetis-4.0.3/install/lib/libparmetis.so;/home/administrator/Desktop/Software/parmetis-4.0.3/install/lib/libmetis.so" \
  -DTPL_ENABLE_COMBBLASLIB=OFF \
  -DTPL_ENABLE_NVSHMEM=OFF \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -Denable_complex16=ON \
  -DXSDK_INDEX_SIZE=32 \
  -Denable_single=ON \
  -DTPL_ENABLE_SYMATCHLIB=ON \
  -DTPL_SYMATCH_INCLUDE_DIRS="/home/administrator/Desktop/Research/superlu_dist-symatch/superlu_dist-symatch/matching/symatch/inc;/home/administrator/Desktop/Research/superlu_dist-symatch/superlu_dist-symatch/matching/symatch/util;/home/administrator/Desktop/Research/superlu_dist-symatch/superlu_dist-symatch/matching/lib/matching" \
  -DTPL_SYMATCH_LIBRARIES="/home/administrator/Desktop/Research/superlu_dist-symatch/superlu_dist-symatch//matching/lib/matching/lib/libsuitor.a"
     
make pddrive -j
make pddrive-v1 -j


