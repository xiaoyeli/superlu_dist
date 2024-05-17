module load rocm

MYROOT=$PWD

export PATH=$PATH:$MYROOT/openmpi-4.1.5/bin
export CC="$MYROOT/openmpi-4.1.5/bin/mpicc"
export CPP="$MYROOT/openmpi-4.1.5/bin/mpicxx"
export FTN="$MYROOT/openmpi-4.1.5/bin/mpif90"
export LD_LIBRARY_PATH=$MYROOT/openmpi-4.1.5/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$MYROOT/openmpi-4.1.5/lib:$LIBRARY_PATH  	

export BLAS_LIB=$MYROOT/OpenBLAS/libopenblas.so
export LAPACK_LIB=$MYROOT/OpenBLAS/libopenblas.so
export OPENMPFLAG=fopenmp

export ParMETIS_DIR=$MYROOT/parmetis-4.0.3/install
export METIS_DIR=$ParMETIS_DIR
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/../metis/include;$ParMETIS_DIR/include"
export PARMETIS_LIBRARIES=$ParMETIS_DIR/lib/libparmetis.so
SLU_ENABLE_HIP=TRUE


cd $MYROOT
rm -rf openmpi-4.1.5
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2
bzip2 -d openmpi-4.1.5.tar.bz2
tar -xvf openmpi-4.1.5.tar 
cd openmpi-4.1.5/ 
./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=gcc CXX=g++ F77=gfortran FC=gfortran --enable-mpi1-compatibility --disable-dlopen


cd $MYROOT
rm -rf parmetis-4.0.3*
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3/
mkdir -p install
make config shared=1 cc=$CC cxx=$CPP prefix=$PWD/install
make clean
make install
cp ./build/Linux-x86_64/libmetis/libmetis.* ./install/lib/.
cp ./metis/include/metis.h ./install/include/.


cd $MYROOT
rm -rf OpenBLAS
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make PREFIX=. USE_OPENMP=1 CC=$CC CXX=$CPP FC=$FTN -j32
make PREFIX=. USE_OPENMP=1 CC=$CC CXX=$CPP FC=$FTN install -j32


cd $MYROOT
rm -rf superlu_dist
git clone https://github.com/xiaoyeli/superlu_dist.git
cd superlu_dist
mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
        -DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
        -DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=1 -DPROFlevel=0 -DDEBUGlevel=0" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_CXX_COMPILER=$CPP \
        -DHIP_HIPCC_FLAGS="--amdgpu-target=gfx908 -I${MYROOT}/openmpi-4.1.5/include -I${MYROOT}/superlu_dist/SRC/cuda -I${MYROOT}/superlu_dist/SRC/hip" \
        -DTPL_ENABLE_HIPLIB=$SLU_ENABLE_HIP \
        -DCMAKE_INSTALL_PREFIX=. \
        -DCMAKE_INSTALL_LIBDIR=./lib \
        -DCMAKE_C_COMPILER=$CC \
        -DCMAKE_Fortran_COMPILER=$FTN \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
        -DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
        -DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
        -Denable_complex16=ON \
        -DTPL_ENABLE_LAPACKLIB=ON \
        -Denable_single=ON \
        -DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
        -DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
make install -j32