#!/bin/bash

##################################################
##################################################
export ModuleEnv='cleanlinux-unknown-openmpi-gnu'
BuildExample=1 # whether to build all examples
MPIFromSource=1 # whether to build openmpi from source

if [[ $(cat /etc/os-release | grep "PRETTY_NAME") != *"Ubuntu"* && $(cat /etc/os-release | grep "PRETTY_NAME") != *"Debian"* ]]; then
	echo "This script can only be used for Ubuntu or Debian systems"
	exit
fi

##################################################
##################################################


export SuperLUROOT=$PWD

############### Yang's tr4 machine
if [ $ModuleEnv = 'cleanlinux-unknown-openmpi-gnu' ]; then
	
	CC=gcc-8
	FTN=gfortran-8
	CPP=g++-8

	if [[ $MPIFromSource = 1 ]]; then
		export PATH=$PATH:$SuperLUROOT/openmpi-4.0.1/bin
		export MPICC="$SuperLUROOT/openmpi-4.0.1/bin/mpicc"
		export MPICXX="$SuperLUROOT/openmpi-4.0.1/bin/mpicxx"
		export MPIF90="$SuperLUROOT/openmpi-4.0.1/bin/mpif90"
		export LD_LIBRARY_PATH=$SuperLUROOT/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
		export LIBRARY_PATH=$SuperLUROOT/openmpi-4.0.1/lib:$LIBRARY_PATH  	

	else

		#######################################
		#  define the following as needed
		export MPICC=
		export MPICXX=
		export MPIF90=
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
		export LIBRARY_PATH=$LIBRARY_PATH 
		export PATH=$PATH 
		########################################

		if [[ -z "$MPICC" ]]; then
			echo "Line: ${LINENO} of $BASH_SOURCE: It seems that openmpi will not be built from source, please set MPICC, MPICXX, MPIF90, PATH, LIBRARY_PATH, LD_LIBRARY_PATH for your OpenMPI build correctly above. Make sure OpenMPI > 4.0.0 is used and compiled with CC=$CC, CXX=$CPP and FC=$FTN."
			exit
		fi
	fi
	export BLAS_LIB=$SuperLUROOT/OpenBLAS/libopenblas.so
	export LAPACK_LIB=$SuperLUROOT/OpenBLAS/libopenblas.so
	export LD_LIBRARY_PATH=$SuperLUROOT/OpenBLAS/:$LD_LIBRARY_PATH
	OPENMPFLAG=fopenmp


fi
###############





#set up environment variables, these are also needed when running GPTune 
################################### 

export ParMETIS_DIR=$SuperLUROOT/parmetis-4.0.3/install
export PARMETIS_INCLUDE_DIRS="$ParMETIS_DIR/include"
export PARMETIS_LIBRARIES="$ParMETIS_DIR/lib/libparmetis.so;$ParMETIS_DIR/lib/libmetis.so"



# install dependencies using apt-get and virtualenv
###################################

apt-get update -y 
apt-get upgrade -y 
apt-get dist-upgrade -y  
apt-get install dialog apt-utils -y 
apt-get install build-essential software-properties-common -y 
add-apt-repository ppa:ubuntu-toolchain-r/test -y 
apt-get update -y 
apt-get install gcc-8 g++-8 gfortran-8 -y  
# apt-get install gcc-9 g++-9 gfortran-9 -y  
# apt-get install gcc-10 g++-10 gfortran-10 -y  


apt-get install libffi-dev -y
apt-get install libssl-dev -y

# apt-get install libblas-dev  -y
# apt-get install liblapack-dev -y
apt-get install cmake -y
apt-get install git -y
apt-get install vim -y
apt-get install autoconf automake libtool -y
apt-get install zlib1g-dev -y
apt-get install wget -y
apt-get install libsm6 -y
apt-get install libbz2-dev -y
apt-get install libsqlite3-dev -y
apt-get install jq -y


# manually install dependencies from cmake and make
###################################
cd $SuperLUROOT
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN -j32
make PREFIX=. CC=$CC CXX=$CPP FC=$FTN install -j32


if [[ $MPIFromSource = 1 ]]; then
	cd $SuperLUROOT
	wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
	bzip2 -d openmpi-4.0.1.tar.bz2
	tar -xvf openmpi-4.0.1.tar 
	cd openmpi-4.0.1/ 
	./configure --prefix=$PWD --enable-mpi-interface-warning --enable-shared --enable-static --enable-cxx-exceptions CC=$CC CXX=$CPP F77=$FTN FC=$FTN --enable-mpi1-compatibility --disable-dlopen
	make -j32
	make install
fi



	cd $SuperLUROOT

	#### the following server is often down, so switch to the github repository 
	wget https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/parmetis/4.0.3-4/parmetis_4.0.3.orig.tar.gz
	tar -xf parmetis_4.0.3.orig.tar.gz
	cd parmetis-4.0.3/
	cp $SuperLUROOT/PATCH/parmetis/CMakeLists.txt .
	mkdir -p install
	make config shared=1 cc=$MPICC cxx=$MPICXX prefix=$PWD/install
	make install 
	cd ../
	cp $PWD/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.so $PWD/parmetis-4.0.3/install/lib/.
	cp $PWD/parmetis-4.0.3/metis/include/metis.h $PWD/parmetis-4.0.3/install/include/.


	cd $SuperLUROOT
	mkdir -p build_cpu
	cd build_cpu
	rm -rf CMakeCache.txt
	rm -rf DartConfiguration.tcl
	rm -rf CTestTestfile.cmake
	rm -rf cmake_install.cmake
	rm -rf CMakeFiles
	cmake .. \
		-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
		-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_CXX_COMPILER=$MPICXX \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
		-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES 
	make pddrive3d



	cd $SuperLUROOT
	mkdir -p build_gpu_no_nvshmem
	cd build_gpu_no_nvshmem
	rm -rf CMakeCache.txt
	rm -rf DartConfiguration.tcl
	rm -rf CTestTestfile.cmake
	rm -rf cmake_install.cmake
	rm -rf CMakeFiles
	cmake .. \
		-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
		-DCMAKE_C_FLAGS="-DGPU_SOLVE -std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_CXX_COMPILER=$MPICXX \
		-DCMAKE_C_COMPILER=$MPICC \
		-DCMAKE_Fortran_COMPILER=$MPIF90 \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		-DTPL_BLAS_LIBRARIES="${BLAS_LIB}" \
		-DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
		-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
		-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES \
		-DTPL_ENABLE_NVSHMEM=OFF \
		-DCMAKE_CUDA_FLAGS="-I$SuperLUROOT/openmpi-4.0.1/include -ccbin=$MPICXX" \
		-DCMAKE_CUDA_ARCHITECTURES=70 \   
		-DTPL_ENABLE_CUDALIB=ON 
	make pddrive3d


	# cd $SuperLUROOT
	# mkdir -p build_gpu_nvshmem
	# cd build_gpu_nvshmem
	# rm -rf CMakeCache.txt
	# rm -rf DartConfiguration.tcl
	# rm -rf CTestTestfile.cmake
	# rm -rf cmake_install.cmake
	# rm -rf CMakeFiles
# cmake .. \
#   -DCMAKE_C_FLAGS="-DGPU_SOLVE -std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0 -DAdd_ -I${NVSHMEM_HOME}/include" \
#   -DCMAKE_CXX_COMPILER=CC \
#   -DCMAKE_C_COMPILER=cc \
#   -DCMAKE_Fortran_COMPILER=ftn \
#   -DXSDK_ENABLE_Fortran=ON \
#   -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
#   -DTPL_ENABLE_LAPACKLIB=ON \
#   -DBUILD_SHARED_LIBS=ON \
#   -DTPL_ENABLE_CUDALIB=ON \
#   -DCMAKE_CUDA_FLAGS="-I${NVSHMEM_HOME}/include -I$SuperLUROOT/openmpi-4.0.1/include -ccbin=$MPICXX" \
#   -DCMAKE_CUDA_ARCHITECTURES=80 \
