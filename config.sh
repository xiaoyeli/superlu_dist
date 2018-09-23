 module unload darshan
 module swap craype-haswell craype-mic-knl
 module load cray-fftw
 module load gsl
 module load cray-hdf5-parallel
 module load idl
 module load craype-hugepages2M
 module unload cray-libsci
 cd SRC
 make clean
 make
 cd ../EXAMPLE
 make clean
 make 

 
