#git clone https://github.com/spack/spack.git
export FORCE_UNSAFE_CONFIGURE=1
export SPACK_ROOT=/home/administrator/Desktop/spack
export PATH=${SPACK_ROOT}/bin:${PATH}
source ${SPACK_ROOT}/share/spack/setup-env.sh
spack install hpcviewer
spack install hpctoolkit +mpi
spack bootstrap
source ${SPACK_ROOT}/share/spack/setup-env.sh
