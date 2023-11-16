#pragma once 
#include <mpi.h>
template<typename Ftype>
struct diagFactBufs_t {
    Ftype* BlockLFactor;
    Ftype* BlockUFactor;
    // Add other members as needed
};

template <typename T>
MPI_Datatype get_mpi_type()
{
    throw std::runtime_error("Unsupported type");
}

template <>
MPI_Datatype get_mpi_type<int>()
{
    return MPI_INT;
}

template <>
MPI_Datatype get_mpi_type<float>()
{
    return MPI_FLOAT;
}

template <>
MPI_Datatype get_mpi_type<double>()
{
    return MPI_DOUBLE;
}

template <>
MPI_Datatype get_mpi_type<complex>()
{
    return MPI_C_COMPLEX;
}

template <>
MPI_Datatype get_mpi_type<doublecomplex>()
{
    return MPI_C_DOUBLE_COMPLEX;
}



