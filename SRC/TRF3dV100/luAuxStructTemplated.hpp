#pragma once 
#include <mpi.h>
#include "superlu_defs.h"
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



template <typename Ftype>
using AnormType = typename std::conditional<
    std::is_same<Ftype, complex>::value || std::is_same<Ftype, double complex>::value,
    float,
    double
>::type;

template <typename Ftype>
using trf3dpartitionType 
template <typename Ftype>
using trf3dpartitionType = typename std::conditional<
    std::is_same<Ftype, double>::value,
    dtrf3dpartition_t,
    typename std::conditional<
        std::is_same<Ftype, float>::value,
        strf3dpartition_t,
        typename std::conditional<
            std::is_same<Ftype, complex>::value,
            ctrf3dpartition_t,
            ztrf3dpartition_t
        >::type
    >::type
>::type;

template <typename Ftype>
using LUStruct_type = typename std::conditional<
    std::is_same<Ftype, double>::value,
    dLUstruct_t,
    typename std::conditional<
        std::is_same<Ftype, float>::value,
        sLUstruct_t,
        typename std::conditional<
            std::is_same<Ftype, complex>::value,
            cLUstruct_t,
            zLUstruct_t
        >::type
    >::type
>::type;

template <typename Ftype>
using LocalLU_type = typename std::conditional<
    std::is_same<Ftype, double>::value,
    dLocalLU_t,
    typename std::conditional<
        std::is_same<Ftype, float>::value,
        sLocalLU_t,
        typename std::conditional<
            std::is_same<Ftype, complex>::value,
            cLocalLU_t,
            zLocalLU_t
        >::type
    >::type
>::type;

template <typename Ftype>
using LUValSubBuf_type = typename std::conditional<
    std::is_same<Ftype, double>::value,
    dLUValSubBuf_t,
    typename std::conditional<
        std::is_same<Ftype, float>::value,
        sLUValSubBuf_t,
        typename std::conditional<
            std::is_same<Ftype, complex>::value,
            cLUValSubBuf_t,
            zLUValSubBuf_t
        >::type
    >::type
>::type;

template <typename Ftype>
using diagFactBufs_type = typename std::conditional<
    std::is_same<Ftype, double>::value,
    ddiagFactBufs_t,
    typename std::conditional<
        std::is_same<Ftype, float>::value,
        sdiagFactBufs_t,
        typename std::conditional<
            std::is_same<Ftype, complex>::value,
            cdiagFactBufs_t,
            zdiagFactBufs_t
        >::type
    >::type
>::type;


