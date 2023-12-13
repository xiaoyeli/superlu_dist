#pragma once 
#include <mpi.h>
#include "superlu_defs.h"
#include "superlu_zdefs.h"
template<typename Ftype>
struct diagFactBufs_t {
    Ftype* BlockLFactor;
    Ftype* BlockUFactor;
    // Add other members as needed
};
struct complex; 
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

// template <>
// MPI_Datatype get_mpi_type<complex>()
// {
//     return MPI_C_COMPLEX;
// }

template <>
MPI_Datatype get_mpi_type<doublecomplex>()
{
    return MPI_C_DOUBLE_COMPLEX;
}

// AnormType<double> 

template <typename Ftype>
using AnormType = typename std::conditional<
    std::is_same<Ftype, float>::value,
    float,
    typename std::conditional<
        std::is_same<Ftype, double>::value || std::is_same<Ftype, doublecomplex>::value,
        double,
        float  // Default to float
    >::type
>::type;

// template <typename Ftype>
// using trf3dpartitionType 
template <typename Ftype>
using trf3dpartitionType = typename std::conditional<
    std::is_same<Ftype, double>::value,
    dtrf3Dpartition_t,
    typename std::conditional<
        std::is_same<Ftype, float>::value,
        strf3Dpartition_t,
        typename std::conditional<
            std::is_same<Ftype, doublecomplex>::value,
            ztrf3Dpartition_t,
            void
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
            std::is_same<Ftype, doublecomplex>::value,
            zLUstruct_t,
            void
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
            std::is_same<Ftype, doublecomplex>::value,
            zLocalLU_t,
            void 
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
            std::is_same<Ftype, doublecomplex>::value,
            zLUValSubBuf_t,
            void 
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
            std::is_same<Ftype, doublecomplex>::value,
            zdiagFactBufs_t,
            void 
        >::type
    >::type
>::type;


