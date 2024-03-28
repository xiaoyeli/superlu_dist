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

template <typename Ftype>
using threshPivValType = typename std::conditional<
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

// Define one<T> for different types double(1.0), float(1.0f), doublecomplex(1.0, 0.0), complex(1.0f, 0.0f)
// Generic template for one
template<typename T>
T one();

// Specialization for double
template<>
double one<double>() {
    return 1.0;
}

// Specialization for float
template<>
float one<float>() {
    return 1.0f;
}

// Specialization for std::complex<double>
template<>
doublecomplex one<doublecomplex>() {
    doublecomplex z = {1.0, 0.0};
    return z;
}


template<typename T>
T zeroT();

// Specialization for double
template<>
double zeroT<double>() {
    return 0.0;
}

// Specialization for float
template<>
float zeroT<float>() {
    return 0.0f;
}

// Specialization for std::complex<double>
template<>
doublecomplex zeroT<doublecomplex>() {
    doublecomplex z = {0.0, 0.0};
    return z;
}

template <typename T>
__device__
T atomicAddT(T* address, T val);

// Specialization for double
template<>
double atomicAddT<double>(double* address, double val) {
    return atomicAdd(address, val);
}

// Specialization for float
template<>
float atomicAddT<float>(float* address, float val) {
    return atomicAdd(address, val);
}

// Specialization for std::complex<double>
template<>
doublecomplex atomicAddT<doublecomplex>(doublecomplex* address, doublecomplex val) {
    // doublecomplex out = *address;
    
    atomicAdd (&address->r, val.r);
    atomicAdd (&address->i, val.i);
    return *address;
}


// External Operator Overload for '-'
__host__ __device__
doublecomplex operator-(const doublecomplex& a, const doublecomplex& b) {
    return {a.r - b.r, a.i - b.i};
}

// External Operator Overload for '=='
__host__ __device__
bool operator==(const doublecomplex& a, const doublecomplex& b) {
    return (a.r == b.r) && (a.i == b.i);
}

// External Operator Overload for '/'
__host__ __device__
doublecomplex operator/(const doublecomplex& a, const doublecomplex& b) {
    double denom = b.r * b.r + b.i * b.i;
    return {(a.r * b.r + a.i * b.i) / denom, (a.i * b.r - a.r * b.i) / denom};
}

__host__ __device__
doublecomplex operator-(const doublecomplex& a) {
    return {-a.r, -a.i};
}

// Note: The assignment operator '=' cannot be overloaded as an external operator.
// It must be a member function.

// External Operator Overload for '*='
__host__ __device__
doublecomplex& operator*=(doublecomplex& a, const doublecomplex& b) {
    double tr = a.r * b.r - a.i * b.i;
    double ti = a.r * b.i + a.i * b.r;
    a.r = tr;
    a.i = ti;
    return a;
}

// External Operator Overload for '-='
__host__ __device__
doublecomplex& operator-=(doublecomplex& a, const doublecomplex& b) {
    a.r -= b.r;
    a.i -= b.i;
    return a;
}


// Template for general case (not yet defined)
// template<typename T>
// double sqnorm(T value);

// // Specialization for float
// template<>
double sqnorm(float value) {
    return (double) value * (double) value;
}

// Specialization for double
// template<>
double sqnorm(double value) {
    return value * value;
}

// Specialization for doublecomplex
// template<>
double sqnorm(doublecomplex value) {
    return value.r * value.r + value.i * value.i;
}


// template <typename Ftype>
// void setDiagToThreshold(Ftype* diagptr, threshPivValType<Ftype> thresh);


// template <>
void setDiagToThreshold(double* diagptr, double thresh) {
    if (*diagptr < 0)
        *diagptr = -thresh;
    else
        *diagptr = thresh; 
}

// template <>
void setDiagToThreshold(float* diagptr, float thresh) {
    if (*diagptr < 0)
        *diagptr = -thresh;
    else
        *diagptr = thresh; 
    // *diagptr = thresh;
}


void setDiagToThreshold(doublecomplex* diagptr, double thresh) {
    doublecomplex z = {thresh, 0.0};
    if (diagptr->r < 0)
        *diagptr = -z;
    else
        *diagptr = z; 
    
    *diagptr = z;
}

