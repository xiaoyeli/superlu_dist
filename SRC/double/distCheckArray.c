//#pragma once // warning: #pragma once in main file [-Wpragma-once-outside-header]
/**
 * @file mpi_functions.c
 * @brief Functions for checking array equality in MPI communication
 */
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#ifndef SELF_TEST
#include "superlu_ddefs.h"
/**
 * @brief Compares two integers for equality.
 *
 * @param a Void pointer to the first integer
 * @param b Void pointer to the second integer
 * @return int Returns 0 if the integers are equal, 1 otherwise
 */
int compareInt_t(void *a, void *b)
{
    int_t val1 = *(int_t *)a;
    int_t val2 = *(int_t *)b;
    if (val1 == val2)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}
#endif

/**
 * @brief Compares two integers for equality.
 *
 * @param a Void pointer to the first integer
 * @param b Void pointer to the second integer
 * @return int Returns 0 if the integers are equal, 1 otherwise
 */
int compareInt(void *a, void *b)
{
    int val1 = *(int *)a;
    int val2 = *(int *)b;
    if (val1 == val2)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

/**
 * @brief Compares two doubles for equality.
 *
 * @param a Void pointer to the first double
 * @param b Void pointer to the second double
 * @return int Returns 0 if the doubles are equal, 1 otherwise
 */
int compareDouble(void *a, void *b)
{
    double val1 = *(double *)a;
    double val2 = *(double *)b;
    if (val1 == val2)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

/**
 * @brief Checks whether arrays at two MPI ranks are identical.
 *
 * This function is used to check if a copy of an array at two different MPI ranks are the same.
 * It uses MPI_Send and MPI_Recv to transfer data between ranks, then compares the arrays.
 *
 * @param arr Void pointer to the array to be compared
 * @param length The length of the array
 * @param datatype MPI_Datatype of the array elements
 * @param src_rank The source rank that has the original array
 * @param dest_rank The destination rank that has the copied array
 * @param communicator The MPI_Comm communicator that includes both ranks
 * @param compare A function pointer to the function used to compare elements.
 *                Should take two void pointers and return 0 if they are equal and a non-zero value otherwise.
 * @return int Returns 0 if arrays are identical, 1 otherwise.
 */
int dist_checkArrayEq(void *arr, int length, MPI_Datatype datatype, 
int src_rank, int dest_rank, MPI_Comm communicator, int (*compare)(void *, void *))
{

    // function implementation
    int my_rank, result = 0;
    MPI_Comm_rank(communicator, &my_rank);

    // return if I am not the source or destination rank
    if (my_rank != src_rank && my_rank != dest_rank)
    {
        return 0;
    }

    void *received_arr = NULL;
    int is_null = (arr == NULL);
    // Check whether the array is NULL; if its NULL, then the other rank should also have a NULL array
    if (my_rank == src_rank)
    {
        // printf("src_rank = %d, dest_rank = %d, is_null = %d\n", src_rank, dest_rank, is_null);
        MPI_Send(&is_null, 1, MPI_INT, dest_rank, 0, communicator);
        if (is_null)
        {
            return 0;
        }
    }
    else if (my_rank == dest_rank)
    {
        int src_null;
        MPI_Recv(&src_null, 1, MPI_INT, src_rank, 0, communicator, MPI_STATUS_IGNORE);
        if (src_null != is_null)
        {
            printf("Array is NULL on one rank but not the other: Dest Rank= %d \n", dest_rank);
            assert(0);
            return 1;
        }
        if (is_null)
        {
            return 0;
        }
    }

    // MPI_Bcast(&is_null, 1, MPI_INT, src_rank, communicator);
    int datatype_size;
    MPI_Type_size(datatype, &datatype_size);

    if (my_rank == src_rank)
    {
        MPI_Send(arr, length, datatype, dest_rank, 0, communicator);
    }
    else if (my_rank == dest_rank)
    {
        received_arr = malloc(length*datatype_size);
        if (received_arr == NULL)
        {
            fprintf(stderr, "Failed to allocate memory\n");
            MPI_Abort(communicator, EXIT_FAILURE);
        }
        MPI_Recv(received_arr, length, datatype, src_rank, 0, communicator, MPI_STATUS_IGNORE);

        for (int i = 0; i < length; i++)
        {
            char *addr1 = (char *)arr + i * datatype_size;
            char *addr2 = (char *)received_arr + i * datatype_size;
            if (compare(addr1, addr2) != 0)
            {
                result = 1;
                assert(0);
                break;
            }
        }

        free(received_arr);
    }

    // printf("Rank %d: result = %d\n", my_rank, result);
    return result;
}


#ifdef SELF_TEST
#include <stdio.h>


int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Test 1: Same arrays of integers at ranks 0 and 1
    int length = 5;
    int *arr1 = rank < 2 ? malloc(length * sizeof(int)) : NULL;
    if (rank < 2) {
        for (int i = 0; i < length; i++) {
            arr1[i] = i;
        }
    }
    int src = 0, dest = 1;
    int result = dist_checkArrayEq(arr1, length, MPI_INT, src, dest, MPI_COMM_WORLD, compareInt);
    if (rank == dest) {
        printf("Test 1 (same integer arrays): %s\n", result == 0 ? "Passed" : "Failed");
    }
    free(arr1);

    // Test 2: Different arrays of doubles at ranks 0 and 1
    double *arr2 = rank < 2 ? malloc(length * sizeof(double)) : NULL;
    if (rank < 2) {
        for (int i = 0; i < length; i++) {
            arr2[i] = rank == 0 ? (double) i : (double) i + 0.1;
        }
    }
    result = dist_checkArrayEq(arr2, length, MPI_DOUBLE, src, dest, MPI_COMM_WORLD, compareDouble);
    printf("==== Rank %d: result = %d\n", rank, result);
    if (rank == dest) 
    {
        printf("Test 2: rank %d (different double arrays): %s\n", rank, result == 0 ? "Failed" : "Passed");
    }
    free(arr2);

    // Test 3: Same arrays of doubles at ranks 0 and 1
    double *arr3 = rank < 2 ? malloc(length * sizeof(double)) : NULL;
    if (rank < 2) {
        for (int i = 0; i < length; i++) {
            arr3[i] = (double) i;
        }
    }
    result = dist_checkArrayEq(arr3, length, MPI_DOUBLE, src, dest, MPI_COMM_WORLD, compareDouble);
    if (rank == dest) {
        printf("Test 3 (same double arrays): %s\n", result == 0 ? "Passed" : "Failed");
    }
    free(arr3);

    // Test 4: Different lengths at ranks 0 and 1
    int arr4_length = rank == 0 ? length : length + 1;
    int *arr4 = rank < 2 ? malloc(arr4_length * sizeof(int)) : NULL;
    if (rank < 2) {
        for (int i = 0; i < arr4_length; i++) {
            arr4[i] = i;
        }
    }
    result = dist_checkArrayEq(arr4, arr4_length, MPI_INT, src, dest, MPI_COMM_WORLD, compareInt);
    if (rank == dest) {
        printf("Test 4 (different length arrays): %s\n", result == 0 ? "Failed" : "Passed");
    }
    free(arr4);

    // Test 5: One array is NULL at ranks 0 and 1
    int *arr5 = rank == 0 ? NULL : malloc(length * sizeof(int));
    if (rank == 1) {
        for (int i = 0; i < length; i++) {
            arr5[i] = i;
        }
    }
    result = dist_checkArrayEq(arr5, length, MPI_INT, src, dest, MPI_COMM_WORLD, compareInt);
    if (rank == dest) {
        printf("Test 5 (one NULL array): %s\n", result == 0 ? "Failed" : "Passed");
    }
    if (rank == 1) {
        free(arr5);
    }


    MPI_Finalize();
    return 0;
}
#endif 
