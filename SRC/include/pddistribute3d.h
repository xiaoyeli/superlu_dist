#pragma once 

#include "superlu_ddefs.h"

// Defining DIST3D temoprarily to track changes in this file while fully
// integrating the 3D code
// #define DIST3D
float pddistribute3d(superlu_dist_options_t *options, int_t n, SuperMatrix *A,
                     dScalePermstruct_t *ScalePermstruct,
                     Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct,
                     gridinfo3d_t *grid3d);


int_t dReDistribute_A3d(SuperMatrix *A, dScalePermstruct_t *ScalePermstruct,
                        Glu_freeable_t *Glu_freeable, int_t *xsup, int_t *supno,
                        gridinfo3d_t *grid3d, int_t *colptr[], int_t *rowind[],
                        double *a[]);

void dpropagate_A_to_LU3d(
    dLUstruct_t *LUstruct,
    int_t *xa,
    int_t *asub,
    double *a,
    superlu_dist_options_t* options,
    gridinfo3d_t *grid3d,
    int_t nsupers, 
    float *mem_use);
int_t ComputeLDAspa_Ilsum( int_t nsupers, int_t* ilsum,  gridinfo3d_t* grid3d) ;

// void propagateDataThroughMatrixBlocks(int_t nsupers, dLUstruct_t *LUstruct, gridinfo3d_t* grid3d, int_t *xusub, int_t *usub, int_t **ToSendR, 
//     int_t *ToSendD, int_t *Urb_length, int_t *rb_marker, int_t *Urb_fstnz, int_t *Ucbs, int_t *ToRecv);

void propagateDataThroughMatrixBlocks(int_t nsupers, Glu_freeable_t *Glu_freeable, dLUstruct_t *LUstruct, gridinfo3d_t* grid3d,  
int_t *Urb_length, int_t *rb_marker, int_t *Urb_fstnz, int_t *Ucbs,
int **ToSendR,  int *ToSendD,  int *ToRecv);
int_t checkDist3DLUStruct(dLUstruct_t* LUstruct, gridinfo3d_t* grid3d);

void dbcastPermutedSparseA(SuperMatrix *A, 
                          dScalePermstruct_t *ScalePermstruct,
                          Glu_freeable_t *Glu_freeable, 
                          dLUstruct_t *LUstruct, gridinfo3d_t *grid3d);

void dnewTrfPartitionInit(int_t nsupers,  dLUstruct_t *LUstruct, gridinfo3d_t *grid3d);


int compareInt_t(void *a, void *b);
int compareInt(void *a, void *b);
int compareDouble(void *a, void *b);
int dist_checkArrayEq(void *arr, int length, MPI_Datatype datatype, int src_rank, int dest_rank, MPI_Comm communicator, int (*compare)(void *, void *));
