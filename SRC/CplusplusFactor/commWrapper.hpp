#pragma once 
#include "mpi.h"

enum collAlg { SYNC, ASYNC, RING, RINGMOD};
enum collCommStatus { Started, Finished, Uninitialized };
class bcastStruct
{
public: 
    MPI_Request request; 
    MPI_Status  status;
    MPI_Comm comm; 
    MPI_Datatype dtype; 
    collAlg bcastAlgm; 
    collCommStatus bcastStatus; 
    bcastStruct(MPI_Comm comm_, MPI_Datatype dtype_, collAlg algm);
    bcastStruct() {};
    int init(void* buffer, int root, int count);
    int test();
    int wait(); 
    bool isFinished();
private: 
    void* buffer;   // doesn't own it; 
    int root;
    int count;
    int commSize;
    int myRank;
};

// class reduceStruct
// {

// };

