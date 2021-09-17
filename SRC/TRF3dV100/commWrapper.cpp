#include "commWrapper.hpp"


bcastStruct::bcastStruct(MPI_Comm comm_, 
        MPI_Datatype dtype_,collAlg algm_) : 
        comm(comm_), dtype(dtype_), bcastAlgm(algm_)
{
    // get the comm size 
    bcastStatus = Finished; 
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &myRank);
}
int bcastStruct::init(void* buffer_, int root_, int count_)
{
    buffer = buffer_;
    root = root_;
    count = count_; 
    bcastStatus = Started; 
    switch (bcastAlgm)
    {
    // in the case of SYNC communication takes place 
    // at the initialization itself
    case SYNC:
        MPI_Bcast(buffer, count, dtype, root, comm);
        bcastStatus =Finished; 
        return 1; 
        break;
    case ASYNC:
        MPI_Ibcast(buffer, count, dtype, root, comm, &request);
        break;
    
    default:
        break;
    }
    return 0;
}

int bcastStruct::test()
{
    int flag; 
    switch (bcastAlgm)
    {
    // in the case of SYNC communication takes place only when
    // the wait is placed     
    case SYNC:
        break;
    case ASYNC:
        MPI_Test(&request, &flag, &status);
        if(flag)
        {
            bcastStatus =Finished; 
            return 1; 
        }
        break;
    default:
        break;
    }
    return 0; 
}
int bcastStruct::wait()
{
    switch (bcastAlgm)
    {
    // in the case of SYNC communication takes place only when
    // the wait is placed     
    case SYNC:
        break;
    case ASYNC:
        MPI_Wait(&request, &status);
        bcastStatus =Finished; 
        break;
    default:
        break;
    }
    return 0; 
    
}

bool bcastStruct::isFinished()
{
    return bcastStatus==Finished; 
}