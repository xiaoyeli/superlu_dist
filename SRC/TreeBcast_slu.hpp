#ifndef __SUPERLU_TREEBCAST
#define __SUPERLU_TREEBCAST

// #include "asyncomm.hpp"
// #include "blas.hpp"
// #include "timer.h"
#include "superlu_defs.h"
#ifdef oneside
#include "oneside.h"
#endif

#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <string>
#include <memory>
//#include <random>

// options to switch from a flat bcast/reduce tree to a binary tree
#ifndef FTREE_LIMIT
#define FTREE_LIMIT 8
#endif

namespace SuperLU_ASYNCOMM {

    // Basic data types
  typedef    int  Int;
    // IO
  extern  std::ofstream  statusOFS;
    // Commonly used
  const Int DEG_TREE = 2; //number of children of each tree node

  extern std::map< MPI_Comm , std::vector<Int> > commGlobRanks;

  template< typename T>
    class TreeBcast_slu{
      protected:
        std::vector<MPI_Request> recvRequests_;
        std::vector<MPI_Status> recvStatuses_;
        std::vector<Int> recvDoneIdx_;
        std::vector<T *> recvDataPtrs_;
        std::vector<T> recvTempBuffer_;
        Int recvPostedCount_;
        Int recvCount_;

        std::vector<MPI_Request> sendRequests_;
        std::vector<MPI_Status> sendStatuses_;
        std::vector<Int> sendDoneIdx_;
        std::vector<T *> sendDataPtrs_;
        std::vector<T> sendTempBuffer_;
        Int sendPostedCount_;
        Int sendCount_;

        bool done_;
        bool fwded_;
        bool isReady_;

        MPI_Comm comm_;
        Int myRoot_;
        //not sure about this one
        Int mainRoot_;
        std::vector<Int> myDests_;

        Int myRank_;
        Int msgSize_;
        Int tag_;

        MPI_Datatype type_;


      protected:
    	virtual void buildTree(Int * ranks, Int rank_cnt)=0;
    	virtual void buildTree(Int * ranks, Int rank_cnt, Int Pc)=0;

      public:
        static TreeBcast_slu<T> * Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize,double rseed);
        static TreeBcast_slu<T> * Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize,double rseed, Int Pc);

        TreeBcast_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt,Int msgSize);
        TreeBcast_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt,Int msgSize, Int Pc);
        
        TreeBcast_slu();
        TreeBcast_slu(const TreeBcast_slu & Tree);
        virtual ~TreeBcast_slu();
        virtual TreeBcast_slu * clone() const = 0; 
        virtual void Copy(const TreeBcast_slu & Tree);
        virtual void Reset();


        virtual inline Int GetNumMsgToRecv();
        virtual inline Int GetNumRecvMsg();
        virtual inline Int GetNumMsgToSend();
        virtual inline Int GetNumSendMsg();
        inline void SetDataReady(bool rdy);
        inline void SetTag(Int tag);
        inline Int GetTag();
        Int * GetDests();
        Int GetDest(Int i);
        Int GetDestCount();
        Int GetRoot();
        bool IsRoot();
        void SetMsgSize(Int msgSize){ this->msgSize_ = msgSize;}
        Int GetMsgSize();
        bool IsReady(){ return this->isReady_;}

        //async wait and forward
 	virtual void AllocateBuffer();
	virtual void cleanupBuffers();
	virtual void allocateRequest();
#ifdef oneside
	virtual void forwardMessageOneSide(T * locBuffer, Int msgSize,  int* iam_col, int *BCcount, long* BCbase, int* maxrecvsz, int Pc);
	virtual void forwardMessageOneSideU(T * locBuffer, Int msgSize,  int* iam_col, int *BCcount, long* BCbase, int* maxrecvsz, int Pc);
#endif
        virtual void forwardMessageSimple(T * locBuffer, Int msgSize);
	virtual void waitSendRequest();	

    };

  template< typename T>
    class FTreeBcast2: public TreeBcast_slu<T>{
      protected:
        virtual void buildTree(Int * ranks, Int rank_cnt);
        virtual void buildTree(Int * ranks, Int rank_cnt,Int Pc);

      public:
        FTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
        FTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, Int Pc);
        virtual FTreeBcast2<T> * clone() const;
    };

  template< typename T>
    class BTreeBcast2: public TreeBcast_slu<T>{
      protected:
        virtual void buildTree(Int * ranks, Int rank_cnt);
        virtual void buildTree(Int * ranks, Int rank_cnt, Int Pc);

      public:
        BTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
        BTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, Int Pc);
        virtual BTreeBcast2<T> * clone() const;
    };

  template< typename T>
    class ModBTreeBcast2: public TreeBcast_slu<T>{
      protected:
        double rseed_;
        virtual void buildTree(Int * ranks, Int rank_cnt);

      public:
        ModBTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed);
        virtual ModBTreeBcast2<T> * clone() const;
    };
} // namespace SuperLU_ASYNCOMM

#include "TreeBcast_slu_impl.hpp"

#endif // __SUPERLU_TREEBCAST
