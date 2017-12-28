#ifndef _PEXSI_TREE_V2_HPP_
#define _PEXSI_TREE_V2_HPP_

#include "environment.hpp"
 #include "blas.hpp"
#include "timer.h"
#include "superlu_defs.h"


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



namespace PEXSI{


  extern std::map< MPI_Comm , std::vector<int> > commGlobRanks;



  template< typename T>
    class TreeBcast_v2{
      protected:
        std::vector<MPI_Request> recvRequests_;
        std::vector<MPI_Status> recvStatuses_;
        std::vector<int> recvDoneIdx_;
        std::vector<T *> recvDataPtrs_;
        std::vector<T> recvTempBuffer_;
        Int recvPostedCount_;
        Int recvCount_;

        std::vector<MPI_Request> sendRequests_;
        std::vector<MPI_Status> sendStatuses_;
        std::vector<int> sendDoneIdx_;
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

#ifdef COMM_PROFILE_BCAST
      protected:
        Int myGRoot_;
        Int myGRank_;
      public:
        inline void SetGlobalComm(const MPI_Comm & pGComm);
#endif

      protected:
        virtual void buildTree(Int * ranks, Int rank_cnt)=0;


      public:
        static TreeBcast_v2<T> * Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize,double rseed);
        TreeBcast_v2();
        TreeBcast_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt,Int msgSize);
        TreeBcast_v2(const TreeBcast_v2 & Tree);
        virtual ~TreeBcast_v2();
        virtual TreeBcast_v2 * clone() const = 0; 
        virtual void Copy(const TreeBcast_v2 & Tree);
        virtual void Reset();


        virtual inline Int GetNumMsgToRecv();
        virtual inline Int GetNumRecvMsg();
        virtual inline Int GetNumMsgToSend();
        virtual inline Int GetNumSendMsg();
        inline void SetDataReady(bool rdy);
        inline void SetTag(Int tag);
        inline int GetTag();
        Int * GetDests();
        Int GetDest(Int i);
        Int GetDestCount();
        Int GetRoot();
        bool IsRoot();
        bool StartForward();
        void SetMsgSize(Int msgSize){ this->msgSize_ = msgSize;}
        Int GetMsgSize();
        bool IsDone();
        bool IsReady(){ return this->isReady_;}


        virtual void SetLocalBuffer(T * locBuffer);
        virtual T * GetLocalBuffer();

        //async wait and forward
        virtual bool Progress();
		virtual void AllocateBuffer();															
        //blocking wait
        void Wait();


        virtual void cleanupBuffers();

		virtual void allocateRequest();
		virtual void forwardMessageSimple(T * locBuffer);	
		virtual void waitSendRequest();	

		
      protected:
        virtual void postRecv();
        virtual void forwardMessage();
        virtual void copyLocalBuffer(T* destBuffer);
        virtual bool isMessageForwarded();
        virtual bool IsDataReceived();
    };


  template< typename T>
    class FTreeBcast2: public TreeBcast_v2<T>{
      protected:
        virtual void buildTree(Int * ranks, Int rank_cnt);

      public:
        FTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
        virtual FTreeBcast2<T> * clone() const;
    };

  template< typename T>
    class BTreeBcast2: public TreeBcast_v2<T>{
      protected:
        virtual void buildTree(Int * ranks, Int rank_cnt);

      public:
        BTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
        virtual BTreeBcast2<T> * clone() const;
    };

  template< typename T>
    class ModBTreeBcast2: public TreeBcast_v2<T>{
      protected:
        double rseed_;
        virtual void buildTree(Int * ranks, Int rank_cnt);

      public:
        ModBTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed);
        virtual ModBTreeBcast2<T> * clone() const;
    };


  template< typename T>
  void TreeBcast_Waitsome(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeBcast_v2<T> > > & arrTrees, std::list<int> & doneIdx, std::vector<bool> & finishedFlags);

  template< typename T>
  void TreeBcast_Testsome(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeBcast_v2<T> > > & arrTrees, std::list<int> & doneIdx, std::vector<bool> & finishedFlags);

  template< typename T>
  void TreeBcast_Testsome(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeBcast_v2<T> > > & arrTrees, std::list<int> & doneIdx, std::vector<int> & finishedEpochs);

  template< typename T>
  void TreeBcast_Waitall(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeBcast_v2<T> > > & arrTrees);





}//namespace PEXSI

#include "TreeBcast_v2_impl.hpp"
#endif
