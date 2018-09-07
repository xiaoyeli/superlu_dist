#ifndef __SUPERLU_TREEREDUCE
#define __SUPERLU_TREEREDUCE

// #include "asyncomm.hpp"
// #include "timer.h"
#include "TreeBcast_slu.hpp"

#include <vector>
#include <map>
#include <algorithm>
#include <string>
//#include <random>


namespace SuperLU_ASYNCOMM{

  template< typename T>
    class TreeReduce_slu: public TreeBcast_slu<T>{
      protected:
        bool isAllocated_;
        bool isBufferSet_;

      public:
        static TreeReduce_slu<T> * Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize,double rseed);

        TreeReduce_slu();
        TreeReduce_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
        TreeReduce_slu(const TreeReduce_slu & Tree);

        virtual ~TreeReduce_slu();
        virtual TreeReduce_slu * clone() const = 0; 
        virtual void Copy(const TreeReduce_slu & Tree);
        virtual void Reset();


        bool IsAllocated(){return this->isAllocated_;}
        virtual inline Int GetNumMsgToSend(){return this->myRank_==this->myRoot_?0:1;}
        virtual inline Int GetNumMsgToRecv(){return this->GetDestCount();}

        virtual T * GetLocalBuffer();


		
		virtual void forwardMessageSimple(T * locBuffer, Int msgSize);
		virtual void allocateRequest();	
		virtual void waitSendRequest();
    };


template< typename T>
class FTreeReduce_slu: public TreeReduce_slu<T>{
protected:
  virtual void buildTree(Int * ranks, Int rank_cnt);
public:
  FTreeReduce_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
  virtual FTreeReduce_slu<T> * clone() const;
};



template< typename T>
class BTreeReduce_slu: public TreeReduce_slu<T>{
protected:
  virtual void buildTree(Int * ranks, Int rank_cnt);

public:
  BTreeReduce_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);


  virtual BTreeReduce_slu<T> * clone() const;



};


template< typename T>
class ModBTreeReduce_slu: public TreeReduce_slu<T>{
protected:
  double rseed_;
  virtual void buildTree(Int * ranks, Int rank_cnt);
  
public:
  ModBTreeReduce_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed);
  virtual void Copy(const ModBTreeReduce_slu<T> & Tree);
  virtual ModBTreeReduce_slu<T> * clone() const;

};

}//namespace SuperLU_ASYNCOMM

#include "TreeReduce_slu_impl.hpp"

#endif // __SUPERLU_TREEREDUCE

