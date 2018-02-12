#ifndef _PEXSI_REDUCE_TREE_V2_HPP_
#define _PEXSI_REDUCE_TREE_V2_HPP_

#include "environment.hpp"
// #include "timer.h"
#include "TreeBcast_v2.hpp"

#include <vector>
#include <map>
#include <algorithm>
#include <string>
//#include <random>



namespace ASYNCOMM{



  template< typename T>
    class TreeReduce_v2: public TreeBcast_v2<T>{
      protected:
        bool isAllocated_;
        bool isBufferSet_;

      public:
        static TreeReduce_v2<T> * Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize,double rseed);

        TreeReduce_v2();
        TreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
        TreeReduce_v2(const TreeReduce_v2 & Tree);

        virtual ~TreeReduce_v2();
        virtual TreeReduce_v2 * clone() const = 0; 
        virtual void Copy(const TreeReduce_v2 & Tree);
        virtual void Reset();


        bool IsAllocated(){return this->isAllocated_;}
        virtual inline Int GetNumMsgToSend(){return this->myRank_==this->myRoot_?0:1;}
        virtual inline Int GetNumMsgToRecv(){return this->GetDestCount();}

        virtual T * GetLocalBuffer();


		
		virtual void forwardMessageSimple(T * locBuffer);
		virtual void allocateRequest();	
		virtual void waitSendRequest();
    };


template< typename T>
class FTreeReduce_v2: public TreeReduce_v2<T>{
protected:
  virtual void buildTree(Int * ranks, Int rank_cnt);
public:
  FTreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
  virtual FTreeReduce_v2<T> * clone() const;
};



template< typename T>
class BTreeReduce_v2: public TreeReduce_v2<T>{
protected:
  virtual void buildTree(Int * ranks, Int rank_cnt);

public:
  BTreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);


  virtual BTreeReduce_v2<T> * clone() const;



};


template< typename T>
class ModBTreeReduce_v2: public TreeReduce_v2<T>{
protected:
  double rseed_;
  virtual void buildTree(Int * ranks, Int rank_cnt);
  
public:
  ModBTreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed);
  virtual void Copy(const ModBTreeReduce_v2<T> & Tree);
  virtual ModBTreeReduce_v2<T> * clone() const;

};


template< typename T>
class PalmTreeReduce_v2: public TreeReduce_v2<T>{
protected:

  virtual void buildTree(Int * ranks, Int rank_cnt);


public:
  PalmTreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize);
  virtual PalmTreeReduce_v2<T> * clone() const;




};


}//namespace ASYNCOMM

#include "TreeReduce_v2_impl.hpp"
#endif
