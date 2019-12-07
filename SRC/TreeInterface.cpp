#include "TreeReduce_slu.hpp"
#include "dcomplex.h"

namespace SuperLU_ASYNCOMM{
	
	
	
#ifdef __cplusplus
	extern "C" {
#endif

	BcTree BcTree_Create(MPI_Comm comm, Int* ranks, Int rank_cnt, Int msgSize, double rseed, char precision){
		assert(msgSize>0);
		if(precision=='d'){
			TreeBcast_slu<double>* BcastTree = TreeBcast_slu<double>::Create(comm,ranks,rank_cnt,msgSize,rseed);
			return (BcTree) BcastTree;
		}
		if(precision=='z'){
			TreeBcast_slu<doublecomplex>* BcastTree = TreeBcast_slu<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);		
			return (BcTree) BcastTree;
		}
	}

	void BcTree_Destroy(BcTree Tree, char precision){
		if(precision=='d'){
			TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
			delete BcastTree; 
		}
		if(precision=='z'){
			TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
			delete BcastTree; 
		}

	}		
	
	void BcTree_SetTag(BcTree Tree, Int tag, char precision){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		BcastTree->SetTag(tag); 
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		BcastTree->SetTag(tag); 
		}
	}


	void BcTree_IsRoot(BcTree Tree, char precision, yes_no_t* rel){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		*rel=BcastTree->myRank_ == BcastTree->myRoot_ ?YES:NO;
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		*rel=BcastTree->myRank_ == BcastTree->myRoot_ ?YES:NO;
		}
	}

	
		void BcTree_forwardMessageSimple(BcTree Tree, void* localBuffer, Int msgSize, char precision){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		BcastTree->forwardMessageSimple((double*)localBuffer,msgSize);	
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		BcastTree->forwardMessageSimple((doublecomplex*)localBuffer,msgSize);	
		}	
	}

#ifdef GPU_ACC	

__CUDA__	void BcTree_IsRoot_Device(BcTree Tree, char precision, yes_no_t* rel){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		*rel=BcastTree->myRank_ == BcastTree->myRoot_ ?YES:NO;
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		*rel=BcastTree->myRank_ == BcastTree->myRoot_ ?YES:NO;
		}
	}

__CUDA__	void BcTree_forwardMessageSimple_Device(BcTree Tree, void* localBuffer, Int msgSize, char precision){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		BcastTree->forwardMessageSimpleDevice((double*)localBuffer,msgSize);	
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		BcastTree->forwardMessageSimpleDevice((doublecomplex*)localBuffer,msgSize);	
		}	
	}	
#endif	
	

	void BcTree_waitSendRequest(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		BcastTree->waitSendRequest();		
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		BcastTree->waitSendRequest();		
		}
	}
	
	
	
	void BcTree_allocateRequest(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		BcastTree->allocateRequest();
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		BcastTree->allocateRequest();
		}			
	}	
	
	void BcTree_getDestCount(BcTree Tree, char precision, int* cnt){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		*cnt=BcastTree->myDestsSize_;					
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		*cnt=BcastTree->myDestsSize_;			
		}
	}	

	void BcTree_GetMsgSize(BcTree Tree, char precision, int* cnt){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		*cnt=BcastTree->msgSize_;					
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		*cnt=BcastTree->msgSize_;					
		}
	}		
	

	StdList StdList_Init(){
		std::list<int_t>* lst = new std::list<int_t>();
		return (StdList) lst;
	}
	void StdList_Pushback(StdList lst, int_t dat){
		std::list<int_t>* list = (std::list<int_t>*) lst;
		list->push_back(dat);
	}
	
	void StdList_Pushfront(StdList lst, int_t dat){
		std::list<int_t>* list = (std::list<int_t>*) lst;
		list->push_front(dat);
	}
	
	int_t StdList_Popfront(StdList lst){
		std::list<int_t>* list = (std::list<int_t>*) lst;
		int_t dat = -1;
		if((*list).begin()!=(*list).end()){
			dat = (*list).front();
			list->pop_front();
		}
		return dat;		
	}	
	
	yes_no_t StdList_Find(StdList lst, int_t dat){
		std::list<int_t>* list = (std::list<int_t>*) lst;
		for (std::list<int_t>::iterator itr = (*list).begin(); itr != (*list).end(); /*nothing*/){
			if(*itr==dat)return YES;
			++itr;
		}
		return NO;
	}

	int_t StdList_Size(StdList lst){
		std::list<int_t>* list = (std::list<int_t>*) lst;
		return list->size();
	}	


	yes_no_t StdList_Empty(StdList lst){
		std::list<int_t>* list = (std::list<int_t>*) lst;
		return (*list).begin()==(*list).end()?YES:NO;
	}		
	

	RdTree RdTree_Create(MPI_Comm comm, Int* ranks, Int rank_cnt, Int msgSize, double rseed, char precision){
		assert(msgSize>0);
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = TreeReduce_slu<double>::Create(comm,ranks,rank_cnt,msgSize,rseed);
		return (RdTree) ReduceTree;
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = TreeReduce_slu<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);
		return (RdTree) ReduceTree;
		}
	}
	
	void RdTree_Destroy(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		delete ReduceTree; 
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		delete ReduceTree;
		}		
	}	
	

	void RdTree_SetTag(RdTree Tree, Int tag, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		ReduceTree->SetTag(tag); 		
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		ReduceTree->SetTag(tag); 
		}
	}

	void  RdTree_GetDestCount(RdTree Tree, char precision, int* cnt){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		*cnt=ReduceTree->myDestsSize_;	
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		*cnt=ReduceTree->myDestsSize_;
		}
	}	
	
	void  RdTree_GetMsgSize(RdTree Tree, char precision, int* cnt){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		*cnt=ReduceTree->msgSize_;		
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		*cnt=ReduceTree->msgSize_;		
		}
	}		
	
	

	void RdTree_IsRoot(RdTree Tree, char precision, yes_no_t *rel){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		*rel=ReduceTree->myRank_ == ReduceTree->myRoot_ ?YES:NO;
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		*rel=ReduceTree->myRank_ == ReduceTree->myRoot_ ?YES:NO;
		}
	}


	void RdTree_forwardMessageSimple(RdTree Tree, void* localBuffer, Int msgSize, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		ReduceTree->forwardMessageSimple((double*)localBuffer,msgSize);	
		}
		if(precision=='z'){TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		ReduceTree->forwardMessageSimple((doublecomplex*)localBuffer,msgSize);	
		}
	}
	
#ifdef GPU_ACC	



__CUDA__	void  RdTree_GetDestCount_Device(RdTree Tree, char precision, int* cnt){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		*cnt=ReduceTree->myDestsSize_;	
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		*cnt=ReduceTree->myDestsSize_;
		}
	}	
	
__CUDA__	void  RdTree_GetMsgSize_Device(RdTree Tree, char precision, int* cnt){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		*cnt=ReduceTree->msgSize_;		
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		*cnt=ReduceTree->msgSize_;		
		}
	}	

__CUDA__	void BcTree_GetMsgSize_Device(BcTree Tree, char precision, int* cnt){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		*cnt=BcastTree->msgSize_;					
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		*cnt=BcastTree->msgSize_;					
		}
	}		
	
__CUDA__ void BcTree_getDestCount_Device(BcTree Tree, char precision, int* cnt){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		*cnt=BcastTree->myDestsSize_;					
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		*cnt=BcastTree->myDestsSize_;			
		}
	}	

__CUDA__ void RdTree_forwardMessageSimple_Device(RdTree Tree, void* localBuffer, Int msgSize, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		ReduceTree->forwardMessageSimpleDevice((double*)localBuffer,msgSize);	
		}
		if(precision=='z'){TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		ReduceTree->forwardMessageSimpleDevice((doublecomplex*)localBuffer,msgSize);	
		}
	}

__CUDA__ void RdTree_IsRoot_Device(RdTree Tree, char precision, yes_no_t *rel){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		*rel=ReduceTree->myRank_ == ReduceTree->myRoot_ ?YES:NO;
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		*rel=ReduceTree->myRank_ == ReduceTree->myRoot_ ?YES:NO;
		}
	}
	
#endif	
	
	
	void RdTree_allocateRequest(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		ReduceTree->allocateRequest();
		}		
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		ReduceTree->allocateRequest();		
		}
		
	}

	void RdTree_waitSendRequest(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		ReduceTree->waitSendRequest();			
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		ReduceTree->waitSendRequest();	
		}		
	}
	
#ifdef __cplusplus
	}
#endif
	
} //namespace SuperLU_ASYNCOMM

