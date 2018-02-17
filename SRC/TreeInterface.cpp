#include "TreeReduce_v2.hpp"
#include "dcomplex.h"

namespace ASYNCOMM{
	
	
	
#ifdef __cplusplus
	extern "C" {
#endif

	BcTree BcTree_Create(MPI_Comm comm, Int* ranks, Int rank_cnt, Int msgSize, double rseed, char precision){
		assert(msgSize>0);
		if(precision=='d'){
			TreeBcast_v2<double>* BcastTree = TreeBcast_v2<double>::Create(comm,ranks,rank_cnt,msgSize,rseed);
			return (BcTree) BcastTree;
		}
		if(precision=='z'){
			TreeBcast_v2<doublecomplex>* BcastTree = TreeBcast_v2<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);		
			return (BcTree) BcastTree;
		}
	}

	void BcTree_Destroy(BcTree Tree, char precision){
		if(precision=='d'){
			TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
			delete BcastTree; 
		}
		if(precision=='z'){
			TreeBcast_v2<doublecomplex>* BcastTree = (TreeBcast_v2<doublecomplex>*) Tree;
			delete BcastTree; 
		}

	}		
	
	void BcTree_SetTag(BcTree Tree, Int tag, char precision){
		if(precision=='d'){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->SetTag(tag); 
		}
		if(precision=='z'){
		TreeBcast_v2<doublecomplex>* BcastTree = (TreeBcast_v2<doublecomplex>*) Tree;
		BcastTree->SetTag(tag); 
		}
	}


	yes_no_t BcTree_IsRoot(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		return BcastTree->IsRoot()?YES:NO;
		}
		if(precision=='z'){
		TreeBcast_v2<doublecomplex>* BcastTree = (TreeBcast_v2<doublecomplex>*) Tree;
		return BcastTree->IsRoot()?YES:NO;
		}
	}

	
	void BcTree_forwardMessageSimple(BcTree Tree, void* localBuffer, Int msgSize, char precision){
		if(precision=='d'){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->forwardMessageSimple((double*)localBuffer,msgSize);	
		}
		if(precision=='z'){
		TreeBcast_v2<doublecomplex>* BcastTree = (TreeBcast_v2<doublecomplex>*) Tree;
		BcastTree->forwardMessageSimple((doublecomplex*)localBuffer,msgSize);	
		}	
	}

	void BcTree_waitSendRequest(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->waitSendRequest();		
		}
		if(precision=='z'){
		TreeBcast_v2<doublecomplex>* BcastTree = (TreeBcast_v2<doublecomplex>*) Tree;
		BcastTree->waitSendRequest();		
		}
	}
	
	
	
	void BcTree_allocateRequest(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->allocateRequest();
		}
		if(precision=='z'){
		TreeBcast_v2<doublecomplex>* BcastTree = (TreeBcast_v2<doublecomplex>*) Tree;
		BcastTree->allocateRequest();
		}			
	}	
	
	int BcTree_getDestCount(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		return BcastTree->GetDestCount();					
		}
		if(precision=='z'){
		TreeBcast_v2<doublecomplex>* BcastTree = (TreeBcast_v2<doublecomplex>*) Tree;
		return BcastTree->GetDestCount();					
		}
	}	

	int BcTree_GetMsgSize(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		return BcastTree->GetMsgSize();					
		}
		if(precision=='z'){
		TreeBcast_v2<doublecomplex>* BcastTree = (TreeBcast_v2<doublecomplex>*) Tree;
		return BcastTree->GetMsgSize();					
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
		TreeReduce_v2<double>* ReduceTree = TreeReduce_v2<double>::Create(comm,ranks,rank_cnt,msgSize,rseed);
		return (RdTree) ReduceTree;
		}
		if(precision=='z'){
		TreeReduce_v2<doublecomplex>* ReduceTree = TreeReduce_v2<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);
		return (RdTree) ReduceTree;
		}
	}
	
	void RdTree_Destroy(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		delete ReduceTree; 
		}
		if(precision=='z'){
		TreeReduce_v2<doublecomplex>* ReduceTree = (TreeReduce_v2<doublecomplex>*) Tree;
		delete ReduceTree;
		}		
	}	
	

	void RdTree_SetTag(RdTree Tree, Int tag, char precision){
		if(precision=='d'){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->SetTag(tag); 		
		}
		if(precision=='z'){
		TreeReduce_v2<doublecomplex>* ReduceTree = (TreeReduce_v2<doublecomplex>*) Tree;
		ReduceTree->SetTag(tag); 
		}
	}

	int  RdTree_GetDestCount(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		return ReduceTree->GetDestCount();		
		}
		if(precision=='z'){
		TreeReduce_v2<doublecomplex>* ReduceTree = (TreeReduce_v2<doublecomplex>*) Tree;
		return ReduceTree->GetDestCount();		
		}
	}	
	
	int  RdTree_GetMsgSize(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		return ReduceTree->GetMsgSize();		
		}
		if(precision=='z'){
		TreeReduce_v2<doublecomplex>* ReduceTree = (TreeReduce_v2<doublecomplex>*) Tree;
		return ReduceTree->GetMsgSize();		
		}
	}		
	
	

	yes_no_t RdTree_IsRoot(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		return ReduceTree->IsRoot()?YES:NO;
		}
		if(precision=='z'){
		TreeReduce_v2<doublecomplex>* ReduceTree = (TreeReduce_v2<doublecomplex>*) Tree;
		return ReduceTree->IsRoot()?YES:NO;
		}
	}


	void RdTree_forwardMessageSimple(RdTree Tree, void* localBuffer, Int msgSize, char precision){
		if(precision=='d'){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->forwardMessageSimple((double*)localBuffer,msgSize);	
		}
		if(precision=='z'){TreeReduce_v2<doublecomplex>* ReduceTree = (TreeReduce_v2<doublecomplex>*) Tree;
		ReduceTree->forwardMessageSimple((doublecomplex*)localBuffer,msgSize);	
		}
	}
	void RdTree_allocateRequest(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->allocateRequest();
		}		
		if(precision=='z'){
		TreeReduce_v2<doublecomplex>* ReduceTree = (TreeReduce_v2<doublecomplex>*) Tree;
		ReduceTree->allocateRequest();		
		}
		
	}

	void RdTree_waitSendRequest(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->waitSendRequest();			
		}
		if(precision=='z'){
		TreeReduce_v2<doublecomplex>* ReduceTree = (TreeReduce_v2<doublecomplex>*) Tree;
		ReduceTree->waitSendRequest();	
		}		
	}
	
	


#ifdef __cplusplus
	}
#endif
	
	
	



} //namespace ASYNCOMM

