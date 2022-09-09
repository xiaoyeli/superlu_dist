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
		if(precision=='s'){
			TreeBcast_slu<float>* BcastTree = TreeBcast_slu<float>::Create(comm,ranks,rank_cnt,msgSize,rseed);
			return (BcTree) BcastTree;
		}
		if(precision=='z'){
			TreeBcast_slu<doublecomplex>* BcastTree = TreeBcast_slu<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);		
			return (BcTree) BcastTree;
		}
		return 0;
	}

	void BcTree_Destroy(BcTree Tree, char precision){
		if(precision=='d'){
			TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
			delete BcastTree; 
		}
		if(precision=='s'){
			TreeBcast_slu<float>* BcastTree = (TreeBcast_slu<float>*) Tree;
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
	    if(precision=='s'){
	        TreeBcast_slu<float>* BcastTree = (TreeBcast_slu<float>*) Tree;
		BcastTree->SetTag(tag); 
	    }
	    if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		BcastTree->SetTag(tag); 
	    }
	}


	yes_no_t BcTree_IsRoot(BcTree Tree, char precision){
		if(precision=='d'){
		  TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		  return BcastTree->IsRoot()?YES:NO;
		}
		if(precision=='s'){
		  TreeBcast_slu<float>* BcastTree = (TreeBcast_slu<float>*) Tree;
		  return BcastTree->IsRoot()?YES:NO;
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		return BcastTree->IsRoot()?YES:NO;
		}
		return NO;
	}

	
	void BcTree_forwardMessageSimple(BcTree Tree, void* localBuffer, Int msgSize, char precision){
	    if(precision=='d'){
	      TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
	      BcastTree->forwardMessageSimple((double*)localBuffer,msgSize);	
	    }
	    if(precision=='s'){
	      TreeBcast_slu<float>* BcastTree = (TreeBcast_slu<float>*) Tree;
	      BcastTree->forwardMessageSimple((float*)localBuffer,msgSize);	
	    }
	    if(precision=='z'){
	      TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
	      BcastTree->forwardMessageSimple((doublecomplex*)localBuffer,msgSize);	
	    }	
	}

	void BcTree_waitSendRequest(BcTree Tree, char precision) {
	  if(precision=='d'){
	    TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
	    BcastTree->waitSendRequest();		
	  }
	  if(precision=='s'){
	    TreeBcast_slu<float>* BcastTree = (TreeBcast_slu<float>*) Tree;
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
	  if(precision=='s'){
	    TreeBcast_slu<float>* BcastTree = (TreeBcast_slu<float>*) Tree;
	    BcastTree->allocateRequest();
	  }
	  if(precision=='z'){
	    TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
	    BcastTree->allocateRequest();
	  }			
	}	
	
	int BcTree_getDestCount(BcTree Tree, char precision){
	  if(precision=='d'){
	    TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
	    return BcastTree->GetDestCount();					
	  }
	  if(precision=='s'){
	    TreeBcast_slu<float>* BcastTree = (TreeBcast_slu<float>*) Tree;
	    return BcastTree->GetDestCount();					
	  }
	  if(precision=='z'){
	    TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
	    return BcastTree->GetDestCount();					
	  }
	  return 0;
	}	

	int BcTree_GetMsgSize(BcTree Tree, char precision){
	  if(precision=='d'){
	    TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
	    return BcastTree->GetMsgSize();					
	  }
	  if(precision=='s'){
	    TreeBcast_slu<float>* BcastTree = (TreeBcast_slu<float>*) Tree;
	    return BcastTree->GetMsgSize();					
	  }
	  if(precision=='z'){
	    TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
	    return BcastTree->GetMsgSize();					
	  }
	  return 0;
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
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = TreeReduce_slu<float>::Create(comm,ranks,rank_cnt,msgSize,rseed);
	    return (RdTree) ReduceTree;
	  }
	  if(precision=='z'){
	    TreeReduce_slu<doublecomplex>* ReduceTree = TreeReduce_slu<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);
	    return (RdTree) ReduceTree;
	  }
	  return 0;
	}
	
	void RdTree_Destroy(RdTree Tree, char precision){
	  if(precision=='d'){
	    TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
	    delete ReduceTree; 
	  }
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = (TreeReduce_slu<float>*) Tree;
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
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = (TreeReduce_slu<float>*) Tree;
	    ReduceTree->SetTag(tag); 		
	  }
	  if(precision=='z'){
	    TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
	    ReduceTree->SetTag(tag); 
	  }
	}

	int  RdTree_GetDestCount(RdTree Tree, char precision){
	  if(precision=='d'){
	    TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
	    return ReduceTree->GetDestCount();		
	  }
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = (TreeReduce_slu<float>*) Tree;
	    return ReduceTree->GetDestCount();		
	  }
	  if(precision=='z'){
	    TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
	    return ReduceTree->GetDestCount();		
	  }
	  return 0;
	}	
	
	int  RdTree_GetMsgSize(RdTree Tree, char precision){
	  if(precision=='d'){
	    TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
	    return ReduceTree->GetMsgSize();		
	  }
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = (TreeReduce_slu<float>*) Tree;
	    return ReduceTree->GetMsgSize();		
	  }
	  if(precision=='z'){
	    TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
	    return ReduceTree->GetMsgSize();		
	  }
	  return 0;
	}		
	
	yes_no_t RdTree_IsRoot(RdTree Tree, char precision){
	  if(precision=='d'){
	    TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
	    return ReduceTree->IsRoot()?YES:NO;
	  }
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = (TreeReduce_slu<float>*) Tree;
	    return ReduceTree->IsRoot()?YES:NO;
	  }
	  if(precision=='z'){
	    TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
	    return ReduceTree->IsRoot()?YES:NO;
	  }
	  return NO;
	}

	void RdTree_forwardMessageSimple(RdTree Tree, void* localBuffer, Int msgSize, char precision){
	  if(precision=='d'){
	    TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
	    ReduceTree->forwardMessageSimple((double*)localBuffer,msgSize);	
	  }
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = (TreeReduce_slu<float>*) Tree;
	    ReduceTree->forwardMessageSimple((float*)localBuffer,msgSize);	
	  }
	  if(precision=='z'){TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
	    ReduceTree->forwardMessageSimple((doublecomplex*)localBuffer,msgSize);	
	  }
	}

	void RdTree_allocateRequest(RdTree Tree, char precision){
	  if(precision=='d'){
	    TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
	    ReduceTree->allocateRequest();
	  }		
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = (TreeReduce_slu<float>*) Tree;
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
	  if(precision=='s'){
	    TreeReduce_slu<float>* ReduceTree = (TreeReduce_slu<float>*) Tree;
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

