#include "TreeReduce_v2.hpp"


namespace PEXSI{
	
	
	
#ifdef __cplusplus
	extern "C" {
#endif

	void TreeTest(void *tree) {
		std::cout<<" ahhh good! "<<std::endl;
	}


	BcTree BcTree_Create(MPI_Comm comm, Int* ranks, Int rank_cnt, Int msgSize, double rseed){
		TreeBcast_v2<double>* BcastTree = TreeBcast_v2<double>::Create(comm,ranks,rank_cnt,msgSize,rseed);
		return (BcTree) BcastTree;
	}

	void BcTree_SetTag(BcTree Tree, Int tag){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->SetTag(tag); 
	}

	yes_no_t BcTree_Progress(BcTree Tree){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		bool done = BcastTree->Progress();
		// std::cout<<done<<std::endl;
		return done ? YES : NO;	
	}

	void BcTree_AllocateBuffer(BcTree Tree){
		TreeBcast_v2<double>* BcastLTree = (TreeBcast_v2<double>*) Tree;
		BcastLTree->AllocateBuffer(); 
	}
	
	
	// Int BcTree_Iprobe(BcTree Tree, MPI_Status* status){
		// TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		// Int flag;	

		// MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, BcastTree->comm_, &flag, status);
		// if(flag!=0){
		// printf("hahah %5d", flag);		
		// fflush(stdout);
		// }
		// return flag;
	// }


	void BcTree_SetDataReady(BcTree Tree){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->SetDataReady(true);
	}

	void BcTree_SetLocalBuffer(BcTree Tree, void* localBuffer){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->SetLocalBuffer( (double*) localBuffer);
	}

	yes_no_t BcTree_IsRoot(BcTree Tree){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		return BcastTree->IsRoot()?YES:NO;
	}

	yes_no_t BcTree_StartForward(BcTree Tree){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		return BcastTree->StartForward()?YES:NO;
	}



	void BcTree_Testsome(StdList TreeIdx, BcTree* ArrTrees, int* Outcount, int* FinishedTrees){
		 std::list<int>* treeIdx = (std::list<int>*)TreeIdx;
		 int i=0, idone=0;
		 bool done;
		 TreeBcast_v2<double>*  curTree;
		
		  for (std::list<int>::iterator itr = (*treeIdx).begin(); itr != (*treeIdx).end(); /*nothing*/){
			curTree = (TreeBcast_v2<double>*) ArrTrees[*itr];
			assert(curTree!=nullptr);
			done = curTree->Progress();
			if(done){
				FinishedTrees[idone] = *itr; /*store finished tree numbers */
				++idone;
				itr = (*treeIdx).erase(itr);			
			}else{
				++itr;	
			}
			++i;
		  }	
		  *Outcount = idone;
	}

	
	void BcTree_forwardMessageSimple(BcTree Tree, void* localBuffer){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->forwardMessageSimple((double*)localBuffer);		
	}

	void BcTree_waitSendRequest(BcTree Tree){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->waitSendRequest();		
	}
	
	
	
	void BcTree_allocateRequest(BcTree Tree){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		BcastTree->allocateRequest();			
	}	
	
	int BcTree_getDestCount(BcTree Tree){
		TreeBcast_v2<double>* BcastTree = (TreeBcast_v2<double>*) Tree;
		return BcastTree->GetDestCount();			
	}	


	StdList StdList_Init(){
		std::list<int>* lst = new std::list<int>();
		return (StdList) lst;
	}
	void StdList_Pushback(StdList lst, int dat){
		std::list<int>* list = (std::list<int>*) lst;
		list->push_back(dat);
	}

	yes_no_t StdList_Find(StdList lst, int dat){
		std::list<int>* list = (std::list<int>*) lst;
		for (std::list<int>::iterator itr = (*list).begin(); itr != (*list).end(); /*nothing*/){
			if(*itr==dat)return YES;
			++itr;
		}
		return NO;
	}

	int StdList_Size(StdList lst){
		std::list<int>* list = (std::list<int>*) lst;
		return list->size();
	}	


	RdTree RdTree_Create(MPI_Comm comm, Int* ranks, Int rank_cnt, Int msgSize, double rseed){
		TreeReduce_v2<double>* ReduceTree = TreeReduce_v2<double>::Create(comm,ranks,rank_cnt,msgSize,rseed);
		return (RdTree) ReduceTree;
	}

	void RdTree_SetTag(RdTree Tree, Int tag){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->SetTag(tag); 
	}
	
	int  RdTree_GetTag(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		return ReduceTree->GetTag();
	}
	

	int  RdTree_GetDestCount(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		return ReduceTree->GetDestCount();
	}	
	

	yes_no_t RdTree_Progress(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		bool done = ReduceTree->Progress();
		// std::cout<<done<<std::endl;
		return done ? YES : NO;	
	}
	
	// void RdTree_PostRecv(RdTree Tree){
		// TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		// ReduceTree->postRecv();
	// }	


	void RdTree_SetDataReady(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->SetDataReady(true);
	}

	
	void RdTree_AllocRecvBuffers(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->AllocRecvBuffers();		
	}	
	
	
	void RdTree_SetLocalBuffer(RdTree Tree, void* localBuffer){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->SetLocalBuffer( (double*) localBuffer);
	}

	yes_no_t RdTree_IsRoot(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		return ReduceTree->IsRoot()?YES:NO;
	}

	
	yes_no_t RdTree_IsReady(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		return ReduceTree->IsReady()?YES:NO;
	}	
	
	
	yes_no_t RdTree_StartForward(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		return ReduceTree->StartForward()?YES:NO;
	}



	void RdTree_Testsome(StdList TreeIdx, RdTree* ArrTrees, int* Outcount, int* FinishedTrees){
		 std::list<int>* treeIdx = (std::list<int>*)TreeIdx;
		 int i=0, idone=0;
		 bool done;
		 TreeReduce_v2<double>*  curTree;
		
		  for (std::list<int>::iterator itr = (*treeIdx).begin(); itr != (*treeIdx).end(); /*nothing*/){
			curTree = (TreeReduce_v2<double>*) ArrTrees[*itr];
			assert(curTree!=nullptr);
			done = curTree->Progress();
			
			// if(*itr==9977){
				// std::cout<<"still good"<<curTree->IsRoot()<<std::endl;
				
			// }			
			
			if(done){
				FinishedTrees[idone] = *itr; /*store finished tree numbers */
				++idone;
				itr = (*treeIdx).erase(itr);			
			}else{
				++itr;	
			}
			++i;
		  }	
		  *Outcount = idone;
	}
	

	void RdTree_forwardMessageSimple(RdTree Tree, void* localBuffer){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->forwardMessageSimple((double*)localBuffer);		
	}
	void RdTree_allocateRequest(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->allocateRequest();			
	}

	void RdTree_waitSendRequest(RdTree Tree){
		TreeReduce_v2<double>* ReduceTree = (TreeReduce_v2<double>*) Tree;
		ReduceTree->waitSendRequest();		
	}
	
	


#ifdef __cplusplus
	}
#endif
	
	
	



} //namespace PEXSI

