#include "TreeReduce_slu.hpp"
#include "dcomplex.h"
#ifdef oneside
#include "oneside.h"
#include <math.h>
using namespace std;
#endif
#include "mpi.h"
namespace SuperLU_ASYNCOMM{
	
	
	
#ifdef __cplusplus
	extern "C" {
#endif

	BcTree BcTree_Create_oneside(MPI_Comm comm, Int* ranks, Int rank_cnt, Int msgSize, double rseed, char precision, int* BufSize, int Pc){
		assert(msgSize>0);
		if(precision=='d'){
			TreeBcast_slu<double>* BcastTree = TreeBcast_slu<double>::Create(comm,ranks,rank_cnt,msgSize,rseed);
 			//printf("Out my root is %d/%d\n",BcastTree->GetRoot(),BcastTree->GetRoot()/Pc);
 			BufSize[BcastTree->GetRoot()/Pc] += 1;
			return (BcTree) BcastTree;
		}
		if(precision=='z'){
			TreeBcast_slu<doublecomplex>* BcastTree = TreeBcast_slu<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);		
 			BufSize[BcastTree->GetRoot()/Pc] += 1;
			return (BcTree) BcastTree;
		}
	}

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


	yes_no_t BcTree_IsRoot(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		return BcastTree->IsRoot()?YES:NO;
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		return BcastTree->IsRoot()?YES:NO;
		}
	}

	
#ifdef oneside
	void BcTree_forwardMessageOneSide(BcTree Tree, void* localBuffer, Int msgSize, char precision, int* iam_col, int* BCcount, long* BCbase, int* maxrecvsz, int Pc){
		if(precision=='d'){
			TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
            double *sendbuf = (double*) localBuffer;
            double *sendbufval;
            double checksum = 0.0;
            if ( !(sendbufval = (double*)SUPERLU_MALLOC( (msgSize+1) * sizeof(double))) )
                    ABORT("Malloc fails for sendbuf[]");
            for(Int i = 0; i<msgSize;++i){
                if(std::isnan(sendbuf[i])) {
                    sendbufval[i] = sendbuf[i];
                    //printf("isnan,sendbuf=%lf,ori=%lf",sendbufval[i],sendbuf[i]);
                    //fflush(stdout);
                    continue;
                }
                    sendbufval[i] = sendbuf[i];
                    //printf("sendbuf=%lf,ori=%lf",sendbufval[i],sendbuf[i]);
                    //fflush(stdout);
                    checksum += sendbuf[i];
            }
            sendbufval[msgSize] = checksum;
            //printf("\n HERE!!! send=%lf,%lf,loc=%lf\n",sendbufval[0],sendbufval[msgSize],checksum);
            //fflush(stdout);
			////msgSize += 1;
            BcastTree->forwardMessageOneSide(sendbufval,msgSize, iam_col, BCcount, BCbase, maxrecvsz, Pc);	
            //printf("END HERE!!! send=%lf,%lf,loc=%lf\n",sendbufval[0],sendbufval[msgSize],checksum);
            //fflush(stdout);
			//BcastTree->forwardMessageOneSide((double*)localBuffer,msgSize, iam_col, BCcount, BCbase, maxrecvsz, Pc);	
		}
		if(precision=='z'){
			TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
			BcastTree->forwardMessageOneSide((doublecomplex*)localBuffer,msgSize, iam_col, BCcount, BCbase, maxrecvsz, Pc);	
		}	
	}
	void RdTree_forwardMessageOneSide(RdTree Tree, void* localBuffer, Int msgSize, char precision, int* iam_row, int* RDcount, long* RDbase, int* maxrecvsz, int Pc){
		if(precision=='d'){
		        TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
                double *sendbuf = (double*) localBuffer;
                double *sendbufval;
                double checksum = 0.0;
                if ( !(sendbufval = (double*)SUPERLU_MALLOC( (msgSize+1) * sizeof(double))) )
                        ABORT("Malloc fails for sendbuf[]");
                for(Int i = 0; i<msgSize;++i){
                    sendbufval[i] = sendbuf[i];
                    if(std::isnan(sendbuf[i])) {
                        //printf("isnan,sendbuf=%lf,ori=%lf",sendbufval[i],sendbuf[i]);
                        //fflush(stdout);
                        continue;
                    }
                        //printf("sendbuf=%lf,ori=%lf",sendbufval[i],sendbuf[i]);
                        //fflush(stdout);
                    checksum += sendbuf[i];
                }
                sendbufval[msgSize] = checksum;
                //printf("\n HERE!!! send=%lf,%lf,loc=%lf\n",sendbufval[0],sendbufval[msgSize],checksum);
                //fflush(stdout);
			    //msgSize += 1;
		        ReduceTree->forwardMessageOneSide(sendbufval, msgSize, iam_row, RDcount, RDbase, maxrecvsz, Pc);	
		        //ReduceTree->forwardMessageOneSide((double*)localBuffer,msgSize, iam_row, RDcount, RDbase, maxrecvsz, Pc);	
		}
		if(precision=='z'){
		    TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		    ReduceTree->forwardMessageOneSide((doublecomplex*)localBuffer,msgSize, iam_row, RDcount, RDbase, maxrecvsz, Pc);	
		}
	}
#endif
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
	
	int BcTree_getDestCount(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		return BcastTree->GetDestCount();					
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
		return BcastTree->GetDestCount();					
		}
	}	

	int BcTree_GetMsgSize(BcTree Tree, char precision){
		if(precision=='d'){
		TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
		return BcastTree->GetMsgSize();					
		}
		if(precision=='z'){
		TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
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
	
	RdTree RdTree_Create_oneside(MPI_Comm comm, Int* ranks, Int rank_cnt, Int msgSize, double rseed, char precision, int* BufSize_rd, int Pc){
		assert(msgSize>0);
		if(precision=='d'){
		        TreeReduce_slu<double>* ReduceTree = TreeReduce_slu<double>::Create(comm,ranks,rank_cnt,msgSize,rseed);
                        //int dn_rank;
                        //MPI_Comm_rank(MPI_COMM_WORLD,&dn_rank);
                        for(Int i=0;i<ReduceTree->GetDestCount();++i){
 			   //     printf("Total=%d, Pc=%d,iam=%d, my root is %d/%d\n",ReduceTree->GetDestCount(), Pc, dn_rank,ReduceTree->GetDest(i),ReduceTree->GetDest(i)%Pc);
                           //     fflush(stdout);
                                BufSize_rd[ReduceTree->GetDest(i)%Pc] += 1;
		        }
                        
                        return (RdTree) ReduceTree;
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = TreeReduce_slu<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);
		return (RdTree) ReduceTree;
		}
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

	int  RdTree_GetDestCount(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		return ReduceTree->GetDestCount();		
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		return ReduceTree->GetDestCount();		
		}
	}	
	
	int  RdTree_GetMsgSize(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		return ReduceTree->GetMsgSize();		
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		return ReduceTree->GetMsgSize();		
		}
	}		
	
	

	yes_no_t RdTree_IsRoot(RdTree Tree, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		return ReduceTree->IsRoot()?YES:NO;
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		return ReduceTree->IsRoot()?YES:NO;
		}
	}
	void RdTree_forwardMessageSimple(RdTree Tree, void* localBuffer, Int msgSize, char precision){
		if(precision=='d'){
		TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
		ReduceTree->forwardMessageSimple((double*)localBuffer,msgSize);	
		}
		if(precision=='z'){
		TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		ReduceTree->forwardMessageSimple((doublecomplex*)localBuffer,msgSize);	
		}
	}
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

