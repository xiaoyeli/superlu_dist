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

#ifdef oneside        
    uint32_t update_crc_32(uint32_t crc_32_val, unsigned char c);
    uint16_t update_crc_16(uint16_t crc_16_val, unsigned char c);
    uint8_t crc_8(unsigned char * crc_8_val,size_t b);
    uint16_t crc_16(unsigned char * crc_16_val,size_t b);
    uint32_t crc_32(unsigned char * crc_32_val,size_t b);

    //unsigned int calcul_hash(const void* buffer, size_t length)
    //{
    //    unsigned int const seed = 0;   /* or any other value */
    //    unsigned int const hash = XXH32(buffer, length, seed);
    //    return hash;
    //}
    //unsigned long long calcul_hash(const void* buffer, size_t length)
    //{
    //        unsigned long long const seed = 0;   /* or any other value */
    //        unsigned long long const hash = XXH64(buffer, length, seed);
    //        return hash;
    //}
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
	void BcTree_forwardMessageOneSide(BcTree Tree, double* localBuffer, Int msgSize, char precision, int* iam_col, int* BCcount, long* BCbase, int* maxrecvsz, int Pc){
		if(precision=='d'){
			TreeBcast_slu<double>* BcastTree = (TreeBcast_slu<double>*) Tree;
	 		//double t1;
            //t1 = SuperLU_timer_();
            //printf("k=%lf,sum=%lf\n", localBuffer[0], localBuffer[XK_H-1]);
            //fflush(stdout);
            
            //localBuffer[XK_H-1] = calcul_hash(&localBuffer[XK_H],sizeof(double)*(msgSize-XK_H));
            //localBuffer[XK_H-1] = localBuffer[msgSize-1];//calcul_hash(&localBuffer[XK_H],sizeof(double)*(msgSize-XK_H));
            //if (localBuffer[XK_H-1] == calcul_hash(&localBuffer[XK_H],sizeof(double)*(msgSize-XK_H))){
            //
            //printf("EUQAL!!!! k=%lf,size=%d,sum=%lf, realwum=%llu\n",localBuffer[0],msgSize-XK_H, localBuffer[XK_H-1], calcul_hash(&localBuffer[XK_H],sizeof(double)*(msgSize-XK_H)));
            //fflush(stdout);
            //}
            localBuffer[XK_H-1] = crc_16((unsigned char*)&localBuffer[XK_H],sizeof(double)*(msgSize-XK_H));
            BcastTree->forwardMessageOneSide((double*)localBuffer,msgSize, iam_col, BCcount, BCbase, maxrecvsz, Pc);	
	        //onesidecomm_bc += SuperLU_timer_() - t1;
		}
		if(precision=='z'){
            localBuffer[XK_H*2-2] = crc_8((unsigned char*)&localBuffer[XK_H*2],sizeof(double)*2*(msgSize-XK_H));
            //int iam;
            //MPI_Comm_rank(MPI_COMM_WORLD, &iam);
            //
            //printf("iam=%d, In Oneside interface k=%lf, checksum start at %d, size %d, checksum=%lf\n",iam, localBuffer[0],XK_H*2, sizeof(double)*2*(msgSize-XK_H),localBuffer[XK_H*2-2]);
            //fflush(stdout);
            //
            TreeBcast_slu<doublecomplex>* BcastTree = (TreeBcast_slu<doublecomplex>*) Tree;
            //for(int i=0; i<msgSize*2; i++){
            //    printf("iam=%d, k=%lf, val[%d]=%lf\n",iam, localBuffer[0], i,localBuffer[i]);
            //    fflush(stdout);
            //}
            BcastTree->forwardMessageOneSideU((doublecomplex*)localBuffer,msgSize, iam_col, BCcount, BCbase, maxrecvsz, Pc);	
            //printf("iam=%d, End In Oneside interface k=%lf, checksum=%lf\n",iam,localBuffer[0],localBuffer[XK_H*2-2]);
            //fflush(stdout);
		}	
	}

	
    void RdTree_forwardMessageOneSide(RdTree Tree, double* localBuffer, Int msgSize, char precision, int* iam_row, int* RDcount, long* RDbase, int* maxrecvsz, int Pc){
		if(precision=='d'){
		        TreeReduce_slu<double>* ReduceTree = (TreeReduce_slu<double>*) Tree;
                //////printf("\n HERE!!! send=%lf,%lf,loc=%lf\n",sendbufval[0],sendbufval[msgSize],checksum);
                //////fflush(stdout);
	 		    //double t1;
                //t1 = SuperLU_timer_();
                //printf("k=%lf,sum=%lf\n", localBuffer[0], localBuffer[LSUM_H-1]);
                //fflush(stdout);
                //localBuffer[LSUM_H-1]=localBuffer[msgSize-1] ;//calcul_hash(&localBuffer[LSUM_H],sizeof(double)*(msgSize-LSUM_H));
                //localBuffer[LSUM_H-1]=calcul_hash(&localBuffer[LSUM_H],sizeof(double)*(msgSize-LSUM_H));
                localBuffer[LSUM_H-1]=crc_16((unsigned char*)&localBuffer[LSUM_H],sizeof(double)*(msgSize-LSUM_H));
		        ReduceTree->forwardMessageOneSide((double*)localBuffer, msgSize, iam_row, RDcount, RDbase, maxrecvsz, Pc);	
		        //onesidecomm_bc += SuperLU_timer_() - t1;
        }
		if(precision=='z'){
            localBuffer[LSUM_H*2-2] = crc_8((unsigned char*)&localBuffer[LSUM_H*2],sizeof(double)*2*(msgSize-LSUM_H));
            //localBuffer[LSUM_H*2-2] = crc_8((unsigned char*)&localBuffer[LSUM_H*2],sizeof(double)*2*(msgSize-LSUM_H));
		    TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
		    ReduceTree->forwardMessageOneSideU((doublecomplex*)localBuffer,msgSize, iam_row, RDcount, RDbase, maxrecvsz, Pc);	
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
 			        //printf("Total=%d, Pc=%d,iam=%d, my root is %d/%d\n",ReduceTree->GetDestCount(), Pc, dn_rank,ReduceTree->GetDest(i),ReduceTree->GetDest(i)%Pc);
                    //fflush(stdout);
                    BufSize_rd[ReduceTree->GetDest(i)%Pc] += 1;
		        }
                        
                        return (RdTree) ReduceTree;
		}
		if(precision=='z'){
		    TreeReduce_slu<doublecomplex>* ReduceTree = TreeReduce_slu<doublecomplex>::Create(comm,ranks,rank_cnt,msgSize,rseed);
            //int dn_rank;
            //MPI_Comm_rank(MPI_COMM_WORLD,&dn_rank);
            for(Int i=0;i<ReduceTree->GetDestCount();++i){
 		        //printf("Total=%d, Pc=%d,iam=%d, my root is %d/%d\n",ReduceTree->GetDestCount(), Pc, dn_rank,ReduceTree->GetDest(i),ReduceTree->GetDest(i)%Pc);
                //fflush(stdout);
                BufSize_rd[ReduceTree->GetDest(i)%Pc] += 1;
		    }
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

