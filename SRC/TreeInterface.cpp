#include "TreeReduce_slu.hpp"
#include "dcomplex.h"

#ifdef onesided
#include "onesided.h"
#include <math.h>
using namespace std;

#endif

namespace SuperLU_ASYNCOMM{
	
	
	
#ifdef __cplusplus
	extern "C" {
#endif

#ifdef onesided
#define         CRC_POLY_16             0xA001
#define         CRC_START_16            0x0000
#define         CRC_START_8             0x00
int             crc_tab16_init          = 0;
uint16_t         crc_tab16[256];
uint8_t sht75_crc_table[256] = {

    0,   49,  98,  83,  196, 245, 166, 151, 185, 136, 219, 234, 125, 76,  31,  46,
    67,  114, 33,  16,  135, 182, 229, 212, 250, 203, 152, 169, 62,  15,  92,  109,
    134, 183, 228, 213, 66,  115, 32,  17,  63,  14,  93,  108, 251, 202, 153, 168,
    197, 244, 167, 150, 1,   48,  99,  82,  124, 77,  30,  47,  184, 137, 218, 235,
    61,  12,  95,  110, 249, 200, 155, 170, 132, 181, 230, 215, 64,  113, 34,  19,
    126, 79,  28,  45,  186, 139, 216, 233, 199, 246, 165, 148, 3,   50,  97,  80,
    187, 138, 217, 232, 127, 78,  29,  44,  2,   51,  96,  81,  198, 247, 164, 149,
    248, 201, 154, 171, 60,  13,  94,  111, 65,  112, 35,  18,  133, 180, 231, 214,
    122, 75,  24,  41,  190, 143, 220, 237, 195, 242, 161, 144, 7,   54,  101, 84,
    57,  8,   91,  106, 253, 204, 159, 174, 128, 177, 226, 211, 68,  117, 38,  23,
    252, 205, 158, 175, 56,  9,   90,  107, 69,  116, 39,  22,  129, 176, 227, 210,
    191, 142, 221, 236, 123, 74,  25,  40,  6,   55,  100, 85,  194, 243, 160, 145,
    71,  118, 37,  20,  131, 178, 225, 208, 254, 207, 156, 173, 58,  11,  88,  105,
    4,   53,  102, 87,  192, 241, 162, 147, 189, 140, 223, 238, 121, 72,  27,  42,
    193, 240, 163, 146, 5,   52,  103, 86,  120, 73,  26,  43,  188, 141, 222, 239,
    130, 179, 224, 209, 70,  119, 36,  21,  59,  10,  89,  104, 255, 206, 157, 172
};
/*
 * static void init_crc16_tab( void );
 *
 * For optimal performance uses the CRC16 routine a lookup table with values
 * that can be used directly in the XOR arithmetic in the algorithm. This
 * lookup table is calculated by the init_crc16_tab() routine, the first time
 * the CRC function is called.
 */

void init_crc16_tab( void ) {

    uint16_t i;
    uint16_t j;
    uint16_t crc;
    uint16_t c;

    for (i=0; i<256; i++) {

        crc = 0;
        c   = i;

        for (j=0; j<8; j++) {

            if ( (crc ^ c) & 0x0001 ) crc = ( crc >> 1 ) ^ CRC_POLY_16;
            else                      crc =   crc >> 1;

            c = c >> 1;
        }

        crc_tab16[i] = crc;
    }

    crc_tab16_init = 1;

}  /* init_crc16_tab */

/*
 * uint16_t crc_16( const unsigned char *input_str, size_t num_bytes );
 *
 * The function crc_16() calculates the 16 bits CRC16 in one pass for a byte
 * string of which the beginning has been passed to the function. The number of
 * bytes to check is also a parameter. The number of the bytes in the string is
 * limited by the constant SIZE_MAX.
 */
uint16_t crc_16( const unsigned char *input_str, size_t num_bytes ) {

    uint16_t crc;
    const unsigned char *ptr;
    size_t a;

    if ( ! crc_tab16_init ) init_crc16_tab();

    crc = CRC_START_16;
    ptr = input_str;

    if ( ptr != NULL ) for (a=0; a<num_bytes; a++) {

        crc = (crc >> 8) ^ crc_tab16[ (crc ^ (uint16_t) *ptr++) & 0x00FF ];
    }

    return crc;

}  /* crc_16 */

uint8_t crc_8( const unsigned char *input_str, size_t num_bytes ) {

    size_t a;
    uint8_t crc;
    const unsigned char *ptr;

    crc = CRC_START_8;
    ptr = input_str;

    if ( ptr != NULL ) for (a=0; a<num_bytes; a++) {

        crc = sht75_crc_table[(*ptr++) ^ crc];
    }

    return crc;

}  /* crc_8 */




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
            //}
            //localBuffer[XK_H-1] = crc_8((unsigned char*)&localBuffer[XK_H],sizeof(double)*(msgSize-XK_H));
            localBuffer[XK_H-1] = crc_8((unsigned char*)&localBuffer[XK_H],sizeof(double)*(msgSize-XK_H));
            //printf("k=%lf,size=%d,sum=%lf\n",localBuffer[0],msgSize-XK_H, localBuffer[XK_H-1]);
            //fflush(stdout);
            BcastTree->forwardMessageOneSide((double*)localBuffer,msgSize, iam_col, BCcount, BCbase, maxrecvsz, Pc);
	        //onesidecomm_bc += SuperLU_timer_() - t1;
		}
		if(precision=='z'){
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
            localBuffer[XK_H*2-2] = crc_8((unsigned char*)&localBuffer[XK_H*2],sizeof(double)*2*(msgSize-XK_H));
            //localBuffer[XK_H*2-2] = crc_16((unsigned char*)&localBuffer[XK_H*2],sizeof(double)*2*(msgSize-XK_H));
            BcastTree->forwardMessageOneSideU((doublecomplex*)localBuffer,msgSize, iam_col, BCcount, BCbase, maxrecvsz, Pc);
            //printf("iam=%d, End In Oneside interface k=%lf, checksum=%lf\n",iam,localBuffer[0],localBuffer[XK_H*2-2]);
            //fflush(stdout);
		}
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
                //printf("k=%lf,size=%d,sum=%lf\n",localBuffer[0],msgSize-XK_H, localBuffer[LSUM_H-1]);
                //fflush(stdout);
		        ReduceTree->forwardMessageOneSide((double*)localBuffer, msgSize, iam_row, RDcount, RDbase, maxrecvsz, Pc);
		        //onesidecomm_bc += SuperLU_timer_() - t1;
        }
		if(precision=='z'){
            //localBuffer[LSUM_H*2-2] = crc_8((unsigned char*)&localBuffer[LSUM_H*2],sizeof(double)*2*(msgSize-LSUM_H));
		    TreeReduce_slu<doublecomplex>* ReduceTree = (TreeReduce_slu<doublecomplex>*) Tree;
            localBuffer[LSUM_H*2-2] = crc_8((unsigned char*)&localBuffer[LSUM_H*2],sizeof(double)*2*(msgSize-LSUM_H));
            //localBuffer[LSUM_H*2-2] = crc_16((unsigned char*)&localBuffer[LSUM_H*2],sizeof(double)*2*(msgSize-LSUM_H));
		    ReduceTree->forwardMessageOneSideU((doublecomplex*)localBuffer,msgSize, iam_row, RDcount, RDbase, maxrecvsz, Pc);
		}
	}
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
		return 0;
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
        return 0;
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
		return NO;
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
		return 0;
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
		return 0;
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
		return 0;
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
		return NO;
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

