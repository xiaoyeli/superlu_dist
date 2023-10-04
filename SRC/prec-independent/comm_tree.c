
#include "dcomplex.h"
#include "superlu_defs.h"

	void C_BcTree_Create_nv(C_Tree* tree, MPI_Comm comm, int* ranks, int rank_cnt, int msgSize, char precision, int* needrecv){
		assert(msgSize>0);

      int nprocs = 0;
      MPI_Comm_size(comm, &nprocs);
	  tree->comm_=comm;
	  tree->msgSize_=msgSize;
	  MPI_Comm_rank(comm,&tree->myRank_);
      tree->myRoot_= -1; 
      tree->tag_=-1;
      tree->destCnt_=0;
      tree->myDests_[0]=-1;
      tree->myDests_[1]=-1;
	  tree->sendRequests_[0]=MPI_REQUEST_NULL;
	  tree->sendRequests_[1]=MPI_REQUEST_NULL;
      tree->empty_= NO;  // non-empty if rank_cnt>1
	  if(precision=='d'){
	  tree->type_=MPI_DOUBLE;
	  }
	  if(precision=='s'){
	tree->type_=MPI_FLOAT;
	  }
	  if(precision=='z'){
	tree->type_=MPI_C_DOUBLE_COMPLEX;
	  }
	 if(precision=='c'){
	     //MPI_Type_contiguous( sizeof(complex), MPI_BYTE, &tree->type_ );
	     tree->type_=MPI_C_COMPLEX;
	 }

      tree->myIdx = 0;
      int ii=0;
	  int child,root;
	  //printf("Tree,rank_cnt=%d\n",rank_cnt);
        for (ii=0;ii<rank_cnt;ii++)
		  if(tree->myRank_ == ranks[ii]){
			  tree->myIdx = ii;
              //printf("Tree,tree->myIdx=%d\n",tree->myIdx);
			  break;
		  }
	  for (ii=0;ii<DEG_TREE;ii++){
		  if((tree->myIdx)*DEG_TREE+1+ii<rank_cnt){
			   child = ranks[(tree->myIdx)*DEG_TREE+1+ii];
			   tree->myDests_[tree->destCnt_++]=child;
               //*mysendmsg_num+=1;
                //printf("Tree,destCnt=%d,mymsg=%d\n",tree->destCnt_,*mysendmsg_num);
		  }
	  }
	  if(tree->myIdx!=0){
		  tree->myRoot_ = ranks[(int)floor((double)(tree->myIdx-1.0)/(double)DEG_TREE)];
		  *needrecv=1;
	  }else{
		  tree->myRoot_ = tree->myRank_;
	  }

       // int myIdx = 0;
       // int ii=0;
       // int child,root;
       // for (ii=0;ii<rank_cnt;ii++)
       //     if(tree->myRank_ == ranks[ii]){
       //         myIdx = ii;
       //         break;
       //     }
       // for (ii=0;ii<DEG_TREE;ii++){
       //     if(myIdx*DEG_TREE+1+ii<rank_cnt){
       //         child = ranks[myIdx*DEG_TREE+1+ii];
       //         tree->myDests_[tree->destCnt_++]=child;
       //         *mysendmsg_num+=1;
       //         printf("Tree,destCnt=%d,mymsg=%d\n",tree->destCnt_,*mysendmsg_num);
       //     }
       // }
       // if(myIdx!=0){
       //     tree->myRoot_ = ranks[(int)floor((double)(myIdx-1.0)/(double)DEG_TREE)];
       // }else{
       //     tree->myRoot_ = tree->myRank_;
       // }

    }

void C_BcTree_Create(C_Tree* tree, MPI_Comm comm, int* ranks, int rank_cnt, int msgSize, char precision){
    assert(msgSize>0);

    int nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    tree->comm_=comm;
    tree->msgSize_=msgSize;
    MPI_Comm_rank(comm,&tree->myRank_);
    tree->myRoot_= -1;
    tree->tag_=-1;
    tree->destCnt_=0;
    tree->myDests_[0]=-1;
    tree->myDests_[1]=-1;
    tree->sendRequests_[0]=MPI_REQUEST_NULL;
    tree->sendRequests_[1]=MPI_REQUEST_NULL;
    tree->empty_= NO;  // non-empty if rank_cnt>1
    if(precision=='d'){
        MPI_Type_contiguous( sizeof(double), MPI_BYTE, &tree->type_ );
    }
    if(precision=='z'){
        MPI_Type_contiguous( sizeof(doublecomplex), MPI_BYTE, &tree->type_ );
    }
    MPI_Type_commit( &tree->type_ );

    tree->myIdx = 0;
    int ii=0;
    int child,root;
    for (ii=0;ii<rank_cnt;ii++)
        if(tree->myRank_ == ranks[ii]){
            tree->myIdx = ii;
            break;
        }
    for (ii=0;ii<DEG_TREE;ii++){
        if(tree->myIdx*DEG_TREE+1+ii<rank_cnt){
            child = ranks[tree->myIdx*DEG_TREE+1+ii];
            tree->myDests_[tree->destCnt_++]=child;
        }
    }
    if(tree->myIdx!=0){
        tree->myRoot_ = ranks[(int)floor((double)(tree->myIdx-1.0)/(double)DEG_TREE)];
    }else{
        tree->myRoot_ = tree->myRank_;
    }

}

	void C_BcTree_Nullify(C_Tree* tree){
	  tree->msgSize_=-1;
	  tree->myRank_=-1;
      tree->myRoot_= -1; 
      tree->tag_=-1;
      tree->destCnt_=0;
      tree->myDests_[0]=-1;
      tree->myDests_[1]=-1;
	  tree->sendRequests_[0]=MPI_REQUEST_NULL;
	  tree->sendRequests_[1]=MPI_REQUEST_NULL;
      tree->empty_= YES; 
	  tree->comm_=MPI_COMM_NULL;
	  tree->type_=MPI_DATATYPE_NULL; 
	}	

	yes_no_t C_BcTree_IsRoot(C_Tree* tree){
		return tree->myRoot_ == tree->myRank_?YES:NO;
	}
	
	void C_BcTree_forwardMessageSimple(C_Tree* tree, void* localBuffer, int msgSize){
        MPI_Status status;
		int flag;
		for( int idxRecv = 0; idxRecv < tree->destCnt_; ++idxRecv ){
          int iProc = tree->myDests_[idxRecv];
          // Use Isend to send to multiple targets
          int error_code = MPI_Isend( localBuffer, msgSize, tree->type_, 
              iProc, tree->tag_,tree->comm_, &tree->sendRequests_[idxRecv] );
			  
			  if(getenv("COMM_TREE_MPI_WAIT"))
			  	  MPI_Wait(&tree->sendRequests_[idxRecv],&status) ; 
			  else
				  MPI_Test(&tree->sendRequests_[idxRecv],&flag,&status) ; 
				  
			  
			  // std::cout<<tree->myRank_<<" FWD to "<<iProc<<" on tag "<<tree->tag_<<std::endl;
        } // for (iProc)
	}

	void C_BcTree_waitSendRequest(C_Tree* tree){
        MPI_Status status;
		for( int idxRecv = 0; idxRecv < tree->destCnt_; ++idxRecv ){
			  MPI_Wait(&tree->sendRequests_[idxRecv],&status) ; 
        } // for (iProc)
	}
	
	void C_RdTree_Create(C_Tree* tree, MPI_Comm comm, int* ranks, int rank_cnt, int msgSize, char precision){
		assert(msgSize>0);

      int nprocs = 0;
      MPI_Comm_size(comm, &nprocs);
	  tree->comm_=comm;
	  tree->msgSize_=msgSize;
	  MPI_Comm_rank(comm,&tree->myRank_);
      tree->myRoot_= -1; 
      tree->tag_=-1;
      tree->destCnt_=0;
      tree->myDests_[0]=-1;
      tree->myDests_[1]=-1;
	  tree->sendRequests_[0]=MPI_REQUEST_NULL;
	  tree->sendRequests_[1]=MPI_REQUEST_NULL;
      tree->empty_= NO;  // non-empty if rank_cnt>1
	  if(precision=='d'){
		  tree->type_=MPI_DOUBLE;
	  }
	  if(precision=='c'){
	      //MPI_Type_contiguous( sizeof(float), MPI_BYTE, &tree->type_ );
	      tree->type_=MPI_C_COMPLEX;
	  }
	  if(precision=='z'){
	      tree->type_=MPI_C_DOUBLE_COMPLEX;
	  }
	  if(precision=='s'){
	      tree->type_=MPI_FLOAT;
	  }	  
      int myIdx = 0;
      int ii=0; 
	  int child,root;
	  for (ii=0;ii<rank_cnt;ii++)
		  if(tree->myRank_ == ranks[ii]){
              tree->myIdx = ii;
			  break;
		  }

		  
	  for (ii=0;ii<DEG_TREE;ii++){
		  if(tree->myIdx*DEG_TREE+1+ii<rank_cnt){
			   child = ranks[tree->myIdx*DEG_TREE+1+ii];
			   tree->myDests_[tree->destCnt_++]=child;
		  }		
	  }		  
		  
	  if(tree->myIdx!=0){
		  tree->myRoot_ = ranks[(int)floor((double)(tree->myIdx-1.0)/(double)DEG_TREE)];
	  }else{
		  tree->myRoot_ = tree->myRank_;
	  }
    }

void C_RdTree_Create_nv(C_Tree* tree, MPI_Comm comm, int* ranks, int rank_cnt, int msgSize, char precision, int* needrecvrd,int* needsendrd){
    assert(msgSize>0);

    int nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    tree->comm_=comm;
    tree->msgSize_=msgSize;
    MPI_Comm_rank(comm,&tree->myRank_);
    tree->myRoot_= -1;
    tree->tag_=-1;
    tree->destCnt_=0;
    tree->myDests_[0]=-1;
    tree->myDests_[1]=-1;
    tree->sendRequests_[0]=MPI_REQUEST_NULL;
    tree->sendRequests_[1]=MPI_REQUEST_NULL;
    tree->empty_= NO;  // non-empty if rank_cnt>1
    
	if(precision=='d'){
	    tree->type_=MPI_DOUBLE;
	}
	if(precision=='s'){
        MPI_Type_contiguous( sizeof(float), MPI_BYTE, &tree->type_ );
	}
	if(precision=='z'){
	    tree->type_=MPI_DOUBLE_COMPLEX;
	}
	if(precision=='s'){
	    tree->type_=MPI_FLOAT;
	}
    //int myIdx = 0;
    int ii=0;
    int child,root;
    for (ii=0;ii<rank_cnt;ii++)
        if(tree->myRank_ == ranks[ii]){
            tree->myIdx = ii;
            break;
        }

    for (ii=0;ii<DEG_TREE;ii++){
        if(tree->myIdx*DEG_TREE+1+ii<rank_cnt){
            child = ranks[tree->myIdx*DEG_TREE+1+ii];
            tree->myDests_[tree->destCnt_++]=child;
        }
    }
    *needrecvrd=tree->destCnt_;

    if(tree->myIdx!=0){
        tree->myRoot_ = ranks[(int)floor((double)(tree->myIdx-1.0)/(double)DEG_TREE)];
        *needsendrd+=1;
    }else{
        tree->myRoot_ = tree->myRank_;
    }
    //*mysendmsg_num_rd+=1;
}

	void C_RdTree_Nullify(C_Tree* tree){
	  tree->msgSize_=-1;
	  tree->myRank_=-1;
      tree->myRoot_= -1; 
      tree->tag_=-1;
      tree->destCnt_=0;
      tree->myDests_[0]=-1;
      tree->myDests_[1]=-1;
	  tree->sendRequests_[0]=MPI_REQUEST_NULL;
	  tree->sendRequests_[1]=MPI_REQUEST_NULL;
      tree->empty_= YES; 
	  tree->comm_=MPI_COMM_NULL;
	  tree->type_=MPI_DATATYPE_NULL; 
	}	


	yes_no_t C_RdTree_IsRoot(C_Tree* tree){
		return tree->myRoot_ == tree->myRank_?YES:NO;
	}


	void C_RdTree_forwardMessageSimple(C_Tree* Tree, void* localBuffer, int msgSize){
        MPI_Status status;
		int flag;
		if(Tree->myRank_!=Tree->myRoot_){	
			  //forward to my root if I have reseived everything
			  int iProc = Tree->myRoot_;
			  // Use Isend to send to multiple targets

			  int error_code = MPI_Isend(localBuffer, msgSize, Tree->type_, 
				  iProc, Tree->tag_,Tree->comm_, &Tree->sendRequests_[0] );
					
					if(getenv("COMM_TREE_MPI_WAIT"))
						MPI_Wait(&Tree->sendRequests_[0],&status) ; 
					else
						MPI_Test(&Tree->sendRequests_[0],&flag,&status) ; 					  
						
				  // std::cout<<Tree->myRank_<<" FWD to "<<iProc<<" on tag "<<Tree->tag_<<std::endl;
		}
	}

	void C_RdTree_waitSendRequest(C_Tree* Tree){
        MPI_Status status;		
		if(Tree->myRank_!=Tree->myRoot_){  // not sure about this if condition
		  MPI_Wait(&Tree->sendRequests_[0],&status) ; 
        }			
	}
	
