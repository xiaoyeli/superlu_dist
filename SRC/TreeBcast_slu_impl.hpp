#ifndef __SUPERLU_TREEBCAST_IMPL
#define __SUPERLU_TREEBCAST_IMPL


namespace SuperLU_ASYNCOMM {

  template< typename T> 
    TreeBcast_slu<T>::TreeBcast_slu(){
      comm_ = MPI_COMM_NULL;
      myRank_=-1;
      myRoot_ = -1; 
      msgSize_ = -1;
      recvCount_ = -1;
      sendCount_ = -1;
      recvPostedCount_ = -1;
      sendPostedCount_ = -1;
      tag_=-1;
      mainRoot_=-1;
      isReady_ = false;
      recvDataPtrs_.assign(1,NULL);
      recvRequests_.assign(1,MPI_REQUEST_NULL);
      fwded_=false;
      done_ = false;


      MPI_Type_contiguous( sizeof(T), MPI_BYTE, &type_ );
      MPI_Type_commit( &type_ );

    }

  template< typename T> 
    TreeBcast_slu<T>::TreeBcast_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt,Int msgSize):TreeBcast_slu(){
      comm_ = pComm;
      MPI_Comm_rank(comm_,&myRank_);
      msgSize_ = msgSize;
      recvCount_ = 0;
      sendCount_ = 0;
      recvPostedCount_ = 0;
      sendPostedCount_ = 0;
      mainRoot_=ranks[0];
#ifdef CHECK_MPI_ERROR
          MPI_Errhandler_set(this->comm_, MPI_ERRORS_RETURN);
          MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
#endif
    }


  template< typename T> 
    TreeBcast_slu<T>::TreeBcast_slu(const TreeBcast_slu & Tree){
      this->Copy(Tree);
    }

  template< typename T> 
    inline void TreeBcast_slu<T>::Copy(const TreeBcast_slu & Tree){
      this->comm_ = Tree.comm_;
      this->myRank_ = Tree.myRank_;
      this->myRoot_ = Tree.myRoot_; 
      this->msgSize_ = Tree.msgSize_;

      this->recvCount_ = Tree.recvCount_;
      this->sendCount_ = Tree.sendCount_;
      this->recvPostedCount_ = Tree.recvPostedCount_;
      this->sendPostedCount_ = Tree.sendPostedCount_;
      this->tag_= Tree.tag_;
      this->mainRoot_= Tree.mainRoot_;
      this->isReady_ = Tree.isReady_;
      this->myDests_ = Tree.myDests_;

      this->recvRequests_ = Tree.recvRequests_;
      this->recvTempBuffer_ = Tree.recvTempBuffer_;
      this->sendRequests_ = Tree.sendRequests_;
      this->recvDataPtrs_ = Tree.recvDataPtrs_;
      if(Tree.recvDataPtrs_[0]==(T*)Tree.recvTempBuffer_.data()){
        this->recvDataPtrs_[0]=(T*)this->recvTempBuffer_.data();
      }

      this->fwded_= Tree.fwded_;
      this->done_= Tree.done_;
    }

  template< typename T> 
    inline void TreeBcast_slu<T>::Reset(){
      assert(done_);
      cleanupBuffers();
      done_=false;
      fwded_=false;
      recvDataPtrs_.assign(GetNumMsgToRecv(),NULL);
      recvRequests_.assign(GetNumMsgToRecv(),MPI_REQUEST_NULL);
      sendDataPtrs_.assign(GetNumMsgToSend(),NULL);
      sendRequests_.assign(GetNumMsgToSend(),MPI_REQUEST_NULL);
      // isAllocated_=false;
      isReady_=false;
      recvCount_ = 0;
      sendCount_ = 0;
      recvPostedCount_ = 0;
      sendPostedCount_ = 0;
    }


  template< typename T> 
    TreeBcast_slu<T>::~TreeBcast_slu(){
      cleanupBuffers();
      MPI_Type_free( &type_ );
    }

  template< typename T> 
    inline Int TreeBcast_slu<T>::GetNumRecvMsg(){
      return recvCount_;
    }
  template< typename T> 
    inline Int TreeBcast_slu<T>::GetNumMsgToSend(){
      return this->GetDestCount();
    }

  template< typename T> 
    inline Int TreeBcast_slu<T>::GetNumMsgToRecv(){
      return 1;//always one even for root//myRank_==myRoot_?0:1;
    }

  template< typename T> 
    inline Int TreeBcast_slu<T>::GetNumSendMsg(){
      return sendCount_;
    }
  template< typename T> 
    inline void TreeBcast_slu<T>::SetDataReady(bool rdy){ 
      isReady_=rdy;
    }
  template< typename T> 
    inline void TreeBcast_slu<T>::SetTag(Int tag){
      tag_ = tag;
    }
  template< typename T> 
    inline Int TreeBcast_slu<T>::GetTag(){
      return tag_;
    }


  template< typename T> 
    inline Int * TreeBcast_slu<T>::GetDests(){
      return &myDests_[0];
    }
  template< typename T> 
    inline Int TreeBcast_slu<T>::GetDest(Int i){
      return myDests_[i];
    }
  template< typename T> 
    inline Int TreeBcast_slu<T>::GetDestCount(){
      return this->myDests_.size();
    }
  template< typename T> 
    inline Int TreeBcast_slu<T>::GetRoot(){
      return this->myRoot_;
    }

  template< typename T> 
    inline bool TreeBcast_slu<T>::IsRoot(){
      return this->myRoot_==this->myRank_;
    }
	

  template< typename T> 
    inline Int TreeBcast_slu<T>::GetMsgSize(){
      return this->msgSize_;
    }

	
	
  template< typename T> 
    inline void TreeBcast_slu<T>::forwardMessageSimple(T * locBuffer, Int msgSize){
        MPI_Status status;
		Int flag;
		for( Int idxRecv = 0; idxRecv < this->myDests_.size(); ++idxRecv ){
          Int iProc = this->myDests_[idxRecv];
          // Use Isend to send to multiple targets
          Int error_code = MPI_Isend( locBuffer, msgSize, this->type_, 
              iProc, this->tag_,this->comm_, &this->sendRequests_[idxRecv] );
			  
			  MPI_Test(&this->sendRequests_[idxRecv],&flag,&status) ; 
			  
			  // MPI_Wait(&this->sendRequests_[idxRecv],&status) ; 
			  // std::cout<<this->myRank_<<" FWD to "<<iProc<<" on tag "<<this->tag_<<std::endl;
        } // for (iProc)
    }	  

  template< typename T> 
    inline void TreeBcast_slu<T>::waitSendRequest(){
        MPI_Status status;
		for( Int idxRecv = 0; idxRecv < this->myDests_.size(); ++idxRecv ){
			  MPI_Wait(&this->sendRequests_[idxRecv],&status) ; 
        } // for (iProc)
    }	
	
	

 

  template< typename T> 
    inline void TreeBcast_slu<T>::allocateRequest(){
        if(this->sendRequests_.size()!=this->GetDestCount()){
          this->sendRequests_.resize(this->GetDestCount());
        }
		this->sendRequests_.assign(this->GetDestCount(),MPI_REQUEST_NULL);
    }
	
	

	
	
	
  template< typename T> 
    inline void TreeBcast_slu<T>::cleanupBuffers(){
      this->recvRequests_.clear();
      this->recvStatuses_.clear();
      this->recvDoneIdx_.clear();
      this->recvDataPtrs_.clear();
      this->recvTempBuffer_.clear();
      this->sendRequests_.clear();
      this->sendStatuses_.clear();
      this->sendDoneIdx_.clear();
      this->sendDataPtrs_.clear();
      this->sendTempBuffer_.clear();

      this->recvRequests_.shrink_to_fit();
      this->recvStatuses_.shrink_to_fit();
      this->recvDoneIdx_.shrink_to_fit();
      this->recvDataPtrs_.shrink_to_fit();
      this->recvTempBuffer_.shrink_to_fit();
      this->sendRequests_.shrink_to_fit();
      this->sendStatuses_.shrink_to_fit();
      this->sendDoneIdx_.shrink_to_fit();
      this->sendDataPtrs_.shrink_to_fit();
      this->sendTempBuffer_.shrink_to_fit();
	  
	  this->myDests_.clear();
	  
    }


  template< typename T> 
    inline void TreeBcast_slu<T>::AllocateBuffer()
    {

      if(!this->IsRoot()){

        if(this->recvDataPtrs_[0]==NULL){
          this->recvTempBuffer_.resize(this->msgSize_);
          this->recvDataPtrs_[0] = (T*)this->recvTempBuffer_.data();
        }
      }
    }	
	
  template< typename T>
    inline TreeBcast_slu<T> * TreeBcast_slu<T>::Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed){
      //get communicator size
      Int nprocs = 0;
      MPI_Comm_size(pComm, &nprocs);

      if(nprocs<=FTREE_LIMIT){
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
        statusOFS<<"FLAT TREE USED"<<std::endl;
#endif

        return new FTreeBcast2<T>(pComm,ranks,rank_cnt,msgSize);

      }
      else{
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
        statusOFS<<"BINARY TREE USED"<<std::endl;
#endif
		return new BTreeBcast2<T>(pComm,ranks,rank_cnt,msgSize);
      }
    }



  template< typename T>
    inline void FTreeBcast2<T>::buildTree(Int * ranks, Int rank_cnt){
      Int idxStart = 0;
      Int idxEnd = rank_cnt;
      this->myRoot_ = ranks[0];
      if(this->IsRoot() ){
        this->myDests_.insert(this->myDests_.end(),&ranks[1],&ranks[0]+rank_cnt);
      }
#if (defined(BCAST_VERBOSE)) 
      statusOFS<<"My root is "<<this->myRoot_<<std::endl;
      statusOFS<<"My dests are ";
      for(Int i =0;i<this->myDests_.size();++i){statusOFS<<this->myDests_[i]<<" ";}
      statusOFS<<std::endl;
#endif
    }



  template< typename T>
    FTreeBcast2<T>::FTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeBcast_slu<T>(pComm,ranks,rank_cnt,msgSize){
      //build the binary tree;
      buildTree(ranks,rank_cnt);
    }


  template< typename T>
    inline FTreeBcast2<T> * FTreeBcast2<T>::clone() const{
      FTreeBcast2 * out = new FTreeBcast2(*this);
      return out;
    } 








  template< typename T>
    inline BTreeBcast2<T>::BTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeBcast_slu<T>(pComm,ranks,rank_cnt,msgSize){
      //build the binary tree;
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline BTreeBcast2<T> * BTreeBcast2<T>::clone() const{
      BTreeBcast2<T> * out = new BTreeBcast2<T>(*this);
      return out;
    }


  template< typename T>
    inline void BTreeBcast2<T>::buildTree(Int * ranks, Int rank_cnt){

      Int myIdx = 0;
      Int ii=0; 
	  Int child,root;
	  for (ii=0;ii<rank_cnt;ii++)
		  if(this->myRank_ == ranks[ii]){
			  myIdx = ii;
			  break;
		  }
	  for (ii=0;ii<DEG_TREE;ii++){
		  if(myIdx*DEG_TREE+1+ii<rank_cnt){
			   child = ranks[myIdx*DEG_TREE+1+ii];
			   this->myDests_.push_back(child);
		  }		
	  }
	  	   
	  if(myIdx!=0){
		  this->myRoot_ = ranks[(Int)floor((double)(myIdx-1.0)/(double)DEG_TREE)];
	  }else{
		  this->myRoot_ = this->myRank_;
	  } 
	  
    }



  template< typename T>
    ModBTreeBcast2<T>::ModBTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed):TreeBcast_slu<T>(pComm,ranks,rank_cnt,msgSize){
      //build the binary tree;
      MPI_Comm_rank(this->comm_,&this->myRank_);
      this->rseed_ = rseed;
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline ModBTreeBcast2<T> * ModBTreeBcast2<T>::clone() const{
      ModBTreeBcast2 * out = new ModBTreeBcast2(*this);
      return out;
    }

  template< typename T>
    inline void ModBTreeBcast2<T>::buildTree(Int * ranks, Int rank_cnt){

      Int idxStart = 0;
      Int idxEnd = rank_cnt;

      //sort the ranks with the modulo like operation
      if(rank_cnt>1){
        Int new_idx = (Int)this->rseed_ % (rank_cnt - 1) + 1; 
        Int * new_start = &ranks[new_idx];
        std::rotate(&ranks[1], new_start, &ranks[0]+rank_cnt);
      }

      Int prevRoot = ranks[0];
      while(idxStart<idxEnd){
        Int curRoot = ranks[idxStart];
        Int listSize = idxEnd - idxStart;

        if(listSize == 1){
          if(curRoot == this->myRank_){
            this->myRoot_ = prevRoot;
            break;
          }
        }
        else{
          Int halfList = floor(ceil(double(listSize) / 2.0));
          Int idxStartL = idxStart+1;
          Int idxStartH = idxStart+halfList;

          if(curRoot == this->myRank_){
            if ((idxEnd - idxStartH) > 0 && (idxStartH - idxStartL)>0){
              Int childL = ranks[idxStartL];
              Int childR = ranks[idxStartH];

              this->myDests_.push_back(childL);
              this->myDests_.push_back(childR);
            }
            else if ((idxEnd - idxStartH) > 0){
              Int childR = ranks[idxStartH];
              this->myDests_.push_back(childR);
            }
            else{
              Int childL = ranks[idxStartL];
              this->myDests_.push_back(childL);
            }
            this->myRoot_ = prevRoot;
            break;
          } 

          //not true anymore ?
          //first half to 
         // TIMER_START(FIND_RANK);
          Int * pos = std::find(&ranks[idxStartL], &ranks[idxStartH], this->myRank_);
         // TIMER_STOP(FIND_RANK);
          if( pos != &ranks[idxStartH]){
            idxStart = idxStartL;
            idxEnd = idxStartH;
          }
          else{
            idxStart = idxStartH;
          }
          prevRoot = curRoot;
        }

      }

#if (defined(REDUCE_VERBOSE))
      statusOFS<<"My root is "<<this->myRoot_<<std::endl;
      statusOFS<<"My dests are ";
      for(Int i =0;i<this->myDests_.size();++i){statusOFS<<this->myDests_[i]<<" ";}
      statusOFS<<std::endl;
#endif
    }


} //namespace SuperLU_ASYNCOMM


#endif
