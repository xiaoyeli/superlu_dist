#ifndef _PEXSI_TREE_IMPL_V2_HPP_
#define _PEXSI_TREE_IMPL_V2_HPP_

#define CHECK_MPI_ERROR

namespace PEXSI{
#ifdef COMM_PROFILE_BCAST
  template< typename T> 
    inline void TreeBcast_v2<T>::SetGlobalComm(const MPI_Comm & pGComm){
      if(commGlobRanks.count(comm_)==0){
        MPI_Group group2 = MPI_GROUP_NULL;
        MPI_Comm_group(pGComm, &group2);
        MPI_Group group1 = MPI_GROUP_NULL;
        MPI_Comm_group(comm_, &group1);

        Int size;
        MPI_Comm_size(comm_,&size);
        vector<int> globRanks(size);
        vector<int> Lranks(size);
        for(int i = 0; i<size;++i){Lranks[i]=i;}
        MPI_Group_translate_ranks(group1, size, &Lranks[0],group2, &globRanks[0]);
        commGlobRanks[comm_] = globRanks;
      }
      myGRoot_ = commGlobRanks[comm_][myRoot_];
      myGRank_ = commGlobRanks[comm_][myRank_];
    }
#endif



  template< typename T> 
    TreeBcast_v2<T>::TreeBcast_v2(){
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
    TreeBcast_v2<T>::TreeBcast_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt,Int msgSize):TreeBcast_v2(){
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
    TreeBcast_v2<T>::TreeBcast_v2(const TreeBcast_v2 & Tree){
      this->Copy(Tree);
    }

  template< typename T> 
    inline void TreeBcast_v2<T>::Copy(const TreeBcast_v2 & Tree){
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
    inline void TreeBcast_v2<T>::Reset(){
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
    TreeBcast_v2<T>::~TreeBcast_v2(){
      cleanupBuffers();
      MPI_Type_free( &type_ );
    }

  template< typename T> 
    inline Int TreeBcast_v2<T>::GetNumRecvMsg(){
      return recvCount_;
    }
  template< typename T> 
    inline Int TreeBcast_v2<T>::GetNumMsgToSend(){
      return this->GetDestCount();
    }

  template< typename T> 
    inline Int TreeBcast_v2<T>::GetNumMsgToRecv(){
      return 1;//always one even for root//myRank_==myRoot_?0:1;
    }

  template< typename T> 
    inline Int TreeBcast_v2<T>::GetNumSendMsg(){
      return sendCount_;
    }
  template< typename T> 
    inline void TreeBcast_v2<T>::SetDataReady(bool rdy){ 
      isReady_=rdy;
    }
  template< typename T> 
    inline void TreeBcast_v2<T>::SetTag(Int tag){
      tag_ = tag;
    }
  template< typename T> 
    inline int TreeBcast_v2<T>::GetTag(){
      return tag_;
    }

  template< typename T> 
    inline bool TreeBcast_v2<T>::IsDone(){
      return done_;
    }


  template< typename T> 
    inline bool TreeBcast_v2<T>::IsDataReceived(){
      bool retVal = false;
      if(myRank_==myRoot_){
        retVal = isReady_;
      }
      else if(recvCount_ == 1){
        retVal = true;
      }
      else if(recvRequests_[0] == MPI_REQUEST_NULL ){
        //post the recv
        postRecv();
        retVal = false;
      }
      else if(recvRequests_[0] != MPI_REQUEST_NULL ){
#if ( _DEBUGlevel_ >= 1 ) || defined(BCAST_VERBOSE)
        statusOFS<<myRank_<<" TESTING RECV on tag "<<tag_<<std::endl;
#endif
        //test
        int flag = 0;
        MPI_Status stat;
        int mpierr = MPI_Test(&recvRequests_[0],&flag,&stat);
        assert(mpierr==MPI_SUCCESS);
        if(flag==1){
          this->recvCount_++;
        }

        retVal = flag==1;
        if(recvCount_==recvPostedCount_){
          //mark that we are ready to send / forward
          isReady_ = true;
        }
      }
      return retVal;
    }

  template< typename T> 
    inline Int * TreeBcast_v2<T>::GetDests(){
      return &myDests_[0];
    }
  template< typename T> 
    inline Int TreeBcast_v2<T>::GetDest(Int i){
      return myDests_[i];
    }
  template< typename T> 
    inline Int TreeBcast_v2<T>::GetDestCount(){
      return this->myDests_.size();
    }
  template< typename T> 
    inline Int TreeBcast_v2<T>::GetRoot(){
      return this->myRoot_;
    }

  template< typename T> 
    inline bool TreeBcast_v2<T>::IsRoot(){
      return this->myRoot_==this->myRank_;
    }

  template< typename T> 
    inline Int TreeBcast_v2<T>::GetMsgSize(){
      return this->msgSize_;
    }

  template< typename T> 
    inline void TreeBcast_v2<T>::forwardMessage( ){
      if(this->isReady_){
#if ( _DEBUGlevel_ >= 1 ) || defined(BCAST_VERBOSE)
        statusOFS<<this->myRank_<<" FORWARDING on tag "<<this->tag_<<std::endl;
#endif
        if(this->sendRequests_.size()!=this->GetDestCount()){
          this->sendRequests_.assign(this->GetDestCount(),MPI_REQUEST_NULL);
        }

        for( Int idxRecv = 0; idxRecv < this->myDests_.size(); ++idxRecv ){
          Int iProc = this->myDests_[idxRecv];
          // Use Isend to send to multiple targets
          int error_code = MPI_Isend( this->recvDataPtrs_[0], this->msgSize_, this->type_, 
              iProc, this->tag_,this->comm_, &this->sendRequests_[idxRecv] );
#ifdef CHECK_MPI_ERROR
          if(error_code!=MPI_SUCCESS){
            char error_string[BUFSIZ];
            int length_of_error_string, error_class;

            MPI_Error_class(error_code, &error_class);
            MPI_Error_string(error_class, error_string, &length_of_error_string);
            statusOFS<<error_string<<std::endl;
            MPI_Error_string(error_code, error_string, &length_of_error_string);
            statusOFS<<error_string<<std::endl;
            gdb_lock();
          }
#endif

#if ( _DEBUGlevel_ >= 1 ) || defined(BCAST_VERBOSE)
          statusOFS<<this->myRank_<<" FWD to "<<iProc<<" on tag "<<this->tag_<<std::endl;
#endif
#ifdef COMM_PROFILE_BCAST
          PROFILE_COMM(this->myGRank_,commGlobRanks[this->comm_][iProc],this->tag_,this->msgSize_);
#endif
          this->sendPostedCount_++;
        } // for (iProc)
        this->fwded_ = true;
      }
    }

  template< typename T> 
    inline void TreeBcast_v2<T>::cleanupBuffers(){
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
    }


  template< typename T> 
    inline void TreeBcast_v2<T>::SetLocalBuffer(T * locBuffer){
      //if recvDataPtrs_[0] has been allocated as a temporary buffer
      if(this->recvDataPtrs_[0]!=NULL && this->recvDataPtrs_[0]!=locBuffer){
        //If we have received some data, we need to copy 
        //it to the new buffer
        if(this->recvCount_>0){
          copyLocalBuffer(locBuffer);
        }

        //If data hasn't been forwarded yet, 
        //it is safe to clear recvTempBuffer_ now
        if(!this->fwded_){
          this->recvTempBuffer_.clear(); 
        }
      }

      this->recvDataPtrs_[0] = locBuffer;
    }


  template< typename T> 
    inline bool TreeBcast_v2<T>::isMessageForwarded(){
      bool retVal=false;

      if(!this->fwded_){
        //If data has been received but not forwarded 
        if(IsDataReceived()){
          forwardMessage();
        }
        retVal = false;
      }
      else{
        //If data has been forwared, check for completion of send requests
        int destCount = this->myDests_.size();
        int completed = 0;
        if(destCount>0){
          //test the send requests
          int flag = 0;

          this->sendDoneIdx_.resize(this->GetDestCount());
#ifndef CHECK_MPI_ERROR
          MPI_Testsome(destCount,this->sendRequests_.data(),&completed,this->sendDoneIdx_.data(),MPI_STATUSES_IGNORE);
#else
          this->sendStatuses_.resize(destCount);
          int error_code = MPI_Testsome(destCount,this->sendRequests_.data(),&completed,this->sendDoneIdx_.data(),this->sendStatuses_.data());
          if(error_code!=MPI_SUCCESS){
            char error_string[BUFSIZ];
            int length_of_error_string, error_class;

            MPI_Error_class(error_code, &error_class);
            MPI_Error_string(error_class, error_string, &length_of_error_string);
            statusOFS<<error_string<<std::endl;
            MPI_Error_string(error_code, error_string, &length_of_error_string);
            statusOFS<<error_string<<std::endl;

            //now check the status
            for(int i = 0; i<this->sendStatuses_.size();i++){
              error_code = this->sendStatuses_[i].MPI_ERROR;
              if(error_code != MPI_SUCCESS){
                MPI_Error_class(error_code, &error_class);
                MPI_Error_string(error_class, error_string, &length_of_error_string);
                statusOFS<<error_string<<std::endl;
                MPI_Error_string(error_code, error_string, &length_of_error_string);
                statusOFS<<error_string<<std::endl;
              }
            }
            gdb_lock();
          }
#endif

        }
        this->sendCount_ += completed;
        retVal = this->sendCount_ == this->sendPostedCount_;

      }
      return retVal;
    }

  //async wait and forward
  template< typename T> 
    inline bool TreeBcast_v2<T>::Progress(){
      bool retVal = this->done_;

      if(!retVal){
        retVal = isMessageForwarded();

        if(retVal){
          //if the local buffer has been set by the user, but the temporary 
          //buffer was already in use, we can clear it now
          if(this->recvTempBuffer_.size()>0){ 
            if(this->recvDataPtrs_[0]!=(T*)this->recvTempBuffer_.data()){
              this->recvTempBuffer_.clear();
            }
          }

          //free the unnecessary arrays
          this->sendRequests_.clear();
#if ( _DEBUGlevel_ >= 1 ) || defined(BCAST_VERBOSE)
          statusOFS<<this->myRank_<<" EVERYTHING COMPLETED on tag "<<this->tag_<<std::endl;
#endif
        }


      }

      this->done_ = retVal;
      return retVal;

    }

  //blocking wait
  template< typename T> 
    inline void TreeBcast_v2<T>::Wait(){
      if(!this->done_){
        while(!Progress());
      }
    }

  template< typename T> 
    inline T* TreeBcast_v2<T>::GetLocalBuffer(){
      assert(this->recvDataPtrs_.size()>0);
      assert(this->recvDataPtrs_[0]!=nullptr);
      return this->recvDataPtrs_[0];
    }

  template< typename T> 
    inline void TreeBcast_v2<T>::postRecv()
    {
#if ( _DEBUGlevel_ >= 1 ) || defined(BCAST_VERBOSE)
      statusOFS<<this->myRank_<<" POSTING RECV on tag "<<this->tag_<<std::endl;
#endif
      if(this->recvCount_<1 && this->recvRequests_[0]==MPI_REQUEST_NULL && !this->IsRoot() ){

        if(this->recvDataPtrs_[0]==NULL){
          this->recvTempBuffer_.resize(this->msgSize_);
          this->recvDataPtrs_[0] = (T*)this->recvTempBuffer_.data();
        }
        int error_code = MPI_Irecv( (char*)this->recvDataPtrs_[0], this->msgSize_, this->type_, 
            this->myRoot_, this->tag_,this->comm_, &this->recvRequests_[0] );
#ifdef CHECK_MPI_ERROR
          if(error_code!=MPI_SUCCESS){
            char error_string[BUFSIZ];
            int length_of_error_string, error_class;

            MPI_Error_class(error_code, &error_class);
            MPI_Error_string(error_class, error_string, &length_of_error_string);
            statusOFS<<error_string<<std::endl;
            MPI_Error_string(error_code, error_string, &length_of_error_string);
            statusOFS<<error_string<<std::endl;
            gdb_lock();
          }
#endif


        this->recvPostedCount_=1;
      }
    }



  template< typename T> 
    inline void TreeBcast_v2<T>::copyLocalBuffer(T* destBuffer){
      std::copy((T*)this->recvDataPtrs_[0],(T*)this->recvDataPtrs_[0]+this->msgSize_,destBuffer);
    }


  template< typename T>
    inline TreeBcast_v2<T> * TreeBcast_v2<T>::Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed){
      //get communicator size
      Int nprocs = 0;
      MPI_Comm_size(pComm, &nprocs);

#if defined(FTREE)
      return new FTreeBcast2<T>(pComm,ranks,rank_cnt,msgSize);
#elif defined(MODBTREE)
      return new ModBTreeBcast2<T>(pComm,ranks,rank_cnt,msgSize,rseed);
#elif defined(BTREE)
      return new BTreeBcast2<T>(pComm,ranks,rank_cnt,msgSize);
#endif


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
        return new ModBTreeBcast2<T>(pComm,ranks,rank_cnt,msgSize, rseed);
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
      for(int i =0;i<this->myDests_.size();++i){statusOFS<<this->myDests_[i]<<" ";}
      statusOFS<<std::endl;
#endif
    }



  template< typename T>
    FTreeBcast2<T>::FTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeBcast_v2<T>(pComm,ranks,rank_cnt,msgSize){
      //build the binary tree;
      buildTree(ranks,rank_cnt);
    }


  template< typename T>
    inline FTreeBcast2<T> * FTreeBcast2<T>::clone() const{
      FTreeBcast2 * out = new FTreeBcast2(*this);
      return out;
    } 








  template< typename T>
    inline BTreeBcast2<T>::BTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeBcast_v2<T>(pComm,ranks,rank_cnt,msgSize){
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

      Int idxStart = 0;
      Int idxEnd = rank_cnt;



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

          if( this->myRank_ < ranks[idxStartH]){
            idxStart = idxStartL;
            idxEnd = idxStartH;
          }
          else{
            idxStart = idxStartH;
          }
          prevRoot = curRoot;
        }

      }

#if (defined(BCAST_VERBOSE))
      statusOFS<<"My root is "<<this->myRoot_<<std::endl;
      statusOFS<<"My dests are ";
      for(int i =0;i<this->myDests_.size();++i){statusOFS<<this->myDests_[i]<<" ";}
      statusOFS<<std::endl;
#endif
    }





  template< typename T>
    ModBTreeBcast2<T>::ModBTreeBcast2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed):TreeBcast_v2<T>(pComm,ranks,rank_cnt,msgSize){
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
          TIMER_START(FIND_RANK);
          Int * pos = std::find(&ranks[idxStartL], &ranks[idxStartH], this->myRank_);
          TIMER_STOP(FIND_RANK);
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
      for(int i =0;i<this->myDests_.size();++i){statusOFS<<this->myDests_[i]<<" ";}
      statusOFS<<std::endl;
#endif
    }



  template< typename T>
   void TreeBcast_Waitsome(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeBcast_v2<T> > > & arrTrees, std::list<int> & doneIdx, std::vector<bool> & finishedFlags){
      doneIdx.clear();
      auto all_done = [](const std::vector<bool> & boolvec){
        return std::all_of(boolvec.begin(), boolvec.end(), [](bool v) { return v; });
      };

      while(doneIdx.empty() && !all_done(finishedFlags) ){

        //for(auto it = finishedFlags.begin();it!=finishedFlags.end();it++){
        //  statusOFS<<(*it?"1":"0")<<" ";
        //}
        //statusOFS<<std::endl;

        for(int i = 0; i<treeIdx.size(); i++){
          Int idx = treeIdx[i];
          auto & curTree = arrTrees[idx];
          if(curTree!=nullptr){
            bool done = curTree->Progress();
            if(done){
              if(!finishedFlags[i]){
                doneIdx.push_back(i);
                finishedFlags[i] = true;
              }
            }
          }
          else{
            finishedFlags[i] = true;
          }
        }
      }
    }

  template< typename T>
  void TreeBcast_Testsome(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeBcast_v2<T> > > & arrTrees, std::list<int> & doneIdx, std::vector<bool> & finishedFlags){
      doneIdx.clear();
      for(int i = 0; i<treeIdx.size(); i++){
        Int idx = treeIdx[i];
        auto & curTree = arrTrees[idx];
        if(curTree!=nullptr){
          bool done = curTree->Progress();
          if(done){
            if(!finishedFlags[i]){
              doneIdx.push_back(i);
              finishedFlags[i] = true;
            }
          }
        }
        else{
          finishedFlags[i] = true;
        }
      }
    }

  template< typename T>
  void TreeBcast_Testsome(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeBcast_v2<T> > > & arrTrees, std::list<int> & doneIdx, std::vector<int> & finishedEpochs){
      doneIdx.clear();
      assert(finishedEpochs.size()==treeIdx.size()+1);
      Int curEpoch = ++finishedEpochs.back();
      for(int i = 0; i<treeIdx.size(); i++){
        Int idx = treeIdx[i];
        auto & curTree = arrTrees[idx];
        if(curTree!=nullptr){
          bool done = curTree->Progress();
          if(done){
            if(finishedEpochs[i]<=0){
              doneIdx.push_back(i);
              finishedEpochs[i] = curEpoch;
            }
          }
        }
        else{
          finishedEpochs[i] = curEpoch;
        }
      }
    }



  template< typename T>
   void TreeBcast_Waitall(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeBcast_v2<T> > > & arrTrees){
     std::list<int> doneIdx;
     std::vector<bool> finishedFlags(treeIdx.size(),false);
     
      doneIdx.clear();
      auto all_done = [](const std::vector<bool> & boolvec){
        return std::all_of(boolvec.begin(), boolvec.end(), [](bool v) { return v; });
      };

     while(!all_done(finishedFlags) ){
        for(int i = 0; i<treeIdx.size(); i++){
          Int idx = treeIdx[i];
          auto & curTree = arrTrees[idx];
          if(curTree!=nullptr){
            bool done = curTree->Progress();
            if(done){
              if(!finishedFlags[i]){
                doneIdx.push_back(i);
                finishedFlags[i] = true;
              }
            }
          }
          else{
            finishedFlags[i] = true;
          }
        }
      }
    }

} //namespace PEXSI


#endif
