#ifndef _PEXSI_REDUCE_TREE_IMPL_V2_HPP_
#define _PEXSI_REDUCE_TREE_IMPL_V2_HPP_

#define _SELINV_TAG_COUNT_ 17

namespace PEXSI{
  template<typename T>
    TreeReduce_v2<T>::TreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeBcast_v2<T>(pComm,ranks,rank_cnt,msgSize){
      this->sendDataPtrs_.assign(1,NULL);
      this->sendRequests_.assign(1,MPI_REQUEST_NULL);
      this->isAllocated_=false;
      this->isBufferSet_=false;
    }




  template<typename T>
    TreeReduce_v2<T>::TreeReduce_v2(const TreeReduce_v2<T> & Tree){
      this->Copy(Tree);
    }

  template<typename T>
    TreeReduce_v2<T>::TreeReduce_v2():TreeBcast_v2<T>(){
    }

  template<typename T>
    TreeReduce_v2<T>::~TreeReduce_v2(){
      this->cleanupBuffers();
    }

  template<typename T>
    inline void TreeReduce_v2<T>::Copy(const TreeReduce_v2<T> & Tree){
      ((TreeBcast_v2<T>*)this)->Copy(*(const TreeBcast_v2<T>*)&Tree);

      this->sendDataPtrs_.assign(1,NULL);
      this->sendRequests_.assign(1,MPI_REQUEST_NULL);
      this->isAllocated_= Tree.isAllocated_;
      this->isBufferSet_= Tree.isBufferSet_;

      this->cleanupBuffers();
    }



  template<typename T>
    inline void TreeReduce_v2<T>::postRecv(){
      if(this->GetDestCount()>this->recvPostedCount_){
        for( Int idxRecv = 0; idxRecv < this->myDests_.size(); ++idxRecv ){
          Int iProc = this->myDests_[idxRecv];
          int error_code = MPI_Irecv( (char*)this->recvDataPtrs_[idxRecv], this->msgSize_, this->type_, 
              iProc, this->tag_,this->comm_, &this->recvRequests_[idxRecv] );
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


          this->recvPostedCount_++;
        } // for (iProc)
      }
    }




  template<typename T>
    inline void TreeReduce_v2<T>::reduce( Int idxRecv, Int idReq){
      //add thing to my data
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
      {
        statusOFS<<"[tag="<<this->tag_<<"] "<<"Contribution received:"<<std::endl;
        for(Int i = 0; i<this->msgSize_;i++){
          statusOFS<<this->recvDataPtrs_[idxRecv][i]<<" ";
          if(i%10==0){statusOFS<<std::endl;}
        }
        statusOFS<<std::endl;
      }
#endif
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
      {
        statusOFS<<"[tag="<<this->tag_<<"] "<<"Reduced before:"<<std::endl;
        for(Int i = 0; i<this->msgSize_;i++){
          statusOFS<<this->sendDataPtrs_[0][i]<<" ";
          if(i%10==0){statusOFS<<std::endl;}
        }
        statusOFS<<std::endl;
      }
#endif

      blas::Axpy(this->msgSize_, ONE<T>(), this->recvDataPtrs_[idxRecv], 1, this->sendDataPtrs_[0], 1 );
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
      {
        statusOFS<<"[tag="<<this->tag_<<"] "<<"Reduced after:"<<std::endl;
        for(Int i = 0; i<this->msgSize_;i++){
          statusOFS<<this->sendDataPtrs_[0][i]<<" ";
          if(i%10==0){statusOFS<<std::endl;}
        }
        statusOFS<<std::endl;
      }
#endif

    }

  template<typename T>
    inline void TreeReduce_v2<T>::forwardMessage(){ 
      if(this->isReady_){
        if(this->myRank_!=this->myRoot_){
          //forward to my root if I have reseived everything
          Int iProc = this->myRoot_;
          // Use Isend to send to multiple targets
          if(this->sendDataPtrs_.size()<1){
            this->sendDataPtrs_.assign(1,NULL);
          }

          int msgsz = this->sendDataPtrs_[0]==NULL?0:this->msgSize_;

          int error_code = MPI_Isend((char*)this->sendDataPtrs_[0], msgsz, this->type_, 
              iProc, this->tag_,this->comm_, &this->sendRequests_[0] );
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


          this->sendPostedCount_++;
#ifdef COMM_PROFILE
          PROFILE_COMM(this->myGRank_,this->myGRoot_,this->tag_,msgsz);
#endif

#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
          statusOFS<<this->myRank_<<" FWD to "<<iProc<<" on tag "<<this->tag_<<std::endl;
#endif
        }
        this->fwded_ = true;
      }
    }

  template< typename T> 
    inline bool TreeReduce_v2<T>::IsDataReceived(){
      bool retVal = false;
      if(this->isReady_){
        if(this->recvCount_== this->GetDestCount()){
//          if(this->tag_==12){gdb_lock();}
//          if(this->tag_==9){gdb_lock();}
          retVal = true;
        }
        else if(this->recvCount_<this->recvPostedCount_){
//          if(this->tag_==12){gdb_lock();}
//          if(this->tag_==9){gdb_lock();}
          //mpi_test_some on recvRequests_
          int recvCount = -1;
          int reqCnt = this->recvRequests_.size();//this->recvPostedCount_-this->recvCount_;//GetDestCount();
          //          assert(reqCnt <= this->recvRequests_.size());

          int error_code = MPI_Testsome(reqCnt,&this->recvRequests_[0],&recvCount,&this->recvDoneIdx_[0],&this->recvStatuses_[0]);

#ifdef CHECK_MPI_ERROR
          if(error_code!=MPI_SUCCESS){
            char error_string[BUFSIZ];
            int length_of_error_string, error_class;

            MPI_Error_class(error_code, &error_class);
            MPI_Error_string(error_class, error_string, &length_of_error_string);
            statusOFS<<error_string<<std::endl;
            MPI_Error_string(error_code, error_string, &length_of_error_string);
            statusOFS<<error_string<<std::endl;

            //now check the status
            for(int i = 0; i<this->recvStatuses_.size();i++){
              error_code = this->recvStatuses_[i].MPI_ERROR;
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


          //if something has been received, accumulate and potentially forward it
          for(Int i = 0;i<recvCount;++i ){
            Int idx = this->recvDoneIdx_[i];

            if(idx!=MPI_UNDEFINED){
              Int size = 0;
              MPI_Get_count(&this->recvStatuses_[i], MPI_BYTE, &size);


#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
              statusOFS<<this->myRank_<<" RECVD from "<<this->recvStatuses_[i].MPI_SOURCE<<" on tag "<<this->tag_<<std::endl;
#endif
              if(size>0){
                //resize if needed
                if(this->sendDataPtrs_.size()<1){
                  this->sendDataPtrs_.assign(1,NULL);
                }

                //If sendDataPtrs is 0, allocate to the size of what has been received
                if(this->sendDataPtrs_[0]==NULL){
                  this->sendTempBuffer_.resize(this->msgSize_);
                  this->sendDataPtrs_[0] = (T*)&this->sendTempBuffer_[0];
                  Int nelem = this->msgSize_;
                  std::fill(this->sendDataPtrs_[0],this->sendDataPtrs_[0]+nelem,ZERO<T>());
                }

                //This is where the handle would be called
                reduce(idx,i);

              }

              this->recvCount_++;
            }
          }

          if(this->recvCount_== this->GetDestCount()){
            retVal = true;
          }
          else{
            retVal = false;
          }
        }
        else if(this->recvPostedCount_<this->GetDestCount()){
//          if(this->tag_==12){gdb_lock();}
//          if(this->tag_==9){gdb_lock();}
          this->postRecv();
          retVal = false;
        }
      }
      return retVal;
    }

  template< typename T> 
    inline bool TreeReduce_v2<T>::Progress(){


      bool retVal = false;
      if(this->done_){
        retVal = true;
      }
      else{
        //Do we need this ?
        AllocRecvBuffers();

        if(this->isAllocated_){
          if(this->myRank_==this->myRoot_ && this->isAllocated_){
            this->isReady_=true;
            this->isBufferSet_=true;
          }

          if(this->isReady_ && this->isBufferSet_){
            if(this->IsDataReceived()){

              //free the unnecessary arrays
              this->recvTempBuffer_.clear();
              this->recvRequests_.clear();
              this->recvStatuses_.clear();
              this->recvDoneIdx_.clear();

              if(this->isMessageForwarded()){
                retVal = true;
              }
            }
          }
        }
      }

      if(retVal){
        this->done_ = retVal;
        //TODO do some smart cleanup here
      }
      return retVal;
    }

  template< typename T> 
    inline T * TreeReduce_v2<T>::GetLocalBuffer(){ 
      return this->sendDataPtrs_[0];
    }

  template< typename T> 
    inline void TreeReduce_v2<T>::SetLocalBuffer(T * locBuffer){
      if(this->sendDataPtrs_.size()<1){
        this->sendDataPtrs_.assign(1,NULL);
      }


      if(!this->IsRoot()){
        //if not root, we need to allocate a temp buffer anyway
        if(this->sendDataPtrs_[0]==NULL){
          this->sendTempBuffer_.resize(this->msgSize_);
          this->sendDataPtrs_[0] = (T*)&this->sendTempBuffer_[0];
          Int nelem = this->msgSize_;
          std::fill(this->sendDataPtrs_[0],this->sendDataPtrs_[0]+nelem,ZERO<T>());
        }
        if(!this->isBufferSet_){
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
          {
            statusOFS<<"[tag="<<this->tag_<<"] "<<"Buffer before:"<<std::endl;
            for(Int i = 0; i<this->msgSize_;i++){
              statusOFS<<this->sendDataPtrs_[0][i]<<" ";
              if(i%10==0){statusOFS<<std::endl;}
            }
            statusOFS<<std::endl;
          }
#endif
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
          {
            statusOFS<<"[tag="<<this->tag_<<"] "<<"External buffer:"<<std::endl;
            for(Int i = 0; i<this->msgSize_;i++){
              statusOFS<<locBuffer[i]<<" ";
              if(i%10==0){statusOFS<<std::endl;}
            }
            statusOFS<<std::endl;
          }
#endif


          blas::Axpy(this->msgSize_, ONE<T>(), locBuffer, 1, this->sendDataPtrs_[0], 1 );
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
          {
            statusOFS<<"[tag="<<this->tag_<<"] "<<"Buffer after:"<<std::endl;
            for(Int i = 0; i<this->msgSize_;i++){
              statusOFS<<this->sendDataPtrs_[0][i]<<" ";
              if(i%10==0){statusOFS<<std::endl;}
            }
            statusOFS<<std::endl;
          }
#endif


        }

        this->isBufferSet_= true;
      }
      else{

        if(this->sendDataPtrs_[0]!=NULL && this->sendDataPtrs_[0]!=locBuffer){
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
          {
            statusOFS<<"[tag="<<this->tag_<<"] "<<"ROOT Buffer before:"<<std::endl;
            for(Int i = 0; i<this->msgSize_;i++){
              statusOFS<<this->sendDataPtrs_[0][i]<<" ";
              if(i%10==0){statusOFS<<std::endl;}
            }
            statusOFS<<std::endl;
          }
#endif
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
          {
            statusOFS<<"[tag="<<this->tag_<<"] "<<"ROOT External buffer:"<<std::endl;
            for(Int i = 0; i<this->msgSize_;i++){
              statusOFS<<locBuffer[i]<<" ";
              if(i%10==0){statusOFS<<std::endl;}
            }
            statusOFS<<std::endl;
          }
#endif


          blas::Axpy(this->msgSize_, ONE<T>(), this->sendDataPtrs_[0], 1, locBuffer, 1 );
          this->sendTempBuffer_.clear(); 
          this->sendDataPtrs_[0] = locBuffer;
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
          {
            statusOFS<<"[tag="<<this->tag_<<"] "<<"ROOT Buffer after:"<<std::endl;
            for(Int i = 0; i<this->msgSize_;i++){
              statusOFS<<this->sendDataPtrs_[0][i]<<" ";
              if(i%10==0){statusOFS<<std::endl;}
            }
            statusOFS<<std::endl;
          }
#endif


        }

      }
    }

  template< typename T> 
    inline void TreeReduce_v2<T>::AllocRecvBuffers(){
      if(!this->isAllocated_){
        this->recvDataPtrs_.assign(this->GetDestCount(),NULL);
        this->recvTempBuffer_.resize(this->GetDestCount()*this->msgSize_);

        for( Int idxRecv = 0; idxRecv < this->GetDestCount(); ++idxRecv ){
          this->recvDataPtrs_[idxRecv] = (T*)&(this->recvTempBuffer_[idxRecv*this->msgSize_]);
        }

        this->recvRequests_.assign(this->GetDestCount(),MPI_REQUEST_NULL);
        this->recvStatuses_.resize(this->GetDestCount());
        this->recvDoneIdx_.resize(this->GetDestCount());

        this->sendRequests_.assign(1,MPI_REQUEST_NULL);

        this->isAllocated_ = true;
      }
    }

  template< typename T> 
    inline void TreeReduce_v2<T>::Reset(){
      TreeBcast_v2<T>::Reset();
      this->isAllocated_=false;
      this->isBufferSet_=false;
    }

  template< typename T> 
    inline bool TreeReduce_v2<T>::isMessageForwarded(){
      bool retVal=false;
//      if(this->tag_==12){gdb_lock();}
//      if(this->tag_==9){gdb_lock();}

      if(!this->fwded_){
        //If data has been received but not forwarded 
        if(this->IsDataReceived()){
          this->forwardMessage();
        }
        retVal = false;
      }
      else{
        //If data has been forwared, check for completion of send requests
        int destCount = this->myRank_==this->myRoot_?0:1;
        int completed = 0;
        if(destCount>0){
          //test the send requests
          int flag = 0;

          this->sendDoneIdx_.resize(destCount);

//          if(this->tag_==12){gdb_lock();}

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

        //MPI_Testall(destCount,sendRequests_.data(),&flag,MPI_STATUSES_IGNORE);
        //retVal = flag==1;
      }
      return retVal;
    }


  template< typename T>
    inline TreeReduce_v2<T> * TreeReduce_v2<T>::Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed){
      //get communicator size
      Int nprocs = 0;
      MPI_Comm_size(pComm, &nprocs);

#if defined(FTREE)
      return new FTreeReduce_v2<T>(pComm,ranks,rank_cnt,msgSize);
#elif defined(MODBTREE)
      return new ModBTreeReduce_v2<T>(pComm,ranks,rank_cnt,msgSize, rseed);
#elif defined(BTREE)
      return new BTreeReduce<T>(pComm,ranks,rank_cnt,msgSize);
#elif defined(PALMTREE)
      return new PalmTreeReduce_v2<T>(pComm,ranks,rank_cnt,msgSize);
#endif


      if(nprocs<=FTREE_LIMIT){
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
        statusOFS<<"FLAT TREE USED"<<std::endl;
#endif
        return new FTreeReduce_v2<T>(pComm,ranks,rank_cnt,msgSize);
      }
      else{
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
        statusOFS<<"BINARY TREE USED"<<std::endl;
#endif
        return new ModBTreeReduce_v2<T>(pComm,ranks,rank_cnt,msgSize, rseed);
      }
    }

  template< typename T>
    FTreeReduce_v2<T>::FTreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeReduce_v2<T>(pComm, ranks, rank_cnt, msgSize){
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline FTreeReduce_v2<T> * FTreeReduce_v2<T>::clone() const{
      FTreeReduce_v2<T> * out = new FTreeReduce_v2<T>(*this);
      return out;
    }



  template< typename T>
    inline void FTreeReduce_v2<T>::buildTree(Int * ranks, Int rank_cnt){

      Int idxStart = 0;
      Int idxEnd = rank_cnt;

      this->myRoot_ = ranks[0];

      if(this->myRank_==this->myRoot_){
        this->myDests_.insert(this->myDests_.end(),&ranks[1],&ranks[0]+rank_cnt);
      }

#if (defined(REDUCE_VERBOSE))
      statusOFS<<"My root is "<<this->myRoot_<<std::endl;
      statusOFS<<"My dests are ";
      for(int i =0;i<this->myDests_.size();++i){statusOFS<<this->myDests_[i]<<" ";}
      statusOFS<<std::endl;
#endif
    }

  template< typename T>
    inline void FTreeReduce_v2<T>::postRecv()
    {
      if(this->isAllocated_ && this->GetDestCount()>this->recvPostedCount_){
        int error_code = MPI_Irecv( (char*)this->recvDataPtrs_[0], this->msgSize_, this->type_, 
            MPI_ANY_SOURCE, this->tag_,this->comm_, &this->recvRequests_[0] );
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


        this->recvPostedCount_++;
      }
    }

  template< typename T>
    inline void FTreeReduce_v2<T>::AllocRecvBuffers(){
      if(!this->isAllocated_){
        this->recvDataPtrs_.assign(1,NULL);
        this->recvTempBuffer_.resize(this->msgSize_);

        this->recvDataPtrs_[0] = (T*)&(this->recvTempBuffer_[0]);

        this->recvRequests_.assign(1,MPI_REQUEST_NULL);
        this->recvStatuses_.resize(1);
        this->recvDoneIdx_.resize(1);
        this->sendRequests_.assign(1,MPI_REQUEST_NULL);

        this->isAllocated_ = true;
      }
    }


  template< typename T>
    inline bool FTreeReduce_v2<T>::Progress(){

      bool retVal = false;
      if(this->done_){
        retVal = true;
      }
      else{

        this->AllocRecvBuffers();

        if(this->isAllocated_){
          if(this->myRank_==this->myRoot_ && this->isAllocated_){
            this->isBufferSet_=true;
            this->isReady_=true;
          }

          if(this->isReady_ && this->isBufferSet_){
            if(this->IsDataReceived()){


              //free the unnecessary arrays
              this->recvTempBuffer_.clear();
              this->recvRequests_.clear();
              this->recvStatuses_.clear();
              this->recvDoneIdx_.clear();

              if(this->isMessageForwarded()){
                retVal = true;
              }
            }
            //else if(this->recvPostedCount_<this->GetDestCount()){
            //  //TODO check this
            //  if(this->recvPostedCount_==this->recvCount_){
            //    this->postRecv();
            //  }
            //}
          }
        }
      }

      if(retVal){
        this->done_ = retVal;
        //TODO do some smart cleanup
      }
      return retVal;
    }


  template< typename T>
    BTreeReduce_v2<T>::BTreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeReduce_v2<T>(pComm, ranks, rank_cnt, msgSize){
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline BTreeReduce_v2<T> * BTreeReduce_v2<T>::clone() const{
      BTreeReduce_v2<T> * out = new BTreeReduce_v2<T>(*this);
      return out;
    }

  template< typename T>
    inline void BTreeReduce_v2<T>::buildTree(Int * ranks, Int rank_cnt){
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

#if (defined(REDUCE_VERBOSE))
      statusOFS<<"My root is "<<this->myRoot_<<std::endl;
      statusOFS<<"My dests are ";
      for(int i =0;i<this->myDests_.size();++i){statusOFS<<this->myDests_[i]<<" ";}
      statusOFS<<std::endl;
#endif
    }


  template< typename T>
    ModBTreeReduce_v2<T>::ModBTreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed):TreeReduce_v2<T>(pComm, ranks, rank_cnt, msgSize){
      this->rseed_ = rseed;
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline void ModBTreeReduce_v2<T>::Copy(const ModBTreeReduce_v2<T> & Tree){
      ((TreeReduce_v2<T>*)this)->Copy(*((const TreeReduce_v2<T>*)&Tree));
      this->rseed_ = Tree.rseed_;
    }

  template< typename T>
    inline ModBTreeReduce_v2<T> * ModBTreeReduce_v2<T>::clone() const{
      ModBTreeReduce_v2<T> * out = new ModBTreeReduce_v2<T>(*this);
      return out;
    }

  template< typename T>
    inline void ModBTreeReduce_v2<T>::buildTree(Int * ranks, Int rank_cnt){

      Int idxStart = 0;
      Int idxEnd = rank_cnt;

      //sort the ranks with the modulo like operation
      if(rank_cnt>1){
        //generate a random position in [1 .. rand_cnt]
        //Int new_idx = (int)((rand()+1.0) * (double)rank_cnt / ((double)RAND_MAX+1.0));
        //srand(ranks[0]+rank_cnt);
        //Int new_idx = rseed_%(rank_cnt-1)+1;

        //Int new_idx = (int)((rank_cnt - 0) * ( (double)this->rseed_ / (double)RAND_MAX ) + 0);// (this->rseed_)%(rank_cnt-1)+1;
        //Int new_idx = (Int)rseed_ % (rank_cnt - 1) + 1; 
        //      Int new_idx = (int)((rank_cnt - 0) * ( (double)this->rseed_ / (double)RAND_MAX ) + 0);// (this->rseed_)%(rank_cnt-1)+1;
        Int new_idx = (int)(this->rseed_)%(rank_cnt-1)+1;

        Int * new_start = &ranks[new_idx];
        //        for(int i =0;i<rank_cnt;++i){statusOFS<<ranks[i]<<" ";} statusOFS<<std::endl;

        //        Int * new_start = std::lower_bound(&ranks[1],&ranks[0]+rank_cnt,ranks[0]);
        //just swap the two chunks   r[0] | r[1] --- r[new_start-1] | r[new_start] --- r[end]
        // becomes                   r[0] | r[new_start] --- r[end] | r[1] --- r[new_start-1] 
        std::rotate(&ranks[1], new_start, &ranks[0]+rank_cnt);
        //        for(int i =0;i<rank_cnt;++i){statusOFS<<ranks[i]<<" ";} statusOFS<<std::endl;
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
    PalmTreeReduce_v2<T>::PalmTreeReduce_v2(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeReduce_v2<T>(pComm,ranks,rank_cnt,msgSize){
      //build the binary tree;
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline PalmTreeReduce_v2<T> * PalmTreeReduce_v2<T>::clone() const{
      PalmTreeReduce_v2<T> * out = new PalmTreeReduce_v2<T>(*this);
      return out;
    }


  template< typename T>
    inline void PalmTreeReduce_v2<T>::buildTree(Int * ranks, Int rank_cnt){
      Int numLevel = floor(log2(rank_cnt));
      Int numRoots = 0;
      for(Int level=0;level<numLevel;++level){
        numRoots = std::min( rank_cnt, numRoots + (Int)pow(2,level));
        Int numNextRoots = std::min(rank_cnt,numRoots + (Int)pow(2,(level+1)));
        Int numReceivers = numNextRoots - numRoots;
        for(Int ip = 0; ip<numRoots;++ip){
          Int p = ranks[ip];
          for(Int ir = ip; ir<numReceivers;ir+=numRoots){
            Int r = ranks[numRoots+ir];
            if(r==this->myRank_){
              this->myRoot_ = p;
            }

            if(p==this->myRank_){
              this->myDests_.push_back(r);
            }
          }
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
    void TreeReduce_Waitsome(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeReduce_v2<T> > > & arrTrees, std::list<int> & doneIdx, std::vector<bool> & finishedFlags){
      doneIdx.clear();
      auto all_done = [](const std::vector<bool> & boolvec){
        return std::all_of(boolvec.begin(), boolvec.end(), [](bool v) { return v; });
      };

      while(doneIdx.empty() && !all_done(finishedFlags) ){
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
    void TreeReduce_Testsome(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeReduce_v2<T> > > & arrTrees, std::list<int> & doneIdx, std::vector<bool> & finishedFlags){
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
    void TreeReduce_Waitall(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeReduce_v2<T> > > & arrTrees){
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

  template< typename T>
    void TreeReduce_ProgressAll(std::vector<Int> & treeIdx, std::vector< std::shared_ptr<TreeReduce_v2<T> > > & arrTrees){

      for(int i = 0; i<treeIdx.size(); i++){
        Int idx = treeIdx[i];
        auto & curTree = arrTrees[idx];
        if(curTree!=nullptr){
          bool done = curTree->Progress();
        }
      }
    }



} //namespace PEXSI
#endif
