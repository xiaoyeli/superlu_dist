#ifndef __SUPERLU_TREEREDUCE_IMPL
#define __SUPERLU_TREEREDUCE_IMPL

namespace SuperLU_ASYNCOMM {
	
  template<typename T>
    TreeReduce_slu<T>::TreeReduce_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeBcast_slu<T>(pComm,ranks,rank_cnt,msgSize){
      this->sendDataPtrs_.assign(1,NULL);
      this->sendRequests_.assign(1,MPI_REQUEST_NULL);
      this->isAllocated_=false;
      this->isBufferSet_=false;
    }




  template<typename T>
    TreeReduce_slu<T>::TreeReduce_slu(const TreeReduce_slu<T> & Tree){
      this->Copy(Tree);
    }

  template<typename T>
    TreeReduce_slu<T>::TreeReduce_slu():TreeBcast_slu<T>(){
    }

  template<typename T>
    TreeReduce_slu<T>::~TreeReduce_slu(){
      this->cleanupBuffers();
    }

  template<typename T>
    inline void TreeReduce_slu<T>::Copy(const TreeReduce_slu<T> & Tree){
      ((TreeBcast_slu<T>*)this)->Copy(*(const TreeBcast_slu<T>*)&Tree);

      this->sendDataPtrs_.assign(1,NULL);
      this->sendRequests_.assign(1,MPI_REQUEST_NULL);
      this->isAllocated_= Tree.isAllocated_;
      this->isBufferSet_= Tree.isBufferSet_;

      this->cleanupBuffers();
    }
	
  template< typename T> 
    inline void TreeReduce_slu<T>::forwardMessageSimple(T * locBuffer, Int msgSize){
        MPI_Status status;
		Int flag;
		if(this->myRank_!=this->myRoot_){
			// if(this->recvCount_== this->GetDestCount()){		
			  //forward to my root if I have reseived everything
			  Int iProc = this->myRoot_;
			  // Use Isend to send to multiple targets

			  Int error_code = MPI_Isend(locBuffer, msgSize, this->type_, 
				  iProc, this->tag_,this->comm_, &this->sendRequests_[0] );
				  
				  MPI_Test(&this->sendRequests_[0],&flag,&status) ; 
				  
				  // std::cout<<this->myRank_<<" FWD to "<<iProc<<" on tag "<<this->tag_<<std::endl;
				  
				 // MPI_Wait(&this->sendRequests_[0],&status) ; 
				  
			// }
		}
      }
	
 

  template< typename T> 
    inline void TreeReduce_slu<T>::allocateRequest(){
        if(this->sendRequests_.size()==0){
          this->sendRequests_.resize(1);
        }
		this->sendRequests_.assign(1,MPI_REQUEST_NULL);
    }
		
	
  template< typename T> 
    inline void TreeReduce_slu<T>::waitSendRequest(){
        MPI_Status status;		
        if(this->sendRequests_.size()>0){
		  MPI_Wait(&this->sendRequests_[0],&status) ; 
        }	
	}	



  template< typename T> 
    inline T * TreeReduce_slu<T>::GetLocalBuffer(){ 
      return this->sendDataPtrs_[0];
    }


  template< typename T> 
    inline void TreeReduce_slu<T>::Reset(){
      TreeBcast_slu<T>::Reset();
      this->isAllocated_=false;
      this->isBufferSet_=false;
    }


  template< typename T>
    inline TreeReduce_slu<T> * TreeReduce_slu<T>::Create(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed){
      //get communicator size
      Int nprocs = 0;
      MPI_Comm_size(pComm, &nprocs);

      if(nprocs<=FTREE_LIMIT){
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
        statusOFS<<"FLAT TREE USED"<<std::endl;
#endif
        return new FTreeReduce_slu<T>(pComm,ranks,rank_cnt,msgSize);
      }
      else{
#if ( _DEBUGlevel_ >= 1 ) || defined(REDUCE_VERBOSE)
        statusOFS<<"BINARY TREE USED"<<std::endl;
#endif
        // return new ModBTreeReduce_slu<T>(pComm,ranks,rank_cnt,msgSize, rseed);
		return new BTreeReduce_slu<T>(pComm,ranks,rank_cnt,msgSize);
      }
    }

  template< typename T>
    FTreeReduce_slu<T>::FTreeReduce_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeReduce_slu<T>(pComm, ranks, rank_cnt, msgSize){
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline FTreeReduce_slu<T> * FTreeReduce_slu<T>::clone() const{
      FTreeReduce_slu<T> * out = new FTreeReduce_slu<T>(*this);
      return out;
    }



  template< typename T>
    inline void FTreeReduce_slu<T>::buildTree(Int * ranks, Int rank_cnt){

      Int idxStart = 0;
      Int idxEnd = rank_cnt;

      this->myRoot_ = ranks[0];

      if(this->myRank_==this->myRoot_){
        this->myDests_.insert(this->myDests_.end(),&ranks[1],&ranks[0]+rank_cnt);
      }

#if (defined(REDUCE_VERBOSE))
      statusOFS<<"My root is "<<this->myRoot_<<std::endl;
      statusOFS<<"My dests are ";
      for(Int i =0;i<this->myDests_.size();++i){statusOFS<<this->myDests_[i]<<" ";}
      statusOFS<<std::endl;
#endif
    }

  template< typename T>
    BTreeReduce_slu<T>::BTreeReduce_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize):TreeReduce_slu<T>(pComm, ranks, rank_cnt, msgSize){
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline BTreeReduce_slu<T> * BTreeReduce_slu<T>::clone() const{
      BTreeReduce_slu<T> * out = new BTreeReduce_slu<T>(*this);
      return out;
    }


	template< typename T>
    inline void BTreeReduce_slu<T>::buildTree(Int * ranks, Int rank_cnt){
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
	  
	  // for(Int i =0;i<this->myDests_.size();++i){std::cout<<this->myRank_<<" "<<this->myDests_[i]<<" "<<std::endl;}

	  // {std::cout<<this->myRank_<<" "<<this->myRoot_<<" "<<std::endl;}	  
	  
    }


  template< typename T>
    ModBTreeReduce_slu<T>::ModBTreeReduce_slu(const MPI_Comm & pComm, Int * ranks, Int rank_cnt, Int msgSize, double rseed):TreeReduce_slu<T>(pComm, ranks, rank_cnt, msgSize){
      this->rseed_ = rseed;
      buildTree(ranks,rank_cnt);
    }

  template< typename T>
    inline void ModBTreeReduce_slu<T>::Copy(const ModBTreeReduce_slu<T> & Tree){
      ((TreeReduce_slu<T>*)this)->Copy(*((const TreeReduce_slu<T>*)&Tree));
      this->rseed_ = Tree.rseed_;
    }

  template< typename T>
    inline ModBTreeReduce_slu<T> * ModBTreeReduce_slu<T>::clone() const{
      ModBTreeReduce_slu<T> * out = new ModBTreeReduce_slu<T>(*this);
      return out;
    }

  template< typename T>
    inline void ModBTreeReduce_slu<T>::buildTree(Int * ranks, Int rank_cnt){

      Int idxStart = 0;
      Int idxEnd = rank_cnt;

      //sort the ranks with the modulo like operation
      if(rank_cnt>1){
        //generate a random position in [1 .. rand_cnt]
        //Int new_idx = (Int)((rand()+1.0) * (double)rank_cnt / ((double)RAND_MAX+1.0));
        //srand(ranks[0]+rank_cnt);
        //Int new_idx = rseed_%(rank_cnt-1)+1;

        //Int new_idx = (Int)((rank_cnt - 0) * ( (double)this->rseed_ / (double)RAND_MAX ) + 0);// (this->rseed_)%(rank_cnt-1)+1;
        //Int new_idx = (Int)rseed_ % (rank_cnt - 1) + 1; 
        //      Int new_idx = (Int)((rank_cnt - 0) * ( (double)this->rseed_ / (double)RAND_MAX ) + 0);// (this->rseed_)%(rank_cnt-1)+1;
        Int new_idx = (Int)(this->rseed_)%(rank_cnt-1)+1;

        Int * new_start = &ranks[new_idx];
        //        for(Int i =0;i<rank_cnt;++i){statusOFS<<ranks[i]<<" ";} statusOFS<<std::endl;

        //        Int * new_start = std::lower_bound(&ranks[1],&ranks[0]+rank_cnt,ranks[0]);
        //just swap the two chunks   r[0] | r[1] --- r[new_start-1] | r[new_start] --- r[end]
        // becomes                   r[0] | r[new_start] --- r[end] | r[1] --- r[new_start-1] 
        std::rotate(&ranks[1], new_start, &ranks[0]+rank_cnt);
        //        for(Int i =0;i<rank_cnt;++i){statusOFS<<ranks[i]<<" ";} statusOFS<<std::endl;
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
