/// @file asyncomm.hpp
/// @brief Namespace defining variables used in asynchronous broadcast/reduction tree.
/// @date 2018-09-06
#ifndef __SUPERLU_ASYNCOMM_HPP // allow multiple inclusions
#define __SUPERLU_ASYNCOMM_HPP 

// STL libraries
#include <iostream> 
#include <iomanip> 
#include <fstream>
#include <sstream>
#include <unistd.h>

#include <cfloat>
#include <complex>
#include <string>

#include <set>
#include <map>
#include <stack>
#include <vector>

#include <algorithm>
#include <cmath>

#include <cassert>
#include <stdexcept>
#include <execinfo.h>
//#include <signal.h>
#include <exception>

// For 32-bit and 64-bit integers
#include <stdint.h>

// MPI
#include <mpi.h>


/***********************************************************************
 *  Data types and constants
 **********************************************************************/

/// @namespace ASYNCOMM
/// @brief The main namespace.

namespace SuperLU_ASYNCOMM {

    // Basic data types
    typedef    int                   Int;

    // IO
    extern  std::ofstream  statusOFS;

    // *********************************************************************
    // Define constants
    // *********************************************************************
    // Commonly used
    const Int DEG_TREE = 2; //number of children of each tree node

} // namespace SuperLU_ASYNCOMM

/***********************************************************************
 *  Error handling
 **********************************************************************/

namespace SuperLU_ASYNCOMM {

  inline void gdb_lock(){
    volatile int lock = 1;
    statusOFS<<"LOCKED"<<std::endl;
    while (lock == 1){ }
  }

#if 0
#ifndef _RELEASE_
  void PushCallStack( std::string s );
  void PopCallStack();
  void DumpCallStack();
#endif // ifndef _RELEASE_
#endif

  // We define an output stream that does nothing. This is done so that the 
  // root process can be used to print data to a file's ostream while all other 
  // processes use a null ostream. 
  struct NullStream : std::ostream
  {            
    struct NullStreamBuffer : std::streambuf
    {
      Int overflow( Int c ) { return traits_type::not_eof(c); }
    } nullStreamBuffer_;

    NullStream() 
      : std::ios(&nullStreamBuffer_), std::ostream(&nullStreamBuffer_)
      { }
  };  

  /////////////////////////////////////////////

  class ExceptionTracer
  {
  public:
    ExceptionTracer()
    {
      void * array[25];
      int nSize = backtrace(array, 25);
      char ** symbols = backtrace_symbols(array, nSize);

      for (int i = 0; i < nSize; i++)
      {
	std::cout << symbols[i] << std::endl;
      }

      free(symbols);
    }
  };

  // *********************************************************************
  // Global utility functions 
  // These utility functions do not depend on local definitions
  // *********************************************************************
  // Return the closest integer to a real number
	inline Int iround(double a){ 
		Int b = 0;
		if(a>0) b = (a-Int(a)<0.5)?Int(a):(Int(a)+1);
		else b = (Int(a)-a<0.5)?Int(a):(Int(a)-1);
		return b; 
	}

  // Read the options from command line
	inline void OptionsCreate(Int argc, char** argv, std::map<std::string,std::string>& options){
		options.clear();
		for(Int k=1; k<argc; k=k+2) {
			options[ std::string(argv[k]) ] = std::string(argv[k+1]);
		}
	}

	// Size information.
	// Like sstm.str().length() but without making the copy
	inline Int Size( std::stringstream& sstm ){
		Int length;
		sstm.seekg (0, std::ios::end);
		length = sstm.tellg();
		sstm.seekg (0, std::ios::beg);
		return length;
	}


} // namespace SuperLU_ASYNCOMM

#endif // __SUPERLU_ASYNCOMM_HPP 
