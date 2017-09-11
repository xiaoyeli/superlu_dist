/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin
	 
   This file is part of PEXSI. All rights reserved.

	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:

	 (1) Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 (2) Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 (3) Neither the name of the University of California, Lawrence Berkeley
	 National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
	 be used to endorse or promote products derived from this software without
	 specific prior written permission.

	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	 You are under no obligation whatsoever to provide any bug fixes, patches, or
	 upgrades to the features, functionality or performance of the source code
	 ("Enhancements") to anyone; however, if you choose to make your Enhancements
	 available either publicly, or directly to Lawrence Berkeley National
	 Laboratory, without imposing a separate written license agreement for such
	 Enhancements, then you hereby grant the following license: a non-exclusive,
	 royalty-free perpetual license to install, use, modify, prepare derivative
	 works, incorporate into other computer software, distribute, and sublicense
	 such enhancements or derivative works thereof, in binary and source code form.
*/
/// @file environment.hpp
/// @brief Environmental variables.
/// @date 2012-08-10
#ifndef _PEXSI_ENVIRONMENT_HPP_
#define _PEXSI_ENVIRONMENT_HPP_

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



// *********************************************************************
// Redefine the global macros
// *********************************************************************


// FIXME Always use complex data for pexsi and ppexsi.
#define _USE_COMPLEX_

// The verbose level of debugging information
#ifdef  DEBUG
#define _DEBUGlevel_ DEBUG
#endif

// Release mode. For speed up the calculation and reduce verbose level.
// Note that RELEASE overwrites DEBUG level.
#ifdef RELEASE
#define _RELEASE_
#define _DEBUGlevel -1
#endif

/***********************************************************************
 *  Data types and constants
 **********************************************************************/

/// @namespace PEXSI
/// @brief The main namespace.

namespace PEXSI{

// Basic data types

#ifndef Add_
#define FORTRAN(name) name
#define BLAS(name) name
#define LAPACK(name) name
#else
#define FORTRAN(name) name##_
#define BLAS(name) name##_
#define LAPACK(name) name##_
#endif
typedef    int                   Int;
typedef    int64_t               LongInt;
typedef    double                Real;
typedef    std::complex<double>  Complex; // Must use elemental form of complex
#ifdef _USE_COMPLEX_
typedef    std::complex<double>  Scalar;  // Must use elemental form of complex
#else
typedef    double                Scalar;
#endif

// IO
extern  std::ofstream  statusOFS;
#ifdef GEMM_PROFILE
extern  std::ofstream  statOFS;
#include <deque>
extern std::deque<int > gemm_stat;
#endif

#if defined(COMM_PROFILE) || defined(COMM_PROFILE_BCAST)
extern  std::ofstream  commOFS;
#include <deque>
extern std::deque<int > comm_stat;

#define PROFILE_COMM(sender,receiver,tag,size)\
do{\
  comm_stat.push_back(sender);\
  comm_stat.push_back(receiver);\
  comm_stat.push_back(tag);\
  comm_stat.push_back(size);\
}while(0)

#define HEADER_COMM "Sender\tReceiver\tTag\tSize"
#define LINE_COMM(it) *it<<"\t"<<*(it+1)<<"\t"<<*(it+2)<<"\t"<<*(it+3)

#else

#define PROFILE_COMM(sender,receiver,tag,size)

#endif


// *********************************************************************
// Define constants
// *********************************************************************
// Commonly used
const Int I_ZERO = 0;
const Int I_ONE  = 1;
const Int I_MINUS_ONE  = -1;
const Real D_ZERO = 0.0;
const Real D_ONE  = 1.0;
const Real D_MINUS_ONE  = -1.0;
const Complex Z_ZERO = Complex(0.0, 0.0);
const Complex Z_ONE  = Complex(1.0, 0.0);
const Complex Z_MINUS_ONE  = Complex(-1.0, 0.0);
const Complex Z_I    = Complex(0.0, 1.0);
const Complex Z_MINUS_I    = Complex(0.0, -1.0);
const Scalar SCALAR_ZERO    = static_cast<Scalar>(0.0);
const Scalar SCALAR_ONE     = static_cast<Scalar>(1.0);
const Scalar SCALAR_MINUS_ONE = static_cast<Scalar>(-1.0);

template<typename T>
const T ZERO(){ return static_cast<T>(0.0);};
template<typename T>
const T ONE(){ return static_cast<T>(1.0);};
template<typename T>
const T MINUS_ONE(){ return static_cast<T>(-1.0);};

const char UPPER = 'U';
const char LOWER = 'L';

// Physical constants

const Real au2K = 315774.67;
const Real PI = 3.141592653589793;

} // namespace PEXSI

/***********************************************************************
 *  Error handling
 **********************************************************************/

namespace PEXSI{






  inline void gdb_lock(){
    volatile int lock = 1;
    statusOFS<<"LOCKED"<<std::endl;
    while (lock == 1){ }
  }







#ifndef _RELEASE_
  void PushCallStack( std::string s );
  void PopCallStack();
  void DumpCallStack();
#endif // ifndef _RELEASE_

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
	inline Int iround(Real a){ 
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


} // namespace PEXSI


#endif // _PEXSI_ENVIRONMENT_HPP_
