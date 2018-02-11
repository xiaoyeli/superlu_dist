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

/// @namespace ASYNCOMM
/// @brief The main namespace.

namespace ASYNCOMM{

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

} // namespace ASYNCOMM

/***********************************************************************
 *  Error handling
 **********************************************************************/

namespace ASYNCOMM{






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


} // namespace ASYNCOMM


#endif // _PEXSI_ENVIRONMENT_HPP_
