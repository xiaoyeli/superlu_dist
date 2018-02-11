#include "environment.hpp"
  #include <deque>

namespace ASYNCOMM{

// *********************************************************************
// IO
// *********************************************************************
  std::ofstream  statusOFS;


// *********************************************************************
// Error handling
// *********************************************************************
	// If we are not in RELEASE mode, then implement wrappers for a
	// CallStack
#ifndef _RELEASE_
	std::stack<std::string> callStack;	

	void PushCallStack( std::string s )
	{ callStack.push(s); }

	void PopCallStack()
	{ callStack.pop(); }

	void DumpCallStack()
	{
		std::ostringstream msg;
		while( ! callStack.empty() )
		{
			msg << "Stack[" << callStack.size() << "]: " << callStack.top() << "\n";
			callStack.pop();
		}
		std::cerr << msg.str() << std::endl;
	}

#endif // ifndef _RELEASE_
} // namespace ASYNCOMM
