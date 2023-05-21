#include "systemc.h"
#include "top.h"

#ifdef MTI_SYSTEMC
SC_MODULE_EXPORT(top)
#endif

#ifdef NC_SYSTEMC
NCSC_MODULE_EXPORT(top)
#endif


using namespace sc_core; // SystemC core functions

SC_HAS_PROCESS(top);

top::top(sc_module_name nm) // Constructor
	: sc_module(nm) // initialize module
{
	// Create sub-module instances
	A1 = new A("A1");
	B1 = new B("B1");

	// Connectivity
	A1->init_socket(B1->targ_socket);
}


