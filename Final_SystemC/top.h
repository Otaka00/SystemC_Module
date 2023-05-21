#ifndef _TOP_H_
#define _TOP_H_

#include "systemc.h"
#include "tlm.h"
#include "A.h"
#include "B.h"

using namespace tlm;


struct top : sc_core::sc_module {
public:

	A* A1;
	B* B1;

	SC_CTOR(top); // Constructor

};

#endif
