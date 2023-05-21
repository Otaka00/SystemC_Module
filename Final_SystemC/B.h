#ifndef _B_H_
#define _B_H_

#include "systemc.h"
#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"

using namespace tlm;


class B : public sc_core::sc_module {
public:
    tlm_utils::simple_target_socket<B> targ_socket;


    SC_CTOR(B) {
        SC_REPORT_INFO("A", "Constructing sc_module B");
        targ_socket.register_b_transport(this, &B::b_transport);
    }

    void b_transport( // Blocking transport
        tlm::tlm_generic_payload& payload,
        sc_core::sc_time& tLOCAL
    ) {
        payload.set_response_status(TLM_OK_RESPONSE); // Assume ok
        if (payload.get_data_length() != 1) SC_REPORT_FATAL("mem", "Size!=1");
        uint64         addr = payload.get_address();
        unsigned char* data_ptr = payload.get_data_ptr();
        if (payload.is_read()) {
            SC_REPORT_INFO("B", "Doing a READ transaction");
        }
        else if (payload.is_write()) {
            SC_REPORT_INFO("B", "Doing a WRITE transaction");
        }//endif
    }


};

#endif
