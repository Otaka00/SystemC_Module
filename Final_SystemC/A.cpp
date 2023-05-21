#include "systemc.h"
#include "tlm.h"
#include "A.h"

using namespace tlm;

void A::initiator_thread() {
    tlm_generic_payload payload;
    unsigned int  addr;
    unsigned char data;
    sc_time tLOCAL(SC_ZERO_TIME);

    addr = static_cast<unsigned int>(rand() % 0x1000);
    data = static_cast<unsigned char>(rand() % 256);
    payload.set_address(addr);
    payload.set_data_ptr(&data);
    payload.set_data_length(1);
    payload.set_write();
    SC_REPORT_INFO("A", "Doing a WRITE transaction");
    init_socket->b_transport(payload, tLOCAL);

    if (payload.is_response_error()) {
        SC_REPORT_ERROR("A", "Received error reply.");
    }
}

