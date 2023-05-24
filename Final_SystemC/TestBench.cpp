#include "systemc.h"
#include "tlm.h"
#include "Final_SystemC.cpp"

SC_MODULE(Testbench) {
    Initiator initiator;
    Target target;

    SC_CTOR(Testbench) : initiator("initiator"), target("target") {
        // Connect the sockets
        initiator.init_socket.bind(target.target_socket);

        // Register the b_transport function
        target.target_socket.register_b_transport(&target, &Target::b_transport);
    }

    void b_transport(tlm_generic_payload &payload, sc_time &t, const sc_core::sc_port_b<tlm_fw_transport_if<>>&port) {

        // Check if the payload is a write transaction
        if (payload.is_write()) {
            // Extract the flattened array from the payload
            float* flattened_array = reinterpret_cast<float*>(payload.get_data_ptr());
            int flattened_array_size = payload.get_data_length() / sizeof(float);

            // Print the flattened array received from the target module
            cout << "Flattened Array received in Testbench: ";
            for (int i = 0; i < flattened_array_size; ++i) {
                cout << flattened_array[i] << " ";
            }
            cout << endl;
        }
    }
};

void setMKLThreads() {
    int numThreads = 4; // Set the desired number of threads
    torch::set_num_threads(numThreads);
}

int sc_main(int argc, char* argv[]) {
     setMKLThreads();
    // Create an instance of the Testbench
    Testbench testbench("testbench");

    // Start the simulation
    sc_start();

    return 0;
}