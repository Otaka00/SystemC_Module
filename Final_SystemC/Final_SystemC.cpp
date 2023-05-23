#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/data/example.h>
#include <torch/csrc/jit/serialization/import.h>
#include "systemc.h"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"

using namespace sc_core;
using namespace tlm;

SC_MODULE(Initiator) {
    tlm_utils::simple_initiator_socket<Initiator> init_socket;

    SC_CTOR(Initiator) : init_socket("init_socket") {
        SC_THREAD(sendDataThread);
    }

    void sendDataThread() {
        sc_time tLOCAL(SC_ZERO_TIME);
        unsigned int addr = static_cast<unsigned int>(rand() % 0x100);
        cout << "address: " << addr << "\n";

        std::vector<float> input_data = { 1.0,2.0,3.0,4.0 }; // example input data

        int input_size = input_data.size() * sizeof(float);
        cout << input_data << input_size;
        // Create a payload for the transaction
        tlm_generic_payload payload;
        payload.set_data_ptr(reinterpret_cast<unsigned char*>(input_data.data()));
        payload.set_data_length(input_size);
        payload.set_write();

        // Send the payload through the init_socket
        SC_REPORT_INFO("A", "Doing a WRITE transaction");
        init_socket->b_transport(payload, tLOCAL);

       /* const float* data_ptr = reinterpret_cast<float*>(payload.get_data_ptr());
        cout << "Data pointer: " << data_ptr << "\n";

        std::vector<float> soft_output_data;
        soft_output_data.resize(payload.get_data_length() / sizeof(float));
        std::copy(data_ptr, data_ptr + soft_output_data.size(), soft_output_data.begin());

        cout << "New Output: ";
        for (int i = 0; i < soft_output_data.size(); ++i)
            cout << soft_output_data[i] << " ";*/

        cout << "\n---------------------------------------------------------------------------------\n";
        // Extract the input data from the payload
        float* input_data2 = reinterpret_cast<float*>(payload.get_data_ptr());
        int input_size2 = payload.get_data_length() / sizeof(float);
        cout << "Inputs received in initiator module: ";
        for (int i = 0; i < input_size2; ++i) 
          cout << input_data2[i] << " ";
 
        cout << endl;
        // Handle response or check for errors
        if (payload.is_response_error())
            SC_REPORT_ERROR("A", "Received error reply.");
        else
            SC_REPORT_INFO("A", "Received correct reply.");
    }
};

SC_MODULE(Target) {
    tlm_utils::simple_target_socket<Target> target_socket;

    SC_CTOR(Target) : target_socket("target_socket") {
        target_socket.register_b_transport(this, &Target::b_transport);
    }
    std::vector<float> flattened_array;

    void b_transport(tlm_generic_payload & payload, sc_time & tLOCAL) {

        // Extract the input data from the payload
        float* input_data = reinterpret_cast<float*>(payload.get_data_ptr());
        int input_size = payload.get_data_length() / sizeof(float);

        cout << "Input received in Target module: ";
        for (int i = 0; i < input_size; ++i) {
            cout << input_data[i] << " ";
        }
        cout << endl;

        torch::jit::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load("E:\\courses\\Graduation Project\\trained_model.pt");
            // module = torch:: ::load("trained_model.pt");

        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n" << e.msg();
            return;
        }
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({ 1,1, 28, 28 }));;
        std::cout << "Inputs: " << inputs << "\n";
        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        at::Tensor soft_output = torch::softmax(output, 1);

        std::cout << output << '\n';
        std::cout << "Predictions: " << soft_output << '\n';

        // Access the underlying data as a C++ array 
        auto soft_output_accessor = soft_output.accessor<float, 2>();
        std::vector<std::vector<float>> soft_output_array;
        // Iterate over the tensor elements and populate the array 
        for (int i = 0; i < soft_output_accessor.size(0); ++i) {
            std::vector<float> row;
            for (int j = 0; j < soft_output_accessor.size(1); ++j)
                row.push_back(soft_output_accessor[i][j]);
            soft_output_array.push_back(row);
        }

        for (const auto& row : soft_output_array) {
            cout << "2D array: ";
            for (const auto& value : row)
                cout << value << " ";
            cout << endl;
        }

        ////////////////////////// Flattening to 1D array
        // Calculate the total number of elements
        int total_elements = soft_output_array.size() * soft_output_array[0].size();

        // Create a contiguous 1D array and copy the elements
        int index = 0;
        flattened_array.resize(total_elements);
        for (int j = 0; j < total_elements; j++) {
            flattened_array[j] = soft_output_array[0][j];
            cout << flattened_array[j];
        }

        // Set the data pointer and length in the payload
        payload.set_data_ptr(reinterpret_cast<unsigned char*>(flattened_array.data()));
        payload.set_data_length(total_elements * sizeof(float));
        cout << "flattened array: ";
            for (int i = 0; i < total_elements; i++) {
                cout << flattened_array[i];
            }

        // Extract the input data from the payload
       // float* input_data = reinterpret_cast<float*>(payload.get_data_ptr());
       // int input_size = payload.get_data_length() / sizeof(float);

        int batch_size = 1;
        int num_channels = 1;
        int height = 32;
        int width = 32;

        if (payload.is_read())
            SC_REPORT_INFO("B", "Doing a READ transaction");

        else if (payload.is_write())
            SC_REPORT_INFO("B", "Doing a WRITE transaction");

        // Set the response status
        payload.set_response_status(TLM_OK_RESPONSE);

        // Send the response back through the target_socket
        //target_socket->b_transport(payload, tLOCAL);
    }
};

void setMKLThreads() {
    int numThreads = 4; // Set the desired number of threads
    torch::set_num_threads(numThreads);
}

int sc_main(int argc, char* argv[]) {
    setMKLThreads();
    // Create instances of the initiator and target modules
    Initiator initiator("initiator");
    Target target("target");

    // Connect the sockets
    initiator.init_socket.bind(target.target_socket);

    // Start simulation
    sc_start();

    return 0;
}