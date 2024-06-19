// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "grpc_client.h"
#include "timings_reports.hpp"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                                                                                                                                        \
    {                                                                                                                                                                              \
        tc::Error err = (X);                                                                                                                                                       \
        if (!err.IsOk())                                                                                                                                                           \
        {                                                                                                                                                                          \
            std::cerr << "error: " << (MSG) << ": " << err << std::endl;                                                                                                           \
            exit(1);                                                                                                                                                               \
        }                                                                                                                                                                          \
    }

int main(int argc, char** argv)
{
    bool verbose = false;
    std::string url("localhost:8001");
    tc::Headers http_headers;
    uint32_t client_timeout = 0;
    bool use_ssl = false;
    std::string root_certificates;
    std::string private_key;
    std::string certificate_chain;
    grpc_compression_algorithm compression_algorithm = grpc_compression_algorithm::GRPC_COMPRESS_NONE;
    bool use_cached_channel = true;

    // We use a simple model that takes 2 input tensors of 16 integers
    // each and returns 2 output tensors of 16 integers each. One output
    // tensor is the element-wise sum of the inputs and one output is
    // the element-wise difference.
    std::string model_name = "simple_int32";
    std::string model_version = "";

    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    tc::SslOptions ssl_options = tc::SslOptions();
    std::string err;
    if (use_ssl)
    {
        ssl_options.root_certificates = root_certificates;
        ssl_options.private_key = private_key;
        ssl_options.certificate_chain = certificate_chain;
        err = "unable to create secure grpc client";
    }
    else
    {
        err = "unable to create grpc client";
    }
    
    FAIL_IF_ERR(tc::InferenceServerGrpcClient::Create(&client, url, verbose, use_ssl, ssl_options, tc::KeepAliveOptions(), use_cached_channel), err);
    
    std::vector<int32_t> input0_data { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::vector<int32_t> input1_data { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<int64_t> shape { 1, 16 };

    std::vector<int64_t> times;
    for(int n=0; n<100; ++n)
    {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i)
        {
            // Initialize the inputs with the data.
            tc::InferInput* input0;
            tc::InferInput* input1;

            FAIL_IF_ERR(tc::InferInput::Create(&input0, "INPUT0", shape, "INT32"), "unable to get INPUT0");
            std::shared_ptr<tc::InferInput> input0_ptr;
            input0_ptr.reset(input0);

            FAIL_IF_ERR(tc::InferInput::Create(&input1, "INPUT1", shape, "INT32"), "unable to get INPUT1");
            std::shared_ptr<tc::InferInput> input1_ptr;
            input1_ptr.reset(input1);

            FAIL_IF_ERR(input0_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&input0_data[0]), input0_data.size() * sizeof(int32_t)), "unable to set data for INPUT0");
            FAIL_IF_ERR(input1_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&input1_data[0]), input1_data.size() * sizeof(int32_t)), "unable to set data for INPUT1");

            // Generate the outputs to be requested.
            tc::InferRequestedOutput* output0;
            tc::InferRequestedOutput* output1;

            FAIL_IF_ERR(tc::InferRequestedOutput::Create(&output0, "OUTPUT0"), "unable to get 'OUTPUT0'");
            std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
            output0_ptr.reset(output0);
            
            FAIL_IF_ERR(tc::InferRequestedOutput::Create(&output1, "OUTPUT1"), "unable to get 'OUTPUT1'");
            std::shared_ptr<tc::InferRequestedOutput> output1_ptr;
            output1_ptr.reset(output1);

            // The inference settings. Will be using default for now.
            tc::InferOptions options(model_name);
            options.model_version_ = model_version;
            options.client_timeout_ = client_timeout;

            std::vector<tc::InferInput*> inputs = { input0_ptr.get(), input1_ptr.get() };
            std::vector<const tc::InferRequestedOutput*> outputs = { output0_ptr.get(), output1_ptr.get() };

            tc::InferResult* results;
            FAIL_IF_ERR(client->Infer(&results, options, inputs, outputs, http_headers, compression_algorithm), "unable to run model");
            std::shared_ptr<tc::InferResult> results_ptr;
            results_ptr.reset(results);

            // Get pointers to the result returned...
            int32_t* output0_data;
            size_t output0_byte_size;
            FAIL_IF_ERR(results_ptr->RawData("OUTPUT0", (const uint8_t**)&output0_data, &output0_byte_size), "unable to get result data for 'OUTPUT0'");
            if (output0_byte_size != 64)
            {
                std::cerr << "error: received incorrect byte size for 'OUTPUT0': " << output0_byte_size << std::endl;
                exit(1);
            }

            int32_t* output1_data;
            size_t output1_byte_size;
            FAIL_IF_ERR(results_ptr->RawData("OUTPUT1", (const uint8_t**)&output1_data, &output1_byte_size), "unable to get result data for 'OUTPUT1'");
            if (output1_byte_size != 64)
            {
                std::cerr << "error: received incorrect byte size for 'OUTPUT1': " << output1_byte_size << std::endl;
                exit(1);
            }

            for (size_t i = 0; i < 16; ++i)
            {
                if ((input0_data[i] + input1_data[i]) != *(output0_data + i))
                {
                    std::cerr << "error: incorrect sum" << std::endl;
                    exit(1);
                }
                if ((input0_data[i] - input1_data[i]) != *(output1_data + i))
                {
                    std::cerr << "error: incorrect difference" << std::endl;
                    exit(1);
                }
            }
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        times.push_back(elapsed.count());
    }

    timings::print_stats(times);
    return 0;
}