#include <benchmark/benchmark.h>
#include "grpc_client.h"


#define FAIL_IF_ERR(X, MSG)                                                                                                                                                        \
{                                                                    \
    triton::client::Error err = (X);                                 \
    if (!err.IsOk())                                                 \
    {                                                                \
        std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
        exit(1);                                                     \
    }                                                                \
}

static void benchmark_triton_client(benchmark::State& state)
{
    bool verbose = false;
    std::string url("localhost:8001");
    triton::client::Headers http_headers;
    uint32_t client_timeout = 0;
    bool use_ssl = false;
    std::string root_certificates;
    std::string private_key;
    std::string certificate_chain;
    grpc_compression_algorithm compression_algorithm = grpc_compression_algorithm::GRPC_COMPRESS_NONE;
    bool use_cached_channel = true;
    std::string model_name = "simple_int32";
    std::string model_version = "";

    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<triton::client::InferenceServerGrpcClient> client;
    triton::client::SslOptions ssl_options = triton::client::SslOptions();
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
    
    FAIL_IF_ERR(triton::client::InferenceServerGrpcClient::Create(&client, url, verbose, use_ssl, ssl_options, triton::client::KeepAliveOptions(), use_cached_channel), err);
    
    std::vector<int32_t> input0_data { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::vector<int32_t> input1_data { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<int64_t> shape { 1, 16 };


    for (auto _ : state)
    {
        // Initialize the inputs with the data.
        triton::client::InferInput* input0;
        triton::client::InferInput* input1;

        FAIL_IF_ERR(triton::client::InferInput::Create(&input0, "INPUT0", shape, "INT32"), "unable to get INPUT0");
        std::shared_ptr<triton::client::InferInput> input0_ptr;
        input0_ptr.reset(input0);

        FAIL_IF_ERR(triton::client::InferInput::Create(&input1, "INPUT1", shape, "INT32"), "unable to get INPUT1");
        std::shared_ptr<triton::client::InferInput> input1_ptr;
        input1_ptr.reset(input1);

        FAIL_IF_ERR(input0_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&input0_data[0]), input0_data.size() * sizeof(int32_t)), "unable to set data for INPUT0");
        FAIL_IF_ERR(input1_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&input1_data[0]), input1_data.size() * sizeof(int32_t)), "unable to set data for INPUT1");

        // Generate the outputs to be requested.
        triton::client::InferRequestedOutput* output0;
        triton::client::InferRequestedOutput* output1;

        FAIL_IF_ERR(triton::client::InferRequestedOutput::Create(&output0, "OUTPUT0"), "unable to get 'OUTPUT0'");
        std::shared_ptr<triton::client::InferRequestedOutput> output0_ptr;
        output0_ptr.reset(output0);
        
        FAIL_IF_ERR(triton::client::InferRequestedOutput::Create(&output1, "OUTPUT1"), "unable to get 'OUTPUT1'");
        std::shared_ptr<triton::client::InferRequestedOutput> output1_ptr;
        output1_ptr.reset(output1);

        // The inference settings. Will be using default for now.
        triton::client::InferOptions options(model_name);
        options.model_version_ = model_version;
        options.client_timeout_ = client_timeout;

        std::vector<triton::client::InferInput*> inputs = { input0_ptr.get(), input1_ptr.get() };
        std::vector<const triton::client::InferRequestedOutput*> outputs = { output0_ptr.get(), output1_ptr.get() };

        triton::client::InferResult* results;
        FAIL_IF_ERR(client->Infer(&results, options, inputs, outputs, http_headers, compression_algorithm), "unable to run model");
        std::shared_ptr<triton::client::InferResult> results_ptr;
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
                throw std::runtime_error("error: incorrect sum");

            if ((input0_data[i] - input1_data[i]) != *(output1_data + i))
                throw std::runtime_error("error: incorrect difference");
        }
    }
}

BENCHMARK(benchmark_triton_client)
    ->Threads(1)
    ->Iterations(10'000)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->ReportAggregatesOnly(false)
    ->DisplayAggregatesOnly(false);

BENCHMARK_MAIN();
