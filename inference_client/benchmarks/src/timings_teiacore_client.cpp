#include <teiacare/inference_client/client_factory.hpp>
#include "timings_reports.hpp"

int main(int argc, char** argv)
{
    auto client = tc::infer::client_factory::create_client("localhost:8001");

    std::vector<int32_t> data_0 { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::vector<int32_t> data_1 { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<int64_t> shape { 1, 16 };

    std::vector<int64_t> times;
    for(int n=0; n<100; ++n)
    {
        auto start = std::chrono::steady_clock::now();
        for(int i=0; i<100; ++i)
        {
            tc::infer::infer_request request;
            request.model_name = "simple_int32";
            request.model_version = "1";
            request.add_input_tensor(data_0.data(), data_0.size(), shape, "INPUT0");
            request.add_input_tensor(data_1.data(), data_1.size(), shape, "INPUT1");

            tc::infer::infer_response response;
            try
            {
                response = client->infer(request);
            }
            catch(...)
            {
                return EXIT_FAILURE;
            }

            auto output0_data = response.output_tensors[0].as<int32_t>();
            auto output1_data = response.output_tensors[1].as<int32_t>();
            for (size_t i = 0; i < 16; ++i)
            {
                if ((data_0[i] + data_1[i]) != *(output0_data + i))
                {
                    std::cerr << "error: incorrect sum" << std::endl;
                    exit(1);
                }
                if ((data_0[i] - data_1[i]) != *(output1_data + i))
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
    return EXIT_SUCCESS;
}
