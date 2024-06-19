#include <teiacare/inference_client/client_factory.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::level_enum::trace);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::info("Running client_example_simple_inference");

    auto client = tc::infer::client_factory::create_client("localhost:8001");

    {
        std::vector<int32_t> data_0 { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        std::vector<int32_t> data_1 { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        std::vector<int64_t> shape { 1, 16 };

        tc::infer::infer_request request;
        request.model_name = "simple_int32";
        request.model_version = "1";
        request.id = "REQUEST_0";    
        request.add_input_tensor(data_0.data(), data_0.size(), shape, "INPUT0");
        request.add_input_tensor(data_1.data(), data_1.size(), shape, "INPUT1");

        tc::infer::infer_response response;
        try
        {
            response = client->infer(request);
        }
        catch(const std::runtime_error& ex)
        {
            spdlog::error("Unable to perform inference: {}", ex.what());
            return EXIT_FAILURE;
        }

        spdlog::info("Model name: {}", response.model_name);
        spdlog::info("Model version: {}", response.model_version);
        spdlog::info("Output layers");
        for (const auto& output : response.output_tensors)
        {
            spdlog::info("- Name: {}", output.name());
            spdlog::info("- DataType: {}", output.datatype().str());
            spdlog::info("- Shape: [{}]", fmt::join(output.shape(), ", "));
            spdlog::info("- Output layer data");

            const int32_t* output_data = output.as<int32_t>();
            for (auto i = 0; i < output.data_size(); ++i)
            {
                spdlog::debug("  {}: {}", i, output_data[i]);
            }
        }
    }

    {
        std::vector<int8_t> data_0 { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        std::vector<int8_t> data_1 { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        std::vector<int64_t> shape { 1, 16 };

        tc::infer::infer_request request;
        request.model_name = "simple_int8";
        request.model_version = "1";
        request.id = "REQUEST_0";    
        request.add_input_tensor(data_0.data(), data_0.size(), shape, "INPUT0");
        request.add_input_tensor(data_1.data(), data_1.size(), shape, "INPUT1");

        tc::infer::infer_response response;
        try
        {
            response = client->infer(request);
        }
        catch(const std::runtime_error& ex)
        {
            spdlog::error("Unable to perform inference: {}", ex.what());
            return EXIT_FAILURE;
        }
        
        spdlog::info("Model name: {}", response.model_name);
        spdlog::info("Model version: {}", response.model_version);
        spdlog::info("Output layers");
        for (const auto& output : response.output_tensors)
        {
            spdlog::info("- Name: {}", output.name());
            spdlog::info("- DataType: {}", output.datatype().str());
            spdlog::info("- Shape: [{}]", fmt::join(output.shape(), ", "));
            spdlog::info("- Output layer data");

            const int8_t* output_data = output.as<int8_t>();
            for (auto i = 0; i < output.data_size(); ++i)
            {
                spdlog::debug("  {}: {}", i, output_data[i]);
            }
        }
    }

/*
    {
        std::vector<const char*> data_0 { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15" };
        std::vector<char> data_1 { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        std::vector<int64_t> shape { 1, 16 };

        tc::infer::infer_request request;
        request.model_name = "simple_string";
        request.model_version = "1";
        request.id = "string";    
        request.add_input_tensor(data_0.data(), data_0.size(), shape, "INPUT0");
        request.add_input_tensor(data_1.data(), data_1.size(), shape, "INPUT1");
        
        tc::infer::infer_response response;
        try
        {
            response = client->infer(request);
        }
        catch(const std::runtime_error& ex)
        {
            spdlog::error("Unable to perform inference: {}", ex.what());
            return EXIT_FAILURE;
        }
        
        spdlog::info("Model name: {}", response.model_name);
        spdlog::info("Model version: {}", response.model_version);
        spdlog::info("Output layers");
        for (const auto& output : response.output_tensors)
        {
            spdlog::info("- Name: {}", output.name);
            spdlog::info("- DataType: {}", output.datatype.str());
            spdlog::info("- Shape: [{}]", fmt::join(output.shape, ", "));
            spdlog::info("- Output layer data");
            auto size = std::accumulate(output.shape.begin(), output.shape.end(), 1, std::multiplies<>());
            
            int8_t* output_data = output.as<int8_t>();
            for (auto i = 0; i < size; ++i)
            {
                spdlog::debug("  {}: {}", i, output_data[i]);
            }
        }
    }
*/

    return EXIT_SUCCESS;
}
