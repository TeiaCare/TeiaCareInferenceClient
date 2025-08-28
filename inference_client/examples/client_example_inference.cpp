// Copyright 2024 TeiaCare
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <teiacare/inference_client/client_factory.hpp>

#include <spdlog/spdlog.h>

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::level_enum::trace);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::info("Running client_example_inference");

    auto client = tc::infer::create_client("localhost:8001", std::chrono::seconds(30));
    if (!client->is_server_live() || !client->is_server_ready())
    {
        spdlog::error("Server is not available");
        return 1;
    }

    std::string model_name = "yolov5x_face_person_trt";
    std::string model_version = "2";
    const auto model_metadata = client->model_metadata(model_name, model_version);
    spdlog::info("Model available: '{}' (available versions: '{}' platform '{}')", model_metadata.model_name, fmt::join(model_metadata.model_versions, ", "), model_metadata.platform);

    for (auto input : model_metadata.inputs)
    {
        spdlog::info("Input tensor: {} (datatype: {}, shape: {})", input.name, input.datatype, fmt::join(input.shape, ", "));
    }

    for (auto output : model_metadata.outputs)
    {
        spdlog::info("Output tensor: {} (datatype: {}, shape: {})", output.name, output.datatype, fmt::join(output.shape, ", "));
    }

    std::vector<float> data(3 * 640 * 640, 1.0f); // fill with dummy data
    std::vector<int64_t> shape{1, 3, 640, 640};

    tc::infer::infer_request request;
    request.model_name = model_name;
    request.model_version = model_version;
    request.id = "REQUEST_0";
    request.add_input_tensor(data.data(), data.size(), shape, "images");

    tc::infer::infer_response response;
    try
    {
        response = client->infer(request, std::chrono::seconds(30));
    }
    catch (const std::runtime_error& ex)
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

    return EXIT_SUCCESS;
}
