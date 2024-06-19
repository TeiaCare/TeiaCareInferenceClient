#pragma once

#include <teiacare/inference_client/infer_tensor.hpp>

#include <vector>
#include <string>

namespace tc::infer
{
struct infer_response
{
    std::string model_name;
    std::string model_version;
    std::string id;
    std::vector<infer_tensor> output_tensors;

    void add_output_tensor(const infer_tensor& output)
    {
        output_tensors.push_back(output);
    }
};

}