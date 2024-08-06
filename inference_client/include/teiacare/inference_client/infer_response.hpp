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

#pragma once

#include <teiacare/inference_client/infer_tensor.hpp>

#include <string>
#include <vector>

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
