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

#include <teiacare/inference_client/data_type.hpp>
#include <teiacare/inference_client/infer_tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tc::infer
{
struct infer_request
{
    std::string model_name;
    std::string model_version;
    std::string id;
    std::vector<infer_tensor> input_tensors;

    inline void add_input_tensor(std::byte* data, const size_t size, const std::vector<int64_t>& shape, data_type data_type, const std::string& name)
    {
        input_tensors.emplace_back(std::vector<std::byte>(data, data + size), shape, data_type, name);
    }

    template <typename T>
    inline void add_input_tensor(T* data, const size_t size, const std::vector<int64_t>& shape, const std::string& name)
    {
        add_input_tensor(std::bit_cast<std::byte*>(data), size * sizeof(T), shape, cast_to_data_type<T>::type, name);
    }
};

}
