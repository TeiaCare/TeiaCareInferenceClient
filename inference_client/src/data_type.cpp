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

#include <teiacare/inference_client/data_type.hpp>

namespace tc::infer
{
[[nodiscard]] data_type::value data_type::from_string(const std::string& str_value)
{
    if (auto str_type = to_type.find(str_value); str_type != to_type.end())
        return str_type->second;

    return data_type::Unknown;
}

[[nodiscard]] std::string data_type::str() const
{
    return to_string.at(_value);
}

std::map<data_type::value, std::string> data_type::to_string{
    {data_type::Bool, "BOOL"},
    {data_type::Uint8, "UINT8"},
    {data_type::Uint16, "UINT16"},
    {data_type::Uint32, "UINT32"},
    {data_type::Uint64, "UINT64"},
    {data_type::Int8, "INT8"},
    {data_type::Int16, "INT16"},
    {data_type::Int32, "INT32"},
    {data_type::Int64, "INT64"},
    {data_type::Fp16, "FP16"},
    {data_type::Fp32, "FP32"},
    {data_type::Fp64, "FP64"},
    {data_type::String, "BYTES"},
    {data_type::Unknown, "Unknown"}};

std::map<std::string, data_type::value> data_type::to_type{
    {"BOOL", data_type::Bool},
    {"UINT8", data_type::Uint8},
    {"UINT16", data_type::Uint16},
    {"UINT32", data_type::Uint32},
    {"UINT64", data_type::Uint64},
    {"INT8", data_type::Int8},
    {"INT16", data_type::Int16},
    {"INT32", data_type::Int32},
    {"INT64", data_type::Int64},
    {"FP16", data_type::Fp16},
    {"FP32", data_type::Fp32},
    {"FP64", data_type::Fp64},
    {"BYTES", data_type::String},
    {"Unknown", data_type::Unknown}};

}
