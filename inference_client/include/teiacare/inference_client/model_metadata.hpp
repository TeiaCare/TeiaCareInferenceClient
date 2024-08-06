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

#include <cstdint>
#include <string>
#include <vector>

namespace tc::infer
{
struct model_metadata
{
    struct tensor_metadata
    {
        std::string name;
        std::string datatype;
        std::vector<int64_t> shape;
    };

    std::string model_name;
    std::vector<std::string> model_versions;
    std::string platform;
    std::vector<tensor_metadata> inputs;
    std::vector<tensor_metadata> outputs;
};

}
