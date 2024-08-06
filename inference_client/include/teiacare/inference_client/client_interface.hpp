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

#include <teiacare/inference_client/infer_request.hpp>
#include <teiacare/inference_client/infer_response.hpp>
#include <teiacare/inference_client/model_metadata.hpp>
#include <teiacare/inference_client/server_metadata.hpp>
#include <teiacare/inference_client/timeout_error.hpp>

#include <chrono>
#include <string>
#include <vector>

namespace tc::infer
{
class client_interface
{
public:
    virtual ~client_interface() = default;

    virtual bool is_server_live() = 0;
    virtual bool is_server_ready() = 0;
    virtual tc::infer::server_metadata server_metadata() = 0;
    virtual std::vector<std::string> model_list() = 0;
    virtual bool is_model_ready(const std::string& model_name, const std::string& model_version) = 0;
    virtual bool model_load(const std::string& model_name, const std::string& model_version) = 0;
    virtual bool model_unload(const std::string& model_name, const std::string& model_version) = 0;
    virtual tc::infer::model_metadata model_metadata(const std::string& model_name, const std::string& model_version) = 0;
    virtual tc::infer::infer_response infer(const tc::infer::infer_request& infer_request, std::chrono::milliseconds infer_timeout = std::chrono::seconds(1)) = 0;
};

}
