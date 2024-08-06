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

#include "grpc_client.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace tc::infer
{
class grpc_client_async final : public grpc_client
{
public:
    explicit grpc_client_async(const std::string& channel_address);
    ~grpc_client_async();

    bool is_server_live() override;
    bool is_server_ready() override;
    tc::infer::server_metadata server_metadata() override;
    std::vector<std::string> model_list() override;
    bool is_model_ready(const std::string& model_name, const std::string& model_version) override;
    bool model_load(const std::string& model_name, const std::string& model_version) override;
    bool model_unload(const std::string& model_name, const std::string& model_version) override;
    tc::infer::model_metadata model_metadata(const std::string& model_name, const std::string& model_version) override;
    tc::infer::infer_response infer(const tc::infer::infer_request& infer_request, std::chrono::milliseconds timeout) override;

protected:
    void rpc_handler(const std::shared_ptr<grpc::CompletionQueue>& cq);
    void check_status(grpc::Status rpc_status) const;

private:
    std::unique_ptr<inference::GRPCInferenceService::Stub> _stub;
    std::unique_ptr<tc::infer::tensor_converter> _tensor_converter;

    // std::shared_ptr<grpc::CompletionQueue> _async_completion_queue;
    // std::vector<std::thread> _async_grpc_threads;
    // unsigned num_async_threads = 1u;
};

}
