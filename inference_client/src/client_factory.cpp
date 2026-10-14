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

#include "grpc_client.hpp"
#include <grpcpp/create_channel.h>
#include <grpcpp/support/channel_arguments.h>

#include <limits>

namespace tc::infer
{
std::unique_ptr<client_interface> create_client(const std::string& uri, std::chrono::milliseconds rpc_timeout)
{
    // gRPC caps received messages at 4 MB by default; an inference reply can be
    // larger (an instance-segmentation model returns its mask prototypes as one
    // tensor, ~4.5 MB for yolo11-seg at 640). Same limit as the Triton C++ client.
    grpc::ChannelArguments arguments;
    arguments.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
    std::shared_ptr<grpc::ChannelInterface> channel = grpc::CreateCustomChannel(uri, grpc::InsecureChannelCredentials(), arguments);
    std::unique_ptr<inference::GRPCInferenceService::StubInterface> stub = inference::GRPCInferenceService::NewStub(channel);
    return std::make_unique<tc::infer::grpc_client>(std::move(stub), rpc_timeout);
}

std::unique_ptr<client_interface> create_client(void* rpc_stub, std::chrono::milliseconds rpc_timeout)
{
    std::unique_ptr<inference::GRPCInferenceService::StubInterface> stub = std::unique_ptr<inference::GRPCInferenceService::StubInterface>(reinterpret_cast<inference::GRPCInferenceService::StubInterface*>(rpc_stub));
    return std::make_unique<tc::infer::grpc_client>(std::move(stub), rpc_timeout);
}

// #if defined(UNIT_TESTS)
// std::unique_ptr<client_interface> create_client(std::unique_ptr<inference::GRPCInferenceService::StubInterface> rpc_stub, std::chrono::milliseconds rpc_timeout)
// {
// 	return std::make_unique<tc::infer::grpc_client>(std::move(rpc_stub), rpc_timeout);
// }
// #endif

}
