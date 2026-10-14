// Copyright 2026 TeiaCare
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

// A client made by create_client(uri) must accept replies larger than gRPC's
// 4 MB default receive limit: an instance-segmentation model returns its mask
// prototypes as one tensor (yolo11-seg: 32x160x160 FP32 = 3.3 MB, ~4.5 MB with
// the detections), which the default limit rejects with RESOURCE_EXHAUSTED.

#include <teiacare/inference_client/client_factory.hpp>
#include <teiacare/inference_client/client_interface.hpp>

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>
#include <memory>
#include <services.grpc.pb.h>
#include <string>

namespace
{
constexpr std::size_t large_output_floats = 5 * 1024 * 1024 / sizeof(float); // 5 MB

class large_reply_service final : public inference::GRPCInferenceService::Service
{
    grpc::Status ModelInfer(grpc::ServerContext*, const inference::ModelInferRequest*, inference::ModelInferResponse* response) override
    {
        response->set_model_name("large");
        auto* output = response->add_outputs();
        output->set_name("output1");
        output->set_datatype("FP32");
        output->add_shape(1);
        output->add_shape(static_cast<int64_t>(large_output_floats));
        response->add_raw_output_contents(std::string(large_output_floats * sizeof(float), '\x01'));
        return grpc::Status::OK;
    }
};
}

TEST(LargeResponseTest, ClientFromUriAcceptsRepliesAboveGrpcDefaultLimit)
{
    large_reply_service service;
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    ASSERT_GT(port, 0);

    auto client = tc::infer::create_client("127.0.0.1:" + std::to_string(port), std::chrono::seconds(10));
    tc::infer::infer_request request;
    request.model_name = "large";
    float input = 0.0f;
    request.add_input_tensor(&input, 1, {1}, "images");

    tc::infer::infer_response response;
    ASSERT_NO_THROW(response = client->infer(request, std::chrono::seconds(10)));
    ASSERT_EQ(response.output_tensors.size(), 1u);
    EXPECT_EQ(response.output_tensors[0].shape(), (std::vector<int64_t>{1, static_cast<int64_t>(large_output_floats)}));

    server->Shutdown();
}
