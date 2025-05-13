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
#include <teiacare/inference_client/client_interface.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <services_mock.grpc.pb.h>

/*
TEST(FakeClient, server_live)
{
    inference::MockGRPCInferenceServiceStub stub;
    inference::ServerLiveResponse response;
    response.set_live(true);

    EXPECT_CALL(stub, ServerLive)
        .WillRepeatedly(testing::DoAll(testing::SetArgPointee<2>(response), testing::Return(grpc::Status::OK)));

    FakeClient client(&stub);
    EXPECT_TRUE(client.is_server_live());
    EXPECT_TRUE(client.is_server_live());
    EXPECT_TRUE(client.is_server_live());
}

TEST(FakeClient, server_not_live)
{
    inference::MockGRPCInferenceServiceStub stub;
    inference::ServerLiveResponse response;
    response.set_live(false);

    EXPECT_CALL(stub, ServerLive)
        .WillRepeatedly(testing::DoAll(testing::SetArgPointee<2>(response), testing::Return(grpc::Status::OK)));

    FakeClient client(&stub);
    EXPECT_FALSE(client.is_server_live());
    EXPECT_FALSE(client.is_server_live());
    EXPECT_FALSE(client.is_server_live());
}

TEST(FakeClient, call_deadline_exceeded)
{
    inference::MockGRPCInferenceServiceStub stub;

    EXPECT_CALL(stub, ServerLive)
        .WillOnce(testing::Return(grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "")));

    FakeClient client(&stub);
    EXPECT_THROW(client.is_server_live(), tc::infer::timeout_error);
}

TEST(FakeClient, call_error)
{
    inference::MockGRPCInferenceServiceStub stub;

    EXPECT_CALL(stub, ServerLive)
        .WillOnce(testing::Return(grpc::Status::CANCELLED));

    FakeClient client(&stub);
    EXPECT_THROW(client.is_server_live(), std::runtime_error);
}

TEST(FakeClient, call_exception)
{
    inference::MockGRPCInferenceServiceStub stub;

    EXPECT_CALL(stub, ServerLive)
        .WillOnce(testing::Throw(std::logic_error("")));

    FakeClient client(&stub);
    EXPECT_THROW(client.is_server_live(), std::logic_error);
}
*/

TEST(test_inference_client, create)
{
    EXPECT_NO_THROW((tc::infer::create_client("localhost:8001")));
}

TEST(test_inference_client, create_stub)
{
    EXPECT_NO_THROW((tc::infer::create_client(nullptr)));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
