#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <services_mock.grpc.pb.h>
#include <teiacare/inference_client/client_interface.hpp>

class FakeClient
{
public:
    explicit FakeClient(inference::GRPCInferenceService::StubInterface* stub) : _stub(stub) {}

    bool is_server_live()
    {
        inference::ServerLiveRequest request;
        inference::ServerLiveResponse response;
        grpc::ClientContext context;

        const grpc::Status rpc_status = _stub->ServerLive(&context, request, &response);

        return response.live();
    }

private:
    inference::GRPCInferenceService::StubInterface* _stub;
};

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

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
