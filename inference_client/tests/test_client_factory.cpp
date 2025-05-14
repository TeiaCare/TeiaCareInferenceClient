// filepath: /home/stefanolusardi/TeiaCare/TeiaCareInferenceClient/inference_client/src/test_client_factory.cpp
#include <teiacare/inference_client/client_factory.hpp>
#include <teiacare/inference_client/client_interface.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <services_mock.grpc.pb.h>

using namespace tc::infer;
using namespace testing;

class ClientFactoryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Common setup code if needed
    }

    void TearDown() override
    {
        // Common teardown code if needed
    }
};

TEST_F(ClientFactoryTest, CreateClientFromVoidPointer)
{
    // Create a mock stub
    auto mock_stub = std::make_unique<inference::MockGRPCInferenceServiceStub>();

    // Save raw pointer before transferring ownership
    void* raw_stub_ptr = mock_stub.get();

    // Set any expectations on the mock if needed
    // For example, you might want to expect certain methods to be called when using the client

    // Create the client with the void pointer to the stub
    std::chrono::milliseconds timeout(1000);
    std::unique_ptr<tc::infer::client_interface> client = create_client(raw_stub_ptr, timeout);

    // Verify the client was created successfully
    ASSERT_NE(client, nullptr);

    // Release ownership of the mock stub since it's now owned by the client
    mock_stub.release();

    // Additional verification if needed
    // For example, you could call methods on the client and verify the behavior
}

TEST_F(ClientFactoryTest, CreateClientFromUri)
{
    // Test the URI-based client creation
    std::string test_uri = "localhost:50051";
    std::chrono::milliseconds timeout(1000);

    // This might create an actual gRPC channel, which could be problematic in unit tests
    // Ideally, this would be mocked or tested in integration tests
    std::unique_ptr<tc::infer::client_interface> client = create_client(test_uri, timeout);

    // Verify the client was created successfully
    ASSERT_NE(client, nullptr);
}

// If you want to test the commented-out function for UNIT_TESTS, you could add something like:
/*
TEST_F(ClientFactoryTest, CreateClientFromStubInterface) {
    auto mock_stub = std::make_unique<inference::MockGRPCInferenceServiceStub>();
    std::chrono::milliseconds timeout(1000);

    std::unique_ptr<client_interface> client = create_client(std::move(mock_stub), timeout);

    ASSERT_NE(client, nullptr);
}
*/
