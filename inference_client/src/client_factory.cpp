#include <teiacare/inference_client/client_factory.hpp>
#include "grpc_client.hpp"
#include <grpcpp/create_channel.h>

namespace tc::infer
{
std::unique_ptr<client_interface> create_client(const std::string& uri, std::chrono::milliseconds rpc_timeout) 
{
	std::shared_ptr<grpc::ChannelInterface> channel = grpc::CreateChannel(uri, grpc::InsecureChannelCredentials());
	std::unique_ptr<inference::GRPCInferenceService::StubInterface> stub = inference::GRPCInferenceService::NewStub(channel);
	return std::make_unique<tc::infer::grpc_client>(std::move(stub), rpc_timeout);
}

// #if defined(UNIT_TESTS)
// std::unique_ptr<client_interface> create_client(std::unique_ptr<inference::GRPCInferenceService::StubInterface> rpc_stub, std::chrono::milliseconds rpc_timeout)
// {
// 	return std::make_unique<tc::infer::grpc_client>(std::move(rpc_stub), rpc_timeout);
// }
// #endif

}
