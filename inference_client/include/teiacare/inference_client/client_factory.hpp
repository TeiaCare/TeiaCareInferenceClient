#pragma once

#include <teiacare/inference_client/client_interface.hpp>
#include <memory>
#include <string>
#include <chrono>

namespace tc::infer
{
std::unique_ptr<client_interface> create_client(const std::string& uri, std::chrono::milliseconds rpc_timeout = std::chrono::seconds(5));

// #if defined(UNIT_TESTS)
// #include <services.grpc.pb.h>
// std::unique_ptr<client_interface> create_client(std::unique_ptr<inference::GRPCInferenceService::StubInterface> rpc_stub, std::chrono::milliseconds rpc_timeout = std::chrono::seconds(5));
// #endif

}
