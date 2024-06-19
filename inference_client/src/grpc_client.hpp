#pragma once

#include <grpcpp/support/status.h>
#include <teiacare/inference_client/client_interface.hpp>
#include <services.grpc.pb.h>
#include "tensor_converter.hpp"

#include <vector>
#include <memory>
#include <string>
#include <chrono>

namespace tc::infer
{
class grpc_client : public client_interface
{
public:
    explicit grpc_client(std::unique_ptr<inference::GRPCInferenceService::StubInterface> stub, std::chrono::milliseconds rpc_timeout);
    ~grpc_client();

    bool is_server_live() override;
    bool is_server_ready() override;
    tc::infer::server_metadata server_metadata() override;
    std::vector<std::string> model_list() override;
    bool is_model_ready(const std::string& model_name, const std::string& model_version) override;
    bool model_load(const std::string& model_name, const std::string& model_version) override;
    bool model_unload(const std::string& model_name, const std::string& model_version) override;
    tc::infer::model_metadata model_metadata(const std::string& model_name, const std::string& model_version) override;
    tc::infer::infer_response infer(const tc::infer::infer_request& infer_request, std::chrono::milliseconds infer_timeout) override;

protected:
    void check_status(grpc::Status rpc_status) const;

private:
    std::unique_ptr<inference::GRPCInferenceService::StubInterface> _stub;
    std::unique_ptr<tc::infer::tensor_converter> _tensor_converter;
    std::chrono::milliseconds _rpc_timeout;
};

}
