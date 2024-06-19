#pragma once

#include <teiacare/inference_client/infer_request.hpp>
#include <teiacare/inference_client/infer_response.hpp>
#include <teiacare/inference_client/server_metadata.hpp>
#include <teiacare/inference_client/model_metadata.hpp>
#include <teiacare/inference_client/timeout_error.hpp>

#include <vector>
#include <string>
#include <chrono>

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
