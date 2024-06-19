#include "grpc_client.hpp"
#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/support/status.h>

namespace tc::infer
{
grpc_client::grpc_client(const std::string& channel_address)
    : _stub{ inference::GRPCInferenceService::NewStub(grpc::CreateChannel(channel_address, grpc::InsecureChannelCredentials())) }
    , _tensor_converter{ std::make_unique<tc::infer::tensor_converter>() }
{
    // _async_completion_queue = std::make_unique<grpc::CompletionQueue>();
    // for (auto thread_idx = 0; thread_idx < num_async_threads; ++thread_idx)
    //     _async_grpc_threads.emplace_back(std::thread([this](const std::shared_ptr<grpc::CompletionQueue>& cq) { rpc_handler(cq); }, _async_completion_queue));
}

grpc_client::~grpc_client()
{
    // _async_completion_queue->Shutdown();
    // for (auto&& t : _async_grpc_threads)
    //     if (t.joinable()) t.join();
}

void grpc_client::check_status(grpc::Status rpc_status) const
{
    switch(rpc_status.error_code())
    {
        case grpc::StatusCode::OK:
            break;
        case grpc::StatusCode::DEADLINE_EXCEEDED:
            throw tc::infer::timeout_error(rpc_status.error_message());
            break;
        default:
            throw std::runtime_error(rpc_status.error_message());
            break;
    }
}

bool grpc_client::is_server_live()
{
    inference::ServerLiveRequest request;
    inference::ServerLiveResponse response;
    grpc::ClientContext context;

    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    const grpc::Status rpc_status = _stub->ServerLive(&context, request, &response);
    check_status(rpc_status);

    return response.live();
}

bool grpc_client::is_server_ready()
{
    inference::ServerReadyRequest request;
    inference::ServerReadyResponse response;
    grpc::ClientContext context;

    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    const grpc::Status rpc_status = _stub->ServerReady(&context, request, &response);
    check_status(rpc_status);

    return response.ready();
}

tc::infer::server_metadata grpc_client::server_metadata()
{
    inference::ServerMetadataRequest request;
    inference::ServerMetadataResponse response;
    grpc::ClientContext context;

    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    grpc::Status rpc_status = _stub->ServerMetadata(&context, request, &response);
    check_status(rpc_status);

    const std::vector<std::string> server_extensions = { response.extensions().begin(), response.extensions().end() };
    tc::infer::server_metadata metadata = {response.name(), response.version(), std::move(server_extensions) };

    return metadata;
}

bool grpc_client::is_model_ready(const std::string& model_name, const std::string& model_version)
{
    inference::ModelReadyRequest request;
    inference::ModelReadyResponse response;
    grpc::ClientContext context;

    request.set_name(model_name);
    request.set_version(model_version);

    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    grpc::Status rpc_status = _stub->ModelReady(&context, request, &response);
    check_status(rpc_status);

    return response.ready();
}

std::vector<std::string> grpc_client::model_list()
{
    inference::ModelListRequest request;
    inference::ModelListResponse response;
    grpc::ClientContext context;

    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    grpc::Status rpc_status = _stub->ModelList(&context, request, &response);
    check_status(rpc_status);
    
    const auto models = response.models();
    const std::vector<std::string> model_list(models.begin(), models.end());
    
    return model_list;
}

bool grpc_client::model_load(const std::string& model_name, const std::string&)
{
    inference::ModelLoadRequest request;
    inference::ModelLoadResponse response;
    grpc::ClientContext context;

    request.set_name(model_name);

    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    grpc::Status rpc_status = _stub->ModelLoad(&context, request, &response);
    check_status(rpc_status);

    return true;
}

bool grpc_client::model_unload(const std::string& model_name, const std::string&)
{
    inference::ModelUnloadRequest request;
    inference::ModelUnloadResponse response;
    grpc::ClientContext context;

    request.set_name(model_name);

    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    grpc::Status rpc_status = _stub->ModelUnload(&context, request, &response);
    check_status(rpc_status);

    return true;
}

tc::infer::model_metadata grpc_client::model_metadata(const std::string& model_name, const std::string& model_version)
{
    inference::ModelMetadataRequest request;
    inference::ModelMetadataResponse response;
    grpc::ClientContext context;

    request.set_name(model_name);
    request.set_version(model_version);

    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    grpc::Status rpc_status = _stub->ModelMetadata(&context, request, &response);
    check_status(rpc_status);

    tc::infer::model_metadata metadata;
    metadata.model_name = response.name();
    metadata.model_versions = std::vector<std::string>{ response.versions().begin(), response.versions().end() };
    metadata.platform = response.platform();
    
    for(auto&& input : response.inputs())
    {
        std::vector<int64_t> shape { input.shape().begin(), input.shape().end() };
        metadata.inputs.push_back({input.name(), input.datatype(), shape});
    }

    for(auto&& output : response.outputs())
    {
        std::vector<int64_t> shape { output.shape().begin(), output.shape().end() };
        metadata.outputs.push_back({output.name(), output.datatype(), shape});
    }
    
    return metadata;
}

tc::infer::infer_response grpc_client::infer(const tc::infer::infer_request& infer_request, std::chrono::milliseconds timeout)
{
    inference::ModelInferResponse response;
    grpc::ClientContext context;

    std::map<std::string, std::string> metadata {};
    for (auto&& [key, value] : metadata)
    {
        context.AddMetadata(key, value);
    }
    
    auto request = _tensor_converter->get_infer_request(infer_request);

    context.set_deadline(std::chrono::system_clock::now() + timeout);
    grpc::Status rpc_status = _stub->ModelInfer(&context, request, &response);
    check_status(rpc_status);

    auto infer_response = _tensor_converter->get_infer_response(response);
    return infer_response;
}

void grpc_client::rpc_handler(const std::shared_ptr<grpc::CompletionQueue>& cq)
{
    (void)cq;
    // void* tag;
    // bool ok = false;
    
    // while (cq->Next(&tag, &ok))
    // {
    //     if (!ok)
    //     {
    //         // SPDLOG_WARN("Client stream closed");
    //         break;
    //     }

    //     auto rpc = static_cast<AsyncClientCall*>(tag);
    //     rpc->proceed(ok);
    // }
}

}
