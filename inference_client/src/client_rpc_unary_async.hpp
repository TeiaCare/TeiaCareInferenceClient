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

#pragma once

#include <functional>
#include <grpcpp/client_context.h>

namespace tc::infer
{

struct AsyncClientCall
{
    grpc::ClientContext context;
    grpc::Status result_code;
    virtual bool proceed(bool ok) = 0;
};

template <class ResponseT>
struct AsyncClientCallback : public AsyncClientCall
{
    ResponseT response;
    void set_response_callback(std::function<void(ResponseT)> response_callback)
    {
        _on_response_callback = response_callback;
    }

protected:
    std::function<void(ResponseT)> _on_response_callback;
};

template <class ResponseT>
struct AsyncClientUnaryCall : public AsyncClientCallback<ResponseT>
{
    std::unique_ptr<grpc::ClientAsyncResponseReader<ResponseT>> rpc;

    bool proceed(bool ok) override
    {
        if (ok)
        {
            if (this->result_code.ok())
            {
                // SPDLOG_INFO(this->response.message());
                if (this->_on_response_callback)
                    this->_on_response_callback(this->response);
            }
            else
            {
                // SPDLOG_INFO("RPC failed: (" << this->result_code.error_code() << ") " << this->result_code.error_message());
            }
        }

        return false;
    }
};

}
