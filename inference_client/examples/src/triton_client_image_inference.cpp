// Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// #include <dirent.h>
// #include <getopt.h>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <queue>
#include <string>

#include <vector>
#include <chrono>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "grpc_client.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace tc = triton::client;

namespace
{

enum ScaleType
{
    NONE = 0,
    VGG = 1,
    INCEPTION = 2
};

enum ProtocolType
{
    HTTP = 0,
    GRPC = 1
};

struct ModelInfo
{
    std::string output_name_;
    std::string input_name_;
    std::string input_datatype_;
    // The shape of the input
    int input_c_;
    int input_h_;
    int input_w_;
    // The format of the input
    std::string input_format_;
    int type1_;
    int type3_;
    int max_batch_size_;
};

void Preprocess(const cv::Mat& img,
    const std::string& format,
    int img_type1,
    int img_type3,
    size_t img_channels,
    const cv::Size& img_size,
    const ScaleType scale,
    std::vector<uint8_t>* input_data)
{
    // Image channels are in BGR order. Currently model configuration
    // data doesn't provide any information as to the expected channel
    // orderings (like RGB, BGR). We are going to assume that RGB is the
    // most likely ordering and so change the channels to that ordering.

    cv::Mat sample;
    cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);

    cv::Mat sample_resized;
    cv::resize(sample, sample_resized, img_size);

    cv::Mat sample_type;
    sample_resized.convertTo(sample_type, (img_channels == 3) ? img_type3 : img_type1);

    // Inception scaling
    cv::Mat sample_final;
    sample_final = sample_type.mul(cv::Scalar(1 / 127.5, 1 / 127.5, 1 / 127.5));
    sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);

    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample_final.total() * sample_final.elemSize();
    size_t pos = 0;
    input_data->resize(img_byte_size);

    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i)
    {
        input_bgr_channels.emplace_back(img_size.height, img_size.width, img_type1, &((*input_data)[pos]));
        pos += input_bgr_channels.back().total() * input_bgr_channels.back().elemSize();
    }

    cv::split(sample_final, input_bgr_channels);

    if (pos != img_byte_size)
    {
        std::cerr << "unexpected total size of channels " << pos << ", expecting " << img_byte_size << std::endl;
        exit(1);
    }
}

void Postprocess(const tc::InferResult* result,
    const std::string& filename,
    const size_t batch_size,
    const std::string& output_name,
    size_t topk,
    const bool batching)
{
    if (!result->RequestStatus().IsOk())
    {
        std::cerr << "inference  failed with error: " << result->RequestStatus() << std::endl;
        exit(1);
    }

    // Get and validate the shape and datatype
    std::vector<int64_t> shape;
    tc::Error err = result->Shape(output_name, &shape);
    if (!err.IsOk())
    {
        std::cerr << "unable to get shape for " << output_name << std::endl;
        exit(1);
    }

    // Validate shape. Special handling for non-batch model
    // if (!batching)
    // {
    //     if ((shape.size() != 1) || (shape[0] != (int)topk))
    //     {
    //         std::cerr << "received incorrect shape for " << output_name << std::endl;
    //         exit(1);
    //     }
    // }
    // else
    // {
    //     if ((shape.size() != 2) || (shape[0] != (int)batch_size) || (shape[1] != (int)topk))
    //     {
    //         std::cerr << "received incorrect shape for " << output_name << std::endl;
    //         exit(1);
    //     }
    // }

    std::string datatype;
    err = result->Datatype(output_name, &datatype);
    if (!err.IsOk())
    {
        std::cerr << "unable to get datatype for " << output_name << std::endl;
        exit(1);
    }

    // Validate datatype
    // if (datatype.compare("BYTES") != 0)
    // {
    //     std::cerr << "received incorrect datatype for " << output_name << ": " << datatype << std::endl;
    //     exit(1);
    // }

    std::vector<std::string> result_data;
    err = result->StringData(output_name, &result_data);
    if (!err.IsOk())
    {
        std::cerr << "unable to get data for " << output_name << std::endl;
        exit(1);
    }

    // if (result_data.size() != (topk * batch_size))
    // {
    //     std::cerr << "unexpected number of strings in the result, expected " << (topk * batch_size) << ", got " << result_data.size() << std::endl;
    //     exit(1);
    // }

    std::cout << "Image '" << filename << "':" << std::endl;
    for (size_t k = 0; k < topk; ++k)
    {
        std::istringstream is(result_data[k]);
        int count = 0;
        std::string token;
        while (getline(is, token, ':'))
        {
            if (count == 0)
            {
                std::cout << "    " << token;
            }
            else if (count == 1)
            {
                std::cout << " (" << token << ")";
            }
            else if (count == 2)
            {
                std::cout << " = " << token;
            }
            count++;
        }
        std::cout << std::endl;
    }

    const uint8_t* buffer = nullptr;
    size_t buffer_size;
    result->RawData(output_name, &buffer, &buffer_size);

    for (int i=0; i< buffer_size; ++i)
    {
        std::cout << i << ": " << *buffer << std::endl;
        ++buffer;
    }
}

bool ParseType(const std::string& dtype, int* type1, int* type3)
{
    if (dtype.compare("UINT8") == 0)
    {
        *type1 = CV_8UC1;
        *type3 = CV_8UC3;
    }
    else if (dtype.compare("INT8") == 0)
    {
        *type1 = CV_8SC1;
        *type3 = CV_8SC3;
    }
    else if (dtype.compare("UINT16") == 0)
    {
        *type1 = CV_16UC1;
        *type3 = CV_16UC3;
    }
    else if (dtype.compare("INT16") == 0)
    {
        *type1 = CV_16SC1;
        *type3 = CV_16SC3;
    }
    else if (dtype.compare("INT32") == 0)
    {
        *type1 = CV_32SC1;
        *type3 = CV_32SC3;
    }
    else if (dtype.compare("FP32") == 0)
    {
        *type1 = CV_32FC1;
        *type3 = CV_32FC3;
    }
    else if (dtype.compare("FP64") == 0)
    {
        *type1 = CV_64FC1;
        *type3 = CV_64FC3;
    }
    else
    {
        return false;
    }

    return true;
}

void ParseModelGrpc(const inference::ModelMetadataResponse& model_metadata, const inference::ModelConfigResponse& model_config, const size_t batch_size, ModelInfo* model_info)
{
    if (model_metadata.inputs().size() != 1)
    {
        std::cerr << "expecting 1 input, got " << model_metadata.inputs().size() << std::endl;
        exit(1);
    }

    if (model_metadata.outputs().size() != 1)
    {
        std::cerr << "expecting 1 output, got " << model_metadata.outputs().size() << std::endl;
        exit(1);
    }

    if (model_config.config().input().size() != 1)
    {
        std::cerr << "expecting 1 input in model configuration, got " << model_config.config().input().size() << std::endl;
        exit(1);
    }

    auto input_metadata = model_metadata.inputs(0);
    auto input_config = model_config.config().input(0);
    auto output_metadata = model_metadata.outputs(0);

    if (output_metadata.datatype().compare("FP32") != 0)
    {
        std::cerr << "expecting output datatype to be FP32, model '" << model_metadata.name() << "' output type is '" << output_metadata.datatype() << "'" << std::endl;
        exit(1);
    }

    model_info->max_batch_size_ = model_config.config().max_batch_size();

    // Model specifying maximum batch size of 0 indicates that batching
    // is not supported and so the input tensors do not expect a "N"
    // dimension (and 'batch_size' should be 1 so that only a single
    // image instance is inferred at a time).
    if (model_info->max_batch_size_ == 0)
    {
        if (batch_size != 1)
        {
            std::cerr << "batching not supported for model \"" << model_metadata.name() << "\"" << std::endl;
            exit(1);
        }
    }
    else
    {
        //  model_info->max_batch_size_ > 0
        if (batch_size > (size_t)model_info->max_batch_size_)
        {
            std::cerr << "expecting batch size <= " << model_info->max_batch_size_ << " for model '" << model_metadata.name() << "'" << std::endl;
            exit(1);
        }
    }

    // Output is expected to be a vector. But allow any number of
    // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    // }, { 10, 1, 1 } are all ok).
    bool output_batch_dim = (model_info->max_batch_size_ > 0);
    size_t non_one_cnt = 0;
    for (const auto dim : output_metadata.shape())
    {
        if (output_batch_dim)
        {
            output_batch_dim = false;
        }
        else if (dim == -1)
        {
            std::cerr << "variable-size dimension in model output not supported" << std::endl;
            exit(1);
        }
        else if (dim > 1)
        {
            non_one_cnt += 1;
            if (non_one_cnt > 1)
            {
                std::cerr << "expecting model output to be a vector" << std::endl;
                exit(1);
            }
        }
    }

    // Model input must have 3 dims, either CHW or HWC (not counting the
    // batch dimension), either CHW or HWC
    const bool input_batch_dim = (model_info->max_batch_size_ > 0);
    const int expected_input_dims = 3 + (input_batch_dim ? 1 : 0);
    if (input_metadata.shape().size() != expected_input_dims)
    {
        std::cerr << "expecting input to have " << expected_input_dims << " dimensions, model '" << model_metadata.name() << "' input has " << input_metadata.shape().size()
                  << std::endl;
        exit(1);
    }

    if ((input_config.format() != inference::ModelInput::FORMAT_NCHW) && (input_config.format() != inference::ModelInput::FORMAT_NHWC))
    {
        std::cerr << "unexpected input format " << inference::ModelInput_Format_Name(input_config.format()) << ", expecting "
                  << inference::ModelInput_Format_Name(inference::ModelInput::FORMAT_NHWC) << " or " << inference::ModelInput_Format_Name(inference::ModelInput::FORMAT_NCHW)
                  << std::endl;
        exit(1);
    }

    model_info->output_name_ = output_metadata.name();
    model_info->input_name_ = input_metadata.name();
    model_info->input_datatype_ = input_metadata.datatype();

    if (input_config.format() == inference::ModelInput::FORMAT_NHWC)
    {
        model_info->input_format_ = "FORMAT_NHWC";
        model_info->input_h_ = input_metadata.shape(input_batch_dim ? 1 : 0);
        model_info->input_w_ = input_metadata.shape(input_batch_dim ? 2 : 1);
        model_info->input_c_ = input_metadata.shape(input_batch_dim ? 3 : 2);
    }
    else
    {
        model_info->input_format_ = "FORMAT_NCHW";
        model_info->input_c_ = input_metadata.shape(input_batch_dim ? 1 : 0);
        model_info->input_h_ = input_metadata.shape(input_batch_dim ? 2 : 1);
        model_info->input_w_ = input_metadata.shape(input_batch_dim ? 3 : 2);
    }

    if (!ParseType(model_info->input_datatype_, &(model_info->type1_), &(model_info->type3_)))
    {
        std::cerr << "unexpected input datatype '" << model_info->input_datatype_ << "' for model \"" << model_metadata.name() << std::endl;
        exit(1);
    }
}

void FileToInputData(const std::string& filename, size_t c, size_t h, size_t w, const std::string& format, int type1, int type3, ScaleType scale, std::vector<uint8_t>* input_data)
{
    // Load the specified image.
    std::ifstream file(filename);
    std::vector<char> data;
    file >> std::noskipws;
    std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(), std::back_inserter(data));
    if (data.empty())
    {
        std::cerr << "error: unable to read image file " << filename << std::endl;
        exit(1);
    }

    cv::Mat img = imdecode(cv::Mat(data), 1);
    if (img.empty())
    {
        std::cerr << "error: unable to decode image " << filename << std::endl;
        exit(1);
    }

    // Pre-process the image to match input size expected by the model.
    Preprocess(img, format, type1, type3, c, cv::Size(w, h), scale, input_data);
}

}  // namespace

int main(int argc, char** argv)
{
    bool verbose = false;
    int batch_size = 1;
    int topk = 1000;
    ScaleType scale = ScaleType::INCEPTION;
    std::string preprocess_output_filename;
    std::string model_name = "densenet_onnx";
    std::string model_version = "1";
    std::string url("localhost:8001");
    ProtocolType protocol = ProtocolType::GRPC;
    tc::Headers http_headers;

    std::unique_ptr<tc::InferenceServerGrpcClient> grpc_client;
    tc::Error err = tc::InferenceServerGrpcClient::Create(&grpc_client, url, verbose);
    if (!err.IsOk())
    {
        std::cerr << "error: unable to create client for inference: " << err << std::endl;
        exit(1);
    }

    ModelInfo model_info;
    inference::ModelMetadataResponse model_metadata;
    err = grpc_client->ModelMetadata(&model_metadata, model_name, model_version, http_headers);
    if (!err.IsOk())
    {
        std::cerr << "error: failed to get model metadata: " << err << std::endl;
    }

    inference::ModelConfigResponse model_config;
    err = grpc_client->ModelConfig(&model_config, model_name, model_version, http_headers);
    if (!err.IsOk())
    {
        std::cerr << "error: failed to get model config: " << err << std::endl;
    }
    ParseModelGrpc(model_metadata, model_config, batch_size, &model_info);

    // Collect the names of the image(s).
    std::string image_filename = "images/mug.jpg";

    // Preprocess the images into input data according to model
    // requirements
    std::vector<std::vector<uint8_t>> image_data;
    image_data.emplace_back();
    FileToInputData(image_filename, model_info.input_c_, model_info.input_h_, model_info.input_w_, model_info.input_format_, model_info.type1_, model_info.type3_, scale, &(image_data.back()));

    std::vector<int64_t> shape;
    // Include the batch dimension if required
    if (model_info.max_batch_size_ != 0)
    {
        shape.push_back(batch_size);
    }
    if (model_info.input_format_.compare("FORMAT_NHWC") == 0)
    {
        shape.push_back(model_info.input_h_);
        shape.push_back(model_info.input_w_);
        shape.push_back(model_info.input_c_);
    }
    else
    {
        shape.push_back(model_info.input_c_);
        shape.push_back(model_info.input_h_);
        shape.push_back(model_info.input_w_);
    }

    // std::vector<int64_t> times;
    // for(int i = 0; i < 100; ++i)
    // {
    //     auto start = std::chrono::steady_clock::now();

        tc::InferInput* input;
        err = tc::InferInput::Create(&input, model_info.input_name_, shape, model_info.input_datatype_);
        if (!err.IsOk())
        {
            std::cerr << "unable to get input: " << err << std::endl;
            exit(1);
        }
        std::shared_ptr<tc::InferInput> input_ptr(input);

        tc::InferRequestedOutput* output;
        err = tc::InferRequestedOutput::Create(&output, model_info.output_name_, topk);
        if (!err.IsOk())
        {
            std::cerr << "unable to get output: " << err << std::endl;
            exit(1);
        }    
        std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);

        err = input_ptr->Reset();
        if (!err.IsOk())
        {
            std::cerr << "failed resetting input: " << err << std::endl;
            exit(1);
        }

        err = input_ptr->AppendRaw(image_data[0]);
        if (!err.IsOk())
        {
            std::cerr << "failed setting input: " << err << std::endl;
            exit(1);
        }

        tc::InferResult* result;
        tc::InferOptions options(model_name);
        options.model_version_ = model_version;
        options.request_id_ = "request_densenet_onnx";
        std::vector<tc::InferInput*> inputs = { input_ptr.get() };
        std::vector<const tc::InferRequestedOutput*> outputs = { output_ptr.get() };
        
        err = grpc_client->Infer(&result, options, inputs, outputs, http_headers);
        if (!err.IsOk())
        {
            std::cerr << "failed sending synchronous infer request: " << err << std::endl;
            exit(1);
        }

    //     auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    //     times.push_back(elapsed.count());
    // }
    
    // double sum = std::accumulate(times.begin(), times.end(), 0.0);
    // double mean = sum / times.size();

    // double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    // double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

    // auto [min, max] = std::minmax_element(times.begin(), times.end());

    // std::cout << "=== Elapsed ===" << std::endl;
    // for(auto t : times)
    //     std::cout << t << " ";

    // std::cout << "\n\n=== Mean ===" << std::endl;
    // std::cout << mean << std::endl;

    // std::cout << "\n=== Std. Deviation ===" << std::endl;
    // std::cout << stdev << std::endl;
    
    // std::cout << "\n=== Minimum ===" << std::endl;
    // std::cout << *min << std::endl;

    // std::cout << "\n=== Maximum ===" << std::endl;
    // std::cout << *max << std::endl;

    Postprocess(result, image_filename, batch_size, model_info.output_name_, topk, model_info.max_batch_size_ != 0);

    return 0;
}