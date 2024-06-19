#include "tensor_converter.hpp"
#include <bit>

namespace tc::infer
{
auto tensor_converter::get_infer_request(const tc::infer::infer_request& infer_request) const -> inference::ModelInferRequest
{
    inference::ModelInferRequest request;
    request.set_model_name(infer_request.model_name);
    request.set_model_version(infer_request.model_version);
    request.set_id(infer_request.id);
    request.mutable_parameters()->clear();
    request.mutable_raw_input_contents()->Clear();

    for (const tc::infer::infer_tensor& request_input : infer_request.input_tensors)
    {
        inference::ModelInferRequest_InferInputTensor* tensor = request.add_inputs();
        tensor->set_name(request_input.name());
        tensor->set_datatype(request_input.datatype().str());
        tensor->mutable_parameters()->clear();

        for (auto&& shape : request_input.shape())
        {
            tensor->add_shape(shape);
        }

        const size_t input_size = request_input.data_size();
        tensor_data_converter_call_wrapper<tensor_data_writer>(
            request_input.datatype(), 
            tensor, 
            request_input.raw_data(), 
            input_size);

        // DEBUG
        // auto source_data = request_input.raw_data();
        // auto data = static_cast<const float*>((void*)source_data);        
        // // auto contents = tensor->contents().bytes_contents().data();
        // auto contents = tensor->mutable_contents()->mutable_fp32_contents();
        // contents->Add(&data[0], &data[0]+input_size);
    }

    return request;
}

auto tensor_converter::get_infer_response(const inference::ModelInferResponse& response) const -> tc::infer::infer_response
{
    tc::infer::infer_response infer_response;
    infer_response.model_name = response.model_name();
    infer_response.model_version = response.model_version();
    infer_response.id = response.id();
    infer_response.output_tensors.reserve(response.outputs_size());

    int raw_output_index = 0;
    for (const inference::ModelInferResponse_InferOutputTensor& response_output : response.outputs())
    {
        // Triton Inference Server is only capable to output Raw Output contents instead of using type specific outputs
        if (response.raw_output_contents_size())
        {
            std::string output_content = response.raw_output_contents()[raw_output_index];

            tc::infer::infer_tensor infer_tensor_output(
                std::vector<std::byte>(std::bit_cast<std::byte*>(output_content.data()), std::bit_cast<std::byte*>(output_content.data() + output_content.size())),
                std::vector<int64_t>(response_output.shape().begin(), response_output.shape().end()),
                tc::infer::data_type(response_output.datatype()), 
                response_output.name()
            );

            infer_response.add_output_tensor(std::move(infer_tensor_output));
            ++raw_output_index;
        }
        else
        {
            // TODO: implement tensor_data_reader to support AMD server
            // const size_t output_size = std::accumulate(response_output.shape().begin(), response_output.shape().end(), size_t(1), std::multiplies<>());
            // tensor_data_converter_call_wrapper<tensor_data_reader>(infer_tensor_output.datatype, const_cast<inference::ModelInferResponse_InferOutputTensor*>(&response_output), &infer_tensor_output, output_size);
        }
    }

    return infer_response;
}

}