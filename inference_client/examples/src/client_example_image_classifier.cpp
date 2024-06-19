#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include <vector>
#include <chrono>
#include <iostream>
#include <iostream>

#include <teiacare/inference_client/client_factory.hpp>
#include <teiacare/inference_client/infer_request.hpp>
#include <teiacare/inference_client/infer_response.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

std::vector<uint8_t> load_image(const char* image_path, int width, int height, int channels = 3)
{
    int original_width = 0;
    int original_height = 0;
    int original_channels = 0;
    uint8_t* original_img_data = stbi_load(image_path, &original_width, &original_height, &original_channels, STBI_rgb);
    // if (!original_img_data)
    //     throw std::runtime_error(fmt::format("Unable to load image: {}", image_path));

    size_t img_size = width * height * channels;
    std::vector<uint8_t> img_data = std::vector<uint8_t>(img_size);

    stbir_resize_uint8_linear(original_img_data, original_width, original_height, 0, img_data.data(), width, height, 0, static_cast<stbir_pixel_layout>(channels));
    stbi_image_free(original_img_data);

    return img_data;
}

int main(int argc, char** argv)
{
    auto image_path = "images/dog.jpg";
    const int width = 224;
    const int height = 224;
    const int channels = 3;
    // auto img = load_image(image_path, width, height);
    // stbi_write_jpg("output.jpg", width, height, channels, img.data(), 100);

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    
    cv::Mat sample;
    cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
    
    cv::Mat sample_resized;
    cv::resize(sample, sample_resized, {224, 224});

    cv::Mat sample_type;
    sample_resized.convertTo(sample_type, CV_32FC3);

    // Inception scaling
    cv::Mat sample_final;
    sample_final = sample_type.mul(cv::Scalar(1 / 127.5, 1 / 127.5, 1 / 127.5));
    sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);
    
    size_t img_byte_size = sample_final.total() * sample_final.elemSize() * sizeof(uint8_t);
    std::vector<uint8_t> input_data;
    input_data.resize(img_byte_size);

    size_t pos = 0;
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < channels; ++i)
    {
        input_bgr_channels.emplace_back(height, width, CV_32FC1, &((input_data)[pos]) );
        pos += input_bgr_channels.back().total() * input_bgr_channels.back().elemSize() * sizeof(uint8_t);
    }

    cv::split(sample_final, input_bgr_channels);

    if (pos != img_byte_size)
    {
        throw std::runtime_error("Invalid image size");
    }

    auto client = tc::infer::client_factory::create_client("localhost:8001");

    // std::vector<int64_t> times;
    // for(int i = 0; i < 100; ++i)
    // {
    //     auto start = std::chrono::steady_clock::now();

        tc::infer::infer_request request;
        request.model_name = "densenet_onnx";
        request.model_version = "1";
        request.id = "";    
        request.add_input_tensor((std::byte*)input_data.data(), input_data.size(), { channels, width, height }, tc::infer::data_type::Fp32, "data_0");

        tc::infer::infer_response response;
        try
        {
            response = client->infer(request, std::chrono::seconds(5));
        }
        catch (const tc::infer::timeout_error& ex)
        {
            std::cout << "[Timeout Error]\nUnable to perform inference\n" << ex.what() << std::endl;
            return EXIT_FAILURE;
        }
        catch (const std::runtime_error& ex)
        {
            std::cout << "[Error]\nUnable to perform inference\n" << ex.what() << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Model name: " << response.model_name << std::endl;
        std::cout << "Model version: " << response.model_version << std::endl;
        std::cout << "Output layers" << std::endl;
        for (const auto& output : response.output_tensors)
        {
            std::cout << "- Name: " << output.name() << std::endl;
            std::cout << "- DataType: " << output.datatype().str() << std::endl;
            std::cout << "- Shape: ";
            for (auto shape : output.shape())
                std::cout << shape << " " << std::endl;
            
            std::cout << "- Size: " << output.data_size() << std::endl;
            std::cout << "- Output layer data: " << std::endl;
            
            // Iterate through data pointer after type cast
            // (the memory is owned by the infer_tensor object. There is no need to release it)
            // const float* data = output.as<float>();
            // for (auto i = 0; i < output.data_size(); ++i)
            // {
            //     std::cout << i << ": " << data[i] << std::endl;
            // }

            // Iterate data by converting them as a std::vector<T>
            // (the memory is owned by the infer_tensor object. There is no need to release it)
            int i1 = 0;
            for (auto d : output.data<float>())
            {
                std::cout << i1 << ": " << d << std::endl;
                i1++;
            }

            // Iterate data after a clone (deep copy)
            // (the memory is owned by cloned object.)
            // int i2 = 0;
            // auto data_clone = output.clone_data<float>();
            // for (auto d : data_clone)
            // {
            //     std::cout << i2 << ": " << d << std::endl;
            //     i2++;
            // }
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

    return EXIT_SUCCESS;
}
