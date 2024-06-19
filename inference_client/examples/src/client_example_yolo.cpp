#include <stdexcept>
#include <vector>
#include <chrono>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

#include <teiacare/inference_client/client_factory.hpp>
#include <teiacare/inference_client/infer_request.hpp>
#include <teiacare/inference_client/infer_response.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

struct Detection
{
    size_t label;
    cv::Rect bbox;
    float score;
};

class YOLO
{
public:
    YOLO(float confidence = 0.25f, float nonmaxs = 0.5f, int64_t w = 640, int64_t h = 640, int64_t c = 3)
        : conf{ confidence }
        , nms{ nonmaxs }
        , width{ w }
        , height{ h }
        , channels{ c }
    {
    }

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox, const size_t network_size)
    {
        float r_w = network_size / static_cast<float>(imgSz.width);
        float r_h = network_size / static_cast<float>(imgSz.height);

        int l, r, t, b;
        if (r_h > r_w)
        {
            l = bbox[0] - bbox[2] / 2.f;
            r = bbox[0] + bbox[2] / 2.f;
            t = bbox[1] - bbox[3] / 2.f - (network_size - r_w * imgSz.height) / 2;
            b = bbox[1] + bbox[3] / 2.f - (network_size - r_w * imgSz.height) / 2;
            l /= r_w;
            r /= r_w;
            t /= r_w;
            b /= r_w;
        }
        else
        {
            l = bbox[0] - bbox[2] / 2.f - (network_size - r_h * imgSz.width) / 2;
            r = bbox[0] + bbox[2] / 2.f - (network_size - r_h * imgSz.width) / 2;
            t = bbox[1] - bbox[3] / 2.f;
            b = bbox[1] + bbox[3] / 2.f;
            l /= r_h;
            r /= r_h;
            t /= r_h;
            b /= r_h;
        }
        // Clamp the coordinates within the image bounds
        l = std::max(0, std::min(l, imgSz.width - 1));
        r = std::max(0, std::min(r, imgSz.width - 1));
        t = std::max(0, std::min(t, imgSz.height - 1));
        b = std::max(0, std::min(b, imgSz.height - 1));

        return cv::Rect(l, t, r - l, b - t);
    }

    std::vector<float> preprocess(const cv::Mat& img)
    {
        int w, h, x, y;
        const auto input_layer_sz = width;
        float r_w = static_cast<float>(input_layer_sz) / static_cast<float>(img.cols);
        float r_h = static_cast<float>(input_layer_sz) / static_cast<float>(img.rows);
        if (r_h > r_w)
        {
            w = input_layer_sz;
            h = static_cast<int>(r_w * static_cast<float>(img.rows));
            x = 0;
            y = (input_layer_sz - h) / 2;
        }
        else
        {
            w = static_cast<int>(r_h * static_cast<float>(img.cols));
            h = input_layer_sz;
            x = (input_layer_sz - w) / 2;
            y = 0;
        }
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(rgb, re, re.size(), 0, 0, cv::INTER_CUBIC);
        cv::Mat out(input_layer_sz, input_layer_sz, CV_8UC3, cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        out.convertTo(rgb, CV_32FC3, 1.f / 255.f);

        size_t img_byte_size = rgb.total() * rgb.elemSize() * sizeof(float);
        std::vector<float> input_data;
        input_data.resize(img_byte_size);

        size_t pos = 0;
        std::vector<cv::Mat> input_bgr_channels;
        for (size_t i = 0; i < channels; ++i)
        {
            input_bgr_channels.emplace_back(height, width, CV_32FC1, &((input_data)[pos]));
            pos += input_bgr_channels.back().total() * input_bgr_channels.back().elemSize() * sizeof(float);
        }

        cv::split(rgb, input_bgr_channels);

        if (pos != img_byte_size)
        {
            throw std::runtime_error("Invalid image size");
        }
        return input_data;
    }

    std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size)
    {
        const float* output0 = outputs.front().data();
        const std::vector<int64_t> shape0 = shapes.front();
        const auto offset = 5;
        const auto num_classes = shape0[2] - offset;  // 1 x 25200 x 85

        std::vector<Detection> detections;
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;

        std::vector<std::vector<float>> picked_proposals;

        for (int i = 0; i < shape0[1]; ++i)
        {
            const float* scoresPtr = output0 + 5;
            auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
            float score = *maxSPtr * output0[4];
            if (score > conf)
            {
                boxes.emplace_back(get_rect(frame_size, std::vector<float>(output0, output0 + 4), width));
                int label = maxSPtr - scoresPtr;
                confs.emplace_back(score);
                classIds.emplace_back(label);
            }

            output0 += shape0[2];
        }

        // Perform Non Maximum Suppression and draw predictions.
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confs, conf, nms, indices);
        for (int i = 0; i < indices.size(); i++)
        {
            Detection det;
            int idx = indices[i];
            det.label = classIds[idx];
            det.bbox = boxes[idx];
            det.score = confs[idx];
            detections.emplace_back(det);
        }

        return detections;
    }

    int64_t getW() { return width; }
    int64_t getH() { return height; }
    int64_t getC() { return channels; }
    std::string get_input_layer_name() { return input_layer_name; }

private:
    int64_t width{ 0 };
    int64_t height{ 0 };
    int64_t channels{ 0 };
    const float conf;
    const float nms;
    const std::string input_layer_name = "images";
};

std::string keys = "{ help  h    |                    | Print help message. }"
                   "{ input i    | images/person.png  | Path to input image or video file. }"
                   "{ address ip | 192.168.0.221:8001 | path to triton server, in the format ip:port}"
                   "{ thr        | .5                 | Confidence threshold. }"
                   "{ nms        | .4                 | Non-maximum suppression threshold. }";

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    const auto image_path = parser.get<std::string>("input");
    const auto ip = parser.get<std::string>("ip");

    if (!std::filesystem::exists(image_path))
    {
        std::cerr << "Not existent image " << image_path << std::endl;
        std::exit(1);
    }
    YOLO yolo;

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    auto input_data = yolo.preprocess(img);

    auto client = tc::infer::client_factory::create_client(ip);

    std::vector<int64_t> times;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<float>> outputs;
    const size_t max_runs = 1;
    for (int i = 0; i < max_runs; ++i)
    {
        auto start = std::chrono::steady_clock::now();

        tc::infer::infer_request request;
        request.model_name = "yolov5x_face_oss_res_trt";
        request.model_version = "23";
        request.id = "client_example_" + request.model_name;
        request.add_input_tensor(input_data.data(), input_data.size(), { 1, yolo.getC(), yolo.getW(), yolo.getH() }, yolo.get_input_layer_name());
        tc::infer::infer_response response;
        try
        {
            response = client->infer(request, std::chrono::milliseconds(500));
        }
        catch (const tc::infer::timeout_error& ex)
        {
            std::cout << "[Timeout Error] Unable to perform inference " << ex.what() << std::endl;
            return EXIT_FAILURE;
        }
        catch (const std::runtime_error& ex)
        {
            std::cout << "[Error] Unable to perform inference " << ex.what() << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Model name: " << response.model_name << std::endl;
        std::cout << "Model version: " << response.model_version << std::endl;
        std::cout << "Output layers" << std::endl;
        for (const auto& output : response.output_tensors)
        {
            std::cout << "- Name: " << output.name() << std::endl;
            std::cout << "- DataType: " << output.datatype().str() << std::endl;
            std::cout << "- Shape: " << output.shape().size() << std::endl;
            std::cout << "- Output layer data" << std::endl;
            const float* data = output.as<float>();
            std::cout << " Output tensor shape: " << std::endl;
            size_t num_elements = 1;
            std::vector<int64_t> tensor_shape;
            outputs.emplace_back(std::vector<float>(data, data + output.data_size()));
            shapes.emplace_back(output.shape());
        }
        const auto detections = yolo.postprocess(outputs, shapes, img.size());

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        times.push_back(elapsed.count());
        const std::vector<std::string> classes{ "face", "oss", "res" };
        for (auto detection : detections)
        {
            cv::rectangle(img, detection.bbox, cv::Scalar(255, 0, 0));
            cv::putText(img, classes[detection.label], detection.bbox.tl(), 0, 1, cv::Scalar(255, 0, 255), 3);
        }
        cv::imwrite("processed.png", img);
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

    auto [min, max] = std::minmax_element(times.begin(), times.end());

    std::cout << "=== Elapsed ===" << std::endl;
    for (auto t : times) std::cout << t << " ";

    std::cout << "\n\n=== Mean ===" << std::endl;
    std::cout << mean << std::endl;

    std::cout << "\n=== Std. Deviation ===" << std::endl;
    std::cout << stdev << std::endl;

    std::cout << "\n=== Minimum ===" << std::endl;
    std::cout << *min << std::endl;

    std::cout << "\n=== Maximum ===" << std::endl;
    std::cout << *max << std::endl;

    return EXIT_SUCCESS;
}
