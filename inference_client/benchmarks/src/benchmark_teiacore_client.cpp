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

#include <teiacare/inference_client/client_factory.hpp>

#include <benchmark/benchmark.h>

static void benchmark_teiacare_client(benchmark::State& state)
{
    auto client = tc::infer::client_factory::create_client("localhost:8001");

    std::vector<int32_t> data_0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<int32_t> data_1{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int64_t> shape{1, 16};

    for (auto _ : state)
    {
        tc::infer::infer_request request;
        request.model_name = "simple_int32";
        request.model_version = "1";
        request.add_input_tensor(data_0.data(), data_0.size(), shape, "INPUT0");
        request.add_input_tensor(data_1.data(), data_1.size(), shape, "INPUT1");

        tc::infer::infer_response response;
        response = client->infer(request);

        auto output0_data = response.output_tensors[0].as<int32_t>();
        auto output1_data = response.output_tensors[1].as<int32_t>();
        for (size_t i = 0; i < 16; ++i)
        {
            if ((data_0[i] + data_1[i]) != *(output0_data + i))
                throw std::runtime_error("error: incorrect sum");

            if ((data_0[i] - data_1[i]) != *(output1_data + i))
                throw std::runtime_error("error: incorrect difference");
        }
    }
}

BENCHMARK(benchmark_teiacare_client)
    ->Threads(1)
    ->Iterations(10'000)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->ReportAggregatesOnly(false)
    ->DisplayAggregatesOnly(false);

BENCHMARK_MAIN();
