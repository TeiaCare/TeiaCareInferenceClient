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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace timings
{
inline void print_stats(std::vector<int64_t> times)
{
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

    auto [min, max] = std::minmax_element(times.begin(), times.end());

    std::cout << "=== Elapsed ===" << std::endl;
    for (auto t : times)
        std::cout << t << " ";

    std::cout << "\n\n=== Mean ===" << std::endl;
    std::cout << mean << std::endl;

    std::cout << "\n=== Std. Deviation ===" << std::endl;
    std::cout << stdev << std::endl;

    std::cout << "\n=== Minimum ===" << std::endl;
    std::cout << *min << std::endl;

    std::cout << "\n=== Maximum ===" << std::endl;
    std::cout << *max << std::endl;
}

}
