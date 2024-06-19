#pragma once

#include <vector>
#include <chrono>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

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
    for(auto t : times)
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
