#pragma once

#include <stdexcept>
#include <string>

namespace tc::infer
{
class timeout_error : public std::runtime_error
{
public:
    explicit timeout_error(const std::string arg) : std::runtime_error(arg) {}
    virtual ~timeout_error() noexcept = default;
};

}