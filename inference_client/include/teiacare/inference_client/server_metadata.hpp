#pragma once

#include <string>
#include <vector>

namespace tc::infer
{
struct server_metadata
{
    std::string server_name;
    std::string server_version;
    std::vector<std::string> server_extensions;
};

}