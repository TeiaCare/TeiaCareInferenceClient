#include <teiacare/inference_client/client_factory.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::level_enum::trace);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::info("Running client_example_server_info");

    auto client = tc::infer::client_factory::create_client("localhost:8001");

    try
    {
        // Server Live
        auto is_server_live = client->is_server_live();
        spdlog::debug("\n");
        spdlog::debug("is_server_live: {} ", is_server_live);
        
        // Server Ready
        auto is_server_ready = client->is_server_ready();    
        spdlog::debug("\n");
        spdlog::debug("is_server_ready: {} ", is_server_ready);

        // Server Metadata
        auto server_metadata = client->server_metadata();    
        spdlog::debug("\n");
        spdlog::debug("server_name: {}", server_metadata.server_name);
        spdlog::debug("server_version: {}", server_metadata.server_version);
        spdlog::debug("server_extensions: {}", fmt::join(server_metadata.server_extensions, " "));
    }
    catch(const std::runtime_error& ex)
    {
        spdlog::error("Error. {}", ex.what());
        return EXIT_FAILURE;
    }

    spdlog::info("Example client_example_server_info finished");
    return EXIT_SUCCESS;
}
