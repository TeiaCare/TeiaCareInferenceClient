#include <teiacare/inference_client/client_factory.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::level_enum::trace);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::info("Running client_example_model_info");

    auto client = tc::infer::client_factory::create_client("localhost:8001");

    auto model_name = "simple_int32";
    auto model_version = "1";

    try
    {
        // Model Ready
        auto is_model_ready = client->is_model_ready(model_name, model_version);
        spdlog::debug("\n");
        spdlog::debug("is_model_ready: {}", is_model_ready);

        // Model Metadata
        auto model_metadata = client->model_metadata(model_name, model_version);
        spdlog::debug("\n");
        spdlog::debug("model_metadata:");
        spdlog::debug("model_name: {}", model_metadata.model_name);
        spdlog::debug("model_versions: {}", fmt::join(model_metadata.model_versions, ","));

        spdlog::debug("\n");
        spdlog::debug("input tensor metadata: {}", model_metadata.platform);
        for(auto& input : model_metadata.inputs)
        {
            spdlog::debug("  name: {}", input.name);
            spdlog::debug("    datatype: {}", input.datatype);
            spdlog::debug("    shape: [{}]", fmt::join(input.shape, ","));
        }

        spdlog::debug("\n");
        spdlog::debug("output tensor metadata: {}", model_metadata.platform);
        for(auto& output : model_metadata.outputs)
        {
            spdlog::debug("  name: {}", output.name);
            spdlog::debug("    datatype: {}", output.datatype);
            spdlog::debug("    shape: [{}]", fmt::join(output.shape, ","));
        }

        // Model List
        /* CURRENTLY NOT IMPLEMENTED IN TRITON INFERENCE SERVER (AMD SERVER ONLY) */
        // auto model_list = client->model_list();
        // spdlog::debug("\n");
        // spdlog::debug("model_list:");
        // for (auto&& model : model_list)
        //     spdlog::debug(model);

        // Model Load
        /* CURRENTLY NOT IMPLEMENTED IN TRITON INFERENCE SERVER (AMD SERVER ONLY) */
        // auto is_model_loaded = client->model_load(model_name, model_version);
        // spdlog::debug("\n");
        // spdlog::debug("model_load: {}", is_model_loaded);

        // Model Unload
        /* CURRENTLY NOT IMPLEMENTED IN TRITON INFERENCE SERVER (AMD SERVER ONLY) */
        // auto is_model_unloaded = client->model_unload(model_name, model_version);
        // spdlog::debug("\n");
        // spdlog::debug("model_unload: {}", is_model_unloaded);

    }
    catch(const std::runtime_error& ex)
    {
        spdlog::error("Error. {}", ex.what());
        return EXIT_FAILURE;
    }

    spdlog::info("Example client_example_model_info finished");
    return EXIT_SUCCESS;
}
