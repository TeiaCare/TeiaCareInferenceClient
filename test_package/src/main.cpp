#include <teiacare/inference_client/client_factory.hpp>
#include <iostream>

int main(int, char**)
{
    auto client = tc::infer::client_factory::create_client("localhost:8001");
    std::cout << "Welcome to TeiaCareInferenceClient - This is a Conan package test" << std::endl;
    return 0;
}