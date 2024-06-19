#pragma once

#include <teiacare/inference_client/data_type.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <numeric>
#include <bit>

namespace tc::infer
{
class infer_tensor
{
public:
    explicit infer_tensor(std::vector<std::byte> data, std::vector<int64_t> shape, data_type datatype, const std::string& name)
        : _data{ data }
        , _shape { std::move(shape) }
        , _datatype{ std::move(datatype) }
        , _name { std::move(name) }
    {
    }

    [[nodiscard]]
    inline std::vector<int64_t> shape() const noexcept
    {
        return _shape;
    }

    [[nodiscard]]
    inline data_type datatype() const noexcept
    {
        return _datatype;
    }

    [[nodiscard]]
    inline std::string name() const noexcept
    {
        return _name;
    }

    [[nodiscard]]
    inline size_t data_size() const noexcept
    {
        return std::accumulate(_shape.begin(), _shape.end(), size_t{1}, std::multiplies<>());
    }

    [[nodiscard]]
    inline const std::byte* raw_data() const noexcept
    {
        return _data.data();
    }

    template<typename T>
    [[nodiscard]]
    inline const T* as() const noexcept
    {
        return std::bit_cast<const T*>(_data.data());
    }

    template<typename T>
    [[nodiscard]]
    inline std::vector<T> data() const noexcept
    {
        return std::vector<T>(
            std::bit_cast<T*>(_data.data()), 
            std::bit_cast<T*>(_data.data() + _data.size())
        );
    }

    template<typename T>
    [[nodiscard]]
    inline std::vector<T> clone_data() const noexcept
    {
        const auto size = data_size();
        std::vector<T> clone;
        clone.reserve(size);
        T* data = std::bit_cast<T*>(_data.data());
        for (size_t i = 0; i < size; ++i)
        {
            clone.push_back(data[i]);
        }
        return clone;
    }

private:
    std::vector<std::byte> _data;
    std::vector<int64_t> _shape;
    data_type _datatype;
    std::string _name;
};

}