#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace tc::infer
{
class data_type
{
public:
    enum value : uint8_t
    {
        Bool,
        Uint8,
        Uint16,
        Uint32,
        Uint64,
        Int8,
        Int16,
        Int32,
        Int64,
        Fp16,
        Fp32,
        Fp64,
        String,
        Unknown,
    };

    data_type() : _value{value::Unknown} {}
    data_type(data_type::value value) : _value(value) {}
    data_type(const std::string& str_value) : _value(from_string(str_value)) {}

    constexpr operator value() const { return _value; }

    [[nodiscard]] static data_type::value from_string(const std::string& str_value);
    [[nodiscard]] std::string str() const;

private:
    value _value = value::Unknown;
    static std::map<data_type::value, std::string> to_string;
    static std::map<std::string, data_type::value> to_type;
};

template <typename T>
struct cast_to_data_type;

template <>
struct cast_to_data_type<bool> {
  static constexpr data_type::value type = data_type::value::Bool;
};

// template <>
// struct cast_to_data_type<fp16> {
//   static constexpr data_type::value type = data_type::value::Fp16;
// };

template <>
struct cast_to_data_type<float> {
  static constexpr data_type::value type = data_type::value::Fp32;
};

template <>
struct cast_to_data_type<double> {
  static constexpr data_type::value type = data_type::value::Fp64;
};

template <>
struct cast_to_data_type<int8_t> {
  static constexpr data_type::value type = data_type::value::Int8;
};

template <>
struct cast_to_data_type<int16_t> {
  static constexpr data_type::value type = data_type::value::Int16;
};

template <>
struct cast_to_data_type<int32_t> {
  static constexpr data_type::value type = data_type::value::Int32;
};

template <>
struct cast_to_data_type<int64_t> {
  static constexpr data_type::value type = data_type::value::Int64;
};

template <>
struct cast_to_data_type<uint8_t> {
  static constexpr data_type::value type = data_type::value::Uint8;
};

template <>
struct cast_to_data_type<uint16_t> {
  static constexpr data_type::value type = data_type::value::Uint16;
};

template <>
struct cast_to_data_type<uint32_t> {
  static constexpr data_type::value type = data_type::value::Uint32;
};

template <>
struct cast_to_data_type<uint64_t> {
  static constexpr data_type::value type = data_type::value::Uint64;
};

template <>
struct cast_to_data_type<const char*> {
  static constexpr data_type::value type = data_type::value::String;
};

template <>
struct cast_to_data_type<char> {
  static constexpr data_type::value type = data_type::value::String;
};

}