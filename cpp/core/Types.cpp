#include "types.h"
#include <stdexcept>

namespace mylib::core::types
{
    executorch::aten::ScalarType stringToScalarType(const std::string &dtype)
    {
        if (dtype == "float32") return executorch::aten::ScalarType::Float;
        if (dtype == "uint8") return executorch::aten::ScalarType::Byte;
        if (dtype == "int32") return executorch::aten::ScalarType::Int;
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }

    std::string scalarTypeToString(executorch::aten::ScalarType scalarType)
    {
        switch (scalarType)
        {
            case executorch::aten::ScalarType::Float: return "float32";
            case executorch::aten::ScalarType::Byte: return "uint8";
            case executorch::aten::ScalarType::Int: return "int32";
            default: throw std::runtime_error("Unsupported tensor dtype");
        }
    }

    size_t getElementSize(executorch::aten::ScalarType scalarType)
    {
        switch (scalarType)
        {
            case executorch::aten::ScalarType::Float: return 4;
            case executorch::aten::ScalarType::Byte: return 1;
            case executorch::aten::ScalarType::Int: return 4;
            default: throw std::runtime_error("Unsupported tensor dtype");
        }
    }
} // namespace mylib::core::types
