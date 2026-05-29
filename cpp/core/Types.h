#pragma once

#include <cstdint>
#include <string>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace mylib::core::types
{
    enum DType
    {
        uint8,
        int32,
        float32
    };

    DType parseDType(const std::string &s);
    std::string toString(DType dtype);

    executorch::aten::ScalarType toScalarType(DType dtype);
    DType fromScalarType(executorch::aten::ScalarType st);

    size_t elementSize(DType dtype);

} // namespace mylib::core::types
