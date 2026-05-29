#pragma once

#include <string>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace mylib::core::types
{
    executorch::aten::ScalarType stringToScalarType(const std::string &dtype);
    std::string scalarTypeToString(executorch::aten::ScalarType scalarType);
    size_t getElementSize(executorch::aten::ScalarType scalarType);
} // namespace mylib::core::types
