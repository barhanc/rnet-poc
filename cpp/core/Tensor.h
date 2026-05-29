#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include <jsi/jsi.h>

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace mylib::core::tensor
{
    struct TensorHostObject : public facebook::jsi::HostObject
    {
        std::string dtype_;
        std::vector<std::int32_t> shape_;

        size_t size_;
        std::unique_ptr<std::uint8_t[]> data_;
        executorch::extension::TensorPtr tensor_;

        std::shared_mutex mutex_;

        TensorHostObject(const std::vector<std::int32_t> &shape, const std::string &dtype);
        TensorHostObject(const executorch::aten::Tensor &tensor);

        facebook::jsi::Value get(facebook::jsi::Runtime &rt, const facebook::jsi::PropNameID &name) override;
        std::vector<facebook::jsi::PropNameID> getPropertyNames(facebook::jsi::Runtime &rt) override;
    };

    void install_createTensor(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
    void install_disposeTensor(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
    void install_setTensorFromTypedArray(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
    void install_setTypedArrayFromTensor(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::core::tensor
