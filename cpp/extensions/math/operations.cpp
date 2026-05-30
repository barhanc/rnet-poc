#include "operations.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "core/tensor.h"

namespace mylib::extensions::math::operations
{
    namespace jsi = facebook::jsi;
    using TensorHostObject = mylib::core::tensor::TensorHostObject;

    void install_sigmoid(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "sigmoid";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "Usage: sigmoid(src, dst)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "sigmoid: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "sigmoid: dst must be a Tensor");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            if (src->shape_ != dst->shape_)
            {
                throw jsi::JSError(rt, "sigmoid: src and dst must have the same shape");
            }

            if (src->dtype_ != dst->dtype_)
            {
                throw jsi::JSError(rt, "sigmoid: src and dst must have the same dtype");
            }

            if (src->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "sigmoid: only float32 tensors are supported");
            }

            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "sigmoid: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "sigmoid: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "sigmoid: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "sigmoid: dst tensor has been disposed");
            }

            const auto countElements = std::accumulate(src->shape_.begin(), src->shape_.end(), size_t(1), std::multiplies<size_t>());
            const auto *srcData = reinterpret_cast<const float *>(src->data_.get());
            auto *dstData = reinterpret_cast<float *>(dst->data_.get());

            for (size_t i = 0; i < countElements; ++i)
            {
                dstData[i] = 1.0f / (1.0f + std::exp(-srcData[i]));
            }

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody));
    }

    void install_softmax(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "softmax";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2 && count != 3)
            {
                throw jsi::JSError(rt, "Usage: softmax(src, dst, options?)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "softmax: src must be a Tensor");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "softmax: dst must be a Tensor");
            }

            auto src = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto dst = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            if (src->shape_ != dst->shape_)
            {
                throw jsi::JSError(rt, "softmax: src and dst must have the same shape");
            }

            if (src->dtype_ != dst->dtype_)
            {
                throw jsi::JSError(rt, "softmax: src and dst must have the same dtype");
            }

            if (src->dtype_ != mylib::core::types::DType::float32)
            {
                throw jsi::JSError(rt, "softmax: only float32 tensors are supported");
            }

            if (src->shape_.empty())
            {
                throw jsi::JSError(rt, "softmax: src must have at least one dimension");
            }

            int axis = -1;
            if (count == 3)
            {
                if (!args[2].isObject())
                {
                    throw jsi::JSError(rt, "softmax: options must be an object");
                }

                auto options = args[2].asObject(rt);
                if (options.hasProperty(rt, "axis"))
                {
                    auto axisValue = options.getProperty(rt, "axis");
                    if (!axisValue.isNumber())
                    {
                        throw jsi::JSError(rt, "softmax: options.axis must be a number");
                    }
                    axis = static_cast<int>(axisValue.asNumber());
                }
            }

            const int rank = static_cast<int>(src->shape_.size());
            if (axis < 0)
            {
                axis += rank;
            }

            if (axis < 0 || axis >= rank)
            {
                throw jsi::JSError(rt, "softmax: axis is out of range");
            }

            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock())
            {
                throw jsi::JSError(rt, "softmax: src tensor is currently in use");
            }

            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "softmax: dst tensor is currently in use");
            }

            if (!src->data_)
            {
                throw jsi::JSError(rt, "softmax: src tensor has been disposed");
            }

            if (!dst->data_)
            {
                throw jsi::JSError(rt, "softmax: dst tensor has been disposed");
            }

            const auto *srcData = reinterpret_cast<const float *>(src->data_.get());
            auto *dstData = reinterpret_cast<float *>(dst->data_.get());

            const size_t axisDim = static_cast<size_t>(src->shape_[axis]);
            if (axisDim == 0)
            {
                throw jsi::JSError(rt, "softmax: axis dimension must be greater than zero");
            }

            size_t outer = 1;
            for (int i = 0; i < axis; ++i)
            {
                outer *= static_cast<size_t>(src->shape_[i]);
            }

            size_t inner = 1;
            for (int i = axis + 1; i < rank; ++i)
            {
                inner *= static_cast<size_t>(src->shape_[i]);
            }

            for (size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
            {
                for (size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
                {
                    const size_t base = outerIndex * axisDim * inner + innerIndex;

                    float maxValue = -std::numeric_limits<float>::infinity();
                    for (size_t axisIndex = 0; axisIndex < axisDim; ++axisIndex)
                    {
                        maxValue = std::max(maxValue, srcData[base + axisIndex * inner]);
                    }

                    float sum = 0.0f;
                    for (size_t axisIndex = 0; axisIndex < axisDim; ++axisIndex)
                    {
                        const float value = std::exp(srcData[base + axisIndex * inner] - maxValue);
                        dstData[base + axisIndex * inner] = value;
                        sum += value;
                    }

                    for (size_t axisIndex = 0; axisIndex < axisDim; ++axisIndex)
                    {
                        dstData[base + axisIndex * inner] /= sum;
                    }
                }
            }

            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }
} // namespace mylib::extensions::math::operations