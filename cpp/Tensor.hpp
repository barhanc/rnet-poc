#pragma once

#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <string>
#include <vector>

#include <jsi/jsi.h>

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include "Types.hpp"


namespace mylib::tensor
{
    namespace jsi = facebook::jsi;

    struct TensorHostObject : public jsi::HostObject
    {
        std::string dtype_;
        std::vector<std::int32_t> shape_;

        size_t size_;
        std::unique_ptr<std::uint8_t[]> data_;
        executorch::extension::TensorPtr tensor_;

        std::shared_mutex mutex_;

        TensorHostObject(const std::vector<std::int32_t> &shape, const std::string &dtype)
        {
            executorch::aten::ScalarType scalarType = mylib::types::stringToScalarType(dtype);
            size_t elementSize = mylib::types::getElementSize(scalarType);

            dtype_ = dtype;
            shape_ = shape;

            auto numElements = std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());

            size_ = numElements * elementSize;
            data_ = std::make_unique<std::uint8_t[]>(size_);
            tensor_ = executorch::extension::from_blob(data_.get(), shape_, scalarType);
        }

        TensorHostObject(const executorch::aten::Tensor &tensor)
        {
            dtype_ = mylib::types::scalarTypeToString(tensor.dtype());
            shape_ = std::vector<std::int32_t>(tensor.sizes().begin(), tensor.sizes().end());

            size_ = tensor.nbytes();
            data_ = std::make_unique<std::uint8_t[]>(size_);
            tensor_ = executorch::extension::from_blob(data_.get(), shape_, tensor.dtype());

            std::memcpy(data_.get(), tensor.const_data_ptr(), size_);
        }

        jsi::Value get(jsi::Runtime &rt, const jsi::PropNameID &name) override
        {
            auto nameStr = name.utf8(rt);

            if (nameStr == "shape")
            {
                auto jsArray = jsi::Array(rt, shape_.size());
                for (size_t i = 0; i < shape_.size(); ++i)
                {
                    jsArray.setValueAtIndex(rt, i, static_cast<double>(shape_[i]));
                }
                return jsArray;
            }

            if (nameStr == "dtype")
            {
                return jsi::String::createFromUtf8(rt, dtype_);
            }

            return jsi::Value::undefined();
        }

        std::vector<facebook::jsi::PropNameID> getPropertyNames(jsi::Runtime &rt) override
        {
            std::vector<facebook::jsi::PropNameID> properties;
            properties.push_back(jsi::PropNameID::forAscii(rt, "shape"));
            properties.push_back(jsi::PropNameID::forAscii(rt, "dtype"));
            return properties;
        }
    };

    inline void install_createTensor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "createTensor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "Usage: createTensor(shape, dtype)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isArray(rt))
            {
                throw jsi::JSError(rt, "Expected shape as an array of integers");
            }

            if (!args[1].isString())
            {
                throw jsi::JSError(rt, "Expected dtype as a string");
            }

            auto shapeArray = args[0].asObject(rt).asArray(rt);
            std::vector<std::int32_t> shape;
            for (size_t i = 0; i < shapeArray.length(rt); ++i)
            {
                auto dimValue = shapeArray.getValueAtIndex(rt, i);
                if (!dimValue.isNumber())
                {
                    throw jsi::JSError(rt, "Shape array must contain only numbers");
                }

                if (dimValue.asNumber() <= 0)
                {
                    throw jsi::JSError(rt, "Shape dimensions must be positive integers");
                }

                shape.push_back(static_cast<std::int32_t>(dimValue.asNumber()));
            }

            try
            {
                auto dtype = args[1].asString(rt).utf8(rt);
                auto tensorHostObject = std::make_shared<TensorHostObject>(shape, dtype);
                return jsi::Object::createFromHostObject(rt, tensorHostObject);
            }
            catch (const std::exception &e)
            {
                throw jsi::JSError(rt, "Error creating tensor: " + std::string(e.what()));
            }
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }

    inline void install_disposeTensor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "disposeTensor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
            {
                throw jsi::JSError(rt, "Usage: disposeTensor(tensor)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "Expected a TensorHostObject");
            }

            auto tensorHostObject = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);

            std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Tensor is currently in use and cannot be disposed");
            }

            if (!tensorHostObject->data_)
            {
                throw jsi::JSError(rt, "Tensor has already been disposed");
            }

            tensorHostObject->tensor_.reset();
            tensorHostObject->data_.reset();

            return jsi::Value::undefined();
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }

    inline void install_setTensorFromTypedArray(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "setTensorFromTypedArray";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "Usage: setTensorFromTypedArray(tensor, array)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "Expected a TensorHostObject");
            }

            if (!args[1].isObject())
            {
                throw jsi::JSError(rt, "Expected array to be an object (TypedArray)");
            }

            jsi::Object dataObj = args[1].asObject(rt);
            if (!dataObj.hasProperty(rt, "buffer"))
            {
                throw jsi::JSError(rt, "Expected a TypedArray with a 'buffer' property");
            }

            jsi::ArrayBuffer buffer = dataObj.getProperty(rt, "buffer").asObject(rt).getArrayBuffer(rt);
            size_t byteOffset = 0;
            size_t byteLength = buffer.size(rt);

            if (dataObj.hasProperty(rt, "byteOffset"))
            {
                auto byteOffsetValue = dataObj.getProperty(rt, "byteOffset");
                if (!byteOffsetValue.isNumber())
                {
                    throw jsi::JSError(rt, "Expected 'byteOffset' to be a number");
                }
                byteOffset = static_cast<size_t>(byteOffsetValue.asNumber());
            }

            if (dataObj.hasProperty(rt, "byteLength"))
            {
                auto byteLengthValue = dataObj.getProperty(rt, "byteLength");
                if (!byteLengthValue.isNumber())
                {
                    throw jsi::JSError(rt, "Expected 'byteLength' to be a number");
                }
                byteLength = static_cast<size_t>(byteLengthValue.asNumber());
            }

            auto tensorHostObject = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);

            std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Tensor is currently in use and cannot be written to");
            }

            if (!tensorHostObject->data_)
            {
                throw jsi::JSError(rt, "Tensor has been disposed");
            }

            if (byteLength != tensorHostObject->size_)
            {
                std::string errorMsg = "Data size mismatch: TypedArray is " + std::to_string(byteLength) +
                                       " bytes, but Tensor requires " + std::to_string(tensorHostObject->size_) +
                                       " bytes.";
                throw jsi::JSError(rt, errorMsg);
            }

            std::memcpy(tensorHostObject->data_.get(), buffer.data(rt) + byteOffset, byteLength);

            return jsi::Value::undefined();
        };

        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }

    inline void install_setTypedArrayFromTensor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "setTypedArrayFromTensor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "Usage: setTypedArrayFromTensor(array, tensor)");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "Expected a TensorHostObject");
            }

            if (!args[0].isObject())
            {
                throw jsi::JSError(rt, "Expected array to be an object (TypedArray)");
            }

            jsi::Object dataObj = args[0].asObject(rt);
            if (!dataObj.hasProperty(rt, "buffer"))
            {
                throw jsi::JSError(rt, "Expected a TypedArray with a 'buffer' property");
            }

            jsi::ArrayBuffer buffer = dataObj.getProperty(rt, "buffer").asObject(rt).getArrayBuffer(rt);
            size_t byteOffset = 0;
            size_t byteLength = buffer.size(rt);

            if (dataObj.hasProperty(rt, "byteOffset"))
            {
                auto byteOffsetValue = dataObj.getProperty(rt, "byteOffset");
                if (!byteOffsetValue.isNumber())
                {
                    throw jsi::JSError(rt, "Expected 'byteOffset' to be a number");
                }
                byteOffset = static_cast<size_t>(byteOffsetValue.asNumber());
            }

            if (dataObj.hasProperty(rt, "byteLength"))
            {
                auto byteLengthValue = dataObj.getProperty(rt, "byteLength");
                if (!byteLengthValue.isNumber())
                {
                    throw jsi::JSError(rt, "Expected 'byteLength' to be a number");
                }
                byteLength = static_cast<size_t>(byteLengthValue.asNumber());
            }

            auto tensorHostObject = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Tensor is currently in use and cannot be written to");
            }

            if (!tensorHostObject->data_)
            {
                throw jsi::JSError(rt, "Tensor has been disposed");
            }

            if (byteLength != tensorHostObject->size_)
            {
                std::string errorMsg = "Data size mismatch: TypedArray is " + std::to_string(byteLength) +
                                       " bytes, but Tensor requires " + std::to_string(tensorHostObject->size_) +
                                       " bytes.";
                throw jsi::JSError(rt, errorMsg);
            }

            std::memcpy(buffer.data(rt) + byteOffset, tensorHostObject->data_.get(), byteLength);

            return jsi::Value::undefined();
        };
        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody));
    }
} // namespace mylib::tensor
