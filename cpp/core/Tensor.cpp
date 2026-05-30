#include "Tensor.h"
#include <cstring>
#include <numeric>

namespace mylib::core::tensor
{
    namespace jsi = facebook::jsi;

    TensorHostObject::TensorHostObject(const std::vector<std::int32_t> &shape, mylib::core::types::DType dtype)
    {
        const auto elemSize = mylib::core::types::elementSize(dtype);

        dtype_ = dtype;
        shape_ = shape;

        auto numElements = std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());

        size_ = numElements * elemSize;
        data_ = std::make_unique<std::uint8_t[]>(size_);
        tensor_ = executorch::extension::from_blob(data_.get(), shape_, mylib::core::types::toScalarType(dtype));
    }

    TensorHostObject::TensorHostObject(const executorch::aten::Tensor &tensor)
    {
        dtype_ = mylib::core::types::fromScalarType(tensor.dtype());
        shape_ = std::vector<std::int32_t>(tensor.sizes().begin(), tensor.sizes().end());

        size_ = tensor.nbytes();
        data_ = std::make_unique<std::uint8_t[]>(size_);
        tensor_ = executorch::extension::from_blob(data_.get(), shape_, tensor.dtype());

        std::memcpy(data_.get(), tensor.const_data_ptr(), size_);
    }

    jsi::Value TensorHostObject::get(jsi::Runtime &rt, const jsi::PropNameID &name)
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
            return jsi::String::createFromUtf8(rt, mylib::core::types::toString(dtype_));
        }

        return jsi::Value::undefined();
    }

    std::vector<facebook::jsi::PropNameID> TensorHostObject::getPropertyNames(jsi::Runtime &rt)
    {
        std::vector<facebook::jsi::PropNameID> properties;
        properties.push_back(jsi::PropNameID::forAscii(rt, "shape"));
        properties.push_back(jsi::PropNameID::forAscii(rt, "dtype"));
        return properties;
    }

    void install_createTensor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "createTensor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "createTensor: Usage: createTensor(shape, dtype)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isArray(rt))
            {
                throw jsi::JSError(rt, "createTensor: Expected shape as an array of integers");
            }

            if (!args[1].isString())
            {
                throw jsi::JSError(rt, "createTensor: Expected dtype as a string");
            }

            auto shapeArray = args[0].asObject(rt).asArray(rt);
            std::vector<std::int32_t> shape;
            for (size_t i = 0; i < shapeArray.length(rt); ++i)
            {
                auto dimValue = shapeArray.getValueAtIndex(rt, i);
                if (!dimValue.isNumber())
                {
                    throw jsi::JSError(rt, "createTensor: Shape array must contain only numbers");
                }

                if (dimValue.asNumber() <= 0)
                {
                    throw jsi::JSError(rt, "createTensor: Shape dimensions must be positive integers");
                }

                shape.push_back(static_cast<std::int32_t>(dimValue.asNumber()));
            }

            try
            {
                const auto dtype = mylib::core::types::parseDType(args[1].asString(rt).utf8(rt));
                auto tensorHostObject = std::make_shared<TensorHostObject>(shape, dtype);
                return jsi::Object::createFromHostObject(rt, tensorHostObject);
            }
            catch (const std::exception &e)
            {
                throw jsi::JSError(rt, "createTensor: Error creating tensor: " + std::string(e.what()));
            }
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_disposeTensor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "disposeTensor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
            {
                throw jsi::JSError(rt, "disposeTensor: Usage: disposeTensor(tensor)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "disposeTensor: Expected a TensorHostObject");
            }

            auto tensorHostObject = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);

            std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "disposeTensor: Tensor is currently in use and cannot be disposed");
            }

            if (!tensorHostObject->data_)
            {
                throw jsi::JSError(rt, "disposeTensor: Tensor has already been disposed");
            }

            tensorHostObject->tensor_.reset();
            tensorHostObject->data_.reset();

            return jsi::Value::undefined();
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_setTensorData(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "setTensorData";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "setTensorData: Usage: setTensorData(tensor, array)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "setTensorData: Expected a TensorHostObject");
            }

            if (!args[1].isObject())
            {
                throw jsi::JSError(rt, "setTensorData: Expected array to be an object (TypedArray)");
            }

            jsi::Object dataObj = args[1].asObject(rt);
            if (!dataObj.hasProperty(rt, "buffer"))
            {
                throw jsi::JSError(rt, "setTensorData: Expected a TypedArray with a 'buffer' property");
            }

            jsi::ArrayBuffer buffer = dataObj.getProperty(rt, "buffer").asObject(rt).getArrayBuffer(rt);
            size_t byteOffset = 0;
            size_t byteLength = buffer.size(rt);

            if (dataObj.hasProperty(rt, "byteOffset"))
            {
                auto byteOffsetValue = dataObj.getProperty(rt, "byteOffset");
                if (!byteOffsetValue.isNumber())
                {
                    throw jsi::JSError(rt, "setTensorData: Expected 'byteOffset' to be a number");
                }
                byteOffset = static_cast<size_t>(byteOffsetValue.asNumber());
            }

            if (dataObj.hasProperty(rt, "byteLength"))
            {
                auto byteLengthValue = dataObj.getProperty(rt, "byteLength");
                if (!byteLengthValue.isNumber())
                {
                    throw jsi::JSError(rt, "setTensorData: Expected 'byteLength' to be a number");
                }
                byteLength = static_cast<size_t>(byteLengthValue.asNumber());
            }

            auto tensorHostObject = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);

            std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "setTensorData: Tensor is currently in use and cannot be written to");
            }

            if (!tensorHostObject->data_)
            {
                throw jsi::JSError(rt, "setTensorData: Tensor has been disposed");
            }

            if (byteLength != tensorHostObject->size_)
            {
                std::string errorMsg = "setTensorData: Data size mismatch: TypedArray is " + std::to_string(byteLength) +
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

    void install_getTensorData(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "getTensorData";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "getTensorData: Usage: getTensorData(array, tensor)");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "getTensorData: Expected a TensorHostObject");
            }

            if (!args[0].isObject())
            {
                throw jsi::JSError(rt, "getTensorData: Expected array to be an object (TypedArray)");
            }

            jsi::Object dataObj = args[0].asObject(rt);
            if (!dataObj.hasProperty(rt, "buffer"))
            {
                throw jsi::JSError(rt, "getTensorData: Expected a TypedArray with a 'buffer' property");
            }

            jsi::ArrayBuffer buffer = dataObj.getProperty(rt, "buffer").asObject(rt).getArrayBuffer(rt);
            size_t byteOffset = 0;
            size_t byteLength = buffer.size(rt);

            if (dataObj.hasProperty(rt, "byteOffset"))
            {
                auto byteOffsetValue = dataObj.getProperty(rt, "byteOffset");
                if (!byteOffsetValue.isNumber())
                {
                    throw jsi::JSError(rt, "getTensorData: Expected 'byteOffset' to be a number");
                }
                byteOffset = static_cast<size_t>(byteOffsetValue.asNumber());
            }

            if (dataObj.hasProperty(rt, "byteLength"))
            {
                auto byteLengthValue = dataObj.getProperty(rt, "byteLength");
                if (!byteLengthValue.isNumber())
                {
                    throw jsi::JSError(rt, "getTensorData: Expected 'byteLength' to be a number");
                }
                byteLength = static_cast<size_t>(byteLengthValue.asNumber());
            }

            auto tensorHostObject = args[1].asObject(rt).getHostObject<TensorHostObject>(rt);

            std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "getTensorData: Tensor is currently in use and cannot be read");
            }

            if (!tensorHostObject->data_)
            {
                throw jsi::JSError(rt, "getTensorData: Tensor has been disposed");
            }

            if (byteLength != tensorHostObject->size_)
            {
                std::string errorMsg = "getTensorData: Data size mismatch: TypedArray is " + std::to_string(byteLength) +
                                       " bytes, but Tensor requires " + std::to_string(tensorHostObject->size_) +
                                       " bytes.";
                throw jsi::JSError(rt, errorMsg);
            }

            std::memcpy(buffer.data(rt) + byteOffset, tensorHostObject->data_.get(), byteLength);

            return jsi::Value::undefined();
        };
        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody));
    }

    void install_reshapeTensor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "reshapeTensor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "reshapeTensor: Usage: reshapeTensor(tensor, shape)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "reshapeTensor: expected a TensorHostObject");
            }

            if (!args[1].isObject() || !args[1].asObject(rt).isArray(rt))
            {
                throw jsi::JSError(rt, "reshapeTensor: shape must be an array of numbers");
            }

            auto tensorHostObject = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            auto shapeArray = args[1].asObject(rt).asArray(rt);

            std::vector<std::int32_t> newShape;
            size_t newNumElements = 1;
            for (size_t i = 0; i < shapeArray.length(rt); ++i)
            {
                auto dimValue = shapeArray.getValueAtIndex(rt, i);
                if (!dimValue.isNumber())
                {
                    throw jsi::JSError(rt, "reshapeTensor: Shape array must contain only numbers");
                }
                auto dim = static_cast<std::int32_t>(dimValue.asNumber());
                if (dim <= 0)
                {
                    throw jsi::JSError(rt, "reshapeTensor: Shape dimensions must be positive integers");
                }
                newShape.push_back(dim);
                newNumElements *= dim;
            }

            std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "reshapeTensor: Tensor is currently in use and cannot be reshaped");
            }

            if (!tensorHostObject->data_)
            {
                throw jsi::JSError(rt, "reshapeTensor: Tensor has been disposed");
            }

            size_t currentNumElements = std::accumulate(
                tensorHostObject->shape_.begin(),
                tensorHostObject->shape_.end(),
                size_t(1),
                std::multiplies<size_t>());

            if (newNumElements != currentNumElements)
            {
                throw jsi::JSError(rt, "reshapeTensor: Cannot reshape tensor: total number of elements must remain the same");
            }

            tensorHostObject->shape_ = newShape;
            tensorHostObject->tensor_ = executorch::extension::from_blob(
                tensorHostObject->data_.get(),
                tensorHostObject->shape_,
                mylib::core::types::toScalarType(tensorHostObject->dtype_));

            return jsi::Value::undefined();
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody));
    }

    void install_isTensor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "isTensor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
            {
                throw jsi::JSError(rt, "isTensor: Usage: isTensor(value)");
            }
            bool isTensor = args[0].isObject() && args[0].asObject(rt).isHostObject<TensorHostObject>(rt);
            return jsi::Value(isTensor);
        };
        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody));
    }
} // namespace mylib::core::tensor
