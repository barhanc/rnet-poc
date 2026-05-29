#include "MyLib.h"

#include <algorithm>
#include <numeric>
#include <shared_mutex>

#include <jsi/jsi.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/tag.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

using namespace facebook;

struct DTypeInfo
{
    const char *name;
    size_t elementSize;
    executorch::aten::ScalarType scalarType;
    const char *jsTypedArray;
};

static constexpr DTypeInfo kDTypes[] = {
    {"float32", 4, executorch::aten::ScalarType::Float, "Float32Array"},
    {"uint8", 1, executorch::aten::ScalarType::Byte, "Uint8Array"},
    {"int32", 4, executorch::aten::ScalarType::Int, "Int32Array"},
};

struct ModelHostObject : public jsi::HostObject
{
    std::string modelPath_;
    std::unique_ptr<executorch::extension::Module> etModule_;
    std::mutex mutex_;

    ModelHostObject(const std::string &modelPath)
        : etModule_(std::make_unique<executorch::extension::Module>(modelPath)),
          modelPath_(modelPath)
    {
    }

    jsi::Value get(jsi::Runtime &rt, const jsi::PropNameID &name) override
    {
        auto nameStr = name.utf8(rt);

        if (nameStr == "path")
        {
            return jsi::String::createFromUtf8(rt, modelPath_);
        }

        return jsi::Value::undefined();
    }

    std::vector<facebook::jsi::PropNameID> getPropertyNames(jsi::Runtime &rt) override
    {
        std::vector<facebook::jsi::PropNameID> properties;
        properties.push_back(jsi::PropNameID::forAscii(rt, "path"));
        return properties;
    }
};

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
        auto it = std::find_if(std::begin(kDTypes), std::end(kDTypes),
                               [&](const DTypeInfo &d)
                               { return d.name == dtype; });

        if (it == std::end(kDTypes))
        {
            throw std::runtime_error("Unsupported dtype: " + dtype);
        }

        dtype_ = dtype;
        shape_ = shape;

        auto numElements = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<std::int32_t>());

        size_ = numElements * it->elementSize;
        data_ = std::make_unique<std::uint8_t[]>(size_);
        tensor_ = executorch::extension::from_blob(data_.get(), shape_, it->scalarType);
    }

    TensorHostObject(const executorch::aten::Tensor &tensor)
    {
        auto it = std::find_if(std::begin(kDTypes), std::end(kDTypes),
                               [&](const DTypeInfo &d)
                               { return d.scalarType == tensor.dtype(); });

        if (it == std::end(kDTypes))
        {
            throw std::runtime_error("Unsupported tensor dtype");
        }

        dtype_ = it->name;
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

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime)
    {
        auto registeredBackends = executorch::runtime::get_num_registered_backends();
        if (registeredBackends == 0)
        {
            throw jsi::JSError(jsiRuntime,
                               "ExecuTorch runtime has zero registered backends. "
                               "A delegated model (using e.g. XnnpackBackend) cannot run "
                               "until backend registration symbols are linked in.");
        }

        jsi::Object myModule = jsi::Object(jsiRuntime);

        install_getExecuTorchRegisteredBackends(jsiRuntime, myModule);

        // Model management
        install_loadModel(jsiRuntime, myModule);
        install_disposeModel(jsiRuntime, myModule);
        install_executeModelMethod(jsiRuntime, myModule);
        install_getModelMethodMeta(jsiRuntime, myModule);
        install_getModelMethodNames(jsiRuntime, myModule);

        // Tensor management
        install_createTensor(jsiRuntime, myModule);
        install_disposeTensor(jsiRuntime, myModule);
        install_setTensorFromTypedArray(jsiRuntime, myModule);
        install_setTypedArrayFromTensor(jsiRuntime, myModule);

        jsiRuntime.global().setProperty(jsiRuntime, "__executorch_jsi__", std::move(myModule));
    }

    void install_getExecuTorchRegisteredBackends(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "getExecuTorchRegisteredBackends";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 0)
            {
                throw jsi::JSError(rt, "Usage: getExecuTorchRegisteredBackends()");
            }

            auto registeredCount = executorch::runtime::get_num_registered_backends();
            auto jsArray = jsi::Array(rt, registeredCount);
            for (size_t i = 0; i < registeredCount; ++i)
            {
                auto backendName = executorch::runtime::get_backend_name(i);
                if (!backendName.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(backendName.error());
                    throw jsi::JSError(rt, "Failed to get backend name: " + errorMsg);
                }
                jsArray.setValueAtIndex(rt, i, jsi::String::createFromUtf8(rt, backendName.get()));
            }
            return jsArray;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 0, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_createTensor(jsi::Runtime &rt, jsi::Object &module)
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

    void install_disposeTensor(jsi::Runtime &rt, jsi::Object &module)
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

    void install_setTensorFromTypedArray(jsi::Runtime &rt, jsi::Object &module)
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

    void install_setTypedArrayFromTensor(jsi::Runtime &rt, jsi::Object &module)
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

    void install_loadModel(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "loadModel";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
            {
                throw jsi::JSError(rt, "Usage: loadModel(modelPath)");
            }

            if (!args[0].isString())
            {
                throw jsi::JSError(rt, "Expected model path as a string");
            }

            auto modelPath = args[0].asString(rt).utf8(rt);
            auto modelHostObject = std::make_shared<ModelHostObject>(modelPath);

            auto error = modelHostObject->etModule_->load();
            if (!modelHostObject->etModule_->is_loaded())
            {
                std::string errorMsg = executorch::runtime::to_string(error);
                throw jsi::JSError(rt, "Failed to load model: " + errorMsg);
            }

            return jsi::Object::createFromHostObject(rt, modelHostObject);
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_disposeModel(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "disposeModel";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
            {
                throw jsi::JSError(rt, "Usage: disposeModel(model)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<ModelHostObject>(rt))
            {
                throw jsi::JSError(rt, "Expected a ModelHostObject");
            }

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Model is currently in use");
            }

            if (!modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "Model has already been disposed");
            }

            modelHostObject->etModule_.reset();

            return jsi::Value::undefined();
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_getModelMethodNames(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "getModelMethodNames";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
            {
                throw jsi::JSError(rt, "Usage: getModelMethodNames(model)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<ModelHostObject>(rt))
            {
                throw jsi::JSError(rt, "Expected a ModelHostObject");
            }

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Model is currently in use");
            }

            if (!modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "Model has been disposed");
            }

            auto methodNames = modelHostObject->etModule_->method_names();
            if (!methodNames.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodNames.error());
                throw jsi::JSError(rt, "Failed to get method names: " + errorMsg);
            }

            auto jsArray = jsi::Array(rt, methodNames->size());
            size_t index = 0;
            for (const auto &methodName : methodNames.get())
            {
                jsArray.setValueAtIndex(rt, index, jsi::String::createFromUtf8(rt, methodName));
                ++index;
            }

            return jsArray;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_getModelMethodMeta(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "getModelMethodMeta";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "Usage: getModelMethodMeta(model, methodName)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<ModelHostObject>(rt))
            {
                throw jsi::JSError(rt, "Expected a ModelHostObject");
            }

            if (!args[1].isString())
            {
                throw jsi::JSError(rt, "Expected method name as a string");
            }

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Model is currently in use");
            }

            if (!modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "Model has been disposed");
            }

            auto methodName = args[1].asString(rt).utf8(rt);
            auto methodMeta = modelHostObject->etModule_->method_meta(methodName);
            if (!methodMeta.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodMeta.error());
                throw jsi::JSError(rt, "Failed to get method meta: " + errorMsg);
            }

            auto inputTagsArray = jsi::Array(rt, methodMeta->num_inputs());
            for (size_t i = 0; i < methodMeta->num_inputs(); ++i)
            {
                auto tag = methodMeta->input_tag(i);
                if (!tag.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(tag.error());
                    throw jsi::JSError(rt, "Failed to get input tag for input " + std::to_string(i) + ": " + errorMsg);
                }
                inputTagsArray.setValueAtIndex(rt, i, jsi::String::createFromUtf8(rt, executorch::runtime::tag_to_string(tag.get())));
            }

            auto outputTagsArray = jsi::Array(rt, methodMeta->num_outputs());
            for (size_t i = 0; i < methodMeta->num_outputs(); ++i)
            {
                auto tag = methodMeta->output_tag(i);
                if (!tag.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(tag.error());
                    throw jsi::JSError(rt, "Failed to get output tag for output " + std::to_string(i) + ": " + errorMsg);
                }
                outputTagsArray.setValueAtIndex(rt, i, jsi::String::createFromUtf8(rt, executorch::runtime::tag_to_string(tag.get())));
            }

            auto usesBackendMap = jsi::Object(rt);
            for (size_t i = 0; i < methodMeta->num_backends(); ++i)
            {
                auto backendName = methodMeta->get_backend_name(i);
                if (!backendName.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(backendName.error());
                    throw jsi::JSError(rt, "Failed to get backend name for backend " + std::to_string(i) + ": " + errorMsg);
                }
                usesBackendMap.setProperty(rt, backendName.get(), methodMeta->uses_backend(backendName.get()));
            }

            auto tensorMetaToJS = [](jsi::Runtime &rt, const executorch::runtime::TensorInfo &tensorMeta) -> jsi::Object
            {
                auto jsTensorMeta = jsi::Object(rt);
                jsTensorMeta.setProperty(rt, "name", jsi::String::createFromUtf8(rt, std::string(tensorMeta.name())));
                jsTensorMeta.setProperty(rt, "ndim", static_cast<double>(tensorMeta.sizes().size()));
                jsTensorMeta.setProperty(rt, "nbytes", static_cast<double>(tensorMeta.nbytes()));

                auto it = std::find_if(std::begin(kDTypes), std::end(kDTypes),
                                       [&](const DTypeInfo &d)
                                       { return d.scalarType == tensorMeta.scalar_type(); });
                if (it != std::end(kDTypes))
                {
                    jsTensorMeta.setProperty(rt, "dtype", jsi::String::createFromUtf8(rt, it->name));
                }
                else
                {
                    jsTensorMeta.setProperty(rt, "dtype", jsi::String::createFromUtf8(rt, "not supported"));
                }

                auto jsShapeArray = jsi::Array(rt, tensorMeta.sizes().size());
                for (size_t i = 0; i < tensorMeta.sizes().size(); ++i)
                {
                    jsShapeArray.setValueAtIndex(rt, i, static_cast<double>(tensorMeta.sizes()[i]));
                }
                jsTensorMeta.setProperty(rt, "shape", jsShapeArray);

                return jsTensorMeta;
            };

            auto inputTensorMetaArray = jsi::Array(rt, methodMeta->num_inputs());
            for (size_t i = 0; i < methodMeta->num_inputs(); ++i)
            {
                auto tensorMeta = methodMeta->input_tensor_meta(i);
                if (!tensorMeta.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(tensorMeta.error());
                    throw jsi::JSError(rt, "Failed to get tensor meta for input " + std::to_string(i) + ": " + errorMsg);
                }
                inputTensorMetaArray.setValueAtIndex(rt, i, tensorMetaToJS(rt, tensorMeta.get()));
            }

            auto outputTensorMetaArray = jsi::Array(rt, methodMeta->num_outputs());
            for (size_t i = 0; i < methodMeta->num_outputs(); ++i)
            {
                auto tensorMeta = methodMeta->output_tensor_meta(i);
                if (!tensorMeta.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(tensorMeta.error());
                    throw jsi::JSError(rt, "Failed to get tensor meta for output " + std::to_string(i) + ": " + errorMsg);
                }
                outputTensorMetaArray.setValueAtIndex(rt, i, tensorMetaToJS(rt, tensorMeta.get()));
            }

            auto jsMeta = jsi::Object(rt);

            jsMeta.setProperty(rt, "name", jsi::String::createFromUtf8(rt, methodMeta->name()));
            jsMeta.setProperty(rt, "numInputs", static_cast<double>(methodMeta->num_inputs()));
            jsMeta.setProperty(rt, "numOutputs", static_cast<double>(methodMeta->num_outputs()));
            jsMeta.setProperty(rt, "inputTags", inputTagsArray);
            jsMeta.setProperty(rt, "outputTags", outputTagsArray);
            jsMeta.setProperty(rt, "usesBackend", usesBackendMap);
            jsMeta.setProperty(rt, "inputTensorMeta", inputTensorMetaArray);
            jsMeta.setProperty(rt, "outputTensorMeta", outputTensorMetaArray);

            return jsMeta;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_executeModelMethod(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "executeModelMethod";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count < 2)
            {
                throw jsi::JSError(rt, "Usage: executeModelMethod(model, methodName, ...args)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<ModelHostObject>(rt))
            {
                throw jsi::JSError(rt, "Expected a ModelHostObject");
            }

            if (!args[1].isString())
            {
                throw jsi::JSError(rt, "Expected methodName as a string");
            }

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Model is currently in use");
            }

            if (!modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "Model has been disposed");
            }

            auto methodName = args[1].asString(rt).utf8(rt);
            auto methodMeta = modelHostObject->etModule_->method_meta(methodName);

            if (!methodMeta.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodMeta.error());
                throw jsi::JSError(rt, "Failed to get method meta for '" + methodName + "': " + errorMsg);
            }

            if (count != methodMeta->num_inputs() + 2)
            {
                std::string errorMsg = "Incorrect number of arguments: '" + std::to_string(count - 2) +
                                       "' for method '" + methodName +
                                       "', expected " + std::to_string(methodMeta->num_inputs());
                throw jsi::JSError(rt, errorMsg);
            }

            auto inputs = std::vector<executorch::runtime::EValue>(methodMeta->num_inputs());
            std::vector<std::unique_lock<std::shared_mutex>> tensorLocks;

            for (size_t i = 2; i < count; ++i)
            {
                auto tag = methodMeta->input_tag(i - 2);
                if (!tag.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(tag.error());
                    throw jsi::JSError(rt, "Failed to get input tag for argument " + std::to_string(i - 2) + ": " + errorMsg);
                }

                switch (tag.get())
                {
                case executorch::runtime::Tag::None:
                {
                    if (!args[i].isNull())
                    {
                        throw jsi::JSError(rt, "Expected argument " + std::to_string(i - 2) + " to be null");
                    }
                    inputs[i - 2] = executorch::runtime::EValue();
                    break;
                }
                case executorch::runtime::Tag::Tensor:
                {
                    if (!args[i].isObject() || !args[i].asObject(rt).isHostObject<TensorHostObject>(rt))
                    {
                        throw jsi::JSError(rt, "Expected argument " + std::to_string(i - 2) + " to be a TensorHostObject");
                    }

                    auto tensorHostObject = args[i].asObject(rt).getHostObject<TensorHostObject>(rt);

                    tensorLocks.emplace_back(tensorHostObject->mutex_, std::try_to_lock);
                    if (!tensorLocks.back().owns_lock())
                    {
                        throw jsi::JSError(rt, "Tensor argument " + std::to_string(i - 2) + " is currently in use and cannot be read");
                    }

                    auto tensorMeta = methodMeta->input_tensor_meta(i - 2);

                    if (!tensorMeta.ok())
                    {
                        std::string errorMsg = executorch::runtime::to_string(tensorMeta.error());
                        throw jsi::JSError(rt, "Failed to get tensor meta for argument " + std::to_string(i - 2) + ": " + errorMsg);
                    }

                    if (tensorMeta->scalar_type() != tensorHostObject->tensor_->dtype())
                    {
                        throw jsi::JSError(rt, "Tensor dtype mismatch for argument " + std::to_string(i - 2));
                    }

                    if (tensorMeta->sizes().size() != tensorHostObject->shape_.size())
                    {
                        throw jsi::JSError(rt, "Tensor rank mismatch for argument " + std::to_string(i - 2) +
                                                   ": expected rank " + std::to_string(tensorMeta->sizes().size()) +
                                                   " but got " + std::to_string(tensorHostObject->shape_.size()));
                    }

                    auto ndim = tensorHostObject->tensor_->sizes().size();
                    for (size_t j = 0; j < ndim; ++j)
                    {
                        if (tensorMeta->sizes()[j] != tensorHostObject->shape_[j])
                        {
                            throw jsi::JSError(rt, "Tensor shape mismatch for argument " + std::to_string(i - 2) +
                                                       ": expected dimension " + std::to_string(j) + " to be " +
                                                       std::to_string(tensorMeta->sizes()[j]) + " but got " +
                                                       std::to_string(tensorHostObject->shape_[j]));
                        }
                    }

                    inputs[i - 2] = tensorHostObject->tensor_;
                    break;
                }
                case executorch::runtime::Tag::Double:
                {
                    if (!args[i].isNumber())
                    {
                        throw jsi::JSError(rt, "Expected argument " + std::to_string(i - 2) + " to be a number");
                    }
                    inputs[i - 2] = executorch::runtime::EValue(args[i].asNumber());
                    break;
                }
                case executorch::runtime::Tag::Int:
                {
                    if (!args[i].isNumber())
                    {
                        throw jsi::JSError(rt, "Expected argument " + std::to_string(i - 2) + " to be a number");
                    }
                    inputs[i - 2] = executorch::runtime::EValue(static_cast<int64_t>(args[i].asNumber()));
                    break;
                }
                case executorch::runtime::Tag::Bool:
                {
                    if (!args[i].isBool())
                    {
                        throw jsi::JSError(rt, "Expected argument " + std::to_string(i - 2) + " to be a boolean");
                    }
                    inputs[i - 2] = executorch::runtime::EValue(args[i].asBool());
                    break;
                }
                default:
                {
                    throw jsi::JSError(rt, "Unsupported input type for argument " + std::to_string(i - 2));
                    break;
                }
                }
            }

            auto result = modelHostObject->etModule_->execute(methodName, inputs);

            if (!result.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(result.error());
                throw jsi::JSError(rt, "Method '" + methodName + "' execution failed: " + errorMsg +
                                           ". This may be due to missing required backends - use getModelMethodMeta()" +
                                           " to check required backends and getExecuTorchRegisteredBackends()" +
                                           " to check which backends are registered in the runtime.");
            }

            auto jsOutputArray = jsi::Array(rt, result->size());
            size_t index = 0;
            for (const auto &output : result.get())
            {
                switch (output.tag)
                {
                case executorch::runtime::Tag::None:
                {
                    jsOutputArray.setValueAtIndex(rt, index, jsi::Value::null());
                    break;
                }
                case executorch::runtime::Tag::String:
                {
                    jsOutputArray.setValueAtIndex(rt, index, jsi::String::createFromUtf8(rt, std::string(output.toString())));
                    break;
                }
                case executorch::runtime::Tag::Tensor:
                {
                    auto tensorHostObject = std::make_shared<TensorHostObject>(output.toTensor());
                    jsOutputArray.setValueAtIndex(rt, index, jsi::Object::createFromHostObject(rt, tensorHostObject));
                    break;
                }
                case executorch::runtime::Tag::Double:
                {
                    jsOutputArray.setValueAtIndex(rt, index, output.toDouble());
                    break;
                }
                case executorch::runtime::Tag::Int:
                {
                    jsOutputArray.setValueAtIndex(rt, index, static_cast<double>(output.toInt()));
                    break;
                }
                case executorch::runtime::Tag::Bool:
                {
                    jsOutputArray.setValueAtIndex(rt, index, output.toBool());
                    break;
                }
                case executorch::runtime::Tag::ListBool:
                {
                    auto boolList = jsi::Array(rt, output.toBoolList().size());
                    for (size_t i = 0; i < output.toBoolList().size(); ++i)
                    {
                        boolList.setValueAtIndex(rt, i, output.toBoolList()[i]);
                    }
                    jsOutputArray.setValueAtIndex(rt, index, boolList);
                    break;
                }
                case executorch::runtime::Tag::ListDouble:
                {
                    auto doubleList = jsi::Array(rt, output.toDoubleList().size());
                    for (size_t i = 0; i < output.toDoubleList().size(); ++i)
                    {
                        doubleList.setValueAtIndex(rt, i, output.toDoubleList()[i]);
                    }
                    jsOutputArray.setValueAtIndex(rt, index, doubleList);
                    break;
                }
                default:
                {
                    throw jsi::JSError(rt, "Unsupported return type");
                }
                }

                ++index;
            }

            return jsOutputArray;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }

} // namespace mylib