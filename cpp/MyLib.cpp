#include "MyLib.h"

#include <map>
#include <numeric>
#include <shared_mutex>

#include <jsi/jsi.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/tag.h>

using namespace facebook;

struct ModelHostObject : public jsi::HostObject
{
    std::unique_ptr<executorch::extension::Module> etModule_;
    std::mutex mutex_;

    ModelHostObject(const std::string &modelPath)
        : etModule_(std::make_unique<executorch::extension::Module>(modelPath))
    {
    }
};

struct TensorHostObject : public jsi::HostObject
{
    std::string dtype_;
    std::vector<std::uint8_t> data_;
    std::vector<std::int32_t> shape_;
    std::optional<executorch::extension::TensorPtr> tensor_;

    mutable std::shared_mutex mutex_;

    TensorHostObject(const std::vector<std::int32_t> &shape, const std::string &dtype)
    {
        shape_ = shape;
        dtype_ = dtype;

        std::map<std::string, size_t> dtypeSizeMap = {
            {"float32", 4},
            {"uint8", 1},
            {"int32", 4}};

        if (dtypeSizeMap.find(dtype_) == dtypeSizeMap.end())
        {
            throw std::runtime_error("Unsupported dtype: " + dtype_);
        }

        auto elementSize = dtypeSizeMap[dtype_];
        auto numElements = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<std::int32_t>());

        std::map<std::string, executorch::aten::ScalarType> dtypeScalarTypeMap = {
            {"float32", executorch::aten::ScalarType::Float},
            {"uint8", executorch::aten::ScalarType::Byte},
            {"int32", executorch::aten::ScalarType::Int}};

        data_.resize(numElements * elementSize);
        tensor_ = executorch::extension::from_blob(data_.data(), shape_, dtypeScalarTypeMap[dtype_]);
    }

    TensorHostObject(const executorch::aten::Tensor &tensor)
    {
        std::map<executorch::aten::ScalarType, std::string> scalarTypeDtypeMap = {
            {executorch::aten::ScalarType::Float, "float32"},
            {executorch::aten::ScalarType::Byte, "uint8"},
            {executorch::aten::ScalarType::Int, "int32"}};

        if (scalarTypeDtypeMap.find(tensor.dtype()) == scalarTypeDtypeMap.end())
        {
            throw std::runtime_error("Unsupported tensor dtype");
        }

        shape_ = std::vector<std::int32_t>(tensor.sizes().begin(), tensor.sizes().end());
        dtype_ = scalarTypeDtypeMap[tensor.dtype()];

        data_.resize(tensor.nbytes());
        tensor_ = executorch::extension::from_blob(data_.data(), shape_, tensor.dtype());

        std::memcpy(data_.data(), tensor.const_data_ptr(), tensor.nbytes());
    }
};

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime)
    {
        auto registeredBackends = executorch::runtime::get_num_registered_backends();
        if (registeredBackends == 0)
        {
            throw std::runtime_error(
                "ExecuTorch runtime has zero registered backends. "
                "A delegated model (for example XnnpackBackend) cannot run "
                "until backend registration symbols are linked in.");
        }

        // Log registered backends to console
        fprintf(stderr, "\n✓ ExecuTorch: %zu backends registered\n", registeredBackends);
        for (size_t i = 0; i < registeredBackends; ++i)
        {
            auto backendName = executorch::runtime::get_backend_name(i);
            fprintf(stderr, "  [%zu] %s\n", i, backendName.ok() ? backendName.get() : "<error>");
        }
        fprintf(stderr, "\n");

        jsi::Object myModule = jsi::Object(jsiRuntime);

        install_loadModel(jsiRuntime, myModule);
        install_disposeModel(jsiRuntime, myModule);
        install_executeModelMethod(jsiRuntime, myModule);
        install_getModelMethodMeta(jsiRuntime, myModule);
        install_getModelMethodNames(jsiRuntime, myModule);

        install_createTensor(jsiRuntime, myModule);
        install_setTensorFromTypedArray(jsiRuntime, myModule);
        install_getTypedArrayFromTensor(jsiRuntime, myModule);

        jsiRuntime.global().setProperty(jsiRuntime, "__myModule__", std::move(myModule));
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
                throw jsi::JSError(rt, e.what());
            }
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_setTensorFromTypedArray(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "setTensorFromTypedArray";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "Usage: setTensorFromTypedArray(tensor, data)");
            }

            auto tensorHostObject = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            if (!tensorHostObject || !tensorHostObject->tensor_)
            {
                throw jsi::JSError(rt, "Invalid TensorHostObject");
            }

            if (!args[1].isObject())
            {
                throw jsi::JSError(rt, "Expected data to be an object (TypedArray)");
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

            std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Tensor is currently in use and cannot be written to");
            }

            if (byteLength != tensorHostObject->data_.size())
            {
                std::string errorMsg = "Data size mismatch: TypedArray is " + std::to_string(byteLength) +
                                       " bytes, but Tensor requires " + std::to_string(tensorHostObject->data_.size()) +
                                       " bytes.";
                throw jsi::JSError(rt, errorMsg);
            }

            std::memcpy(tensorHostObject->data_.data(), buffer.data(rt) + byteOffset, byteLength);

            return jsi::Value::undefined();
        };

        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_getTypedArrayFromTensor(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "getTypedArrayFromTensor";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
                throw jsi::JSError(rt, "Usage: getTypedArrayFromTensor(tensor)");

            auto tensorHostObject = args[0].asObject(rt).getHostObject<TensorHostObject>(rt);
            if (!tensorHostObject)
                throw jsi::JSError(rt, "Invalid TensorHostObject");

            std::shared_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Tensor is currently in use and cannot be read from");
            }

            size_t bytes = tensorHostObject->data_.size();

            auto arrayBufferCtor = rt.global().getPropertyAsFunction(rt, "ArrayBuffer");
            auto arrayBufferObj = arrayBufferCtor.callAsConstructor(rt, static_cast<double>(bytes)).asObject(rt);
            auto arrayBuffer = arrayBufferObj.getArrayBuffer(rt);

            std::map<std::string, std::string> dtypeToTypedArrayMap = {
                {"float32", "Float32Array"},
                {"uint8", "Uint8Array"},
                {"int32", "Int32Array"}};

            if (dtypeToTypedArrayMap.find(tensorHostObject->dtype_) == dtypeToTypedArrayMap.end())
            {
                throw jsi::JSError(rt, "Unsupported tensor dtype: " + tensorHostObject->dtype_);
            }

            std::memcpy(arrayBuffer.data(rt), tensorHostObject->data_.data(), bytes);

            auto jsConstructorName = dtypeToTypedArrayMap[tensorHostObject->dtype_];
            auto typedArrayCtor = rt.global().getPropertyAsFunction(rt, jsConstructorName.c_str());

            return typedArrayCtor.callAsConstructor(rt, arrayBufferObj);
        };
        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody));
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

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);
            if (!modelHostObject || !modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "Invalid ModelHostObject");
            }

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Model is currently in use");
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

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);
            if (!modelHostObject || !modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "Invalid ModelHostObject");
            }

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Model is currently in use");
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

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);
            if (!modelHostObject || !modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "Invalid ModelHostObject");
            }

            if (!args[1].isString())
            {
                throw jsi::JSError(rt, "Expected method name as a string");
            }

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Model is currently in use");
            }

            auto methodName = args[1].asString(rt).utf8(rt);
            auto methodMeta = modelHostObject->etModule_->method_meta(methodName);
            if (!methodMeta.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodMeta.error());
                throw jsi::JSError(rt, "Failed to get method meta: " + errorMsg);
            }

            auto jsMeta = jsi::Object(rt);
            
            // Populate metadata with actual information
            jsMeta.setProperty(rt, "num_inputs", static_cast<double>(methodMeta->num_inputs()));
            jsMeta.setProperty(rt, "num_outputs", static_cast<double>(methodMeta->num_outputs()));
            jsMeta.setProperty(rt, "num_backends", static_cast<double>(methodMeta->num_backends()));
            
            // List required backends
            auto backendNames = jsi::Array(rt, methodMeta->num_backends());
            for (size_t i = 0; i < methodMeta->num_backends(); ++i)
            {
                auto backendName = methodMeta->get_backend_name(i);
                backendNames.setValueAtIndex(rt, i, 
                    jsi::String::createFromUtf8(rt, backendName.ok() ? backendName.get() : "<error>")
                );
            }
            jsMeta.setProperty(rt, "backends", backendNames);
            
            // List registered backends (from runtime)
            auto registeredCount = executorch::runtime::get_num_registered_backends();
            auto registeredNames = jsi::Array(rt, registeredCount);
            for (size_t i = 0; i < registeredCount; ++i)
            {
                auto backendName = executorch::runtime::get_backend_name(i);
                registeredNames.setValueAtIndex(rt, i,
                    jsi::String::createFromUtf8(rt, backendName.ok() ? backendName.get() : "<error>")
                );
            }
            jsMeta.setProperty(rt, "registered_backends", registeredNames);

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

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);
            if (!modelHostObject || !modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "Invalid ModelHostObject");
            }

            if (!args[1].isString())
            {
                throw jsi::JSError(rt, "Expected methodName as a string");
            }

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "Model is currently in use");
            }

            auto methodName = args[1].asString(rt).utf8(rt);
            auto methodMeta = modelHostObject->etModule_->method_meta(methodName);

            if (!methodMeta.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodMeta.error());
                throw jsi::JSError(rt, "Failed to get method meta: " + errorMsg);
            }

            if (count != methodMeta->num_inputs() + 2)
            {
                std::string errorMsg = "Incorrect number of arguments: '" + std::to_string(count - 2) +
                                       "' for method '" + methodName +
                                       "', expected " + std::to_string(methodMeta->num_inputs());
                throw jsi::JSError(rt, errorMsg);
            }

            auto inputs = std::vector<executorch::runtime::EValue>(methodMeta->num_inputs());

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
                        throw jsi::JSError(rt, "Expected argument " + std::to_string(i - 2) + " to be an object (TensorHostObject)");
                    }

                    auto tensorHostObject = args[i].asObject(rt).getHostObject<TensorHostObject>(rt);
                    if (!tensorHostObject || !tensorHostObject->tensor_)
                    {
                        throw jsi::JSError(rt, "Expected argument " + std::to_string(i - 2) + " to be a TensorHostObject");
                    }

                    std::unique_lock<std::shared_mutex> lock(tensorHostObject->mutex_, std::try_to_lock);
                    if (!lock.owns_lock())
                    {
                        throw jsi::JSError(rt, "Tensor argument " + std::to_string(i - 2) + " is currently in use and cannot be read");
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

            auto error = modelHostObject->etModule_->load_method(methodName);

            if (!modelHostObject->etModule_->is_method_loaded(methodName))
            {
                std::string errorMsg = executorch::runtime::to_string(error);
                std::string requiredBackends = "[";
                for (size_t i = 0; i < methodMeta->num_backends(); ++i)
                {
                    if (i > 0)
                    {
                        requiredBackends += ", ";
                    }

                    auto backendName = methodMeta->get_backend_name(i);
                    requiredBackends += backendName.ok() ? backendName.get() : "<error>";
                }
                requiredBackends += "]";

                std::string registeredBackends = "[";
                auto registeredCount = executorch::runtime::get_num_registered_backends();
                for (size_t i = 0; i < registeredCount; ++i)
                {
                    if (i > 0)
                    {
                        registeredBackends += ", ";
                    }

                    auto backendName = executorch::runtime::get_backend_name(i);
                    registeredBackends += backendName.ok() ? backendName.get() : "<error>";
                }
                registeredBackends += "]";

                throw jsi::JSError(
                    rt,
                    "Failed to load method '" + methodName + "': " + errorMsg +
                        ". Required backends: " + requiredBackends +
                        ". Registered backends: " + registeredBackends);
            }

            auto result = modelHostObject->etModule_->execute(methodName, inputs);

            if (!result.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(result.error());
                
                // Provide diagnostic info on execution failure
                std::string requiredBackends = "[";
                for (size_t i = 0; i < methodMeta->num_backends(); ++i)
                {
                    if (i > 0) requiredBackends += ", ";
                    auto backendName = methodMeta->get_backend_name(i);
                    requiredBackends += backendName.ok() ? backendName.get() : "<error>";
                }
                requiredBackends += "]";

                std::string registeredBackends = "[";
                auto registeredCount = executorch::runtime::get_num_registered_backends();
                for (size_t i = 0; i < registeredCount; ++i)
                {
                    if (i > 0) registeredBackends += ", ";
                    auto backendName = executorch::runtime::get_backend_name(i);
                    registeredBackends += backendName.ok() ? backendName.get() : "<error>";
                }
                registeredBackends += "]";
                
                throw jsi::JSError(
                    rt,
                    "Method '" + methodName + "' execution failed: " + errorMsg +
                        ". Required backends: " + requiredBackends +
                        ". Registered backends: " + registeredBackends);
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