#include "Model.h"
#include "Tensor.h"
#include "types.h"
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/tag.h>

namespace mylib::core::model
{
    namespace jsi = facebook::jsi;
    using TensorHostObject = mylib::core::tensor::TensorHostObject;

    ModelHostObject::ModelHostObject(const std::string &modelPath)
        : modelPath_(modelPath),
          etModule_(std::make_unique<executorch::extension::Module>(modelPath))
    {
    }

    jsi::Value ModelHostObject::get(jsi::Runtime &rt, const jsi::PropNameID &name)
    {
        auto nameStr = name.utf8(rt);

        if (nameStr == "path")
        {
            return jsi::String::createFromUtf8(rt, modelPath_);
        }

        return jsi::Value::undefined();
    }

    std::vector<facebook::jsi::PropNameID> ModelHostObject::getPropertyNames(jsi::Runtime &rt)
    {
        std::vector<facebook::jsi::PropNameID> properties;
        properties.push_back(jsi::PropNameID::forAscii(rt, "path"));
        return properties;
    }

    void install_loadModel(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "loadModel";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
            {
                throw jsi::JSError(rt, "loadModel: Usage: loadModel(arg0)");
            }

            if (!args[0].isString())
            {
                throw jsi::JSError(rt, "loadModel: Expected arg0 to be a string");
            }

            auto modelPath = args[0].asString(rt).utf8(rt);
            auto modelHostObject = std::make_shared<ModelHostObject>(modelPath);

            auto error = modelHostObject->etModule_->load();
            if (!modelHostObject->etModule_->is_loaded())
            {
                std::string errorMsg = executorch::runtime::to_string(error);
                throw jsi::JSError(rt, "loadModel: Failed to load model: " + errorMsg);
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
                throw jsi::JSError(rt, "disposeModel: Usage: disposeModel(arg0)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<ModelHostObject>(rt))
            {
                throw jsi::JSError(rt, "disposeModel: Expected arg0 to be a ModelHostObject");
            }

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "disposeModel: Model is currently in use");
            }

            if (!modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "disposeModel: Model has already been disposed");
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
                throw jsi::JSError(rt, "getModelMethodNames: Usage: getModelMethodNames(arg0)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<ModelHostObject>(rt))
            {
                throw jsi::JSError(rt, "getModelMethodNames: Expected arg0 to be a ModelHostObject");
            }

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "getModelMethodNames: Model is currently in use");
            }

            if (!modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "getModelMethodNames: Model has been disposed");
            }

            auto methodNames = modelHostObject->etModule_->method_names();
            if (!methodNames.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodNames.error());
                throw jsi::JSError(rt, "getModelMethodNames: Failed to get method names: " + errorMsg);
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
                throw jsi::JSError(rt, "getModelMethodMeta: Usage: getModelMethodMeta(arg0, arg1)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<ModelHostObject>(rt))
            {
                throw jsi::JSError(rt, "getModelMethodMeta: Expected arg0 to be a ModelHostObject");
            }

            if (!args[1].isString())
            {
                throw jsi::JSError(rt, "getModelMethodMeta: Expected arg1 to be a string");
            }

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "getModelMethodMeta: Model is currently in use");
            }

            if (!modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "getModelMethodMeta: Model has been disposed");
            }

            auto methodName = args[1].asString(rt).utf8(rt);
            auto methodMeta = modelHostObject->etModule_->method_meta(methodName);
            if (!methodMeta.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodMeta.error());
                throw jsi::JSError(rt, "getModelMethodMeta: Failed to get method meta: " + errorMsg);
            }

            auto inputTagsArray = jsi::Array(rt, methodMeta->num_inputs());
            for (size_t i = 0; i < methodMeta->num_inputs(); ++i)
            {
                auto tag = methodMeta->input_tag(i);
                if (!tag.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(tag.error());
                    throw jsi::JSError(rt, "getModelMethodMeta: Failed to get input tag for input " + std::to_string(i) + ": " + errorMsg);
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
                    throw jsi::JSError(rt, "getModelMethodMeta: Failed to get output tag for output " + std::to_string(i) + ": " + errorMsg);
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
                    throw jsi::JSError(rt, "getModelMethodMeta: Failed to get backend name for backend " + std::to_string(i) + ": " + errorMsg);
                }
                usesBackendMap.setProperty(rt, backendName.get(), methodMeta->uses_backend(backendName.get()));
            }

            auto tensorMetaToJS = [](jsi::Runtime &rt, const executorch::runtime::TensorInfo &tensorMeta) -> jsi::Object
            {
                auto jsTensorMeta = jsi::Object(rt);
                jsTensorMeta.setProperty(rt, "name", jsi::String::createFromUtf8(rt, std::string(tensorMeta.name())));
                jsTensorMeta.setProperty(rt, "ndim", static_cast<double>(tensorMeta.sizes().size()));
                jsTensorMeta.setProperty(rt, "nbytes", static_cast<double>(tensorMeta.nbytes()));

                try
                {
                    std::string dtypeStr = mylib::core::types::toString(mylib::core::types::fromScalarType(tensorMeta.scalar_type()));
                    jsTensorMeta.setProperty(rt, "dtype", jsi::String::createFromUtf8(rt, dtypeStr));
                }
                catch (const std::exception &)
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
                    throw jsi::JSError(rt, "getModelMethodMeta: Failed to get tensor meta for input " + std::to_string(i) + ": " + errorMsg);
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
                    throw jsi::JSError(rt, "getModelMethodMeta: Failed to get tensor meta for output " + std::to_string(i) + ": " + errorMsg);
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
            if (count != 4)
            {
                throw jsi::JSError(rt, "executeModelMethod: Usage: executeModelMethod(arg0, arg1, arg2, arg3)");
            }

            if (!args[0].isObject() || !args[0].asObject(rt).isHostObject<ModelHostObject>(rt))
            {
                throw jsi::JSError(rt, "executeModelMethod: Expected arg0 to be a ModelHostObject");
            }

            if (!args[1].isString())
            {
                throw jsi::JSError(rt, "executeModelMethod: Expected arg1 to be a string");
            }

            if (!args[2].isObject() || !args[2].asObject(rt).isArray(rt))
            {
                throw jsi::JSError(rt, "executeModelMethod: Expected arg2 to be an array");
            }

            if (!args[3].isObject() || !args[3].asObject(rt).isArray(rt))
            {
                throw jsi::JSError(rt, "executeModelMethod: Expected arg3 to be an array");
            }

            auto modelHostObject = args[0].asObject(rt).getHostObject<ModelHostObject>(rt);

            std::unique_lock<std::mutex> lock(modelHostObject->mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                throw jsi::JSError(rt, "executeModelMethod: arg0 is currently in use");
            }

            if (!modelHostObject->etModule_)
            {
                throw jsi::JSError(rt, "executeModelMethod: arg0 has been disposed");
            }

            auto methodName = args[1].asString(rt).utf8(rt);
            auto methodMeta = modelHostObject->etModule_->method_meta(methodName);
            auto inputsArray = args[2].asObject(rt).asArray(rt);

            if (!methodMeta.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodMeta.error());
                throw jsi::JSError(rt, "executeModelMethod: Failed to get method meta for '" + methodName + "': " + errorMsg);
            }

            if (inputsArray.size(rt) != methodMeta->num_inputs())
            {
                std::string errorMsg = "executeModelMethod: Incorrect size for arg2: got " +
                                       std::to_string(inputsArray.size(rt)) +
                                       ", expected " + std::to_string(methodMeta->num_inputs());
                throw jsi::JSError(rt, errorMsg);
            }

            auto validateTensor = [](jsi::Runtime &rt,
                                     const TensorHostObject *tensorHostObject,
                                     const executorch::runtime::Result<executorch::runtime::TensorInfo> &tensorMeta,
                                     const std::string &identifier)
            {
                if (tensorMeta->scalar_type() != tensorHostObject->tensor_->dtype())
                {
                    throw jsi::JSError(rt, "executeModelMethod: Tensor dtype mismatch for " + identifier);
                }

                if (tensorMeta->sizes().size() != tensorHostObject->shape_.size())
                {
                    throw jsi::JSError(rt, "executeModelMethod: Tensor rank mismatch for " + identifier +
                                               ": expected rank " + std::to_string(tensorMeta->sizes().size()) +
                                               " but got " + std::to_string(tensorHostObject->shape_.size()));
                }

                auto ndim = tensorHostObject->tensor_->sizes().size();
                for (size_t j = 0; j < ndim; ++j)
                {
                    if (tensorMeta->sizes()[j] != tensorHostObject->shape_[j])
                    {
                        throw jsi::JSError(rt, "executeModelMethod: Tensor shape mismatch for " + identifier +
                                                   ": expected dimension " + std::to_string(j) + " to be " +
                                                   std::to_string(tensorMeta->sizes()[j]) + " but got " +
                                                   std::to_string(tensorHostObject->shape_[j]));
                    }
                }
            };

            auto inputs = std::vector<executorch::runtime::EValue>(methodMeta->num_inputs());
            std::vector<std::unique_lock<std::shared_mutex>> tensorLocks;

            for (size_t i = 0; i < methodMeta->num_inputs(); ++i)
            {
                auto tag = methodMeta->input_tag(i);
                auto val = inputsArray.getValueAtIndex(rt, i);

                if (!tag.ok())
                {
                    std::string errorMsg = executorch::runtime::to_string(tag.error());
                    throw jsi::JSError(rt, "executeModelMethod: Failed to get input tag for arg2[" +
                                               std::to_string(i) + "]: " + errorMsg);
                }

                switch (tag.get())
                {
                case executorch::runtime::Tag::None:
                {
                    if (!val.isNull())
                    {
                        throw jsi::JSError(rt, "executeModelMethod: Expected arg2[" +
                                                   std::to_string(i) + "] to be null");
                    }
                    inputs[i] = executorch::runtime::EValue();
                    break;
                }
                case executorch::runtime::Tag::Tensor:
                {
                    if (!val.isObject() || !val.asObject(rt).isHostObject<mylib::core::tensor::TensorHostObject>(rt))
                    {
                        throw jsi::JSError(rt, "executeModelMethod: Expected arg2[" +
                                                   std::to_string(i) + "] to be a TensorHostObject");
                    }

                    auto tensorHostObject = val.asObject(rt).getHostObject<mylib::core::tensor::TensorHostObject>(rt);
                    if (!tensorHostObject->data_)
                    {
                        throw jsi::JSError(rt, "executeModelMethod: arg2[" + std::to_string(i) + "] has been disposed");
                    }

                    tensorLocks.emplace_back(tensorHostObject->mutex_, std::try_to_lock);
                    if (!tensorLocks.back().owns_lock())
                    {
                        throw jsi::JSError(rt, "executeModelMethod: arg2[" + std::to_string(i) +
                                                   "] is currently in use");
                    }

                    auto tensorMeta = methodMeta->input_tensor_meta(i);

                    if (!tensorMeta.ok())
                    {
                        std::string errorMsg = executorch::runtime::to_string(tensorMeta.error());
                        throw jsi::JSError(rt, "executeModelMethod: Failed to get tensor meta for arg2[" +
                                                   std::to_string(i) + "]: " + errorMsg);
                    }

                    validateTensor(rt, tensorHostObject.get(), tensorMeta, "arg2[" + std::to_string(i) + "]");

                    inputs[i] = tensorHostObject->tensor_;
                    break;
                }
                case executorch::runtime::Tag::Double:
                {
                    if (!val.isNumber())
                    {
                        throw jsi::JSError(rt, "executeModelMethod: Expected arg2[" +
                                                   std::to_string(i) + "] to be a number");
                    }
                    inputs[i] = executorch::runtime::EValue(val.asNumber());
                    break;
                }
                case executorch::runtime::Tag::Int:
                {
                    if (!val.isNumber())
                    {
                        throw jsi::JSError(rt, "executeModelMethod: Expected arg2[" +
                                                   std::to_string(i) + "] to be a number");
                    }
                    inputs[i] = executorch::runtime::EValue(static_cast<int64_t>(val.asNumber()));
                    break;
                }
                case executorch::runtime::Tag::Bool:
                {
                    if (!val.isBool())
                    {
                        throw jsi::JSError(rt, "executeModelMethod: Expected arg2[" +
                                                   std::to_string(i) + "] to be a boolean");
                    }
                    inputs[i] = executorch::runtime::EValue(val.asBool());
                    break;
                }
                default:
                {
                    throw jsi::JSError(rt, "executeModelMethod: Unsupported input type for arg2[" + std::to_string(i) + "]");
                }
                }
            }

            auto result = modelHostObject->etModule_->execute(methodName, inputs);

            if (!result.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(result.error());
                throw jsi::JSError(rt, "executeModelMethod: Method '" + methodName + "' execution failed: " + errorMsg +
                                           ". This may be due to missing required backends - use getModelMethodMeta()" +
                                           " to check required backends and getExecuTorchRegisteredBackends()" +
                                           " to check which backends are registered in the runtime.");
            }

            auto tensorOutputsArray = args[3].asObject(rt).asArray(rt);
            auto jsOutputArray = jsi::Array(rt, result->size());

            size_t index = 0;
            size_t tensorOutputIdx = 0;

            for (const auto &output : result.get())
            {
                switch (output.tag)
                {
                case executorch::runtime::Tag::None:
                {
                    jsOutputArray.setValueAtIndex(rt, index, jsi::Value::null());
                    break;
                }
                case executorch::runtime::Tag::Tensor:
                {
                    if (tensorOutputIdx >= tensorOutputsArray.size(rt))
                    {
                        throw jsi::JSError(rt, "executeModelMethod: Not enough tensor output placeholders in arg3");
                    }

                    auto val = tensorOutputsArray.getValueAtIndex(rt, tensorOutputIdx);
                    if (!val.isObject() || !val.asObject(rt).isHostObject<TensorHostObject>(rt))
                    {
                        throw jsi::JSError(rt, "executeModelMethod: Expected arg3[" +
                                                   std::to_string(tensorOutputIdx) + "] to be a TensorHostObject");
                    }

                    auto tensorHostObject = val.asObject(rt).getHostObject<mylib::core::tensor::TensorHostObject>(rt);
                    if (!tensorHostObject->data_)
                    {
                        throw jsi::JSError(rt, "executeModelMethod: arg3[" + std::to_string(tensorOutputIdx) + "] has been disposed");
                    }

                    tensorLocks.emplace_back(tensorHostObject->mutex_, std::try_to_lock);
                    if (!tensorLocks.back().owns_lock())
                    {
                        throw jsi::JSError(rt, "executeModelMethod: arg3[" +
                                                   std::to_string(tensorOutputIdx) +
                                                   "] is currently in use");
                    }

                    auto tensorMeta = methodMeta->output_tensor_meta(index);

                    if (!tensorMeta.ok())
                    {
                        std::string errorMsg = executorch::runtime::to_string(tensorMeta.error());
                        throw jsi::JSError(rt, "executeModelMethod: Failed to get tensor meta for output at index " +
                                                   std::to_string(index) + ": " + errorMsg);
                    }

                    validateTensor(rt, tensorHostObject.get(), tensorMeta, "arg3[" + std::to_string(tensorOutputIdx) + "]");

                    std::memcpy(tensorHostObject->data_.get(),
                                output.toTensor().const_data_ptr(),
                                output.toTensor().nbytes());

                    jsOutputArray.setValueAtIndex(rt, index, jsi::Object::createFromHostObject(rt, tensorHostObject));
                    ++tensorOutputIdx;

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
                default:
                {
                    throw jsi::JSError(rt, "executeModelMethod: Unsupported return type");
                }
                }

                ++index;
            }

            return jsOutputArray;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 4, fnBody);

        module.setProperty(rt, name, fn);
    }
} // namespace mylib::core::model
