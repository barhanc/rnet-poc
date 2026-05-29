#include "Model.h"
#include "Tensor.h"
#include "types.h"
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/tag.h>

namespace mylib::core::model
{
    namespace jsi = facebook::jsi;

    ModelHostObject::ModelHostObject(const std::string &modelPath)
        : etModule_(std::make_unique<executorch::extension::Module>(modelPath)),
          modelPath_(modelPath)
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

                try
                {
                    std::string dtypeStr = mylib::core::types::scalarTypeToString(tensorMeta.scalar_type());
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
                    if (!args[i].isObject() || !args[i].asObject(rt).isHostObject<mylib::core::tensor::TensorHostObject>(rt))
                    {
                        throw jsi::JSError(rt, "Expected argument " + std::to_string(i - 2) + " to be a TensorHostObject");
                    }

                    auto tensorHostObject = args[i].asObject(rt).getHostObject<mylib::core::tensor::TensorHostObject>(rt);

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
                    auto tensorHostObject = std::make_shared<mylib::core::tensor::TensorHostObject>(output.toTensor());
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
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 0, fnBody);

        module.setProperty(rt, name, fn);
    }
} // namespace mylib::core::model
