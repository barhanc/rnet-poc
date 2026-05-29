#include "MyLib.h"
#include <chrono>
#include <thread>
#include <jsi/jsi.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>

using namespace facebook;

struct ModelHostObject : public jsi::HostObject
{
    std::unique_ptr<executorch::extension::Module> etModule_;
    std::mutex mutex_;

    ModelHostObject(const std::string &modelPath)
        : mutex_(),
          etModule_(std::make_unique<executorch::extension::Module>(modelPath))
    {
    }
};

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime)
    {
        jsi::Object myModule = jsi::Object(jsiRuntime);

        install_checkExecuTorch(jsiRuntime, myModule);
        install_loadModel(jsiRuntime, myModule);
        install_disposeModel(jsiRuntime, myModule);
        install_executeModel(jsiRuntime, myModule);
        install_getModelMethodMeta(jsiRuntime, myModule);
        install_getModelMethodNames(jsiRuntime, myModule);

        jsiRuntime.global().setProperty(jsiRuntime, "__myModule__", std::move(myModule));
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

            auto methodName = args[1].asString(rt).utf8(rt);
            auto methodMeta = modelHostObject->etModule_->method_meta(methodName);
            if (!methodMeta.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodMeta.error());
                throw jsi::JSError(rt, "Failed to get method meta: " + errorMsg);
            }

            auto jsMeta = jsi::Object(rt);
            // TODO: Populate jsMeta with relevant metadata fields from methodMeta

            return jsMeta;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_executeModel(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "executeModel";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count < 2)
            {
                throw jsi::JSError(rt, "Usage: executeModel(model, methodName, ...args)");
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

            auto methodName = args[1].asString(rt).utf8(rt);
            auto methodMeta = modelHostObject->etModule_->method_meta(methodName);

            if (!methodMeta.ok())
            {
                std::string errorMsg = executorch::runtime::to_string(methodMeta.error());
                throw jsi::JSError(rt, "Failed to get method meta: " + errorMsg);
            }

            if (count != methodMeta->num_inputs() + 2)
            {
                std::string errorMsg = "Incorrect number of arguments: " + std::to_string(count - 2) +
                                       " for method " + methodName +
                                       ", expected " + std::to_string(methodMeta->num_inputs());
                throw jsi::JSError(rt, errorMsg);
            }

            for (size_t i = 2; i < count; ++i)
            {
            }

            return jsi::Value::undefined();
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }

    void install_checkExecuTorch(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "checkExecuTorch";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 0)
                throw jsi::JSError(rt, "Incorrect number of arguments");

            // 1. Define shape and data
            int32_t sizes[] = {2}; // A 1D tensor with 2 elements
            float data[] = {42.0f, 7.0f};

            // 2. Create an ExecuTorch Tensor Implementation
            // This wraps our raw 'data' array without copying it
            exec_aten::TensorImpl impl(
                exec_aten::ScalarType::Float,
                1, // Number of dimensions
                sizes,
                data,
                nullptr // No special allocator needed for this test
            );

            // 3. Create the actual Tensor object
            exec_aten::Tensor tensor(&impl);

            // 4. Read data back to verify
            float val = tensor.const_data_ptr<float>()[0];

            // 5. Return success message to JavaScript
            std::string result = "ExecuTorch is Live! Tensor[0] = " + std::to_string(val);
            return jsi::String::createFromUtf8(rt, result);
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 0, fnBody);

        module.setProperty(rt, name, fn);
    }
} // namespace mylib