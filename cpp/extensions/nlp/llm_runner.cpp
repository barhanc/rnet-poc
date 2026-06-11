#include "llm_runner.h"
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/stats.h>

namespace mylib::extensions::nlp::llm
{
    namespace jsi = facebook::jsi;

    static jsi::Object statsToJSI(jsi::Runtime &rt, const executorch::extension::llm::Stats &stats)
    {
        jsi::Object obj(rt);
        obj.setProperty(rt, "numPromptTokens", static_cast<double>(stats.num_prompt_tokens));
        obj.setProperty(rt, "numGeneratedTokens", static_cast<double>(stats.num_generated_tokens));
        obj.setProperty(rt, "firstTokenMs", static_cast<double>(stats.first_token_ms));
        obj.setProperty(rt, "inferenceStartMs", static_cast<double>(stats.inference_start_ms));
        obj.setProperty(rt, "inferenceEndMs", static_cast<double>(stats.inference_end_ms));
        obj.setProperty(rt, "modelLoadStartMs", static_cast<double>(stats.model_load_start_ms));
        obj.setProperty(rt, "modelLoadEndMs", static_cast<double>(stats.model_load_end_ms));
        return obj;
    }

    LLMRunnerHostObject::LLMRunnerHostObject(const std::string &modelPath,
                                             const std::string &tokenizerPath)
        : modelPath_(modelPath),
          tokenizerPath_(tokenizerPath)
    {
        auto tokenizer = executorch::extension::llm::load_tokenizer(tokenizerPath);
        if (!tokenizer)
        {
            throw std::runtime_error("LLMRunner: Failed to load runner tokenizer at path: " + tokenizerPath);
        }

        runner_ = executorch::extension::llm::create_text_llm_runner(modelPath, std::move(tokenizer));
        if (!runner_)
        {
            throw std::runtime_error("LLMRunner: Failed to create text llm runner");
        }

        auto loadError = runner_->load();
        if (loadError != executorch::runtime::Error::Ok)
        {
            std::string errorMsg = executorch::runtime::to_string(loadError);
            throw std::runtime_error("LLMRunner: Failed to load model: " + errorMsg);
        }
    }

    jsi::Value LLMRunnerHostObject::get(jsi::Runtime &rt, const jsi::PropNameID &name)
    {
        auto nameStr = name.utf8(rt);

        if (nameStr == "modelPath")
        {
            return jsi::String::createFromUtf8(rt, modelPath_);
        }

        if (nameStr == "tokenizerPath")
        {
            return jsi::String::createFromUtf8(rt, tokenizerPath_);
        }

        if (nameStr == "generate")
        {
            auto self = shared_from_this();
            auto fnBody = [self](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
            {
                if (count < 1)
                {
                    throw jsi::JSError(rt, "LLMRunner.generate: Usage: generate(prompt, config?, onToken?)");
                }

                if (!args[0].isString())
                {
                    throw jsi::JSError(rt, "LLMRunner.generate: Expected prompt to be a string");
                }

                std::string prompt = args[0].asString(rt).utf8(rt);

                executorch::extension::llm::GenerationConfig config;
                if (count > 1 && args[1].isObject())
                {
                    auto configObj = args[1].asObject(rt);
                    if (configObj.hasProperty(rt, "echo"))
                    {
                        config.echo = configObj.getProperty(rt, "echo").asBool();
                    }
                    if (configObj.hasProperty(rt, "ignoreEos"))
                    {
                        config.ignore_eos = configObj.getProperty(rt, "ignoreEos").asBool();
                    }
                    if (configObj.hasProperty(rt, "maxNewTokens"))
                    {
                        config.max_new_tokens = static_cast<int32_t>(configObj.getProperty(rt, "maxNewTokens").asNumber());
                    }
                    if (configObj.hasProperty(rt, "temperature"))
                    {
                        config.temperature = static_cast<float>(configObj.getProperty(rt, "temperature").asNumber());
                    }
                }

                std::function<void(const std::string &)> tokenCallback;
                if (count > 2 && args[2].isObject() && args[2].asObject(rt).isFunction(rt))
                {
                    auto tokenFn = std::make_shared<jsi::Function>(args[2].asObject(rt).asFunction(rt));
                    tokenCallback = [&rt, tokenFn](const std::string &token)
                    {
                        tokenFn->call(rt, jsi::String::createFromUtf8(rt, token));
                    };
                }

                auto finalStats = std::make_shared<executorch::extension::llm::Stats>();
                auto statsCallback = [finalStats](const executorch::extension::llm::Stats &stats)
                {
                    finalStats->num_prompt_tokens = stats.num_prompt_tokens;
                    finalStats->num_generated_tokens = stats.num_generated_tokens;
                    finalStats->first_token_ms = stats.first_token_ms;
                    finalStats->inference_start_ms = stats.inference_start_ms;
                    finalStats->inference_end_ms = stats.inference_end_ms;
                    finalStats->model_load_start_ms = stats.model_load_start_ms;
                    finalStats->model_load_end_ms = stats.model_load_end_ms;
                    finalStats->aggregate_sampling_time_ms = stats.aggregate_sampling_time_ms;
                };

                // Hold the lock for the whole call so dispose() cannot free the
                // runner mid-generation (dispose blocks on this lock until we
                // return). try_to_lock: only one prefill/generate may run at a
                // time, so fail fast instead of queuing. stop() is lock-free and
                // can still interrupt us.
                std::unique_lock<std::mutex> lock(self->mutex_, std::try_to_lock);
                if (!lock.owns_lock())
                {
                    throw jsi::JSError(rt, "LLMRunner.generate: Runner is already in use");
                }
                if (!self->runner_)
                {
                    throw jsi::JSError(rt, "LLMRunner.generate: Runner has been disposed");
                }
                auto error = self->runner_->generate(prompt, config, tokenCallback, statsCallback);

                if (error != executorch::runtime::Error::Ok)
                {
                    std::string errorMsg = executorch::runtime::to_string(error);
                    throw jsi::JSError(rt, "LLMRunner.generate: Failed to generate: " + errorMsg);
                }

                return statsToJSI(rt, *finalStats);
            };
            return jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, "generate"), 1, fnBody);
        }

        if (nameStr == "prefill")
        {
            auto self = shared_from_this();
            auto fnBody = [self](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
            {
                if (count < 1 || !args[0].isString())
                {
                    throw jsi::JSError(rt, "LLMRunner.prefill: Usage: prefill(prompt)");
                }

                std::string prompt = args[0].asString(rt).utf8(rt);

                // Lock held for the whole call, same as generate().
                std::unique_lock<std::mutex> lock(self->mutex_, std::try_to_lock);
                if (!lock.owns_lock())
                {
                    throw jsi::JSError(rt, "LLMRunner.prefill: Runner is already in use");
                }
                if (!self->runner_)
                {
                    throw jsi::JSError(rt, "LLMRunner.prefill: Runner has been disposed");
                }
                auto result = self->runner_->prefill({executorch::extension::llm::make_text_input(prompt)});
                if (result.error() != executorch::runtime::Error::Ok)
                {
                    std::string errorMsg = executorch::runtime::to_string(result.error());
                    throw jsi::JSError(rt, "LLMRunner.prefill: Failed: " + errorMsg);
                }

                return jsi::Value::undefined();
            };
            return jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, "prefill"), 1, fnBody);
        }

        if (nameStr == "stop")
        {
            auto self = shared_from_this();
            auto fnBody = [self](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
            {
                // Intentionally no mutex here: stop() is designed to be called
                // concurrently to interrupt an in-progress generate(). Taking the
                // lock would block until generate() finishes, defeating the point.
                // runner_ is only cleared by dispose() on this same (JS) thread,
                // so reading it lock-free here is safe.
                if (!self->runner_)
                {
                    throw jsi::JSError(rt, "LLMRunner.stop: Runner has been disposed");
                }
                self->runner_->stop();
                return jsi::Value::undefined();
            };
            return jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, "stop"), 0, fnBody);
        }

        if (nameStr == "dispose")
        {
            auto self = shared_from_this();
            auto fnBody = [self](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
            {
                if (count != 0)
                {
                    throw jsi::JSError(rt, "dispose: Usage: dispose()");
                }

                // Signal stop before locking so any in-progress generate() exits
                // quickly; we then block on the lock until it returns and clear
                // the runner, which frees the model. Idempotent: a second
                // dispose() finds a null runner_ and is a no-op.
                if (self->runner_)
                {
                    self->runner_->stop();
                }

                std::unique_lock<std::mutex> lock(self->mutex_);
                self->runner_.reset();

                return jsi::Value::undefined();
            };
            return jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, "dispose"), 0, fnBody);
        }

        return jsi::Value::undefined();
    }

    std::vector<jsi::PropNameID> LLMRunnerHostObject::getPropertyNames(jsi::Runtime &rt)
    {
        std::vector<jsi::PropNameID> properties;
        properties.push_back(jsi::PropNameID::forAscii(rt, "modelPath"));
        properties.push_back(jsi::PropNameID::forAscii(rt, "tokenizerPath"));
        properties.push_back(jsi::PropNameID::forAscii(rt, "prefill"));
        properties.push_back(jsi::PropNameID::forAscii(rt, "generate"));
        properties.push_back(jsi::PropNameID::forAscii(rt, "stop"));
        properties.push_back(jsi::PropNameID::forAscii(rt, "dispose"));
        return properties;
    }

    void install_createLLMRunner(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "createLLMRunner";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
            {
                throw jsi::JSError(rt, "createLLMRunner: Usage: createLLMRunner(modelPath, tokenizerPath)");
            }

            if (!args[0].isString() || !args[1].isString())
            {
                throw jsi::JSError(rt, "createLLMRunner: Expected arguments to be strings");
            }

            auto modelPath = args[0].asString(rt).utf8(rt);
            auto tokenizerPath = args[1].asString(rt).utf8(rt);

            try
            {
                auto runnerInstance = std::make_shared<LLMRunnerHostObject>(modelPath, tokenizerPath);
                return jsi::Object::createFromHostObject(rt, runnerInstance);
            }
            catch (const std::exception &e)
            {
                throw jsi::JSError(rt, std::string("createLLMRunner: ") + e.what());
            }
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody));
    }
} // namespace mylib::extensions::nlp::llm
