#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <jsi/jsi.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>

namespace mylib::extensions::nlp::llm
{
    class LLMRunnerHostObject : public facebook::jsi::HostObject, public std::enable_shared_from_this<LLMRunnerHostObject>
    {
    public:
        std::unique_ptr<executorch::extension::llm::TextLLMRunner> runner_;
        std::mutex mutex_;
        std::string modelPath_;
        std::string tokenizerPath_;

        LLMRunnerHostObject(const std::string &modelPath, const std::string &tokenizerPath);

        facebook::jsi::Value get(facebook::jsi::Runtime &rt, const facebook::jsi::PropNameID &name) override;
        std::vector<facebook::jsi::PropNameID> getPropertyNames(facebook::jsi::Runtime &rt) override;
    };

    void install_createLLMRunner(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::extensions::nlp::llm
