#include "install.h"
#include "llm_runner.h"

namespace mylib::extensions::nlp
{
    namespace jsi = facebook::jsi;

    void install(facebook::jsi::Runtime &rt, facebook::jsi::Object &module)
    {
        jsi::Object myNLPModule = jsi::Object(rt);

        llm::install_createLLMRunner(rt, myNLPModule);

        module.setProperty(rt, "nlp", myNLPModule);
    }
} // namespace mylib::extensions::nlp
