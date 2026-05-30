#include "install.h"
#include "operations.h"

namespace mylib::extensions::math
{
    namespace jsi = facebook::jsi;

    void install(jsi::Runtime &rt, jsi::Object &module)
    {
        jsi::Object myMathModule(rt);

        operations::install_sigmoid(rt, myMathModule);
        operations::install_softmax(rt, myMathModule);

        module.setProperty(rt, "math", myMathModule);
    }
} // namespace mylib::extensions::math