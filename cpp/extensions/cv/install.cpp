#include "install.h"
#include "processing.h"

namespace mylib::extensions::cv
{
    namespace jsi = facebook::jsi;

    void install(facebook::jsi::Runtime &rt, facebook::jsi::Object &module)
    {
        jsi::Object cvModule = jsi::Object(rt);

        processing::install_resize(rt, cvModule);

        module.setProperty(rt, "cv", cvModule);
    }
} // namespace mylib::extensions::cv
