#include "install.h"
#include "processing.h"

namespace mylib::extensions::cv
{
    namespace jsi = facebook::jsi;

    void install(facebook::jsi::Runtime &rt, facebook::jsi::Object &module)
    {
        jsi::Object myCVModule = jsi::Object(rt);

        processing::install_resize(rt, myCVModule);
        processing::install_cvtColor(rt, myCVModule);
        processing::install_toChannelsFirst(rt, myCVModule);
        processing::install_toChannelsLast(rt, myCVModule);
        processing::install_normalize(rt, myCVModule);
        processing::install_nms(rt, myCVModule);
        processing::install_decodeBoxes(rt, myCVModule);
        processing::install_scaleBoxes(rt, myCVModule);

        module.setProperty(rt, "cv", myCVModule);
    }
} // namespace mylib::extensions::cv
