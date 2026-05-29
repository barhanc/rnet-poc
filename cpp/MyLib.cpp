#include "MyLib.h"

#include <opencv2/core.hpp>

#include "core/install.h"
#include "extensions/cv/install.h"

using namespace facebook;

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime)
    {
        jsi::Object myModule = jsi::Object(jsiRuntime);

        mylib::core::install(jsiRuntime, myModule);
        mylib::extensions::cv::install(jsiRuntime, myModule);

        jsiRuntime.global().setProperty(jsiRuntime, "__mylib_jsi__", std::move(myModule));
    }
} // namespace mylib