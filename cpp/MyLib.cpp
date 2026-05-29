#include "MyLib.h"

#include <opencv2/core.hpp>

#include "core/install.h"

using namespace facebook;

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime)
    {
        // OpenCV build/linkage test call
        cv::Mat testMat = cv::Mat::zeros(10, 10, CV_8UC1);
        (void)testMat; // suppress unused warning

        jsi::Object myModule = jsi::Object(jsiRuntime);

        mylib::core::install(jsiRuntime, myModule);

        jsiRuntime.global().setProperty(jsiRuntime, "__mylib_jsi__", std::move(myModule));
    }
} // namespace mylib