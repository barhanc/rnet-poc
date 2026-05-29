#pragma once

#include <jsi/jsi.h>

namespace mylib::extensions::cv::processing
{
    void install_resize(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::extensions::cv::processing
