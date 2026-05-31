#pragma once

#include <jsi/jsi.h>

#include <string>
#include <tuple>

namespace mylib::extensions::cv::box_ops
{
    void install_nms(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::extensions::cv::box_ops
