#pragma once

#include <jsi/jsi.h>

namespace mylib::extensions::cv::box_ops
{
    void install_nms(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
    void install_decodeBoxes(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
    void install_scaleBoxes(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::extensions::cv::box_ops
