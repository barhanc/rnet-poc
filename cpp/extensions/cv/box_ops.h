#pragma once

#include <jsi/jsi.h>

#include <string>
#include <tuple>

namespace mylib::extensions::cv::box_ops
{
    enum class BoxFormat
    {
        XYXY,
        XYWH,
        CXCYWH
    };

    BoxFormat parseBoxFormat(const std::string &s);

    std::tuple<float, float, float, float> decodeToXyxy(
        float a, float b, float c, float d,
        BoxFormat format
    );

    void install_nms(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::extensions::cv::box_ops
