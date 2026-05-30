#pragma once

#include <jsi/jsi.h>

namespace mylib::extensions::math::operations
{
    void install_sigmoid(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
    void install_softmax(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::extensions::math::operations