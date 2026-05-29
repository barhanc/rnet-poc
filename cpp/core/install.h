#pragma once

#include <jsi/jsi.h>

namespace mylib::core
{
    void install(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::core
