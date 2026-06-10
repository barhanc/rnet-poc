#pragma once

#include <jsi/jsi.h>

namespace mylib::extensions::nlp
{
    void install(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::extensions::nlp
