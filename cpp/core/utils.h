#pragma once

#include <jsi/jsi.h>

namespace mylib::core::utils
{
    void install_getExecuTorchRegisteredBackends(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace mylib::core::utils
