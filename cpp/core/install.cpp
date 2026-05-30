#include "install.h"
#include "utils.h"
#include "model.h"
#include "tensor.h"

namespace mylib::core
{
    void install(facebook::jsi::Runtime &rt, facebook::jsi::Object &module)
    {
        utils::install_getExecuTorchRegisteredBackends(rt, module);
        model::install_loadModel(rt, module);
        tensor::install_createTensor(rt, module);
    }
} // namespace mylib::core
