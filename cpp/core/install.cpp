#include "install.h"
#include "utils.h"
#include "Model.h"
#include "Tensor.h"

namespace mylib::core
{
    void install(facebook::jsi::Runtime &rt, facebook::jsi::Object &module)
    {
        utils::install_getExecuTorchRegisteredBackends(rt, module);

        model::install_loadModel(rt, module);
        model::install_disposeModel(rt, module);
        model::install_executeModelMethod(rt, module);
        model::install_getModelMethodMeta(rt, module);
        model::install_getModelMethodNames(rt, module);

        tensor::install_createTensor(rt, module);
        tensor::install_disposeTensor(rt, module);
        tensor::install_setTensorFromTypedArray(rt, module);
        tensor::install_setTypedArrayFromTensor(rt, module);
    }
} // namespace mylib::core
