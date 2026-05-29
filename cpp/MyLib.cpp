#include "MyLib.h"

#include "Model.hpp"
#include "Tensor.hpp"
#include "Utils.hpp"

using namespace facebook;

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime)
    {
        jsi::Object myModule = jsi::Object(jsiRuntime);

        mylib::utils::install_getExecuTorchRegisteredBackends(jsiRuntime, myModule);

        mylib::model::install_loadModel(jsiRuntime, myModule);
        mylib::model::install_disposeModel(jsiRuntime, myModule);
        mylib::model::install_executeModelMethod(jsiRuntime, myModule);
        mylib::model::install_getModelMethodMeta(jsiRuntime, myModule);
        mylib::model::install_getModelMethodNames(jsiRuntime, myModule);

        mylib::tensor::install_createTensor(jsiRuntime, myModule);
        mylib::tensor::install_disposeTensor(jsiRuntime, myModule);
        mylib::tensor::install_setTensorFromTypedArray(jsiRuntime, myModule);
        mylib::tensor::install_setTypedArrayFromTensor(jsiRuntime, myModule);

        jsiRuntime.global().setProperty(jsiRuntime, "__mylib_jsi__", std::move(myModule));
    }
} // namespace mylib