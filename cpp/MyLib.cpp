#include "MyLib.h"

#include <opencv2/core.hpp>

#include "core/Model.h"
#include "core/Tensor.h"
#include "core/Utils.h"

using namespace facebook;

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime)
    {
        // OpenCV build/linkage test call
        cv::Mat testMat = cv::Mat::zeros(10, 10, CV_8UC1);
        (void)testMat; // suppress unused warning

        jsi::Object myModule = jsi::Object(jsiRuntime);

        mylib::core::utils::install_getExecuTorchRegisteredBackends(jsiRuntime, myModule);

        mylib::core::model::install_loadModel(jsiRuntime, myModule);
        mylib::core::model::install_disposeModel(jsiRuntime, myModule);
        mylib::core::model::install_executeModelMethod(jsiRuntime, myModule);
        mylib::core::model::install_getModelMethodMeta(jsiRuntime, myModule);
        mylib::core::model::install_getModelMethodNames(jsiRuntime, myModule);

        mylib::core::tensor::install_createTensor(jsiRuntime, myModule);
        mylib::core::tensor::install_disposeTensor(jsiRuntime, myModule);
        mylib::core::tensor::install_setTensorFromTypedArray(jsiRuntime, myModule);
        mylib::core::tensor::install_setTypedArrayFromTensor(jsiRuntime, myModule);

        jsiRuntime.global().setProperty(jsiRuntime, "__mylib_jsi__", std::move(myModule));
    }
} // namespace mylib