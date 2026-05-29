#ifndef MYLIB_H
#define MYLIB_H
#include <jsi/jsi.h>
#include <vector>

using namespace facebook;

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime);

    void install_checkExecuTorch(jsi::Runtime &rt, jsi::Object &module);
    void install_loadModel(jsi::Runtime &rt, jsi::Object &module);
    void install_executeModel(jsi::Runtime &rt, jsi::Object &module);
    void install_disposeModel(jsi::Runtime &rt, jsi::Object &module);
    void install_getModelMethodMeta(jsi::Runtime &rt, jsi::Object &module);
    void install_getModelMethodNames(jsi::Runtime &rt, jsi::Object &module);

} // namespace mylib

#endif