#ifndef MYLIB_H
#define MYLIB_H
#include <jsi/jsi.h>
#include <vector>

using namespace facebook;

namespace mylib
{
void install(jsi::Runtime &jsiRuntime);

void task1(jsi::Runtime &rt, jsi::Object &module);
void task2(jsi::Runtime &rt, jsi::Object &module);
void task3(jsi::Runtime &rt, jsi::Object &module);
void task4(jsi::Runtime &rt, jsi::Object &module);
void task5(jsi::Runtime &rt, jsi::Object &module);
void task6(jsi::Runtime &rt, jsi::Object &module);
void task7(jsi::Runtime &rt, jsi::Object &module);
void task8(jsi::Runtime &rt, jsi::Object &module);
void task9(jsi::Runtime &rt, jsi::Object &module);
void task10(jsi::Runtime &rt, jsi::Object &module);
void task11(jsi::Runtime &rt, jsi::Object &module);
void task12(jsi::Runtime &rt, jsi::Object &module);
void task13(jsi::Runtime &rt, jsi::Object &module);
void task14(jsi::Runtime &rt, jsi::Object &module);
void installExecutorchModule(jsi::Runtime &rt, jsi::Object &module);

} // namespace mylib

#endif