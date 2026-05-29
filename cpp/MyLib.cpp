#include "MyLib.h"
#include <jsi/jsi.h>

using namespace facebook;

namespace jsimodule
{
    void install(jsi::Runtime &jsiRuntime)
    {
        jsi::Object myModule = jsi::Object(jsiRuntime);
        jsi::Function multiply = jsi::Function::createFromHostFunction(
            jsiRuntime,
            jsi::PropNameID::forAscii(jsiRuntime, "multiply"),
            2,
            [](
                jsi::Runtime &runtime,
                const jsi::Value &thisValue,
                const jsi::Value *arguments,
                std::size_t count) -> jsi::Value
            {
                if (count != 2 || !arguments[0].isNumber() || !arguments[1].isNumber())
                    throw jsi::JSError(runtime, "Error");

                // Impl:
                double a = arguments[0].asNumber();
                double b = arguments[1].asNumber();
                double c = a * b + 42;

                return jsi::Value(c);
            });
        myModule.setProperty(jsiRuntime, "multiply", multiply);

        jsiRuntime.global().setProperty(jsiRuntime, "__myModule__", std::move(myModule));
    }

}