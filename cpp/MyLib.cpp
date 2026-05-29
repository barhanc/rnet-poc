#include "MyLib.h"
#include <jsi/jsi.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/module/module.h>

using namespace facebook;
using namespace ::executorch::extension;

class Date : public jsi::HostObject
{
public:
    jsi::Value get(jsi::Runtime &rt, const jsi::PropNameID &name) override
    {
        if (name.utf8(rt) == "now")
        {
            time_t now = time(0);
            std::string dateTime = ctime(&now);
            return jsi::String::createFromUtf8(rt, dateTime.c_str());
        }
        else if (name.utf8(rt) == "day")
        {
            return jsi::Value(day);
        }
        else if (name.utf8(rt) == "month")
        {
            return jsi::Value(month);
        }
        else if (name.utf8(rt) == "year")
        {
            return jsi::Value(year);
        }

        return jsi::Value::undefined();
    }

    void set(jsi::Runtime &rt, const jsi::PropNameID &name, const jsi::Value &value) override
    {
        if (name.utf8(rt) == "day")
            day = value.asNumber();
        else if (name.utf8(rt) == "month")
            month = value.asNumber();
        else if (name.utf8(rt) == "year")
            year = value.asNumber();
    }

    std::vector<facebook::jsi::PropNameID> getPropertyNames(jsi::Runtime &rt) override
    {
        std::vector<facebook::jsi::PropNameID> properties;
        properties.push_back(facebook::jsi::PropNameID::forAscii(rt, "now"));
        properties.push_back(facebook::jsi::PropNameID::forAscii(rt, "day"));
        properties.push_back(facebook::jsi::PropNameID::forAscii(rt, "month"));
        properties.push_back(facebook::jsi::PropNameID::forAscii(rt, "year"));
        return properties;
    }

    int day;
    int month;
    int year;

    Date()
    {
        time_t rawtime;
        time(&rawtime);
        struct tm *timeinfo = localtime(&rawtime);
        year = timeinfo->tm_year + 1900;
        month = timeinfo->tm_mon;
        day = timeinfo->tm_mday;
    }
    ~Date() override
    {
    }
};

class Infinity : public jsi::HostObject
{
public:
    jsi::Value get(jsi::Runtime &rt, const jsi::PropNameID &name) override
    {
        counter++;
        return jsi::Value(counter);
    }
    void set(jsi::Runtime &rt, const jsi::PropNameID &name, const jsi::Value &value) override
    {
        counter--;
    }
    std::vector<facebook::jsi::PropNameID> getPropertyNames(jsi::Runtime &rt) override
    {
        std::vector<facebook::jsi::PropNameID> properties;
        return properties;
    }

    int counter = 0;

    Infinity()
    {
    }
    ~Infinity() override
    {
    }
};

namespace mylib
{
    void install(jsi::Runtime &jsiRuntime)
    {
        jsi::Object myModule = jsi::Object(jsiRuntime);

        task1(jsiRuntime, myModule);
        task2(jsiRuntime, myModule);
        task3(jsiRuntime, myModule);
        task4(jsiRuntime, myModule);
        task5(jsiRuntime, myModule);
        task6(jsiRuntime, myModule);
        task7(jsiRuntime, myModule);
        task8(jsiRuntime, myModule);
        task9(jsiRuntime, myModule);
        task10(jsiRuntime, myModule);
        task11(jsiRuntime, myModule);
        task12(jsiRuntime, myModule);
        task13(jsiRuntime, myModule);
        task14(jsiRuntime, myModule);
        installExecutorchModule(jsiRuntime, myModule);

        jsiRuntime.global().setProperty(jsiRuntime, "__myModule__", std::move(myModule));
    }

    void task1(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "answerToTheUltimateQuestionOfLifeTheUniverseAndEverything";
        module.setProperty(rt, name, 42);
    }
    void task2(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "isWednesday";
        module.setProperty(rt, name, false);
    }
    void task3(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "workshopName";
        module.setProperty(rt, name, "JSIWorkshops");
    }
    void task4(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "myAwesomeArray";

        auto array = jsi::Array(rt, 3);
        array.setValueAtIndex(rt, 0, 42);
        array.setValueAtIndex(rt, 1, true);
        array.setValueAtIndex(rt, 2, "App.js");

        module.setProperty(rt, name, array);
    }
    void task5(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "giveMeFive";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 0)
                throw jsi::JSError(rt, "Incorrect number of arguments");
            return 5;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 0, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task6(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "sumMeThis";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
                throw jsi::JSError(rt, "Incorrect number of arguments");
            if (!args[0].isNumber() || !args[1].isNumber())
                throw jsi::JSError(rt, "Expected numbers");
            return args[0].asNumber() + args[1].asNumber();
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task7(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "divideMeThis";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
                throw jsi::JSError(rt, "Incorrect number of arguments");
            if (!args[0].isNumber() || !args[1].isNumber())
                throw jsi::JSError(rt, "Expected numbers");
            if (args[1].asNumber() == 0)
                throw jsi::JSError(rt, "Division by 0!");
            return args[0].asNumber() / args[1].asNumber();
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task8(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "reverseMeThis";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
                throw jsi::JSError(rt, "Incorrect number of arguments");
            if (!args[0].isString())
                throw jsi::JSError(rt, "Expected string");

            auto str = args[0].asString(rt).utf8(rt);
            std::reverse(str.begin(), str.end());
            return jsi::String::createFromUtf8(rt, str);
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task9(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "sumMeThisObject";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
                throw jsi::JSError(rt, "Incorrect number of arguments");

            auto inputObj = args[0].asObject(rt);
            if (!inputObj.hasProperty(rt, "firstNum") || !inputObj.hasProperty(rt, "lastNum"))
                throw jsi::JSError(rt, "Required fields do not exist");

            auto num1 = inputObj.getProperty(rt, "firstNum").asNumber();
            auto num2 = inputObj.getProperty(rt, "lastNum").asNumber();
            auto result = num1 + num2;

            auto outputObj = jsi::Object(rt);
            outputObj.setProperty(rt, "result", result);
            return outputObj;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task10(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "sumMeThisArray";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 1)
                throw jsi::JSError(rt, "Incorrect number of arguments");
            if (!args[0].asObject(rt).isArray(rt))
                throw jsi::JSError(rt, "Expected array");

            auto inputArr = args[0].asObject(rt).asArray(rt);
            for (size_t i = 0; i < inputArr.size(rt); i++)
                if (!inputArr.getValueAtIndex(rt, i).isNumber())
                    throw jsi::JSError(rt, "Expected array of numbers");

            auto result = 0;
            for (size_t i = 0; i < inputArr.size(rt); i++)
                result += inputArr.getValueAtIndex(rt, i).asNumber();

            return result;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 1, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task11(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "nativeMap";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 2)
                throw jsi::JSError(rt, "Incorrect number of arguments");
            if (!args[0].asObject(rt).isArray(rt))
                throw jsi::JSError(rt, "Expected array");
            if (!args[1].asObject(rt).isFunction(rt))
                throw jsi::JSError(rt, "Expected function");

            auto inputArr = args[0].asObject(rt).asArray(rt);
            auto fn = args[1].asObject(rt).asFunction(rt);

            auto result = jsi::Array(rt, inputArr.size(rt));
            for (size_t i = 0; i < inputArr.size(rt); i++)
                result.setValueAtIndex(rt, i, fn.call(rt, inputArr.getValueAtIndex(rt, i)));

            return result;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 2, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task12(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "runJsFunction";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 0)
                throw jsi::JSError(rt, "Incorrect number of arguments");

            jsi::Object console = rt.global().getProperty(rt, "console").asObject(rt);
            jsi::Function log = console.getProperty(rt, "log").asObject(rt).asFunction(rt);
            log.call(rt, jsi::String::createFromUtf8(rt, "Hello from C++"));

            return jsi::Value::undefined();
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 0, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task13(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "getDateObject";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 0)
                throw jsi::JSError(rt, "Incorrect number of argumnents");

            auto date = std::make_shared<Date>();
            jsi::Object jsDate = jsi::Object::createFromHostObject(rt, date);
            return jsDate;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 0, fnBody);

        module.setProperty(rt, name, fn);
    }
    void task14(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "getInfinityObject";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 0)
                throw jsi::JSError(rt, "Incorrect number of argumnents");

            auto infinity = std::make_shared<Infinity>();
            jsi::Object jsInfinity = jsi::Object::createFromHostObject(rt, infinity);
            return jsInfinity;
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 0, fnBody);

        module.setProperty(rt, name, fn);
    }
    void installExecutorchModule(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "checkExecutorch";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            if (count != 0)
                throw jsi::JSError(rt, "Incorrect number of arguments");

            // 1. Define shape and data
            int32_t sizes[] = {2}; // A 1D tensor with 2 elements
            float data[] = {42.0f, 7.0f};

            // 2. Create an ExecuTorch Tensor Implementation
            // This wraps our raw 'data' array without copying it
            exec_aten::TensorImpl impl(
                exec_aten::ScalarType::Float,
                1, // Number of dimensions
                sizes,
                data,
                nullptr // No special allocator needed for this test
            );

            // 3. Create the actual Tensor object
            exec_aten::Tensor tensor(&impl);

            // 4. Read data back to verify
            float val = tensor.const_data_ptr<float>()[0];

            // 5. Return success message to JavaScript
            std::string result = "ExecuTorch is Live! Tensor[0] = " + std::to_string(val);
            return jsi::String::createFromUtf8(rt, result);
        };
        auto fn = jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 0, fnBody);

        module.setProperty(rt, name, fn);
    }
} // namespace mylib