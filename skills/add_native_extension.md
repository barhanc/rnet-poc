---
id: add_native_extension
name: Add Native C++ Extension & JSI Bindings
description: How to add custom native C++ operations, write JSI host functions, register them in install maps, and compile.
scope: cpp/extensions/*, src/extensions/*
---

# Skill: Add a Native C++ Extension & JSI Bindings

Use this guide to add custom, performance-critical native operations in C++ and expose them to TypeScript via React Native JSI.

---

## 🚦 Architectural Guidelines & Constraints

Before writing any C++ code, ensure you adhere to the following principles:

1. **Amdahl's Law & Premature Optimization**:
   * Do not implement operations in C++ unless absolutely necessary.
   * Evaluate what percentage of total inference/pipeline time the processing step occupies. If the preprocessing/postprocessing step takes `< 5%` of the total inference budget, write it in **pure TypeScript** to reduce codebase complexity and maintenance overhead.

2. **No Implicit Allocations (C-Style Destination Tensors & JSI Array Returns)**:
   * Native functions must **NEVER** implicitly allocate new tensors or data buffers.
   * If the operation writes numeric/tensor output, the destination tensor must be pre-allocated by the caller and passed as an argument (e.g., `sigmoid(src, dst)`).
   * If the operation does not write to a dense tensor (e.g. Non-Maximum Suppression (NMS)), it should **not** allocate tensors inside C++. Instead, return a plain JavaScript array of primitives (like indices or coordinates) to the TS layer.
   * *Example*: `nms(boxes, scores, options)` returns a `jsi::Array` of indices (e.g., `[0, 4, 12]`) rather than a new tensor. This avoids all native memory management overhead for variable-sized outputs.◊

3. **No Default Parameters in C++**:
   * C++ native functions must **NEVER** define default argument values (e.g. `axis = -1`).
   * Default arguments must be defined explicitly in the TypeScript wrapper layer. This ensures that users trace code using the IDE's "Go to Definition" feature see transparent, fully-specified parameters.

---

## 🛠️ Step-by-Step Implementation

### Step 1: Create the Native Operation Files
Under `cpp/extensions/<domain>/`, create or modify the header and implementation files for your operations:

#### 1. Header (`cpp/extensions/<domain>/operations.h`)
Keep the header clean and specify exact JSI install functions:
```cpp
#pragma once
#include <jsi/jsi.h>

namespace mylib::extensions::<domain>
{
    void install_customOp(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
}
```

#### 2. Source (`cpp/extensions/<domain>/operations.cpp`)
* Extract input and output tensors as `TensorHostObject` pointers.
* Check bounds, shapes, types, and verify that the output tensor is **not the same instance** as the input (no unsafely managed in-place mutation).
* Lock tensors using `std::shared_lock` (for inputs) and `std::unique_lock` (for outputs).

```cpp
#include "operations.h"
#include "core/tensor.h"
#include <algorithm>

namespace mylib::extensions::<domain>
{
    namespace jsi = facebook::jsi;
    using TensorHostObject = mylib::core::tensor::TensorHostObject;

    void install_customOp(jsi::Runtime &rt, jsi::Object &module)
    {
        auto name = "customOp";
        auto fnBody = [](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value
        {
            // 1. Strict argument count validation (No default values here!)
            if (count != 3)
            {
                throw jsi::JSError(rt, "Usage: customOp(src, dst, factor)");
            }

            // 2. Validate input and output types
            auto srcObj = args[0].asObject(rt);
            auto dstObj = args[1].asObject(rt);
            if (!srcObj.isHostObject<TensorHostObject>(rt) || !dstObj.isHostObject<TensorHostObject>(rt))
            {
                throw jsi::JSError(rt, "customOp: Arguments src and dst must be Tensors");
            }

            auto src = srcObj.getHostObject<TensorHostObject>(rt);
            auto dst = dstObj.getHostObject<TensorHostObject>(rt);
            double factor = args[2].asNumber();

            // 3. Prevent in-place mutations
            if (src.get() == dst.get())
            {
                throw jsi::JSError(rt, "customOp: In-place operations (src == dst) are not supported.");
            }

            // 4. Validate metadata compatibility
            if (src->shape_ != dst->shape_ || src->dtype_ != dst->dtype_)
            {
                throw jsi::JSError(rt, "customOp: src and dst shape and dtype must match");
            }

            // 5. Lock underlying buffers
            std::shared_lock<std::shared_mutex> src_lock(src->mutex_, std::try_to_lock);
            std::unique_lock<std::shared_mutex> dst_lock(dst->mutex_, std::try_to_lock);
            if (!src_lock.owns_lock() || !dst_lock.owns_lock())
            {
                throw jsi::JSError(rt, "customOp: Tensors are currently in use");
            }

            if (!src->data_ || !dst->data_)
            {
                throw jsi::JSError(rt, "customOp: Tensor has been disposed");
            }

            // 6. Perform the computation
            const float *srcData = reinterpret_cast<const float *>(src->data_.get());
            float *dstData = reinterpret_cast<float *>(dst->data_.get());
            size_t size = src->size();

            for (size_t i = 0; i < size; ++i)
            {
                dstData[i] = srcData[i] * static_cast<float>(factor);
            }

            // Always return the destination tensor (args[1]) as the JSI result
            return jsi::Value(rt, args[1]);
        };

        module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 3, fnBody));
    }
}
```

---

### Step 2: Register in Extension and Core JSI Installs

1. **Extension Register** (`cpp/extensions/<domain>/install.cpp`):
   ```cpp
   #include "install.h"
   #include "operations.h"

   namespace mylib::extensions::<domain>
   {
       void install(facebook::jsi::Runtime &rt, facebook::jsi::Object &module)
       {
           facebook::jsi::Object subModule(rt);
           install_customOp(rt, subModule);
           module.setProperty(rt, "<domain>", subModule);
       }
   }
   ```

2. **Core Register** ([cpp/MyLib.cpp](../cpp/MyLib.cpp)):
   ```cpp
   #include "extensions/<domain>/install.h"
   // ... inside mylib::install ...
   mylib::extensions::<domain>::install(jsiRuntime, myModule);
   ```

---

### Step 3: TypeScript Bridge & Wrappers
Under `src/extensions/<domain>.ts` or `src/extensions/<domain>/index.ts`:
* **Use the `mylibJsi` Symbol**: You must import and interact with native bindings using the `mylibJsi` symbol exported from [src/native/bridge.ts](../src/native/bridge.ts). **Do not** reference the global `__mylib_jsi__` directly throughout your wrapper files.
* Expose the TypeScript wrapper.
* Handle default values here instead of the C++ layer.
* Mark wrapper functions with the `"worklet";` directive.

```typescript
import { mylibJsi } from '../native/bridge';
import { type Tensor } from '../core/tensor';

/**
 * Applies a custom operation scaling the src tensor by factor.
 * @param src Input Tensor.
 * @param dst Pre-allocated Destination Tensor.
 * @param factor Scale factor. Defaults to 1.0.
 */
export function customOp(src: Tensor, dst: Tensor, factor: number = 1.0): Tensor {
  'worklet';
  return mylibJsi.<domain>.customOp(src, dst, factor);
}
```
