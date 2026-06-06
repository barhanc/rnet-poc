---
id: add_task_pipeline
name: Add TypeScript Task Pipeline & React Hooks
description: How to create end-to-end task pipelines (loading models, pre/post-processing) and wrap them in React Hooks.
scope: src/extensions/*/tasks/*, src/hooks/*
---

# Skill: Add a High-Level Task Pipeline (TypeScript)

Use this guide to construct end-to-end task pipelines (e.g. classification, style transfer, object detection) in TypeScript and wrap them in React hooks.

---

## 🚦 Design Principles & Constraints

When implementing task constructors like `create<Task>` (e.g. `createClassifier`, `createStyleTransfer`), adhere to the following rules:

1. **Pre-allocating Static Tensors (`as const`)**:
   * Statically sized scratch/output tensors required for inference should be pre-allocated inside the constructor body.
   * Allocate them using:
     ```typescript
     const tensors = [
       tensor('float32', shapeA),
       tensor('float32', shapeB),
     ] as const;
     ```
    * **Destructuring & Naming**: It is idiomatic to destructure and name the individual tensors immediately after allocation. 
      * Always prefix tensor variables with a lowercase `t` (e.g. `tReshape`, `tUint8`, `tInput`) to easily distinguish them from raw data buffers or other variables.
      * **Do not** access tensors by index (e.g., `tensors[0]`, `tensors[1]`) throughout the function body; destructuring makes references explicit and readable:
        ```typescript
        const [tReshape, tUint8] = tensors;
        ```

2. **Immediate `dispose()` Definition**:
   * Right after allocating the static tensors, define the `dispose` function immediately. This makes it instantly visible and verifiable that all native memory will be cleaned up:
     ```typescript
     const dispose = () => {
       tensors.forEach((t) => t.dispose());
       preprocessor.dispose();
       model.dispose();
     };
     ```

3. **Dynamic Tensors & `try/finally` Pattern**:
   * If you must allocate dynamically sized tensors during inference execution (e.g. resizing an output tensor to match the input image dimensions), you **must** wrap the execution inside a `try {} finally {}` block.
   * Dispose of the dynamic tensors inside the `finally` block to prevent native memory leaks.
     ```typescript
     const tResize = tensor('uint8', [input.height, input.width, 4]);
     try {
       // Perform work with tResize...
     } finally {
       tResize.dispose();
     }
     ```

4. **Flat Closures (Exactly Two Inner Functions)**:
   * You must define **exactly two** inner functions inside the `create<Task>` constructor:
     1. The `dispose` function (responsible for releasing pre-allocated static tensors and the model runtime).
     2. The task `worklet` executor function (which executes synchronously on the worklet thread).
   * **Do not** implement any other inner helper functions (like `postprocess()`) inside the `create<Task>` body. Defining inner helpers that are subsequently called inside the worklet—where both access the shared variables allocated in the outer constructor scope—creates a tangled chain of implicit dependencies and captures that is extremely difficult to reason about.
   * Write all auxiliary/helper logic as pure, worklet-compatible functions **outside** the `create<Task>` constructor. Any helper functions that are invoked inside the worklet executor thread must contain the `'worklet';` directive to run on the worklet runtime.
   * *(Note: Exposing the asynchronous API for the JS thread is done independently by wrapping the synchronous worklet using `wrapAsync(...)`.)*

5. **No Leaking Raw Tensors to Consumers**:
   * The methods returned by `create<Task>` (such as `classify` or `transferStyle`) **must NEVER** return raw `Tensor` objects to the API consumer.
   * Doing so places the burden of native memory management on the user. Always convert output data to standard JavaScript values/objects (e.g. array of floats, image buffers, strings) before returning.

6. **Minimizing Thread Boundary Crossings**:
   * Passing heavy objects across the two JavaScript runtimes (JS thread vs. Worklet runtime) incurs significant serialization overhead. Avoid doing so unless it is truly required.
   * Only cross this boundary when absolutely necessary—for example, passing an input `ImageBuffer` down to the worklet runtime for processing, and returning a processed output `ImageBuffer` back to the JS thread. Keep all intermediate data representations within the same runtime.

7. **PTE Model Export as a Degree of Freedom (Simplify & Generalize)**:
   * **Do not** treat the `.pte` model as an unchangeable black box. Because we control the PyTorch model export phase, we can reshape the model's inputs and outputs to make the mobile client pipeline as lightweight as possible.
   * **Shift Heavy Ops to PyTorch**: Push complex tensor reshaping, data normalization, activations (e.g. `softmax`), or bounding box decoding into the PyTorch model itself so they execute on native backends (e.g., XNNPACK or CoreML).
   * **Balance Optimization with Generalization (Extensibility)**: While optimizing model exports is preferred, **do not make the input/output contracts so specific that they break extensibility**. Users should be able to run their own custom `.pte` models through our pipelines.
     * *Rule*: Keep contracts generic (e.g., normal dense logits, standard bounding box layouts like `xyxy`/`xywh`, standard floating-point arrays).
     * *Rule*: Handle model-specific configuration parameters (such as unique normalization factors, thresholds, or label arrays) dynamically through the TypeScript task options argument rather than baking them rigidly into JSI C++ code or the model structure.
   * **Target Layouts**: Export models with layouts that align with our image preprocessors (such as standard normalized `[1, 3, H, W]` float32 tensors).

---

## 🛠️ Step-by-Step Implementation Template

### Step 1: Create the Task File (`src/extensions/<domain>/tasks/<task>.ts`)

```typescript
import type { WorkletRuntime } from 'react-native-worklets';

import { tensor } from '../../../core/tensor';
import { loadModel } from '../../../core/model';
import { validateModelSchema, SymbolicTensor } from '../../../core/modelSchema';
import { wrapAsync } from '../../../core/runtime';
import { type ImageBuffer } from '../image';
import { createImagePreprocessor, type ImagePreprocessorOptions } from './preprocessing';

export type MyTaskOptions = ImagePreprocessorOptions & {
  readonly defaultThreshold: number;
};

export type MyTaskModel = {
  readonly modelPath: string;
  readonly taskOpts: MyTaskOptions;
};

export type MyTaskResult = {
  readonly classId: number;
  readonly score: number;
};

// 1. Helper functions MUST be defined OUTSIDE create<Task> and be worklet-compatible
function postprocessOutput(rawData: Float32Array, threshold: number): MyTaskResult[] {
  'worklet';
  const results: MyTaskResult[] = [];
  for (let i = 0; i < rawData.length; i++) {
    if (rawData[i]! > threshold) {
      results.push({ classId: i, score: rawData[i]! });
    }
  }
  return results.sort((a, b) => b.score - a.score);
}

export async function createMyTask(
  config: MyTaskModel,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  runTask: (input: ImageBuffer, options?: { threshold?: number }) => Promise<MyTaskResult[]>;
  runTaskWorklet: (input: ImageBuffer, options?: { threshold?: number }) => MyTaskResult[];
}> {
  const { modelPath, taskOpts } = config;
  const model = await wrapAsync(loadModel, runtime)(modelPath);

  // Validate model schema
  const meta = validateModelSchema(
    model,
    'forward',
    [SymbolicTensor('float32', [1, 3, 'H', 'W'], [3, 'H', 'W'])],
    [SymbolicTensor('float32', [1, 10], [10])],
  );
  const inpShape = meta.inputTensorMeta[0]!.shape;
  const outShape = meta.outputTensorMeta[0]!.shape;

  // 2. Pre-allocate static tensors
  const tensors = [
    tensor('float32', outShape),
  ] as const;

  // Idiomatic destructuring and naming with "t" prefix
  const [tOutput] = tensors;
  const preprocessor = createImagePreprocessor(taskOpts, inpShape);

  // 3. Define dispose() immediately after allocation
  const dispose = () => {
    preprocessor.dispose();
    tensors.forEach((t) => t.dispose());
    model.dispose();
  };

  // 4. Define exactly two inner functions (dispose & runTaskWorklet)
  const runTaskWorklet = (
    input: ImageBuffer,
    options?: { threshold?: number }
  ): MyTaskResult[] => {
    'worklet';
    
    // Process input buffer to input tensor
    const tInput = preprocessor.process(input);
    model.execute('forward', [tInput], [tOutput]);

    const data = tOutput.getData(new Float32Array(tOutput.numel));
    const threshold = options?.threshold ?? taskOpts.defaultThreshold;
    
    // 5. Return standard JS object, never raw Tensor
    return postprocessOutput(data, threshold);
  };

  const runTask = wrapAsync(runTaskWorklet, runtime);

  return { runTask, runTaskWorklet, dispose };
}
```

### Step 2: Create the React Hook Wrapper (`src/hooks/use<Task>.ts`)

Wrap the task pipeline in a custom React Hook using the core hooks `useModelDownload` and `useModel`. This manages downloading, compilation, error tracking, and automatic cleanup of the native memory upon unmounting or config changes.

```typescript
import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import { createMyTask, type MyTaskModel } from '../extensions/<domain>/tasks/<task>';

export function useMyTask(
  config: MyTaskModel, 
  options?: { preventLoad?: boolean }
) {
  // 1. Resolve remote or local asset model path and download progress
  const { localPath, downloadProgress, downloadError } = useModelDownload(
    config.modelPath,
    options?.preventLoad,
  );

  // 2. Instantiate and compile the task pipeline (with automatic lifecycle cleanup)
  const { model, error } = useModel(
    createMyTask,
    localPath ? { ...config, modelPath: localPath } : null,
    [localPath],
  );

  return {
    isReady: !!model,
    error: downloadError || error,
    downloadProgress,
    localPath,
    runTask: model?.runTask,
    runTaskWorklet: model?.runTaskWorklet,
  };
}
```
