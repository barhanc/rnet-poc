# RNET-POC

- [What?](#what)
- [Why?](#why)
- [Design philosophy](#design-philosophy)
- [Structure](#structure)
  - [Lower-level API](#lower-level-api)
  - [Higher-level API](#higher-level-api)
- [Comparison with
  `react-native-executorch`](#comparison-with-react-native-executorch)
  - [Quantitative Metrics](#quantitative-metrics)
  - [Developer Ergonomics](#developer-ergonomics)
  - [Performance](#performance)
- [What's next?](#whats-next)

## What?

This is a proof-of-concept alternative design of 'react-native-executorch'. It
was built over the course of a few days, with heavy use of AI coding agents to
explore designs that could overcome pain points of the existing
'react-native-executorch' library. While this PoC implements only a small
fraction of the original library's features, we hope to demonstrate that the
design choices are suitable for a whole range of use cases, and that the
implementation can be extended to cover all the features of the original
library, and more.

## Why?

'react-native-executorch' is a unique solution in the world of mobile AI, which
tries to bring together the ease of use of libraries like MediaPipe which
provide read-to-use pipelines for common machine learning tasks, and the
flexibility of libraries like ExecuTorch / ONNX Runtime / LiteRT / MLX which
provide low-level APIs to run arbitrary models, all to React Native ecosystem.
There are however some pain points in the design of the library which have
become more apparent recently as more and more models and use cases are being
added to the library. The main pain points which we identified are the
following.

1. **Restricted Arbitrary Model Execution**

    While 'react-native-executorch' should theoretically allow developers to
    run, inspect, and manipulate any model, doing so in practice is highly
    restrictive. The library's native wrapper for user-provided `.pte` files—the
    ExecutorchModule class—exposes only a single forward method. It lacks APIs
    to execute arbitrary model methods or inspect inputs and outputs, both of
    which are foundational utilities of the underlying native ExecuTorch Module
    API. There are also no primitives for manipulating the model inputs/outputs.

2. **Opaque Task-Specific Pipelines**

    Task-specific APIs (such as `ClassificationModule` or
    `InstanceSegmentationModule`) offer a `.fromCustomConfig` method to run
    custom models through pre-built pipelines. However, reverse-engineering
    these pipelines to understand their input/output contracts requires digging
    through multiple abstraction layers all the way down to native C++ code. The
    documentation only describes these contracts declaratively.

    Try it yourself: Trace how the object detection pipeline is implemented in
    the current 'react-native-executorch' source code. Start at
    `ObjectDetectionModule` and use only your IDE's 'Go to Definition'.

3. **High Friction for Library Maintainers & Contributors**

    Adding a new task pipeline currently requires modifying unrelated files
    across the codebase, duplicating boilerplate code, and wrestling with tight
    coupling. This makes extending the library a time-consuming process. This
    friction was recently highlighted and summarized well by the
    [RFC](https://github.com/software-mansion/react-native-executorch/discussions/1189).

This PoC tries to address these pain points by presenting a different design of
the library, which is more modular, more transparent and easier to extend with
new features and task pipelines.

## Design philosophy

The core idea behind the design is the separation of the library into two main
conceptual layers: a lower-level API which provides direct access to the
underlying ExecuTorch capabilities and a way for manipulating model
inputs/outputs as Tensor-like data structures, and a higher-level API which
provides ready-to-use pipelines for common machine learning tasks and React
Native hooks that wrap them.

The lower-level API is designed to be flexible and extensible, allowing
developers and power users to build custom pipelines and utilities on top of it
directly in TypeScript, without needing to implement the pipelines in native
code.

The higher-level API is what most users will interact with. The pipelines there
are implemented as orchestration layers on top of the lower-level API which only
exposes crude bindings to the native code.

## Structure

```text
cpp                         │     src
├── core                    │     ├── core
│   ├── install.h           │     │   ├── model.ts
│   ├── dtype.h             │     │   ├── modelSchema.ts
│   ├── model.h             │     │   ├── runtime.ts
│   ├── tensor.h            │     │   └── tensor.ts
│   └── ...                 │     ├── extensions
└── extensions              │     │   ├── math.ts
│   ├── math                │     │   ├── cv
│   │   ├── install.h       │     │   │   ├── image.ts
│   │   ├── operations.h    │     │   │   ├── ops
│   │   └── ...             │     │   │   │   ├── image.ts
│   ├── cv                  │     │   │   │   └── ...
│   │   ├── install.h       │     │   │   └── tasks   
│   │   ├── image_ops.h     │     │   │       ├── classification.ts
│   │   └── ...             │     │   │       └── ...
│   ├── llm                 │     │   ├── llm
│   │   └── ...             │     │   │   └── ...
│   └── speech              │     │   └── speech
│       └── ...             │     │       └── ...
├── MyLib.h                 │     ├── hooks
└── MyLib.cpp               │     │   ├── useClassifier.ts
                            │     │   └── ...
                            │     ├── native
                            │     │   └── ...
                            │     ├── index.ts
                            │     ├── models.ts
                            │     ├── constants.ts
                            │     └── utils.ts
                            │     
                            │     
```

As illustrated above, both the native (`cpp/`) and TypeScript (`src/`) layers
mirror each other symmetrically, strictly dividing responsibilities into
**Core** and **Extensions**.

- **Core (`cpp/core/`, `src/core/`)**: Implements the absolute bare minimum
  required for the **lower-level API**. It provides the foundational bindings
  for ExecuTorch: primitives for `Tensor` and `Model` management alongside the
  runtime execution logic. It intentionally lacks any task-specific code.
- **Extensions (`cpp/extensions/`, `src/extensions/`)**: Houses all
  domain-specific logic, categorized by field (e.g., `cv`, `math`, `llm`,
  `speech`). This framework relies heavily on **modularity**. An extension
  contains both its specific **lower-level APIs** (like C++ image processing
  operations for computer vision) and **higher-level APIs** (complete,
  ready-to-use task pipelines located in `src/extensions/<domain>/tasks/`).

For standard use cases, the primary entry points to the library are the React
hooks located in `src/hooks/` (e.g., `useClassifier`, `useDetector`). These
serve as accessible, lightweight React wrappers built squarely on top of the
generic task pipelines defined within the extensions.

### Lower-level API

The lower-level API provides direct access to the underlying ExecuTorch
capabilities and primitives for manipulating Tensors and Models. Its design
focuses on performance, modularity, and explicit control, revolving around the
following concepts:

1. **Modular JSI Injection** Unlike `react-native-executorch`—which forcefully
   injects dozens of isolated functions (`global.loadClassification`,
   `global.loadObjectDetection`, etc.) straight onto the global object—this PoC
   exposes a single namespaced global root: `global.__mylib_jsi__`. The C++
   bindings are entirely modular; this root object is assembled piecewise by
   invoking individual `install()` functions for the core module and any
   included extensions (e.g., `mylib::core::install`,
   `mylib::extensions::cv::install`). Furthermore, the native layer deliberately
   exposes standalone functions rather than heavy class instances (with the
   exception of `Model` and `Tensor` HostObjects). To preserve type safety
   without building massive monolithic interfaces, domain files like
   [src/extensions/math.ts](src/extensions/math.ts) simply import this global
   object and wrap the native calls in thin, exported TypeScript functions.

2. **Types, Modularity, and Schema** `Tensor` and `Model` primitives (defined in
   [src/core/tensor.ts](src/core/tensor.ts) and
   [src/core/model.ts](src/core/model.ts)) are natively implemented as JSI
   `HostObjects`. On the JS engine side, they represent nothing more than thin,
   explicitly tracked pointers to C++ heap memory. However, the lower-level API
   is more than just JSI bindings. It actively encapsulates required domain
   typings (e.g., establishing `BoundingBox` type for Computer Vision). More
   importantly, it enforces strict algebraic bounds checking through features
   like `modelSchema.ts`. Instead of blindly throwing `.pte` parameters at the
   ExecuTorch runtime, logic like `validateModelSchema` verifies that models
   match symbolic shape layouts (e.g., `SymbolicTensor('float32', [1, 3, 'H',
   'W'])`) before ever running them.

3. **Software Mansion Worklets Compatibility (`react-native-worklets`)**
   'react-native-executorch' relies on custom, hand-written C++ thread pooling.
   Instead, this PoC shifts threading mechanics entirely onto Software Mansion's
   `react-native-worklets`. Crucially, the underlying `Tensor` and `Model`
   `HostObjects` are fully thread-safe, making them freely shareable across
   JavaScript realms. Because all Native JSI wrappers are decorated with a
   `'worklet';` directive, these bindings are inherently portable. They can be
   immediately dumped onto background threads, or invoked seamlessly inside
   high-intensity `react-native-vision-camera` frame processors.

4. **Fine-grained Memory Control and Imperative Execution** At the core of the
   lower-level API's performance is manual memory tracking. Memory is always
   explicitly allocated from the TS layer—never implicitly via a native C++
   module return. Native module operations utilize strict imperative patterns
   (`fn(src, dst)`). Because the caller explicitly passes in an existing
   destination tensor (`dst`) instead of the function creating and returning a
   new one, developers can allocate memory once and reuse it repeatedly across
   continuous processing loops.

   However, chaining imperative `f(a,b); g(b,c); h(c,d);` becomes quickly
   unreadable. The API mitigates this using expressive `.through` and
   `.throughIf` syntaxes natively bound to the `Tensor` object.

   *A note on disposal:* While JSI `HostObjects` are technically tracked by the
   JavaScript Garbage Collector, relying on automatic cleanup is strongly
   discouraged. The JS GC only tracks the lightweight wrapper, completely
   unaware of the massive C++ memory block (often hundreds of megabytes for
   models) it holds underneath. Therefore, explicit `.dispose()` calls are
   highly recommended to prevent native memory bloat.

   *An example of cleanly allocating, unwrapping, dispatching, and disposing
   explicit memory:*

   ```ts
    // Explicitly allocate expected output/intermediate tensors on the TS layer
    const tensors = [
      tensor('float32', outShape),
      tensor('float32', [3, targetH, targetW]),
      tensor('float32', [targetH, targetW, 3]),
      tensor('uint8', [targetH, targetW, 3]),
      tensor('uint8', [targetH, targetW, 4]),
      tensor('uint8', [height, width, 4]),
    ] as const;

    // Unwrap for strict, strongly-typed usage
    const [tOutput, tReshape, tChanLast, tUint8, tRgba, tResize] = tensors;

    try {
      // Execute chained operations acting solely on the recycled memory destinations
      const data = tOutput
        .copyTo(tReshape)
        .through(toChannelsLast, tChanLast)
        .through(normalize, tUint8, { alpha: 255.0, beta: 0.0 })
        .through(cvtColor, tRgba, 'RGB2RGBA')
        .through(resize, tResize, { mode: 'stretch' })
        .getData(new Uint8Array(tResize.numel));
      // Use `data`...
    } finally {
      // Clean batch disposal (Optional but highly recommended)
      tensors.forEach((t) => t.dispose());
    }
   ```

### Higher-level API

While the lower-level API is aimed at maximum control and performance, the
**Higher-level API** focuses on developer experience, abstraction, and React
lifecycle integration. It represents the task orchestration logic built entirely
on top of the primitives provided by the lower layer.

The design revolves around five major conceptual blocks:

1. **Standalone Task Pipelines (`src/extensions/<domain>/tasks`)** In contrast
   to `react-native-executorch` where adding or modifying a task required diving
   deep into opaque C++ structures, task pipelines in this PoC are authored
   purely in TypeScript.

   A core design choice here is the deliberate **omission of classes**. To
   manage state without boilerplate, tasks are defined as single `create<Task>`
   factory functions (e.g., `createClassifier()`). These functions allocate
   memory, validate tensor layout schemas, and return a clean closure bundle
   exposing exactly what the user needs:

   - An asynchronous execution function (e.g., `classify()`).
   - A mandatory `dispose()` method allowing the consumer to gracefully release
     the internal state and native memory blocks.

   This closure pattern naturally encapsulates the internal state (hiding the
   pre-allocated tensors and the model instance from the outside) while keeping
   the implementation confined to highly readable, single files.

   *Below are direct excerpts from the real implementation of classification
   task pipeline.*

    ```ts
    // src/extensions/cv/tasks/classification.ts
    export type ClassifierOptions<L> = ImagePreprocessorOptions & {
      readonly labels: readonly L[];
    };
    export type ClassifierModel<L> = {
      readonly modelPath: string;
      readonly classifierOpts: ClassifierOptions<L>;
    };
    export type Classification<L> = {
      readonly label: L;
      readonly confidence: number;
    };

    export async function createClassifier<L>(
      config: ClassifierModel<L>,
      runtime?: WorkletRuntime,
    ): Promise<{
      dispose: () => void;
      classify: (input: ImageBuffer) => Promise<Classification<L>[]>;
    }> {
      const { modelPath, classifierOpts } = config;
      const model = await wrapAsync(loadModel, runtime)(modelPath);

      const meta = validateModelSchema(
        model,
        'forward',
        [SymbolicTensor('float32', [1, 3, 'H', 'W'], [3, 'H', 'W'])],
        [SymbolicTensor('float32', [1, 'N'], ['N'])],
      );
      const inpShape = meta.inputTensorMeta[0]!.shape;
      const outShape = meta.outputTensorMeta[0]!.shape;

      const tensors = [
        tensor('float32', outShape), //
        tensor('float32', outShape),
      ] as const;

      const [tLogits, tProbas] = tensors;
      const preprocessor = createImagePreprocessor(classifierOpts, inpShape);

      const dispose = () => {
        preprocessor.dispose();
        tensors.forEach((t) => t.dispose());
        model.dispose();
      };

      const classify = async (input: ImageBuffer): Promise<Classification<L>[]> => {
        const tInput = preprocessor.process(input);
        await wrapAsync(() => {
          'worklet';
          model.execute('forward', [tInput], [tLogits]);
        }, runtime)();

        const probas = tLogits
          .through(softmax, tProbas) //
          .getData(new Float32Array(tProbas.numel));

        return Array.from(probas)
          .map((confidence, index) => ({ confidence, label: classifierOpts.labels[index]! }))
          .sort((a, b) => b.confidence - a.confidence);
      };

      return { classify, dispose };
    }
    ```

2. **React Hooks Integration (`src/hooks`)** The factory closures map perfectly
   to React's lifecycle through a single, generic `useModel` hook. Because
   `useModel` generically handles all the complex asynchronous initialization
   and cleanup, task-specific hooks like `useClassifier` or `useStyleTransfer`
   are incredibly simple to write—they are essentially thin wrappers that just
   pass the task factory to `useModel`. By doing so, `useModel` automatically:
   - Orchestrates optional `.pte` model downloading via `useModelDownload`.
     *(Note: This PoC uses a very crude, temporary downloading mechanism. The
     standalone `resource-fetcher` package from the original
     `react-native-executorch` repository is a vastly superior solution and the
     ideal production candidate).*
   - Asynchronously creates the task bundle off the main thread.
   - Routes the bundle's `dispose()` function into the `useEffect` cleanup
     return, guaranteeing that when a React component unmounts, the underlying
     multi-megabyte C++ tensor memory is freed immediately.

    *Below are direct excerpts from the real implementation of the
    `useClassifier` hook.*

    ```ts
    // src/hooks/useClassifier.ts
    export function useClassifier<L>(config: ClassifierModel<L>, options?: { preventLoad?: boolean }) {
      const { localPath, downloadProgress, downloadError } = useModelDownload(
        config.modelPath,
        options?.preventLoad,
      );
      const { model, error } = useModel(
        createClassifier<L>,
        localPath ? { ...config, modelPath: localPath } : null,
        [localPath],
      );

      return {
        isReady: !!model,
        error: downloadError || error,
        downloadProgress,
        localPath,
        classify: model?.classify,
      };
    }
    ```

3. **Task Design Degrees of Freedom** When authoring a new domain-specific task,
   this architecture provides two clear avenues for handling arbitrary
   complexity:
   - **Delegating to the Model Graph (Schema Enforcement):** Whenever possible,
     we enforce a specific schema during model export to offload operations
     directly into the ExecuTorch graph, greatly simplifying the host-side TS
     infrastructure.
   - **Custom C++ Operations via JSI:** If the pre/post-processing *cannot* be
     represented as a straight-line program (e.g., dynamic control flow, complex
     heuristic loops, or unstructured state manipulations), then we have to
     offload it to a custom C++ implementation. Because extending JSI is trivial
     under this partitioned architecture, we simply implement the complex logic
     in C++ and expose it through our domain-specific JSI bridge (e.g., the `cv`
     or `llm` namespace), orchestrating the final control flow cleanly in
     TypeScript.

4. **Extensibility to Other Domains (LLMs, Speech, etc.)** Looking at the entire
   repository, expanding this architecture beyond Computer Vision to domains
   like LLMs or Speech is remarkably straightforward. Crucially, because native
   code is strictly compartmentalized, we aren't forced to shoehorn every domain
   into the generic `Model` primitive. For instance, ExecuTorch provides
   specialized, stateful native runners for LLMs. We can easily wrap these
   specialized runners inside `cpp/extensions/llm/install.cpp` and expose them
   to TypeScript.

   From there, the closure pattern handles the statefulness elegantly: the
   runner's internal context (like a KV cache) resides safely inside the
   closure's hidden state. For streaming tasks—such as autoregressive text
   generation from an LLM— TypeScript orchestrations leveraging SWM Worklets
   excel. A simple `while` loop can run `generateToken()` synchronously on a
   background thread, yielding tokens incrementally to the UI without blocking
   the main thread or bloating the app binary.

5. **Single Source of Truth for Model Metadata (`src/models.ts` &
   `src/constants.ts`)** Pre-exported models and their exact input constraints
   (e.g., resizing strategies, mean/std normalizations, label maps) are
   meticulously defined as strongly typed TypeScript objects in `models.ts`
   alongside their URL paths. Note that model configurations are centrally
   exported using an intuitive, hierarchical namespace (e.g.,
   `models.classification.EFFICIENTNET_V2_S`, following the generic pattern
   `models.<task>.<MODEL_NAME>.[<PREC_BACKEND>]`).

   Crucially, these objects are typed using explicit domain interfaces (such as
   `ClassifierModel<T>` or `DetectorModel`) exported directly from their
   corresponding pipeline files in `src/extensions/<domain>/tasks`. This strict
   bonding guarantees that when a user passes a configuration into a task
   factory (e.g., `createClassifier(model: ClassifierModel<T>)`), the TypeScript
   compiler statically ensures the pipeline receives the specific schema it was
   designed to process. By pairing the raw `.pte` binary with the exact
   preprocessing and label constants it was trained against, these files act as
   a foolproof single source of truth, stripping away runtime errors and
   developer guesswork.

## Comparison with `react-native-executorch`

To evaluate the design, we compared the core architecture and vision models
(Classification, Style Transfer, Semantic Segmentation, and Object Detection)
against the original library, omitting features unique to
`react-native-executorch` (like LLMs or Audio) for a fair baseline.

### Quantitative Metrics

| Metric | `react-native-mylib` (PoC) | `react-native-executorch` | Difference |
| :--- | :---: | :---: | :---: |
| **Total Files** | 52 | 84 | **-38%** |
| **Total Lines of Code (NLOC)** | 3,932 | 6,741 | **-41%** |
| **Model-Specific NLOC** | 1,219 | 3,286 | **-63%** |
| **Avg. Cyclomatic Complexity** | 8.56 | 7.56 | Balanced (<10) |

- **Massive Code Reduction:** `react-native-executorch` contains roughly **70%
  more code overall**.
- **Where the Fat Was Cut:** The core difference lies in model implementation.
  By utilizing generic Tensor operations in TypeScript via Worklets, this PoC
  cuts model-specific code by **63%**, replacing heavy, custom C++ pipeline
  classes with lightweight TS orchestration.

### Developer Ergonomics

- **Adding New Models**: In this PFoC, introducing a new model family (e.g.,
pose estimation) means writing a single TypeScript file that chains together
existing Tensor primitives. In react-native-executorch, it requires writing C++
headers, implementation files, and JSI host object bindings.

- **Iteration Speed**: Developers can log intermediate Tensor states and tweak
pipeline logic on the fly without ever recompiling C++ binaries.

### Performance

While the model inference time remains identical to the original library (as
both run the same underlying ExecuTorch runtime), this library achieves **up to
a 2x overall pipeline speedup** due to how memory is managed before and after
inference.

*Try it yourself*. We provide a playground in the `/example` application. It
contains:

- **Gallery**: An interactive screen where you can select an image, run any of
  the supported pipelines (Classification, Style Transfer, Object Detection, or
  Semantic Segmentation), inspect the visual outputs, and see precise
  measurements of the execution times.

## What's next?

You decide! This PoC is meant to be a starting point for discussion and
exploration.

![ExecuTorch is coming for you!](media/logo-hero-flame.png)
