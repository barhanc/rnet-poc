---
id: verify_and_build
name: Compilation, Verification & Troubleshooting
description: Workflows for rebuilding typescript and native C++ environments and fixing common runtime JSI issues.
scope: general development, example/
---

# Skill: Verification, Compilation & Troubleshooting

Use this guide to compile, test, verify, and troubleshoot your modifications to the library (both JS/TS and native C++).

---

## 🛠️ Verification & Compilation Workflows

### 1. Build and Typecheck TypeScript
To check types and compile the TypeScript source code:
* **Verify Types (Fast)**:
  ```bash
  yarn typecheck
  ```
* **Build Bundles**:
  ```bash
  yarn prepare
  ```
  *This cleans target directories and builds modules into the `lib/` directory using `react-native-builder-bob`.*
* **When**: After adding/updating TypeScript files under `src/`.
* **Verification**: Ensure no compiler or type-checking errors are raised.

### 2. Native Rebuilding

#### iOS Development
Whenever you modify native files under `cpp/` or update `.podspec`:
```bash
cd example/ios && pod install && cd ../..
```
* **Rebuilding**: Re-run the app from the root workspace or from the `example/` folder:
  ```bash
  # From root:
  yarn example ios
  # OR from example/ directory:
  yarn ios
  ```
* **Clean Build**: If files aren't being picked up:
  * In Xcode: `Product` > `Clean Build Folder` (Cmd + Shift + K).

#### Android Development
Android compiles C++ source files on-demand during application builds:
```bash
# From root:
yarn example android
# OR from example/ directory:
yarn android
```
* **Clean Build**: If caching issues occur:
  ```bash
  cd android && ./gradlew clean && cd ..
  ```

---

## 🔍 Debugging & Log Access

### 1. TypeScript & Worklet Logging
Logging inside worklets (functions annotated with `'worklet';`) behaves slightly differently as they run on a separate runtime thread.
* Use standard `console.log(...)` inside worklets.
* **Caution**: Worklet logs are piped back to the JS console, but passing complex circular objects to `console.log` from a worklet can fail or freeze. Stringify or log specific primitive properties instead.

### 2. Native C++ Debugging
Native JSI crashes, standard outputs, or ExecuTorch errors can be caught in the native log streams:
* **iOS**: Watch outputs directly inside the Xcode Console while running the example target.
* **Android**: Use Android Studio's logcat or run:
  ```bash
  adb logcat *:S ReactNative:V ReactNativeJS:V mylib:V
  ```

---

## 🚦 Common JSI Troubleshooting

### Symptom: `Error: JSI global object '__mylib_jsi__' is not registered.`
* **Cause 1**: The native code wasn't linked or JSI installation failed during loading.
  * *Fix*: Verify that you have registered your new extension/module in [cpp/MyLib.cpp](../cpp/MyLib.cpp).
  * *Fix*: Re-run `pod install` (for iOS) or perform a Gradle sync/clean (for Android) to ensure your C++ changes compiled.
* **Cause 2**: The bridge did not invoke the native installation.
  * *Fix*: Ensure the JS entrypoint triggers the load (see [src/native/bridge.ts](../src/native/bridge.ts)).

### Symptom: `TypeError: Cannot read property '<extension>' of undefined`
* **Cause**: You added an extension in C++ but forgot to register it in your TypeScript bridge bindings.
  * *Fix*: Check that `module.setProperty(rt, "<extension>", subModule)` is called in your native `install.cpp` and that your TS wrapper exports and references it under `mylibJsi.<extension>`.

---

## 📂 Model Hosting & Download Caching

This project does **not** bundle local `.pte` model files inside the React Native application package. Instead, the standard workflow is:

1. **Upload to Hugging Face**:
   * Upload compiled/exported `.pte` models directly to our Hugging Face repository bucket.
   * **Naming Convention**: All `.pte` files must follow the strict name contract:
     `modelname_optionalsize_backend_precision.pte`
     * *Example*: `efficientnet_v2_s_xnnpack_int8.pte`
     * *Example*: `style_transfer_candy_xnnpack_fp32.pte`

2. **Define in Models Manifest** ([src/models.ts](../src/models.ts)):
   * **Single Export Registry Rule**: Define the new model configuration as an internal (private) `const` inside `models.ts`. **Do not** export individual model configuration constants. Instead, expose them solely by registering them under the appropriate category nested inside the main `models` registry object, which is the only exported symbol from `models.ts`.
   * *Example*:
     ```typescript
     // Defined internally (private)
     const MY_MODEL_XNNPACK_FP32: MyTaskModel = { ... };

     // Registered in the exported models object
     export const models = {
       ...
       myTask: {
         MY_MODEL: {
           XNNPACK_FP32: MY_MODEL_XNNPACK_FP32,
         }
       }
     };
     ```

3. **Runtime Downloading & Caching**:
   * The custom hook `useModelDownload(config.modelPath)` handles checks automatically:
     * If the model file is not already in the device cache directory, it downloads it from the Hugging Face URL to `RNFS.CachesDirectoryPath`.
     * If the model file exists in the cache, it bypasses the download and returns the cached local path immediately to load the model.


