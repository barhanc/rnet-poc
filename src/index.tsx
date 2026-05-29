import { createWorkletRuntime, runOnRuntimeAsync } from "react-native-worklets";

const globalObj = globalThis as any;
if (!globalObj.__myModule__) require("./NativeMyLib").default.install();

const jsi = globalObj.__myModule__;

// Create our dedicated background thread for ExecuTorch
const executorchRuntime = createWorkletRuntime("executorch-thread");

export type DType = "float32" | "uint8" | "int32";

export interface NativeTensor {
  // Opaque HostObject reference
}

export interface NativeModel {
  // Opaque HostObject reference
}

export interface MethodDiagnostic {
  methodNames: string[];
  methodMeta: unknown;
}

export const MyLib = {
  loadModel: (path: string): NativeModel => jsi.loadModel(path),
  disposeModel: (model: NativeModel): void => jsi.disposeModel(model),

  getModelMethodNames: (model: NativeModel): string[] =>
    jsi.getModelMethodNames(model),
  getModelMethodMeta: (model: NativeModel, methodName: string): unknown =>
    jsi.getModelMethodMeta(model, methodName),
  executeModelMethod: (
    model: NativeModel,
    methodName: string,
    ...args: (NativeTensor | number | boolean | null)[]
  ): NativeTensor[] => jsi.executeModelMethod(model, methodName, ...args),

  createTensor: (shape: number[], dtype: DType): NativeTensor =>
    jsi.createTensor(shape, dtype),
  setTensorFromTypedArray: (
    tensor: NativeTensor,
    data: ArrayBufferView,
  ): void => jsi.setTensorFromTypedArray(tensor, data),
  getTypedArrayFromTensor: (
    tensor: NativeTensor,
  ): Float32Array | Uint8Array | Int32Array =>
    jsi.getTypedArrayFromTensor(tensor),

  /**
   * Runs the model entirely on the background Worklet thread.
   */
  runInferenceAsync: async (
    model: NativeModel,
    methodName: string,
    ...args: (NativeTensor | number | boolean)[]
  ): Promise<NativeTensor[]> => {
    return runOnRuntimeAsync(
      executorchRuntime,
      (m, method, ...a) => {
        "worklet";
        return jsi.executeModelMethod(m, method, ...a);
      },
      model,
      methodName,
      ...args,
    );
  },

  diagnoseMethod: (
    model: NativeModel,
    methodName: string,
  ): MethodDiagnostic => ({
    methodNames: jsi.getModelMethodNames(model),
    methodMeta: jsi.getModelMethodMeta(model, methodName),
  }),
};
