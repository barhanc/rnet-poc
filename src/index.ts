import { createWorkletRuntime, runOnRuntimeAsync } from "react-native-worklets";

const globalObj = globalThis as any;
if (!globalObj.__executorch_jsi__) {
  require("./NativeMyLib").default.install();
}
const _jsi = globalObj.__executorch_jsi__;
const _executorchRuntime = createWorkletRuntime("executorch-thread");

export type DType = "float32" | "uint8" | "int32";
export type ETValue = NativeTensor | number | boolean | null;

export interface NativeModel {
  path: string;
}

export interface NativeTensor {
  dtype: DType;
  shape: number[];
}

export type TensorMeta = {
  name: string;
  ndim: number;
  nbytes: number;
  dtype: DType;
  shape: number[];
};

export type ModelMethodMeta = {
  name: string;
  numInputs: number;
  numOutputs: number;
  inputTags: string[];
  outputTags: string[];
  usesBackend: Map<string, boolean>;
  inputTensorMeta: TensorMeta[];
  outputTensorMeta: TensorMeta[];
};

export async function loadModel(path: string): Promise<NativeModel> {
  return runOnRuntimeAsync(
    _executorchRuntime,
    (path) => {
      "worklet";
      try {
        return _jsi.loadModel(path);
      } catch (e: any) {
        console.error("Model loading error:", e.message);
      }
    },
    path,
  );
}

export function disposeModel(model: NativeModel): void {
  _jsi.disposeModel(model);
}

export async function executeModelMethod(
  model: NativeModel,
  methodName: string,
  ...args: ETValue[]
): Promise<ETValue[]> {
  return runOnRuntimeAsync(
    _executorchRuntime,
    (rawModel, methodName, ...args) => {
      "worklet";
      try {
        for (let i = 0; i < 100_000_000; i++) {}
        return _jsi.executeModelMethod(rawModel, methodName, ...args);
      } catch (e: any) {
        console.error("Inference error:", e.message);
      }
    },
    model,
    methodName,
    ...args,
  );
}

export function getModelMethodNames(model: NativeModel): string[] {
  return _jsi.getModelMethodNames(model);
}

export function getModelMethodMeta(
  model: NativeModel,
  methodName: string,
): ModelMethodMeta {
  return _jsi.getModelMethodMeta(model, methodName);
}

export function createTensor(shape: number[], dtype: DType): NativeTensor {
  return _jsi.createTensor(shape, dtype);
}

export function setTensorFromTypedArray(
  tensor: NativeTensor,
  data: ArrayBufferView,
): void {
  _jsi.setTensorFromTypedArray(tensor, data);
}

export function setTypedArrayFromTensor(
  data: ArrayBufferView,
  tensor: NativeTensor,
): void {
  _jsi.setTypedArrayFromTensor(data, tensor);
}

export function disposeTensor(tensor: NativeTensor): void {
  _jsi.disposeTensor(tensor);
}

export function getExecuTorchRegisteredBackends(): string[] {
  return _jsi.getExecuTorchRegisteredBackends();
}

export async function runInferenceAsync(
  model: NativeModel,
  methodName: string,
  ...args: ETValue[]
): Promise<NativeTensor[]> {
  return executeModelMethod(model, methodName, ...args) as Promise<
    NativeTensor[]
  >;
}

export const MyLib = {
  loadModel,
  disposeModel,
  executeModelMethod,
  getModelMethodNames,
  getModelMethodMeta,
  createTensor,
  disposeTensor,
  setTensorFromTypedArray,
  setTypedArrayFromTensor,
  getExecuTorchRegisteredBackends,
  runInferenceAsync,
};
