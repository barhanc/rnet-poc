import {
  runOnRuntimeAsync,
  createWorkletRuntime,
  type WorkletRuntime,
} from "react-native-worklets";
import { mylibJsi } from "../native/bridge";
import { Tensor, type DType } from "./Tensor";

export type ModelInput = Tensor | number | boolean | null;
export type ModelOutput = Tensor | number | boolean | null;
export type ModelHostObject = { path: string };

export type TensorMeta = {
  name: string;
  ndim: number;
  nbytes: number;
  dtype: DType;
  shape: number[];
};

export type ExecuTorchTag =
  | "None"
  | "Tensor"
  | "Int"
  | "Double"
  | "Bool"
  | "String"
  | "ListBool"
  | "ListDouble"
  | "ListInt"
  | "ListTensor";

export type ModelMethodMeta = {
  name: string;
  numInputs: number;
  numOutputs: number;
  inputTags: ExecuTorchTag[];
  outputTags: ExecuTorchTag[];
  usesBackend: Map<string, boolean>;
  inputTensorMeta: TensorMeta[];
  outputTensorMeta: TensorMeta[];
};

let mylibWorkletRuntime: WorkletRuntime | null = null;

function getWorkletRuntime() {
  if (!mylibWorkletRuntime) mylibWorkletRuntime = createWorkletRuntime("executorch-thread");
  return mylibWorkletRuntime;
}

export class Model {
  private _hostObject: ModelHostObject;

  private constructor(hostObject: ModelHostObject) {
    this._hostObject = hostObject;
  }

  get path(): string {
    return this._hostObject.path;
  }

  get methodNames(): string[] {
    return mylibJsi.getModelMethodNames(this._hostObject);
  }

  getMethodMeta(methodName: string): ModelMethodMeta {
    return mylibJsi.getModelMethodMeta(this._hostObject, methodName);
  }

  dispose(): void {
    mylibJsi.disposeModel(this._hostObject);
  }

  static load(modelPath: string): Model {
    const nativeModel = mylibJsi.loadModel(modelPath);
    return new Model(nativeModel);
  }

  static async loadAsync(modelPath: string): Promise<Model> {
    const result = await runOnRuntimeAsync(
      getWorkletRuntime(),
      (path: string) => {
        "worklet";
        try {
          return { ok: true, value: mylibJsi.loadModel(path) };
        } catch (error) {
          return { ok: false, error: error instanceof Error ? error.message : String(error) };
        }
      },
      modelPath,
    );
    if (!result.ok) throw new Error(result.error);
    return new Model(result.value!);
  }

  execute(methodName: string, inputs: ModelInput[], tensorOutputs?: Tensor[]): ModelOutput[] {
    const meta = this.getMethodMeta(methodName);
    if (!tensorOutputs) {
      tensorOutputs = meta.outputTensorMeta.map((m) => Tensor.fromEmpty(m.shape, m.dtype));
    }

    const args = inputs.map((input) => (input instanceof Tensor ? input.hostObject : input));
    const buffers = tensorOutputs.map((tensor) => tensor.hostObject);

    const result = mylibJsi.executeModelMethod(this._hostObject, methodName, args, buffers);

    let tensorIdx = 0;
    return result.map((out: any, idx: number) =>
      meta.outputTags[idx] === "Tensor" ? tensorOutputs[tensorIdx++] : out,
    );
  }

  async executeAsync(
    methodName: string,
    inputs: ModelInput[],
    tensorOutputs?: Tensor[],
  ): Promise<ModelOutput[]> {
    const meta = this.getMethodMeta(methodName);
    if (!tensorOutputs) {
      tensorOutputs = meta.outputTensorMeta.map((m) => Tensor.fromEmpty(m.shape, m.dtype));
    }

    const result = await runOnRuntimeAsync(
      getWorkletRuntime(),
      (model: ModelHostObject, name: string, args: any[], buffers: any[]) => {
        "worklet";
        try {
          return { ok: true, value: mylibJsi.executeModelMethod(model, name, args, buffers) };
        } catch (error) {
          return { ok: false, error: error instanceof Error ? error.message : String(error) };
        }
      },
      this._hostObject,
      methodName,
      inputs.map((input) => (input instanceof Tensor ? input.hostObject : input)),
      tensorOutputs.map((tensor) => tensor.hostObject),
    );

    if (!result.ok) {
      throw new Error(result.error);
    }

    let tensorIdx = 0;
    return result.value.map((out: any, idx: number) =>
      meta.outputTags[idx] === "Tensor" ? tensorOutputs[tensorIdx++] : out,
    );
  }
}
