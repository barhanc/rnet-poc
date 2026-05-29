import { runOnRuntimeAsync } from "react-native-worklets";
import { mylibWorkletRuntime, mylibJsi } from "./bridge";
import { Tensor } from "./Tensor";
import type { ModelHostObject, ModelInput, ModelMethodMeta, ModelOutput } from "./types";

export class Model {
  private _hostObject: ModelHostObject;

  private constructor(nativeModel: ModelHostObject) {
    this._hostObject = nativeModel;
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

  static async load(modelPath: string): Promise<Model> {
    const result = await runOnRuntimeAsync(
      mylibWorkletRuntime,
      (path: string) => {
        "worklet";
        try {
          for (let i = 0; i < 500_000_000; i++) {}
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

  async execute(methodName: string, ...inputs: ModelInput[]): Promise<ModelOutput[]> {
    const result = await runOnRuntimeAsync(
      mylibWorkletRuntime,
      (model: ModelHostObject, name: string, ...args: any[]) => {
        "worklet";
        try {
          return { ok: true, value: mylibJsi.executeModelMethod(model, name, ...args) };
        } catch (error) {
          return { ok: false, error: error instanceof Error ? error.message : String(error) };
        }
      },
      this._hostObject,
      methodName,
      ...inputs.map((input) => (input instanceof Tensor ? input.hostObject : input)),
    );
    if (!result.ok) throw new Error(result.error);

    const meta = this.getMethodMeta(methodName);
    const outputs = result.value.map((out: any, idx: number) =>
      meta.outputTags[idx] === "Tensor" ? Tensor.fromHostObject(out) : out,
    );
    return outputs;
  }
}
