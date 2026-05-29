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

  getMethodNames(): string[] {
    return mylibJsi.getModelMethodNames(this._hostObject);
  }

  getMethodMeta(methodName: string): ModelMethodMeta {
    return mylibJsi.getModelMethodMeta(this._hostObject, methodName);
  }

  dispose(): void {
    mylibJsi.disposeModel(this._hostObject);
  }

  async load(modelPath: string): Promise<{ ok: boolean; value?: Model; error?: string }> {
    return runOnRuntimeAsync(
      mylibWorkletRuntime,
      (path: string) => {
        "worklet";
        try {
          return { ok: true, value: mylibJsi.loadModel(this._hostObject, path) };
        } catch (error) {
          return { ok: false, error: error instanceof Error ? error.message : String(error) };
        }
      },
      modelPath,
    );
  }

  async execute(
    methodName: string,
    ...inputs: ModelInput[]
  ): Promise<{ ok: boolean; value?: ModelOutput[]; error?: string }> {
    return runOnRuntimeAsync(
      mylibWorkletRuntime,
      (name: string, args: any) => {
        "worklet";
        try {
          return { ok: true, value: mylibJsi.executeModelMethod(this._hostObject, name, args) };
        } catch (error) {
          return { ok: false, error: error instanceof Error ? error.message : String(error) };
        }
      },
      methodName,
      inputs.map((input) => (input instanceof Tensor ? input._hostObject : input)),
    );
  }
}
