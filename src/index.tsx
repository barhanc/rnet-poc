import { createWorkletRuntime, runOnRuntimeAsync } from "react-native-worklets";

if (!global.__myModule__) require("./NativeMyLib").default.install();

const backgroundRuntime = createWorkletRuntime({ name: "executorch-bg" });
const jsiModule = globalThis.__myModule__!;

export function getModelMethodNames(model: any): string[] {
  return jsiModule.getModelMethodNames(model);
}

export function getModelMethodMeta(model: any, methodName: string): any {
  return jsiModule.getModelMethodMeta(model, methodName);
}

export async function loadModel(path: string): Promise<any> {
  return runOnRuntimeAsync(
    backgroundRuntime,
    (p) => {
      "worklet";
      try {
        return jsiModule.loadModel(p);
      } catch (e: any) {
        console.log("Error loading model:", e.message);
        throw e.message || "Model loading failed";
      }
    },
    path,
  );
}

export async function executeModel(
  model: any,
  methodName: string,
  ...args: any[]
): Promise<number> {
  return runOnRuntimeAsync(
    backgroundRuntime,
    (m, name, a) => {
      "worklet";
      try {
        return jsiModule.executeModel(m, name, ...a);
      } catch (e: any) {
        console.log(`Error executing method ${name}:`, e.message);
        throw e.message || "Execution failed";
      }
    },
    model,
    methodName,
    args,
  );
}

export async function disposeModel(model: any): Promise<void> {
  return runOnRuntimeAsync(
    backgroundRuntime,
    (m) => {
      "worklet";
      try {
        jsiModule.disposeModel(m);
      } catch (e: any) {
        console.log(`Error disposing model:`, e.message);
        throw e.message || "Dispose failed";
      }
    },
    model,
  );
}
