import { mylibJsi } from "./bridge";

export { Model } from "./Model";
export { Tensor } from "./Tensor";
export type { DType, TensorMeta, ModelMethodMeta, ModelInput, ModelOutput, ETTag } from "./types";

/**
 * Gets the list of backends registered in the ExecuTorch runtime.
 */
export function getExecuTorchRegisteredBackends(): string[] {
  try {
    return mylibJsi.getExecuTorchRegisteredBackends();
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    throw new Error(`Failed to get registered backends: ${message}`);
  }
}
