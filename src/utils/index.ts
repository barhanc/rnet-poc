import { mylibJsi } from "../native/bridge";

export function getRegisteredBackends(): string[] {
  return mylibJsi.getExecuTorchRegisteredBackends();
}
