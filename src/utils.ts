import { mylibJsi } from "./bridge";

export function getRegisteredBackends(): string[] {
  return mylibJsi.getExecuTorchRegisteredBackends();
}
