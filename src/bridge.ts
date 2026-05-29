import { createWorkletRuntime } from "react-native-worklets";

const globalObj = globalThis as any;

if (!globalObj.__mylib_jsi__) {
  const NativeMyLib = require("./NativeMyLib").default;
  if (NativeMyLib) NativeMyLib.install();
}

export const mylibJsi = globalObj.__mylib_jsi__;

if (!mylibJsi) {
  throw new Error("JSI global object '__mylib_jsi__' is not registered.");
}

export const mylibWorkletRuntime = createWorkletRuntime("executorch-thread");
