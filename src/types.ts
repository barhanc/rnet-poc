import type { Tensor } from "./Tensor";

export type DType = "float32" | "uint8" | "int32";
export type ModelInput = Tensor | number | boolean | null;
export type ModelOutput = Tensor | number | boolean | string | null | boolean[] | number[];

export interface ModelHostObject {
  path: string;
}

export interface TensorHostObject {
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

export type ETTag =
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
  inputTags: ETTag[];
  outputTags: ETTag[];
  usesBackend: Map<string, boolean>;
  inputTensorMeta: TensorMeta[];
  outputTensorMeta: TensorMeta[];
};
