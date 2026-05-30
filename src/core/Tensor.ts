import { mylibJsi } from "../native/bridge";

declare const tensorBrand: unique symbol;

export type DType = "float32" | "uint8" | "int32";
export type Tensor = {
  readonly dtype: DType;
  readonly shape: number[];
  readonly numel: number;
  dispose(): void;
  reshape(dst: Tensor): Tensor;
  setData(src: Float32Array | Uint8Array | Int32Array): Tensor;
  getData<T extends Float32Array | Uint8Array | Int32Array>(dst: T): T;
  through<R, Args extends any[]>(fn: (t: Tensor, ...args: Args) => R, ...args: Args): R;

  /**
   * @internal
   * Prevents plain JS objects from being cast as Tensors. Tensors should only
   * be created via the `tensor` function exported from this module.
   */
  readonly [tensorBrand]: never;
};

export function tensor(
  dtype: DType,
  shape: number[],
  src?: Float32Array | Uint8Array | Int32Array,
): Tensor {
  "worklet";
  const t: Tensor = mylibJsi.createTensor(shape, dtype);
  if (src) t.setData(src);
  return t;
}
