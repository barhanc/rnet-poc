import { mylibJsi } from "./bridge";
import type { DType, TensorHostObject } from "./types";

export class Tensor {
  private _hostObject: TensorHostObject;

  private constructor(hostObject: TensorHostObject) {
    this._hostObject = hostObject;
  }

  /* @internal */
  get hostObject(): TensorHostObject {
    return this._hostObject;
  }

  get dtype(): DType {
    return this._hostObject.dtype;
  }

  get shape(): number[] {
    return this._hostObject.shape;
  }

  get numel(): number {
    let numel = 1;
    for (const dim of this.shape) {
      numel *= dim;
    }
    return numel;
  }

  dispose(): void {
    mylibJsi.disposeTensor(this._hostObject);
  }

  /* @internal */
  static fromHostObject(hostObject: TensorHostObject): Tensor {
    return new Tensor(hostObject);
  }

  static fromTypedArray(
    data: Float32Array | Uint8Array | Int32Array,
    shape: number[],
    dtype?: DType,
  ): Tensor {
    let resolvedDType: DType;

    if (dtype) {
      resolvedDType = dtype;
    } else {
      if (data instanceof Float32Array) {
        resolvedDType = "float32";
      } else if (data instanceof Uint8Array) {
        resolvedDType = "uint8";
      } else if (data instanceof Int32Array) {
        resolvedDType = "int32";
      } else {
        throw new Error("Unsupported typed array type");
      }
    }

    const hostObject = mylibJsi.createTensor(shape, resolvedDType);
    mylibJsi.setTensorFromTypedArray(hostObject, data);
    return new Tensor(hostObject);
  }

  toTypedArray<T extends Float32Array | Uint8Array | Int32Array>(target: T): T;
  toTypedArray(): Float32Array | Uint8Array | Int32Array;
  toTypedArray(dst?: any) {
    if (dst) {
      mylibJsi.setTypedArrayFromTensor(dst, this._hostObject);
      return dst;
    }

    let array: Float32Array | Uint8Array | Int32Array;
    switch (this.dtype) {
      case "float32":
        array = new Float32Array(this.numel);
        break;
      case "uint8":
        array = new Uint8Array(this.numel);
        break;
      case "int32":
        array = new Int32Array(this.numel);
        break;
      default:
        throw new Error(`Unsupported dtype: ${this.dtype}`);
    }

    mylibJsi.setTypedArrayFromTensor(array, this._hostObject);
    return array;
  }
}
