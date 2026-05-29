import { mylibJsi } from "./bridge";
import type { DType, TensorHostObject } from "./types";

export class Tensor {
  private _hostObject: TensorHostObject;

  private constructor(hostObject: TensorHostObject) {
    this._hostObject = hostObject;
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

  /**
   * @internal
   * Factory method to wrap an existing native JSI host object.
   */
  static fromHostObject(hostObject: TensorHostObject): Tensor {
    return new Tensor(hostObject);
  }

  static fromTypedArray(data: Float32Array | Uint8Array | Int32Array, shape: number[]): Tensor {
    let dtype: DType;
    if (data instanceof Float32Array) {
      dtype = "float32";
    } else if (data instanceof Uint8Array) {
      dtype = "uint8";
    } else if (data instanceof Int32Array) {
      dtype = "int32";
    } else {
      throw new Error("Unsupported typed array type");
    }
    const hostObject = mylibJsi.createTensor(dtype, shape);
    mylibJsi.setTensorFromTypedArray(hostObject, data);
    return new Tensor(hostObject);
  }

  toTypedArray(): Float32Array | Uint8Array | Int32Array {
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
