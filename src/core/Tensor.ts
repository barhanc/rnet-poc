import { mylibJsi } from "../native/bridge";
export type DType = "float32" | "uint8" | "int32";
export type TensorHostObject = { dtype: DType; shape: number[] };

export class Tensor {
  private _hostObject: TensorHostObject;

  private constructor(hostObject: TensorHostObject) {
    this._hostObject = hostObject;
  }

  /**
   * @internal
   */
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

  /**
   * @internal
   */
  static fromHostObject(hostObject: TensorHostObject): Tensor {
    return new Tensor(hostObject);
  }

  static fromEmpty(shape: number[], dtype: DType): Tensor {
    const hostObject = mylibJsi.createTensor(shape, dtype);
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

  setFromTypedArray(data: Float32Array | Uint8Array | Int32Array): void {
    mylibJsi.setTensorFromTypedArray(this._hostObject, data);
  }

  toTypedArray(
    opts: { selfDispose: boolean } = { selfDispose: false },
    dst?: Float32Array | Uint8Array | Int32Array,
  ): Float32Array | Uint8Array | Int32Array {
    if (!dst) {
      switch (this.dtype) {
        case "float32":
          dst = new Float32Array(this.numel);
          break;
        case "uint8":
          dst = new Uint8Array(this.numel);
          break;
        case "int32":
          dst = new Int32Array(this.numel);
          break;
        default:
          throw new Error(`Unsupported dtype: ${this.dtype}`);
      }
    }

    mylibJsi.setTypedArrayFromTensor(dst, this._hostObject);
    if (opts.selfDispose) {
      this.dispose();
    }
    return dst;
  }

  reshape(shape: number[]): this {
    mylibJsi.reshapeTensor(this._hostObject, shape);
    return this;
  }

  through<R, Args extends any[]>(
    opts: { selfDispose: boolean } = { selfDispose: false },
    fn: (src: this, ...args: Args) => R,
    ...args: Args
  ): R {
    const res = fn(this, ...args);
    if (opts.selfDispose) {
      this.dispose();
    }
    return res;
  }
}
