import { mylibJsi } from "../../native/bridge";
import { Tensor } from "../../core/Tensor";

export type ColorConversionCode =
  | "RGBA2RGB"
  | "RGBA2BGR"
  | "BGRA2RGBA"
  | "BGRA2RGB"
  | "BGRA2BGR"
  | "RGB2BGR"
  | "BGR2RGB"
  | "RGB2GRAY"
  | "RGBA2GRAY"
  | "BGR2GRAY"
  | "BGRA2GRAY"
  | "RGB2RGBA"
  | "BGR2RGBA";

const colorConversionCodeToChannels: Record<
  ColorConversionCode,
  { srcChannels: number; dstChannels: number }
> = {
  RGBA2RGB: { srcChannels: 4, dstChannels: 3 },
  RGBA2BGR: { srcChannels: 4, dstChannels: 3 },
  BGRA2RGBA: { srcChannels: 4, dstChannels: 4 },
  BGRA2RGB: { srcChannels: 4, dstChannels: 3 },
  BGRA2BGR: { srcChannels: 4, dstChannels: 3 },
  RGB2BGR: { srcChannels: 3, dstChannels: 3 },
  BGR2RGB: { srcChannels: 3, dstChannels: 3 },
  RGB2GRAY: { srcChannels: 3, dstChannels: 1 },
  RGBA2GRAY: { srcChannels: 4, dstChannels: 1 },
  BGR2GRAY: { srcChannels: 3, dstChannels: 1 },
  BGRA2GRAY: { srcChannels: 4, dstChannels: 1 },
  RGB2RGBA: { srcChannels: 3, dstChannels: 4 },
  BGR2RGBA: { srcChannels: 3, dstChannels: 4 },
};

export type BoxFormat = "xyxy" | "xywh" | "cxcywh";

export type ResizeOptions = {
  width?: number;
  height?: number;
  mode?: "stretch" | "letterbox" | "crop";
  interpolation?: "nearest" | "area" | "cubic" | "lanczos";
  padValue?: number;
};

export type NormalizeOptions = {
  alpha?: number | number[];
  beta?: number | number[];
};

export type NmsOptions = {
  iouThreshold?: number;
  scoreThreshold?: number;
};

const defaultResizeOptions = {
  mode: "stretch",
  interpolation: "lanczos",
  padValue: 0,
} as const;

const defaultNormalizeOptions = {
  alpha: 1 / 255.0,
  beta: 0.0,
} as const;

const defaultNmsOptions = {
  iouThreshold: 0.5,
  scoreThreshold: 0.5,
} as const;

export function resize(src: Tensor, opts?: ResizeOptions, dst?: Tensor): Tensor {
  if (!dst) {
    const dstChannels = src.shape[2]!;
    dst = Tensor.fromEmpty([opts!.height!, opts!.width!, dstChannels], src.dtype);
  }
  mylibJsi.cv.resize(src.hostObject, dst.hostObject, { ...defaultResizeOptions, ...opts });
  return dst;
}

export function cvtColor(src: Tensor, code: ColorConversionCode, dst?: Tensor): Tensor {
  if (!dst) {
    const dstChannels = colorConversionCodeToChannels[code].dstChannels;
    const dstShape = [src.shape[0]!, src.shape[1]!, dstChannels];
    dst = Tensor.fromEmpty(dstShape, src.dtype);
  }
  mylibJsi.cv.cvtColor(src.hostObject, dst.hostObject, code);
  return dst;
}

export function toChannelsFirst(src: Tensor, dst?: Tensor): Tensor {
  if (!dst) {
    const dstShape = [src.shape[2]!, src.shape[0]!, src.shape[1]!];
    dst = Tensor.fromEmpty(dstShape, src.dtype);
  }
  mylibJsi.cv.toChannelsFirst(src.hostObject, dst.hostObject);
  return dst;
}

export function toChannelsLast(src: Tensor, dst?: Tensor): Tensor {
  if (!dst) {
    const dstShape = [src.shape[1]!, src.shape[2]!, src.shape[0]!];
    dst = Tensor.fromEmpty(dstShape, src.dtype);
  }
  mylibJsi.cv.toChannelsLast(src.hostObject, dst.hostObject);
  return dst;
}

export function normalize(src: Tensor, opts?: NormalizeOptions, dst?: Tensor): Tensor {
  if (!dst) {
    dst = Tensor.fromEmpty(src.shape, "float32");
  }
  mylibJsi.cv.normalize(src.hostObject, dst.hostObject, { ...defaultNormalizeOptions, ...opts });
  return dst;
}

export function nms(boxes: Tensor, scores: Tensor, opts?: NmsOptions): number[] {
  return mylibJsi.cv.nms(boxes.hostObject, scores.hostObject, { ...defaultNmsOptions, ...opts });
}

export function decodeBoxes(
  src: Tensor,
  { from, to }: { from: BoxFormat; to: BoxFormat },
  dst?: Tensor,
): Tensor {
  if (!dst) {
    dst = Tensor.fromEmpty(src.shape, src.dtype);
  }
  mylibJsi.cv.decodeBoxes(src.hostObject, dst.hostObject, { from, to });
  return dst;
}

export function scaleBoxes(
  src: Tensor,
  opts: { from: [number, number]; to: [number, number] },
  dst?: Tensor,
): Tensor {
  if (!dst) {
    dst = Tensor.fromEmpty(src.shape, src.dtype);
  }
  mylibJsi.cv.scaleBoxes(src.hostObject, dst.hostObject, opts);
  return dst;
}
