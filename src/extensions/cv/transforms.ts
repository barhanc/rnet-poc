import { mylibJsi } from "../../native/bridge";
import { type Tensor } from "../../core/Tensor";

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

export type BoxFormat = "xyxy" | "xywh" | "cxcywh";

export type ResizeOptions = {
  width?: number;
  height?: number;
  mode?: "stretch" | "letterbox" | "crop";
  interpolation?: "nearest" | "area" | "cubic" | "lanczos" | "linear";
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

export function resize(src: Tensor, dst: Tensor, opts?: ResizeOptions): Tensor {
  "worklet";
  return mylibJsi.cv.resize(src, dst, { ...defaultResizeOptions, ...opts });
}

export function cvtColor(src: Tensor, dst: Tensor, code: ColorConversionCode): Tensor {
  "worklet";
  return mylibJsi.cv.cvtColor(src, dst, code);
}

export function toChannelsFirst(src: Tensor, dst: Tensor): Tensor {
  "worklet";
  return mylibJsi.cv.toChannelsFirst(src, dst);
}

export function toChannelsLast(src: Tensor, dst: Tensor): Tensor {
  "worklet";
  return mylibJsi.cv.toChannelsLast(src, dst);
}

export function normalize(src: Tensor, dst: Tensor, opts?: NormalizeOptions): Tensor {
  "worklet";
  return mylibJsi.cv.normalize(src, dst, { ...defaultNormalizeOptions, ...opts });
}

export function nms(boxes: Tensor, scores: Tensor, opts?: NmsOptions): number[] {
  "worklet";
  return mylibJsi.cv.nms(boxes, scores, { ...defaultNmsOptions, ...opts });
}

export function decodeBoxes(
  src: Tensor,
  dst: Tensor,
  opts: { from: BoxFormat; to: BoxFormat },
): Tensor {
  "worklet";
  return mylibJsi.cv.decodeBoxes(src, dst, opts);
}

export function scaleBoxes(
  src: Tensor,
  dst: Tensor,
  opts: { from: [number, number]; to: [number, number] },
): Tensor {
  "worklet";
  return mylibJsi.cv.scaleBoxes(src, dst, opts);
}
