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

export type BoxFormat = "xyxy" | "xywh" | "cxcywh";

export type ResizeOptions = {
  mode: "stretch" | "letterbox" | "crop";
  interpolation: "nearest" | "area" | "cubic" | "lanczos";
  padValue: number;
};

export type NormalizeOptions = {
  alpha: number | number[];
  beta: number | number[];
};

export type NmsOptions = {
  iouThreshold: number;
  scoreThreshold: number;
};

export function resize(src: Tensor, dst: Tensor, opts: ResizeOptions): Tensor {
  mylibJsi.cv.resize(src.hostObject, dst.hostObject, opts);
  return dst;
}

export function cvtColor(src: Tensor, dst: Tensor, code: ColorConversionCode): Tensor {
  mylibJsi.cv.cvtColor(src.hostObject, dst.hostObject, code);
  return dst;
}

export function toChannelsFirst(src: Tensor, dst: Tensor): Tensor {
  mylibJsi.cv.toChannelsFirst(src.hostObject, dst.hostObject);
  return dst;
}

export function toChannelsLast(src: Tensor, dst: Tensor): Tensor {
  mylibJsi.cv.toChannelsLast(src.hostObject, dst.hostObject);
  return dst;
}

export function normalize(src: Tensor, dst: Tensor, opts: NormalizeOptions): Tensor {
  mylibJsi.cv.normalize(src.hostObject, dst.hostObject, opts);
  return dst;
}

export function nms(boxes: Tensor, scores: Tensor, opts: NmsOptions): number[] {
  return mylibJsi.cv.nms(boxes.hostObject, scores.hostObject, opts);
}

export function decodeBoxes(src: Tensor, dst: Tensor, { from, to }: { from: BoxFormat; to: BoxFormat }): Tensor {
  mylibJsi.cv.decodeBoxes(src.hostObject, dst.hostObject, { from, to });
  return dst;
}

export function scaleBoxes(src: Tensor, dst: Tensor, opts: { from: BoxFormat; to: BoxFormat }): Tensor {
  mylibJsi.cv.scaleBoxes(src.hostObject, dst.hostObject, opts);
  return dst;
}
