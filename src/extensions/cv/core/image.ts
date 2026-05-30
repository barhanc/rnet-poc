import { type ColorConversionCode } from "./transforms";

export type ImageFormat = "rgb" | "rgba" | "bgr" | "bgra";

export const FORMAT_CHANNELS: Record<ImageFormat, number> = {
  rgb: 3,
  rgba: 4,
  bgr: 3,
  bgra: 4,
};

export const FORMAT_CONVERSION: Record<
  ImageFormat,
  Record<ImageFormat, ColorConversionCode | null>
> = {
  rgb: {
    rgb: null,
    rgba: "RGB2RGBA",
    bgr: "RGB2BGR",
    bgra: null,
  },
  rgba: {
    rgb: "RGBA2RGB",
    rgba: null,
    bgr: "RGBA2BGR",
    bgra: null,
  },
  bgr: {
    rgb: "BGR2RGB",
    rgba: "BGR2RGBA",
    bgr: null,
    bgra: null,
  },
  bgra: {
    rgb: "BGRA2RGB",
    rgba: "BGRA2RGBA",
    bgr: "BGRA2BGR",
    bgra: null,
  },
};

export type ImageBuffer = {
  data: Uint8Array;
  width: number;
  height: number;
  format: ImageFormat;
  readonly layout: "hwc";
};
