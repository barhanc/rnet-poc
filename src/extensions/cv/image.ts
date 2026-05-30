export type ImageFormat = "rgb" | "rgba" | "bgr" | "bgra";
export type ImageLayout = "hwc" | "chw";
export type ImageBuffer = {
  data: Uint8Array;
  width: number;
  height: number;
  format: ImageFormat;
  layout: ImageLayout;
};
