export type ImageFormat = "rgb" | "rgba" | "bgr" | "bgra" | "gray";

export type ImageBuffer = {
  data: Uint8Array;
  width: number;
  height: number;
  format?: ImageFormat;
};
