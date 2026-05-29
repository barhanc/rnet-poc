import { mylibJsi } from "../../native/bridge";
import { Tensor } from "../../core/Tensor";

export type ResizeOptions = {
  mode: "stretch" | "letterbox" | "crop";
  interpolation: "nearest" | "area" | "cubic" | "lanczos";
  padValue: number;
};

export function resize(
  src: Tensor,
  dst: Tensor,
  opts: ResizeOptions = { mode: "stretch", interpolation: "nearest", padValue: 0 },
): void {
  mylibJsi.cv.resize(src.hostObject, dst.hostObject, opts);
}
