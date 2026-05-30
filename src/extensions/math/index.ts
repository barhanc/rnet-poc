import { mylibJsi } from "../../native/bridge";
import type { Tensor } from "../../core/Tensor";

export type SoftmaxOptions = {
	axis?: number;
};

export function sigmoid(src: Tensor, dst: Tensor): Tensor {
	"worklet";
	return mylibJsi.math.sigmoid(src, dst);
}

export function softmax(src: Tensor, dst: Tensor, opts?: SoftmaxOptions): Tensor {
	"worklet";
	return mylibJsi.math.softmax(src, dst, { axis: opts?.axis ?? -1 });
}