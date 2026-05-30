import { tensor, type Tensor } from "../../../core/tensor";
import {
  type ImageFormat,
  type ImageBuffer,
  FORMAT_CHANNELS,
  FORMAT_CONVERSION,
} from "../core/image";
import { type ResizeMode, type InterpolationMethod } from "../core/transforms";
import * as cv from "../core/transforms";

export type ImagePreprocessorOptions = {
  resizeMode: ResizeMode;
  interpolation: InterpolationMethod;
  alpha: number | number[];
  beta: number | number[];
  fixedInput?: { width: number; height: number; format: ImageFormat };
};

export function createImagePreprocessor(opts: ImagePreprocessorOptions, modelInputShape: number[]) {
  const { fixedInput, resizeMode, interpolation, alpha, beta } = opts;
  const targetH = modelInputShape[2]!;
  const targetW = modelInputShape[3]!;

  let tSrc: Tensor | null = null;
  if (fixedInput) {
    tSrc = tensor("uint8", [
      fixedInput.height,
      fixedInput.width,
      FORMAT_CHANNELS[fixedInput.format],
    ]);
  }

  const tensors = [
    tensor("uint8", [targetH, targetW, 4]),
    tensor("uint8", [targetH, targetW, 3]),
    tensor("uint8", [3, targetH, targetW]),
    tensor("float32", [3, targetH, targetW]),
    tensor("float32", modelInputShape),
    ...(tSrc ? [tSrc] : []),
  ] as const;

  const [tResize, tColor, tChannels, tNorm, tInput] = tensors;

  const dispose = () => tensors.forEach((t) => t.dispose());
  const process = (input: ImageBuffer): Tensor => {
    "worklet";
    const { data, width, height, format } = input;
    const numChannels = FORMAT_CHANNELS[format];
    const colorCode = FORMAT_CONVERSION[format]["rgb"];
    const src = tSrc ?? tensor("uint8", [height, width, numChannels]);
    try {
      src
        .setData(data)
        .through(cv.resize, tResize, { mode: resizeMode, interpolation: interpolation })
        .throughIf(colorCode !== null, cv.cvtColor, tColor, colorCode!)
        .through(cv.toChannelsFirst, tChannels)
        .through(cv.normalize, tNorm, { alpha, beta })
        .reshape(tInput);
    } finally {
      if (!tSrc) src.dispose();
    }
    return tInput;
  };

  return { process, dispose };
}
