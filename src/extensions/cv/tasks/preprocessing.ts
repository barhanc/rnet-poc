import { tensor, type Tensor } from '../../../core/tensor';

import { type ImageFormat, type ImageBuffer } from '../image';
import {
  type ResizeMode,
  type InterpolationMethod,
  FORMAT_CONVERSION,
  FORMAT_CHANNELS,
  resize,
  cvtColor,
  toChannelsFirst,
  normalize,
} from '../ops/image';

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
    tSrc = tensor('uint8', [
      fixedInput.height,
      fixedInput.width,
      FORMAT_CHANNELS[fixedInput.format],
    ]);
  }

  const tensors = [
    tensor('uint8', [targetH, targetW, 4]),
    tensor('uint8', [targetH, targetW, 3]),
    tensor('uint8', [3, targetH, targetW]),
    tensor('float32', [3, targetH, targetW]),
    tensor('float32', modelInputShape),
    ...(tSrc ? [tSrc] : []),
  ] as const;

  const [tResize, tColor, tChannels, tNorm, tInput] = tensors;

  const dispose = () => tensors.forEach((t) => t.dispose());
  const process = (input: ImageBuffer): Tensor => {
    'worklet';
    const { data, width, height, format } = input;
    const numChannels = FORMAT_CHANNELS[format];
    const colorCode = FORMAT_CONVERSION[format]['rgb'];
    const src = tSrc ?? tensor('uint8', [height, width, numChannels]);
    try {
      src
        .setData(data)
        .through(resize, tResize, { mode: resizeMode, interpolation: interpolation })
        .throughIf(colorCode !== null, cvtColor, tColor, colorCode!)
        .through(toChannelsFirst, tChannels)
        .through(normalize, tNorm, { alpha, beta })
        .reshape(tInput);
    } finally {
      if (!tSrc) src.dispose();
    }
    return tInput;
  };

  return { process, dispose };
}
