import { tensor, type Tensor } from '../../../core/tensor';

import { type ImageBuffer } from '../image';
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
};

export function createImagePreprocessor(opts: ImagePreprocessorOptions, outputShape: number[]) {
  const { resizeMode, interpolation, alpha, beta } = opts;

  // Assuming the output shape is in [N]CHW format.
  const targetH = outputShape.at(-2)!;
  const targetW = outputShape.at(-1)!;

  let tSrc: Tensor | null = null;

  const tensors = [
    tensor('uint8', [targetH, targetW, 4]),
    tensor('uint8', [targetH, targetW, 3]),
    tensor('uint8', [3, targetH, targetW]),
    tensor('float32', [3, targetH, targetW]),
    tensor('float32', outputShape),
  ] as const;

  const [tResize, tColor, tChannels, tNorm, tInput] = tensors;

  const dispose = () => {
    if (tSrc) tSrc.dispose();
    tensors.forEach((t) => t.dispose());
  };

  const process = (input: ImageBuffer): Tensor => {
    'worklet';
    const { data, width, height, format } = input;
    const numChannels = FORMAT_CHANNELS[format];
    const colorCode = FORMAT_CONVERSION[format]['rgb'];

    if (
      !tSrc ||
      tSrc.shape[0] !== height ||
      tSrc.shape[1] !== width ||
      tSrc.shape[2] !== numChannels
    ) {
      if (tSrc) tSrc.dispose();
      tSrc = tensor('uint8', [height, width, numChannels]);
    }

    tSrc
      .setData(data)
      .through(resize, tResize, { mode: resizeMode, interpolation: interpolation })
      .throughIf(colorCode !== null, cvtColor, tColor, colorCode!)
      .through(toChannelsFirst, tChannels)
      .through(normalize, tNorm, { alpha, beta })
      .copyTo(tInput);

    return tInput;
  };

  return { process, dispose };
}
