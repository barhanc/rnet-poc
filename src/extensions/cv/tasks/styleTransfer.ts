import type { WorkletRuntime } from 'react-native-worklets';

import { tensor, type Tensor } from '../../../core/tensor';
import { loadModel } from '../../../core/model';
import { wrapAsync } from '../../../core/runtime';

import { type ImageBuffer } from '../image';
import { createImagePreprocessor, type ImagePreprocessorOptions } from './preprocessing';
import {
  toChannelsLast,
  normalize,
  cvtColor,
  resize,
  type InterpolationMethod,
} from '../ops/image';

export type StyleTransferOptions = Omit<ImagePreprocessorOptions, 'resizeMode'> & {
  readonly resizeMode: 'stretch';
  readonly outAlpha: number | number[];
  readonly outBeta: number | number[];
  readonly outInterpolation: InterpolationMethod;
};
export type StyleTransferModel = {
  readonly modelPath: string;
  readonly opts: StyleTransferOptions;
};

export async function createStyleTransfer(
  config: StyleTransferModel,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  transfer: (input: ImageBuffer) => ImageBuffer;
  transferAsync: (input: ImageBuffer) => Promise<ImageBuffer>;
}> {
  const { modelPath, opts } = config;
  const model = await wrapAsync(loadModel, runtime)(modelPath);

  // Assuming the model has a single input and a single output
  const meta = model.getMethodMeta('forward');
  const inpShape = meta.inputTensorMeta[0]!.shape;
  const outShape = meta.outputTensorMeta[0]!.shape;

  // Assuming the output is in [N]CHW format and has 3 channels (RGB)
  const targetH = outShape.at(-2)!;
  const targetW = outShape.at(-1)!;

  const tensors = [
    tensor('float32', outShape),
    tensor('float32', [3, targetH, targetW]),
    tensor('float32', [targetH, targetW, 3]),
    tensor('uint8', [targetH, targetW, 3]),
    tensor('uint8', [targetH, targetW, 4]),
  ] as const;

  let tResize: Tensor | null = null;

  const [tOutput, tReshape, tChanLast, tUint8, tRgba] = tensors;
  const preprocessor = createImagePreprocessor(opts, inpShape);

  const dispose = () => {
    if (tResize) tResize.dispose();
    tensors.forEach((t) => t.dispose());
    preprocessor.dispose();
    model.dispose();
  };

  const transfer = (input: ImageBuffer): ImageBuffer => {
    'worklet';

    if (!tResize || tResize.shape[0] !== input.height || tResize.shape[1] !== input.width) {
      if (tResize) tResize.dispose();
      tResize = tensor('uint8', [input.height, input.width, 4]);
    }

    const tInput = preprocessor.process(input);
    model.execute('forward', [tInput], [tOutput]);

    const data = tOutput
      .copyTo(tReshape)
      .through(toChannelsLast, tChanLast)
      .through(normalize, tUint8, { alpha: opts.outAlpha, beta: opts.outBeta })
      .through(cvtColor, tRgba, 'RGB2RGBA')
      .through(resize, tResize, { mode: 'stretch', interpolation: opts.outInterpolation })
      .getData(new Uint8Array(tResize.numel));

    return {
      data,
      width: input.width,
      height: input.height,
      format: 'rgba',
      layout: 'hwc',
    };
  };

  const transferAsync = wrapAsync(transfer, runtime);

  return { transfer, transferAsync, dispose };
}
