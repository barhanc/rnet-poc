import type { WorkletRuntime } from 'react-native-worklets';

import { tensor, type Tensor } from '../../../core/tensor';
import { loadModel } from '../../../core/model';
import { validateModelSchema, SymbolicTensor } from '../../../core/modelSchema';
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
  transfer: (input: ImageBuffer) => Promise<ImageBuffer>;
}> {
  const { modelPath, opts } = config;
  const model = await wrapAsync(loadModel, runtime)(modelPath);

  const meta = validateModelSchema(
    model,
    'forward',
    [SymbolicTensor('float32', [1, 3, 'H', 'W'], [3, 'H', 'W'])],
    [SymbolicTensor('float32', [1, 3, 'H', 'W'], [3, 'H', 'W'])],
  );
  const inpShape = meta.inputTensorMeta[0]!.shape;
  const outShape = meta.outputTensorMeta[0]!.shape;

  const targetH = outShape.at(-2)!;
  const targetW = outShape.at(-1)!;

  const tensors = [
    tensor('float32', outShape),
    tensor('float32', [3, targetH, targetW]),
    tensor('float32', [targetH, targetW, 3]),
    tensor('uint8', [targetH, targetW, 3]),
    tensor('uint8', [targetH, targetW, 4]),
  ] as const;

  const [tOutput, tReshape, tChanLast, tUint8, tRgba] = tensors;
  const preprocessor = createImagePreprocessor(opts, inpShape);

  const dispose = () => {
    tensors.forEach((t) => t.dispose());
    preprocessor.dispose();
    model.dispose();
  };

  const transfer = async (input: ImageBuffer): Promise<ImageBuffer> => {
    const tInput = preprocessor.process(input);
    await wrapAsync(() => {
      'worklet';
      model.execute('forward', [tInput], [tOutput]);
    }, runtime)();

    let tResize: Tensor | null = null;
    const data = new Uint8Array(input.height * input.width * 4);
    try {
      tResize = tensor('uint8', [input.height, input.width, 4]);
      tOutput
        .copyTo(tReshape)
        .through(toChannelsLast, tChanLast)
        .through(normalize, tUint8, { alpha: opts.outAlpha, beta: opts.outBeta })
        .through(cvtColor, tRgba, 'RGB2RGBA')
        .through(resize, tResize, { mode: 'stretch', interpolation: opts.outInterpolation })
        .getData(data);
    } finally {
      tResize?.dispose();
    }
    return {
      data,
      width: input.width,
      height: input.height,
      format: 'rgba',
      layout: 'hwc',
    };
  };

  return { transfer, dispose };
}
