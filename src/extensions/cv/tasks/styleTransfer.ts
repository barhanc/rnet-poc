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
  readonly styleTransferOpts: StyleTransferOptions;
};

export async function createStyleTransfer(
  config: StyleTransferModel,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  transfer: (input: ImageBuffer) => ImageBuffer;
  transferAsync: (input: ImageBuffer) => Promise<ImageBuffer>;
}> {
  const { modelPath, styleTransferOpts: opts } = config;
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

  const [tOutput, tReshaped, tChanLast, tUint8, tRgba] = tensors;
  const preprocessor = createImagePreprocessor(opts, inpShape);

  const dispose = () => {
    tensors.forEach((t) => t.dispose());
    preprocessor.dispose();
    model.dispose();
  };

  const postprocess = (
    tOutput: Tensor,
    { inputW, inputH }: { inputW: number; inputH: number },
  ): ImageBuffer => {
    'worklet';
    const data = new Uint8Array(inputH * inputW * 4);
    const tResized = tensor('uint8', [inputH, inputW, 4]);
    try {
      tOutput
        .copyTo(tReshaped)
        .through(toChannelsLast, tChanLast)
        .through(normalize, tUint8, { alpha: opts.outAlpha, beta: opts.outBeta })
        .through(cvtColor, tRgba, 'RGB2RGBA')
        .through(resize, tResized, { mode: 'stretch', interpolation: opts.outInterpolation })
        .getData(data);
    } finally {
      tResized.dispose();
    }
    return {
      data,
      width: inputW,
      height: inputH,
      format: 'rgba',
      layout: 'hwc',
    };
  };

  const transfer = (input: ImageBuffer): ImageBuffer => {
    'worklet';
    const tInput = preprocessor.process(input);
    model.execute('forward', [tInput], [tOutput]);
    return postprocess(tOutput, { inputW: input.width, inputH: input.height });
  };

  // The async variant is just the sync worklet run end-to-end on the worklet
  // runtime. Unlike a plain-array result, the returned `ImageBuffer.data` is a
  // `Uint8Array`; serializing a typed array back out of a worklet relies on the
  // react-native-worklets serialization fix vendored in patches/ (PR swmansion
  // /react-native-reanimated#9475). See `wrapAsync` in core/runtime.
  const transferAsync = wrapAsync(transfer, runtime);

  return { transfer, transferAsync, dispose };
}
