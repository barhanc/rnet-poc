import type { WorkletRuntime } from 'react-native-worklets';

import { tensor, type Tensor } from '../../../core/tensor';
import { loadModel } from '../../../core/model';
import { validateModelSignature, SymbolicTensor } from '../../../core/signature';
import { wrapAsync } from '../../../core/runtime';

import { type ImageBuffer } from '../image';
import { createImagePreprocessor, type ImagePreprocessorOptions } from './preprocessing';
import {
  toChannelsLast,
  normalize,
  resize,
  cvtColor,
  applyColormap,
  type InterpolationMethod,
} from '../ops/image';
import { sigmoid, argmax } from '../../math';

export type SemanticSegmentationOptions<L> = Omit<ImagePreprocessorOptions, 'resizeMode'> & {
  readonly resizeMode: 'stretch';
  readonly outInterpolation: InterpolationMethod;
  readonly labels: readonly L[];
};
export type SemanticSegmentationModel<L> = {
  readonly modelPath: string;
  readonly opts: SemanticSegmentationOptions<L>;
};

export type ColorMap<L extends PropertyKey> = Record<L, [number, number, number, number]>;

export type PartialColorMap<L extends PropertyKey> = Partial<
  Record<L, [number, number, number, number]>
>;

export type SemanticSegmentationResult<L extends PropertyKey> = {
  buffer: ImageBuffer;
  colormap?: ColorMap<L>;
};

export async function createSemanticSegmenter<L extends PropertyKey = string>(
  config: SemanticSegmentationModel<L>,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  segment: (input: ImageBuffer, colormap?: PartialColorMap<L>) => SemanticSegmentationResult<L>;
  segmentAsync: (
    input: ImageBuffer,
    colormap?: PartialColorMap<L>,
  ) => Promise<SemanticSegmentationResult<L>>;
}> {
  const { modelPath, opts } = config;
  const model = await wrapAsync(loadModel, runtime)(modelPath);

  const meta = validateModelSignature(
    model,
    'forward',
    [SymbolicTensor('float32', [1, 3, 'H', 'W'], [3, 'H', 'W'])],
    [SymbolicTensor('float32', [1, 'K', 'H', 'W'], ['K', 'H', 'W'])],
  );
  const inpShape = meta.inputTensorMeta[0]!.shape;
  const outShape = meta.outputTensorMeta[0]!.shape;

  const nClasses = outShape.at(-3)!;
  const targetH = outShape.at(-2)!;
  const targetW = outShape.at(-1)!;

  // We multiply the index by prime numbers to scatter the colors across the RGB space
  // so that consecutive classes get highly distinguishable, deterministic colors.
  const defaultColormap = opts.labels.map(
    (_, i) => [(i * 37) % 255, (i * 73) % 255, (i * 127) % 255, 255] as const,
  );

  const defaultColormapDict = Object.fromEntries(
    opts.labels.map((l, i) => [l, defaultColormap[i]]),
  ) as ColorMap<L>;

  if (nClasses > 1 && opts.labels.length !== nClasses)
    throw new Error(
      `Model outputs ${nClasses} classes, but ${opts.labels.length}` +
        `labels were provided in the configuration.`,
    );

  const tensors = [
    tensor('float32', outShape),
    tensor('float32', [nClasses, targetH, targetW]),
    tensor('float32', [nClasses, targetH, targetW]),
    tensor('float32', [targetH, targetW, nClasses]),
    tensor('uint8', [targetH, targetW, 1]),
    tensor('uint8', [targetH, targetW, 4]),
  ] as const;

  let tResize: Tensor | null = null;
  const [tOutput, tReshape, tSigmoid, tChanLast, tMask, tRgba] = tensors;
  const preprocessor = createImagePreprocessor(opts, inpShape);

  const dispose = () => {
    if (tResize) tResize.dispose();
    tensors.forEach((t) => t.dispose());
    preprocessor.dispose();
    model.dispose();
  };

  const segment = (
    input: ImageBuffer,
    colormap?: PartialColorMap<L>,
  ): SemanticSegmentationResult<L> => {
    'worklet';

    if (!tResize || tResize.shape[0] !== input.height || tResize.shape[1] !== input.width) {
      if (tResize) tResize.dispose();
      tResize = tensor('uint8', [input.height, input.width, 4]);
    }

    const tInput = preprocessor.process(input);
    model.execute('forward', [tInput], [tOutput]);

    let returnColormap: ColorMap<L> | undefined;

    if (nClasses > 1) {
      if (colormap) {
        returnColormap = Object.fromEntries(
          opts.labels.map((l) => [l, colormap[l] ?? [0, 0, 0, 0]]),
        ) as ColorMap<L>;
      } else {
        returnColormap = defaultColormapDict;
      }

      const cmap = opts.labels.map((l) => returnColormap![l]);

      tOutput
        .copyTo(tReshape)
        .through(toChannelsLast, tChanLast)
        .through(argmax, tMask, -1)
        .through(applyColormap, tRgba, cmap);
    } else {
      tOutput
        .copyTo(tReshape)
        .through(sigmoid, tSigmoid)
        .through(toChannelsLast, tChanLast)
        .through(normalize, tMask, { alpha: 255.0 })
        .through(cvtColor, tRgba, 'GRAY2RGBA');
    }

    const data = tRgba
      .through(resize, tResize, { mode: 'stretch', interpolation: opts.outInterpolation })
      .getData(new Uint8Array(tResize.numel));

    return {
      buffer: {
        data,
        width: input.width,
        height: input.height,
        format: 'rgba',
        layout: 'hwc',
      },
      colormap: returnColormap,
    };
  };

  const segmentAsync = wrapAsync(segment, runtime);

  return { segment, segmentAsync, dispose };
}
