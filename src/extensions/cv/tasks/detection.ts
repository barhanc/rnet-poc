import type { WorkletRuntime } from 'react-native-worklets';

import { tensor } from '../../../core/tensor';
import { loadModel } from '../../../core/model';
import { validateModelSchema, SymbolicTensor } from '../../../core/modelSchema';
import { wrapAsync } from '../../../core/runtime';

import { type ImageBuffer } from '../image';
import { createImagePreprocessor, type ImagePreprocessorOptions } from './preprocessing';
import { nms, scaleBox, decodeBox, type BoundingBox, type BoxFormat } from '../ops/boxes';

export type { BoxFormat };

export type DetectorOptions<L, F extends BoxFormat> = Omit<
  ImagePreprocessorOptions,
  'resizeMode'
> & {
  readonly resizeMode: 'stretch';
  readonly labels: readonly L[];
  readonly boxFormat: F;
  readonly defaultIouThreshold: number;
  readonly defaultConfidenceThreshold: number;
};

export type DetectorModel<L, F extends BoxFormat> = {
  readonly modelPath: string;
  readonly detectorOpts: DetectorOptions<L, F>;
};

export type Detection<L, F extends BoxFormat> = {
  readonly box: BoundingBox<F>;
  readonly label: L;
  readonly confidence: number;
};

export async function createDetector<L, F extends BoxFormat>(
  config: DetectorModel<L, F>,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  detect: (
    input: ImageBuffer,
    options?: { confidenceThreshold?: number; iouThreshold?: number },
  ) => Promise<Detection<L, F>[]>;
}> {
  const { modelPath, detectorOpts } = config;
  const model = await wrapAsync(loadModel, runtime)(modelPath);

  const meta = validateModelSchema(
    model,
    'forward',
    [SymbolicTensor('float32', [1, 3, 'H', 'W'], [3, 'H', 'W'])],
    [
      SymbolicTensor('float32', ['N', 4]),
      SymbolicTensor('float32', ['N']),
      SymbolicTensor('float32', ['N']),
    ],
  );

  const inpShape = meta.inputTensorMeta[0]!.shape;
  const outBoxesShape = meta.outputTensorMeta[0]!.shape;
  const outScoresShape = meta.outputTensorMeta[1]!.shape;
  const outClassesShape = meta.outputTensorMeta[2]!.shape;

  const targetH = inpShape.at(-2)!;
  const targetW = inpShape.at(-1)!;

  const tensors = [
    tensor('float32', outBoxesShape),
    tensor('float32', outScoresShape),
    tensor('float32', outClassesShape),
  ] as const;

  const [tBoxes, tScores, tClasses] = tensors;
  const preprocessor = createImagePreprocessor(detectorOpts, inpShape);

  const { boxFormat } = detectorOpts;

  const dispose = () => {
    preprocessor.dispose();
    tensors.forEach((t) => t.dispose());
    model.dispose();
  };

  const detect = async (
    input: ImageBuffer,
    options?: { confidenceThreshold?: number; iouThreshold?: number },
  ): Promise<Detection<L, F>[]> => {
    const tInput = preprocessor.process(input);
    await wrapAsync(() => {
      'worklet';
      model.execute('forward', [tInput], [tBoxes, tScores, tClasses]);
    }, runtime)();

    const boxes = tBoxes.getData(new Float32Array(tBoxes.numel));
    const scores = tScores.getData(new Float32Array(tScores.numel));
    const classes = tClasses.getData(new Float32Array(tClasses.numel));

    const iouThreshold = options?.iouThreshold ?? detectorOpts.defaultIouThreshold;
    const scoreThreshold = options?.confidenceThreshold ?? detectorOpts.defaultConfidenceThreshold;

    const results: Detection<L, F>[] = [];
    const indices = await wrapAsync(() => {
      'worklet';
      return nms(tBoxes, tScores, { boxFormat, scoreThreshold, iouThreshold });
    }, runtime)();

    for (const index of indices) {
      const confidence = scores[index]!;
      const classIdx = Math.round(classes[index]!);
      const label = detectorOpts.labels[classIdx];

      if (label === undefined) {
        throw new Error(
          `Detector: Predicted class index ${classIdx} is out of bounds for` +
            `labels array of size ${detectorOpts.labels.length}.`,
        );
      }

      const a = boxes[index * 4]!;
      const b = boxes[index * 4 + 1]!;
      const c = boxes[index * 4 + 2]!;
      const d = boxes[index * 4 + 3]!;

      results.push({
        label,
        confidence,
        box: scaleBox(
          decodeBox([a, b, c, d], boxFormat),
          { width: targetW, height: targetH },
          { width: input.width, height: input.height },
        ),
      });
    }

    return results;
  };

  return { detect, dispose };
}
