import type { WorkletRuntime } from 'react-native-worklets';

import { tensor, type Tensor } from '../../../core/tensor';
import { loadModel } from '../../../core/model';
import { validateModelSchema, SymbolicTensor } from '../../../core/modelSchema';
import { wrapAsync } from '../../../core/runtime';

import type { ImageBuffer } from '../image';
import { createImagePreprocessor, type ImagePreprocessorOptions } from './preprocessing';
import type { ResizeMode } from '../ops/image';
import { scalePoint, type Point } from '../ops/points';
import { nms, type BoundingBox, type BoxFormat, decodeBox, scaleBox } from '../ops/boxes';

export type { BoxFormat };

export type KeypointDetectorOptions<F extends BoxFormat, L extends PropertyKey> = Omit<
  ImagePreprocessorOptions,
  'resizeMode'
> & {
  readonly resizeMode: Exclude<ResizeMode, 'crop'>;
  readonly boxFormat: F;
  readonly landmarks: readonly L[];
  readonly defaultIouThreshold: number;
  readonly defaultConfidenceThreshold: number;
};

export type KeypointDetectorModel<F extends BoxFormat, L extends PropertyKey> = {
  readonly modelPath: string;
  readonly opts: KeypointDetectorOptions<F, L>;
};

export type Landmarks<L extends PropertyKey> = Record<L, Point & { readonly confidence: number }>;

export type KeypointDetection<F extends BoxFormat, L extends PropertyKey> = {
  readonly box: BoundingBox<F>;
  readonly confidence: number;
  readonly landmarks: Landmarks<L>;
};

function postprocess<F extends BoxFormat, L extends PropertyKey>(
  tBoxes: Tensor,
  tScores: Tensor,
  tKeypoints: Tensor,
  opts: {
    readonly from: { readonly width: number; readonly height: number };
    readonly to: { readonly width: number; readonly height: number };
    readonly boxFormat: F;
    readonly landmarks: readonly L[];
    readonly iouThreshold: number;
    readonly confidenceThreshold: number;
    readonly resizeMode: Exclude<ResizeMode, 'crop'>;
  },
): KeypointDetection<F, L>[] {
  'worklet';

  const nmsGroups = nms(tBoxes, tScores, { ...opts, nmsType: 'weighted' });

  const boxes = tBoxes.getData(new Float32Array(tBoxes.numel));
  const scores = tScores.getData(new Float32Array(tScores.numel));
  const keypoints = tKeypoints.getData(new Float32Array(tKeypoints.numel));

  const results: KeypointDetection<F, L>[] = [];

  for (const group of nmsGroups) {
    const totalScore = group.reduce((total, idx) => total + (scores[idx] ?? 0), 0);
    const weightedBox = new Float32Array(4);
    const weightedKpt = new Float32Array(opts.landmarks.length * 3);

    for (const idx of group) {
      const score = scores[idx]!;
      weightedBox.forEach((v, i) => {
        weightedBox[i] = v + score * boxes[idx * 4 + i]!;
      });
      weightedKpt.forEach((v, i) => {
        weightedKpt[i] = v + score * keypoints[idx * opts.landmarks.length * 3 + i]!;
      });
    }

    if (totalScore > 0) {
      weightedBox.forEach((v, i) => {
        weightedBox[i] = v / totalScore;
      });
      weightedKpt.forEach((v, i) => {
        weightedKpt[i] = v / totalScore;
      });
    }

    const [a, b, c, d] = weightedBox;
    const box = scaleBox(decodeBox([a!, b!, c!, d!], opts.boxFormat), opts);
    const landmarks = {} as Landmarks<L>;

    for (const [i, key] of opts.landmarks.entries()) {
      const point = scalePoint({ x: weightedKpt[i * 3]!, y: weightedKpt[i * 3 + 1]! }, opts);
      const confidence = weightedKpt[i * 3 + 2]!;
      landmarks[key] = { ...point, confidence };
    }

    results.push({ box, confidence: totalScore / group.length, landmarks });
  }

  return results;
}

export async function createKeypointDetector<F extends BoxFormat, L extends PropertyKey>(
  config: KeypointDetectorModel<F, L>,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  detectKeypoints: (
    input: ImageBuffer,
    options?: { confidenceThreshold?: number; iouThreshold?: number },
  ) => Promise<KeypointDetection<F, L>[]>;
  detectKeypointsWorklet: (
    input: ImageBuffer,
    options?: { confidenceThreshold?: number; iouThreshold?: number },
  ) => KeypointDetection<F, L>[];
}> {
  const { modelPath, opts } = config;
  const { landmarks } = opts;
  const model = await wrapAsync(loadModel, runtime)(modelPath);
  const meta = validateModelSchema(
    model,
    'forward',
    [SymbolicTensor('float32', [1, 3, 'H', 'W'])],
    [
      SymbolicTensor('float32', ['N', 4]),
      SymbolicTensor('float32', ['N']),
      SymbolicTensor('float32', ['N', landmarks.length, 3]),
    ],
  );

  const inpShape = meta.inputTensorMeta[0]!.shape;
  const numAnchors = meta.outputTensorMeta[0]!.shape[0]!;

  const targetH = inpShape.at(-2)!;
  const targetW = inpShape.at(-1)!;

  const tensors = [
    tensor('float32', [numAnchors, 4]),
    tensor('float32', [numAnchors]),
    tensor('float32', [numAnchors, landmarks.length, 3]),
  ] as const;

  const [tBoxes, tScores, tKeypoints] = tensors;
  const preprocessor = createImagePreprocessor(opts, inpShape);

  const dispose = () => {
    preprocessor.dispose();
    tensors.forEach((t) => t.dispose());
    model.dispose();
  };

  const detectKeypointsWorklet = (
    input: ImageBuffer,
    options?: { confidenceThreshold?: number; iouThreshold?: number },
  ): KeypointDetection<F, L>[] => {
    'worklet';
    const tInput = preprocessor.process(input);
    model.execute('forward', [tInput], [tBoxes, tScores, tKeypoints]);

    const iouThreshold = options?.iouThreshold ?? opts.defaultIouThreshold;
    const confidenceThreshold = options?.confidenceThreshold ?? opts.defaultConfidenceThreshold;

    return postprocess(tBoxes, tScores, tKeypoints, {
      ...opts,
      iouThreshold,
      confidenceThreshold,
      from: { width: targetW, height: targetH },
      to: { width: input.width, height: input.height },
    });
  };

  const detectKeypoints = wrapAsync(detectKeypointsWorklet, runtime);

  return { detectKeypoints, detectKeypointsWorklet, dispose };
}
