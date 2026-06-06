import type { WorkletRuntime } from 'react-native-worklets';

import { tensor } from '../../../core/tensor';
import { loadModel } from '../../../core/model';
import { validateModelSchema, SymbolicTensor } from '../../../core/modelSchema';
import { wrapAsync } from '../../../core/runtime';

import { type ImageBuffer } from '../image';
import { createImagePreprocessor, type ImagePreprocessorOptions } from './preprocessing';
import { weightedNms, type BoundingBox, type BoxFormat, decodeBox, scaleBox } from '../ops/boxes';
import { scalePoint } from '../ops/points';

import { type ResizeMode } from '../ops/image';
import { sigmoid } from '../../math';

export type FaceDetectorOptions<F extends BoxFormat> = Omit<
  ImagePreprocessorOptions,
  'resizeMode'
> & {
  readonly resizeMode: Exclude<ResizeMode, 'crop'>;
  readonly boxFormat: F;
  readonly defaultMinScoreThreshold: number;
  readonly defaultSuppressionThreshold: number;
};

export type FaceDetectorModel<F extends BoxFormat> = {
  readonly modelPath: string;
  readonly opts: FaceDetectorOptions<F>;
};

export type FaceLandmarks = {
  readonly leftEye: { readonly x: number; readonly y: number };
  readonly rightEye: { readonly x: number; readonly y: number };
  readonly leftEar: { readonly x: number; readonly y: number };
  readonly rightEar: { readonly x: number; readonly y: number };
  readonly noseTip: { readonly x: number; readonly y: number };
  readonly mouthCenter: { readonly x: number; readonly y: number };
};

export type FaceDetection<F extends BoxFormat> = {
  readonly box: BoundingBox<F>;
  readonly confidence: number;
  readonly landmarks: FaceLandmarks;
};

export async function createFaceDetector<F extends BoxFormat>(
  config: FaceDetectorModel<F>,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  detect: (
    input: ImageBuffer,
    options?: { confidenceThreshold?: number; suppressionThreshold?: number },
  ) => Promise<FaceDetection<F>[]>;
  detectWorklet: (
    input: ImageBuffer,
    options?: { confidenceThreshold?: number; suppressionThreshold?: number },
  ) => FaceDetection<F>[];
}> {
  const { modelPath, opts } = config;
  const model = await wrapAsync(loadModel, runtime)(modelPath);
  const meta = validateModelSchema(
    model,
    'forward',
    [SymbolicTensor('float32', [1, 3, 'H', 'W'])],
    // 16 = (4 bbox points) + (6 keypoints) * (2 coordinates per kp)
    [SymbolicTensor('float32', [1, 'N', 16]), SymbolicTensor('float32', [1, 'N', 1])],
  );

  const inpShape = meta.inputTensorMeta[0]!.shape;
  const numAnchors = meta.outputTensorMeta[0]!.shape[1]!;

  const targetH = inpShape.at(-2)!;
  const targetW = inpShape.at(-1)!;

  const tensors = [
    tensor('float32', [1, numAnchors, 16]),
    tensor('float32', [1, numAnchors, 1]),
    tensor('float32', [1, numAnchors, 1]),
  ] as const;

  const [tPoints, tScores, tSigmoid] = tensors;
  const preprocessor = createImagePreprocessor(opts, inpShape);

  const dispose = () => {
    preprocessor.dispose();
    tensors.forEach((t) => t.dispose());
    model.dispose();
  };

  const detectWorklet = (
    input: ImageBuffer,
    options?: { confidenceThreshold?: number; suppressionThreshold?: number },
  ): FaceDetection<F>[] => {
    'worklet';
    const tInput = preprocessor.process(input);
    model.execute('forward', [tInput], [tPoints, tScores]);

    const points = tPoints.getData(new Float32Array(numAnchors * 16));
    const scores = tScores
      .through(sigmoid, tSigmoid) //
      .getData(new Float32Array(numAnchors));

    const scoreThreshold = options?.confidenceThreshold ?? opts.defaultMinScoreThreshold;
    const suppressionThreshold = options?.suppressionThreshold ?? opts.defaultSuppressionThreshold;

    const results: FaceDetection<F>[] = [];
    const nmsOpts = { boxFormat: opts.boxFormat, suppressionThreshold, scoreThreshold };
    const blendGroups = weightedNms(tPoints, tScores, nmsOpts);

    for (let i = 0; i < blendGroups.length; ++i) {
      const group = blendGroups[i]!;
      const weightedPts = new Float32Array(16);

      let totalScore = 0;

      for (let j = 0; j < group.length; ++j) {
        const idx = group[j]!;
        const score = scores[idx]!;
        totalScore += score;
        for (let c = 0; c < 16; ++c)
          weightedPts[c]! += (score / totalScore) * (points[idx * 16 + c]! - weightedPts[c]!);
      }

      const box = scaleBox(
        decodeBox(
          [weightedPts[0]!, weightedPts[1]!, weightedPts[2]!, weightedPts[3]!],
          opts.boxFormat,
        ),
        { width: targetW, height: targetH },
        { width: input.width, height: input.height },
        opts,
      );

      const landmarks: { x: number; y: number }[] = [];
      for (let kp = 0; kp < 6; ++kp) {
        landmarks.push(
          scalePoint(
            { x: weightedPts[4 + kp * 2]!, y: weightedPts[4 + kp * 2 + 1]! },
            { width: targetW, height: targetH },
            { width: input.width, height: input.height },
            opts,
          ),
        );
      }

      results.push({
        box,
        landmarks: {
          leftEye: landmarks[0]!,
          rightEye: landmarks[1]!,
          noseTip: landmarks[2]!,
          mouthCenter: landmarks[3]!,
          leftEar: landmarks[4]!,
          rightEar: landmarks[5]!,
        },
        confidence: totalScore / group.length,
      });
    }

    return results;
  };

  const detect = wrapAsync(detectWorklet, runtime);

  return { detect, detectWorklet, dispose };
}
