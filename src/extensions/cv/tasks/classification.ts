import type { WorkletRuntime } from 'react-native-worklets';

import { tensor } from '../../../core/tensor';
import { loadModel } from '../../../core/model';
import { wrapAsync } from '../../../core/runtime';

import { softmax } from '../../math';
import { type ImageBuffer } from '../image';
import { createImagePreprocessor, type ImagePreprocessorOptions } from './preprocessing';

export type ClassifierOptions<L extends PropertyKey = string> = ImagePreprocessorOptions & {
  readonly labels: readonly L[];
};
export type ClassifierModel<L extends PropertyKey = string> = {
  readonly modelPath: string;
  readonly classifierOpts: ClassifierOptions<L>;
};
export type Classification<L extends PropertyKey = string> = {
  readonly label: L;
  readonly confidence: number;
};

export async function createClassifier<L extends PropertyKey = string>(
  config: ClassifierModel<L>,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  classify: (input: ImageBuffer) => Classification<L>[];
  classifyAsync: (input: ImageBuffer) => Promise<Classification<L>[]>;
}> {
  const { modelPath, classifierOpts } = config;
  const model = await wrapAsync(loadModel, runtime)(modelPath);

  // Assuming the model has a single input and a single output
  const meta = model.getMethodMeta('forward');
  const inpShape = meta.inputTensorMeta[0]!.shape;
  const outShape = meta.outputTensorMeta[0]!.shape;

  const tensors = [
    tensor('float32', outShape), //
    tensor('float32', outShape),
  ] as const;

  const [tLogits, tProbas] = tensors;
  const preprocessor = createImagePreprocessor(classifierOpts, inpShape);

  const dispose = () => {
    preprocessor.dispose();
    tensors.forEach((t) => t.dispose());
    model.dispose();
  };

  const classify = (input: ImageBuffer): Classification<L>[] => {
    'worklet';

    const tInput = preprocessor.process(input);
    model.execute('forward', [tInput], [tLogits]);

    const probas = tLogits
      .through(softmax, tProbas) //
      .getData(new Float32Array(tProbas.numel));

    return Array.from(probas)
      .map((confidence, index) => ({ confidence, label: classifierOpts.labels[index]! }))
      .sort((a, b) => b.confidence - a.confidence);
  };

  const classifyAsync = wrapAsync(classify, runtime);

  return { classify, classifyAsync, dispose };
}
