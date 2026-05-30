import type { WorkletRuntime } from "react-native-worklets";

import { tensor } from "../../../core/tensor";
import { loadModel } from "../../../core/model";
import { wrapAsync } from "../../../core/runtime";

import { type ImageBuffer } from "../core/image";
import { createImagePreprocessor, type ImagePreprocessorOptions } from "./preprocessing";

import * as math from "../../math";

export type ClassifierOptions<L = any> = ImagePreprocessorOptions & { labels: L[] };
export type ClassifierModel<L = any> = { modelPath: string; classifierOpts: ClassifierOptions<L> };
export type Classification<L = any> = { label: L; confidence: number };

export async function createClassifier<L = any>(
  config: ClassifierModel<L>,
  runtime?: WorkletRuntime,
): Promise<{
  dispose: () => void;
  classify: (input: ImageBuffer) => Classification<L>[];
  classifyAsync: (input: ImageBuffer) => Promise<Classification<L>[]>;
}> {
  const { modelPath, classifierOpts } = config;
  const model = await wrapAsync(loadModel, runtime)(modelPath);

  const meta = model.getMethodMeta("forward");
  const inpShape = meta.inputTensorMeta[0]!.shape;
  const outShape = meta.outputTensorMeta[0]!.shape;

  const preprocessor = createImagePreprocessor(classifierOpts, inpShape);

  const tensors = [
    tensor("float32", outShape), //
    tensor("float32", outShape),
  ] as const;
  const [tLogits, tProbas] = tensors;

  const dispose = () => {
    preprocessor.dispose();
    tensors.forEach((t) => t.dispose());
    model.dispose();
  };

  const classify = (input: ImageBuffer): Classification<L>[] => {
    "worklet";

    const tInput = preprocessor.process(input);
    model.execute("forward", [tInput], [tLogits]);
    const probas = tLogits
      .through(math.softmax, tProbas) //
      .getData(new Float32Array(tProbas.numel));

    const result: Classification<L>[] = [];
    for (let i = 0; i < probas.length; i++) {
      result.push({
        confidence: probas[i]!,
        label: classifierOpts.labels[i]!,
      });
    }
    result.sort((a, b) => b.confidence - a.confidence);

    return result;
  };

  const classifyAsync = wrapAsync(classify, runtime);

  return { classify, classifyAsync, dispose };
}
