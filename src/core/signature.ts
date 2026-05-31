import { type DType } from './tensor';
import { type Model, type ExecuTorchTag, type ModelMethodMeta, type TensorMeta } from './model';

export type SymbolicShape = readonly (number | string)[];
export type TensorConstraint = {
  readonly dtype?: DType;
  readonly shapes?: readonly SymbolicShape[];
};

export function SymbolicTensor(
  dtype?: DType,
  ...shapes: readonly SymbolicShape[]
): TensorConstraint {
  return { dtype, shapes };
}

const primitiveTagMap = {
  number: ['Int', 'Double'] as ExecuTorchTag[],
  boolean: ['Bool'] as ExecuTorchTag[],
  null: ['None'] as ExecuTorchTag[],
} as const;

export type ValueConstraint = keyof typeof primitiveTagMap | TensorConstraint;

export function matchShape(actual: number[], ...expected: readonly SymbolicShape[]): boolean {
  return expected.some((shape) => {
    if (actual.length !== shape.length) return false;
    const symbolMap = new Map<string, number>();
    return shape.every((dim, i) => {
      const act = actual[i]!;
      if (typeof dim === 'number') return act === dim;
      if (symbolMap.has(dim)) return symbolMap.get(dim) === act;
      symbolMap.set(dim, act);
      return true;
    });
  });
}

function validateTags(
  method: string,
  side: 'input' | 'output',
  expected: readonly ValueConstraint[],
  actualTags: ExecuTorchTag[],
  tensorMetas: TensorMeta[],
) {
  const numTensors = expected.filter((t) => typeof t === 'object').length;
  if (tensorMetas.length !== numTensors)
    throw new Error(`signature validation: '${method}' ${side}: tensor count mismatch`);

  let tIdx = 0;
  expected.forEach((exp, i) => {
    const act = actualTags[i]!;
    if (typeof exp === 'string' && !primitiveTagMap[exp].includes(act))
      throw new Error(
        `signature validation: '${method}' ${side}[${i}]: expected '${exp}', got '${act}'`,
      );
    if (typeof exp === 'object') {
      if (act !== 'Tensor')
        throw new Error(
          `signature validation: '${method}' ${side}[${i}]: expected Tensor, got '${act}'`,
        );
      const tMeta = tensorMetas[tIdx++]!;
      if (exp.dtype && tMeta.dtype !== exp.dtype)
        throw new Error(
          `signature validation: '${method}' ${side}[${i}]: dtype ${tMeta.dtype}, want ${exp.dtype}`,
        );
      if (exp.shapes?.length && !matchShape(tMeta.shape, ...exp.shapes)) {
        throw new Error(`signature validation: '${method}' ${side}[${i}]: shape mismatch`);
      }
    }
  });
}

export function validateModelSignature(
  model: Model,
  methodName: string,
  expectedInputs: readonly ValueConstraint[],
  expectedOutputs: readonly ValueConstraint[],
): ModelMethodMeta {
  if (!model.getMethodNames().includes(methodName))
    throw new Error(`signature validation: '${methodName}' method not found`);

  const meta = model.getMethodMeta(methodName);

  if (meta.inputTags.length !== expectedInputs.length)
    throw new Error(
      `signature validation: '${methodName}': inputs ${meta.inputTags.length}, want ${expectedInputs.length}`,
    );
  if (meta.outputTags.length !== expectedOutputs.length)
    throw new Error(
      `signature validation: '${methodName}': outputs ${meta.outputTags.length}, want ${expectedOutputs.length}`,
    );

  validateTags(methodName, 'input', expectedInputs, meta.inputTags, meta.inputTensorMeta);
  validateTags(methodName, 'output', expectedOutputs, meta.outputTags, meta.outputTensorMeta);

  return meta;
}
