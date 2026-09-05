import { mylibJsi } from '../../../native/bridge';

declare const llmRunnerBrand: unique symbol;

export type GenerationConfig = {
  readonly echo?: boolean;
  readonly ignoreEos?: boolean;
  readonly maxNewTokens?: number;
  readonly temperature?: number;
};

export type GenerationStats = {
  readonly numPromptTokens: number;
  readonly numGeneratedTokens: number;
  readonly firstTokenMs: number;
  readonly inferenceStartMs: number;
  readonly inferenceEndMs: number;
  readonly modelLoadStartMs: number;
  readonly modelLoadEndMs: number;
};

export type LLMRunner = {
  readonly modelPath: string;
  readonly tokenizerPath: string;
  dispose(): void;
  prefill(prompt: string): void;
  stop(): void;
  generate(
    prompt: string,
    config?: GenerationConfig,
    onToken?: (token: string) => void,
  ): GenerationStats;

  /**
   * @internal
   * Prevents plain JS objects from being cast as LLMRunners
   */
  readonly [llmRunnerBrand]: never;
};

export function createLLMRunner(modelPath: string, tokenizerPath: string): LLMRunner {
  'worklet';
  return mylibJsi.nlp.createLLMRunner(modelPath, tokenizerPath) as LLMRunner;
}

