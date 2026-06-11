import { mylibJsi } from '../../../native/bridge';

export interface GenerationConfig {
  readonly echo?: boolean;
  readonly ignoreEos?: boolean;
  readonly maxNewTokens?: number;
  readonly temperature?: number;
}

export interface GenerationStats {
  readonly numPromptTokens: number;
  readonly numGeneratedTokens: number;
  readonly firstTokenMs: number;
  readonly inferenceStartMs: number;
  readonly inferenceEndMs: number;
  readonly modelLoadStartMs: number;
  readonly modelLoadEndMs: number;
  readonly ttftMs: number;
}

export interface LLMRunner {
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
}

export function createLLMRunner(modelPath: string, tokenizerPath: string): LLMRunner {
  'worklet';
  return mylibJsi.nlp.createLLMRunner(modelPath, tokenizerPath) as LLMRunner;
}
