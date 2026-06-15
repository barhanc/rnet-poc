import { scheduleOnRN, type WorkletRuntime } from 'react-native-worklets';

import { wrapAsync } from '../../../core/runtime';
import {
  createLLMRunner,
  type LLMRunner,
  type GenerationConfig,
  type GenerationStats,
} from '../llm/llmRunner';
import { createJinjaChatFormatter } from '../llm/jinja';
import type { TokenizerChatConfig } from '../llm/tokenizerConfig';

export type ChatMessage = {
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
};

export type ChatFormatter = (
  message: ChatMessage,
  options: { readonly isFirst: boolean },
) => string;

export type LLMModel = {
  readonly modelPath: string;
  readonly tokenizerPath: string;
  readonly tokenizerConfigPath: string;
};

export type LLMChatSessionOptions = {
  readonly initialMessages?: readonly ChatMessage[];
  readonly generationConfig?: GenerationConfig;
  readonly stopTokens?: readonly string[];
};

export type LLMChatSessionConfig = {
  readonly model: Omit<LLMModel, 'tokenizerConfigPath'> & { tokenizerConfig: TokenizerChatConfig };
  readonly options?: LLMChatSessionOptions;
};

export type GenerationResult = {
  readonly response: string;
  readonly stats: GenerationStats;
};

export type LLMChatSession = {
  dispose(): void;
  sendMessage(
    message: string,
    genConfig?: GenerationConfig,
    onToken?: (token: string) => void,
  ): Promise<GenerationResult>;
  getHistory(): readonly ChatMessage[];
  stop(): void;
};

type SessionState = {
  history: ChatMessage[];
};

function generateChatTurn(
  nativeRunner: LLMRunner,
  prompt: string,
  options: {
    readonly genConfig: GenerationConfig;
    readonly stopTokens: readonly string[];
    readonly onToken?: (token: string) => void;
  },
): GenerationResult {
  'worklet';
  const { genConfig, stopTokens, onToken } = options;

  let response = '';

  const callback = (token: string) => {
    if (stopTokens.includes(token)) return;
    response += token;
    if (onToken) scheduleOnRN(onToken, token);
  };

  const stats = nativeRunner.generate(prompt, genConfig, callback);

  return { response, stats };
}

export async function createLLMChatSession(
  config: LLMChatSessionConfig,
  runtime?: WorkletRuntime,
): Promise<LLMChatSession> {
  const { model, options } = config;
  const { modelPath, tokenizerPath, tokenizerConfig } = model;

  const initialMessages = options?.initialMessages ?? [];
  const defaultGenerationConfig = options?.generationConfig;

  const { chatTemplate, bosToken, eosToken } = tokenizerConfig;

  const format = createJinjaChatFormatter(chatTemplate, { bosToken });
  const stopTokens = [...(options?.stopTokens ?? []), ...(eosToken ? [eosToken] : [])];

  const state: SessionState = { history: [] };
  const nativeRunner = await wrapAsync(createLLMRunner, runtime)(modelPath, tokenizerPath);
  const prefill = wrapAsync(nativeRunner.prefill, runtime);

  for (const msg of initialMessages) {
    const fmtMsg = format(msg, { isFirst: state.history.length === 0 });
    if (fmtMsg.length > 0) {
      await prefill(fmtMsg);
    }
    state.history.push(msg);
  }

  const stop = () => nativeRunner.stop();
  const dispose = () => nativeRunner.dispose();
  const runGeneration = wrapAsync(generateChatTurn, runtime);

  const sendMessage = async (
    message: string,
    genConfig?: GenerationConfig,
    onToken?: (token: string) => void,
  ): Promise<GenerationResult> => {
    const userMsg: ChatMessage = { role: 'user', content: message };
    const assistantHeader: ChatMessage = { role: 'assistant', content: '' };

    const fmtUserMsg = format(userMsg, { isFirst: state.history.length === 0 });
    const fmtAssistantHeader = format(assistantHeader, { isFirst: false });

    state.history.push(userMsg);

    const prompt = fmtUserMsg + fmtAssistantHeader;
    const { response, stats } = await runGeneration(nativeRunner, prompt, {
      genConfig: { ...defaultGenerationConfig, ...genConfig },
      stopTokens,
      onToken,
    });

    state.history.push({ role: 'assistant', content: response });

    return { response, stats };
  };

  return {
    stop,
    dispose,
    sendMessage,
    getHistory: () => state.history,
  };
}
