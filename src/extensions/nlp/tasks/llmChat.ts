import { scheduleOnRN, type WorkletRuntime } from 'react-native-worklets';

import { wrapAsync } from '../../../core/runtime';
import {
  createLLMRunner,
  type GenerationConfig,
  type GenerationStats,
  type LLMRunner,
} from '../llm/llmRunner';

export type ChatMessage = {
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
};

export type ChatFormatter = (
  message: ChatMessage,
  options: { readonly isFirst: boolean },
) => string;

export type LLMOptions = {
  readonly modelPath: string;
  readonly tokenizerPath: string;
  readonly initialMessages?: readonly ChatMessage[];
  readonly format: ChatFormatter;
  readonly stopTokens?: readonly string[];
  readonly defaultGenerationConfig?: GenerationConfig;
};

export type LLMModel = {
  readonly modelPath: string;
  readonly tokenizerPath: string;
  readonly tokenizerConfigPath: string;
};

export type GenerationResult = {
  readonly response: string;
  readonly stats: GenerationStats;
};

export interface LLMChatSession {
  dispose(): void;
  sendMessage(
    message: string,
    options?: GenerationConfig,
    onToken?: (token: string) => void,
  ): Promise<GenerationResult>;
  getHistory(): readonly ChatMessage[];
  stop(): void;
}

type SessionState = {
  history: ChatMessage[];
};

function runGenerationWorklet(
  nativeRunner: LLMRunner,
  stopTokens: readonly string[],
  fmtUserMsg: string,
  fmtAssistantHeader: string,
  genConfig: GenerationConfig,
  onToken?: (token: string) => void,
): GenerationResult {
  'worklet';
  if (fmtUserMsg.length > 0) {
    nativeRunner.prefill(fmtUserMsg);
  }
  if (fmtAssistantHeader.length > 0) {
    nativeRunner.prefill(fmtAssistantHeader);
  }

  let response = '';
  const callback = (token: string) => {
    if (stopTokens.includes(token)) return;
    response += token;
    if (onToken) scheduleOnRN(onToken, token);
  };
  const stats = nativeRunner.generate('', genConfig, callback);
  return { response, stats };
}

export async function createLLMChatSession(
  config: LLMOptions,
  runtime?: WorkletRuntime,
): Promise<LLMChatSession> {
  const { modelPath, tokenizerPath, format, initialMessages, defaultGenerationConfig } = config;
  const stopTokens = config.stopTokens ?? [];
  const nativeRunner = await wrapAsync(createLLMRunner, runtime)(modelPath, tokenizerPath);
  const state: SessionState = { history: [] };

  for (const msg of initialMessages ?? []) {
    const formatted = format(msg, { isFirst: state.history.length === 0 });
    if (formatted.length > 0) {
      await wrapAsync(nativeRunner.prefill, runtime)(formatted);
    }
    state.history.push(msg);
  }

  const stop = () => nativeRunner.stop();
  const dispose = () => nativeRunner.dispose();
  const runGeneration = wrapAsync(runGenerationWorklet, runtime);

  const sendMessage = async (
    message: string,
    options?: GenerationConfig,
    onToken?: (token: string) => void,
  ): Promise<GenerationResult> => {
    const sendMessageStartMs = Date.now();

    const userMsg: ChatMessage = { role: 'user', content: message };
    const assistantHeader: ChatMessage = { role: 'assistant', content: '' };

    const fmtUserMsg = format(userMsg, { isFirst: state.history.length === 0 });
    const fmtAssistantHeader = format(assistantHeader, { isFirst: false });

    state.history.push(userMsg);

    const result = await runGeneration(
      nativeRunner,
      stopTokens,
      fmtUserMsg,
      fmtAssistantHeader,
      { ...defaultGenerationConfig, ...options },
      onToken,
    );

    const ttftMs = result.stats.firstTokenMs - sendMessageStartMs;
    const patchedResult: GenerationResult = {
      ...result,
      stats: { ...result.stats, ttftMs },
    };

    state.history.push({ role: 'assistant', content: result.response });
    return patchedResult;
  };

  return {
    stop,
    dispose,
    sendMessage,
    getHistory: () => state.history,
  };
}
