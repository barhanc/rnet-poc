import { scheduleOnRN, type WorkletRuntime } from 'react-native-worklets';

import { wrapAsync } from '../../../core/runtime';
import { createLLMRunner, type GenerationConfig, type GenerationStats } from '../runner';

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
  /**
   * Default generation config applied to every `sendMessage` call. A config
   * passed directly to `sendMessage` overrides these on a per-property basis.
   */
  readonly generationConfig?: GenerationConfig;
  /**
   * Tokens suppressed from the streamed output and the final response, e.g. the
   * model's EOS / turn-end token. ExecuTorch decodes and emits the stop token
   * before halting generation, so without this it leaks into the text.
   */
  readonly stopTokens?: readonly string[];
};

/**
 * A downloadable LLM definition as exposed from `models`. Points at the model
 * weights, the tokenizer, and the `tokenizer_config.json` from which the prompt
 * format (chat template + special tokens) is derived at load time.
 */
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

export async function createLLMChatSession(
  config: LLMOptions,
  runtime?: WorkletRuntime,
): Promise<LLMChatSession> {
  const { modelPath, tokenizerPath, format, initialMessages, generationConfig } = config;
  const stopTokens = config.stopTokens ?? [];
  const nativeRunner = await wrapAsync(createLLMRunner, runtime)(modelPath, tokenizerPath);
  const state: SessionState = { history: [] };

  // Prefill runs on the worklet runtime so it never blocks the JS thread. The
  // host method must be accessed *inside* the worklet — capturing
  // `nativeRunner.prefill` on the JS thread would bind it to the JS runtime.
  const prefill = wrapAsync((text: string): void => {
    'worklet';
    nativeRunner.prefill(text);
  }, runtime);

  for (const msg of initialMessages ?? []) {
    const formatted = format(msg, { isFirst: state.history.length === 0 });
    if (formatted.length > 0) {
      await prefill(formatted);
    }
    state.history.push(msg);
  }

  // The whole prefill + generate loop runs in a single hop on the worklet
  // runtime. `response` is accumulated there (a JS-thread closure variable
  // could not be mutated from the worklet thread) and returned once complete;
  // individual tokens are streamed back to the JS thread via `scheduleOnRN`.
  const runGeneration = wrapAsync(
    (
      fmtUserMsg: string,
      fmtAssistantHeader: string,
      genConfig: GenerationConfig,
      onToken?: (token: string) => void,
    ): GenerationResult => {
      'worklet';
      if (fmtUserMsg.length > 0) {
        nativeRunner.prefill(fmtUserMsg);
      }
      if (fmtAssistantHeader.length > 0) {
        nativeRunner.prefill(fmtAssistantHeader);
      }

      let response = '';
      const stats = nativeRunner.generate('', genConfig, (token: string) => {
        // ExecuTorch emits the stop token before halting; drop it so it never
        // reaches the accumulated response or the stream.
        if (stopTokens.includes(token)) {
          return;
        }
        response += token;
        if (onToken) {
          scheduleOnRN(onToken, token);
        }
      });

      return { response, stats };
    },
    runtime,
  );

  // `stop`/`dispose` stay on the JS thread on purpose: routing them through the
  // (single-threaded) worklet runtime would queue them behind an in-flight
  // `generate` and never interrupt it. The native handlers are lock-free.
  const dispose = () => nativeRunner.dispose();
  const stop = () => nativeRunner.stop();

  const sendMessage = async (
    message: string,
    options?: GenerationConfig,
    onToken?: (token: string) => void,
  ): Promise<GenerationResult> => {
    const userMsg: ChatMessage = { role: 'user', content: message };
    const assistantHeader: ChatMessage = { role: 'assistant', content: '' };

    const fmtUserMsg = format(userMsg, { isFirst: state.history.length === 0 });
    const fmtAssistantHeader = format(assistantHeader, { isFirst: false });

    state.history.push(userMsg);

    const result = await runGeneration(
      fmtUserMsg,
      fmtAssistantHeader,
      { ...generationConfig, ...options },
      onToken,
    );

    state.history.push({ role: 'assistant', content: result.response });
    return result;
  };

  return {
    dispose,
    sendMessage,
    getHistory: () => state.history,
    stop,
  };
}
