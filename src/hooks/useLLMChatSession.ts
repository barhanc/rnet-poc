import { useEffect, useState } from 'react';
import RNFS from 'react-native-fs';

import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import { createLLMChatSession, type ChatFormatter, type ChatMessage } from '../extensions/nlp/tasks/llmChat';
import { createJinjaChatFormatter } from '../extensions/nlp/llm/jinja';
import { parseTokenizerConfig } from '../extensions/nlp/llm/tokenizerConfig';
import type { GenerationConfig } from '../extensions/nlp/llm/llmRunner';

export type UseLLMChatSessionConfig = {
  readonly modelPath: string;
  readonly tokenizerPath: string;
  /**
   * URL/path to the model's `tokenizer_config.json`. Its `chat_template` and
   * special tokens are used to build the prompt formatter automatically, so no
   * template needs to be hand-written. Optional when `format` is supplied.
   */
  readonly tokenizerConfigPath?: string;
  readonly initialMessages?: readonly ChatMessage[];
  readonly generationConfig?: GenerationConfig;
  /** Overrides the formatter derived from `tokenizerConfigPath`. */
  readonly format?: ChatFormatter;
  /** Overrides the stop tokens derived from the tokenizer config's EOS token. */
  readonly stopTokens?: readonly string[];
};

type DerivedFormat = {
  readonly format: ChatFormatter;
  readonly stopTokens: readonly string[];
};

export function useLLMChatSession(
  config: UseLLMChatSessionConfig,
  options?: { readonly preventLoad?: boolean },
) {
  const {
    localPath: localModelPath,
    downloadProgress: modelProgress,
    downloadError: modelError,
  } = useModelDownload(config.modelPath, options?.preventLoad);

  const {
    localPath: localTokenizerPath,
    downloadProgress: tokenizerProgress,
    downloadError: tokenizerError,
  } = useModelDownload(config.tokenizerPath, options?.preventLoad);

  const {
    localPath: localConfigPath,
    downloadProgress: configProgress,
    downloadError: configDownloadError,
  } = useModelDownload(config.tokenizerConfigPath, options?.preventLoad);

  // Derive the prompt formatter + stop tokens from the tokenizer config (unless
  // an explicit `format` override is provided).
  const [derived, setDerived] = useState<DerivedFormat | null>(null);
  const [configError, setConfigError] = useState<Error | null>(null);

  useEffect(() => {
    setConfigError(null);

    if (config.format) {
      setDerived({ format: config.format, stopTokens: config.stopTokens ?? [] });
      return;
    }

    if (!config.tokenizerConfigPath) {
      setDerived(null);
      setConfigError(new Error('useLLMChatSession: provide either `tokenizerConfigPath` or `format`'));
      return;
    }

    if (!localConfigPath) {
      setDerived(null);
      return;
    }

    let isMounted = true;
    RNFS.readFile(localConfigPath, 'utf8')
      .then((text) => {
        const { chatTemplate, bosToken, eosToken } = parseTokenizerConfig(JSON.parse(text));
        const format = createJinjaChatFormatter(chatTemplate, { bosToken });
        if (isMounted) {
          setDerived({
            format,
            stopTokens: config.stopTokens ?? (eosToken ? [eosToken] : []),
          });
        }
      })
      .catch((e) => {
        if (isMounted) setConfigError(e instanceof Error ? e : new Error(String(e)));
      });

    return () => {
      isMounted = false;
    };
  }, [localConfigPath, config.tokenizerConfigPath, config.format, config.stopTokens]);

  const downloadProgress = (modelProgress + tokenizerProgress + configProgress) / 3;
  const downloadError = modelError || tokenizerError || configDownloadError;

  const { model: session, error: loadError } = useModel(
    createLLMChatSession,
    localModelPath && localTokenizerPath && derived
      ? {
          modelPath: localModelPath,
          tokenizerPath: localTokenizerPath,
          initialMessages: config.initialMessages,
          generationConfig: config.generationConfig,
          format: derived.format,
          stopTokens: derived.stopTokens,
        }
      : null,
    [localModelPath, localTokenizerPath, derived],
  );

  return {
    isReady: !!session,
    downloadProgress,
    error: downloadError || configError || loadError,
    localModelPath,
    localTokenizerPath,
    sendMessage: session?.sendMessage,
    getHistory: session?.getHistory,
    stop: session?.stop,
    dispose: session?.dispose,
  };
}
