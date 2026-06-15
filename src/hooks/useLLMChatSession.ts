import { useEffect, useState } from 'react';
import RNFS from 'react-native-fs';

import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import {
  createLLMChatSession,
  type LLMModel,
  type LLMChatSessionOptions,
  type LLMChatSessionConfig,
} from '../extensions/nlp/tasks/llmChat';
import {
  parseTokenizerConfig,
  type TokenizerChatConfig,
} from '../extensions/nlp/llm/tokenizerConfig';

export function useTokenizerConfig(source: string, options?: { preventLoad?: boolean }) {
  const { localPath, downloadProgress, downloadError } = useModelDownload(
    source,
    options?.preventLoad,
  );
  const [config, setConfig] = useState<TokenizerChatConfig | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setConfig(null);
    setError(null);
    if (!localPath) return;

    let isMounted = true;
    RNFS.readFile(localPath, 'utf8')
      .then((text) => {
        if (isMounted) setConfig(parseTokenizerConfig(JSON.parse(text)));
      })
      .catch((e) => {
        if (isMounted) setError(e instanceof Error ? e : new Error(String(e)));
      });
    return () => {
      isMounted = false;
    };
  }, [localPath]);

  return { config, downloadProgress, error: downloadError || error };
}

export function useLLMChatSession(
  model: LLMModel,
  options?: LLMChatSessionOptions & { preventLoad?: boolean },
) {
  const {
    localPath: localModelPath,
    downloadProgress: modelProgress,
    downloadError: modelError,
  } = useModelDownload(model.modelPath, options?.preventLoad);

  const {
    localPath: localTokenizerPath,
    downloadProgress: tokenizerProgress,
    downloadError: tokenizerError,
  } = useModelDownload(model.tokenizerPath, options?.preventLoad);

  const {
    config: tokenizerConfig,
    downloadProgress: configProgress,
    error: configError,
  } = useTokenizerConfig(model.tokenizerConfigPath, { preventLoad: options?.preventLoad });

  const downloadProgress = (modelProgress + tokenizerProgress + configProgress) / 3;
  const downloadError = modelError || tokenizerError || configError;

  const { preventLoad, ...sessionOptions } = options ?? {};

  let sessionConfig: LLMChatSessionConfig | null = null;
  if (localModelPath && localTokenizerPath && tokenizerConfig)
    sessionConfig = {
      model: { modelPath: localModelPath, tokenizerPath: localTokenizerPath, tokenizerConfig },
      options: sessionOptions,
    };

  const { model: session, error: loadError } = useModel(createLLMChatSession, sessionConfig, [
    localModelPath,
    localTokenizerPath,
    tokenizerConfig,
  ]);

  return {
    isReady: !!session,
    downloadProgress,
    error: downloadError || loadError,
    localModelPath,
    localTokenizerPath,
    sendMessage: session?.sendMessage,
    getHistory: session?.getHistory,
    stop: session?.stop,
    dispose: session?.dispose,
  };
}
