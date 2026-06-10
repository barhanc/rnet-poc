import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import { createLLMChatSession, type LLMOptions } from '../extensions/nlp/tasks/llm';

export function useLLMChatSession(
  config: LLMOptions,
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

  const downloadProgress = (modelProgress + tokenizerProgress) / 2;
  const downloadError = modelError || tokenizerError;

  const { model: session, error: loadError } = useModel(
    createLLMChatSession,
    localModelPath && localTokenizerPath
      ? { ...config, modelPath: localModelPath, tokenizerPath: localTokenizerPath }
      : null,
    [localModelPath, localTokenizerPath],
  );

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
