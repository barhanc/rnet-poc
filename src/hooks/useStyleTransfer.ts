import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import { createStyleTransfer, type StyleTransferModel } from '../extensions/cv/tasks/styleTransfer';

export function useStyleTransfer(config: StyleTransferModel, options?: { preventLoad?: boolean }) {
  const { localPath, downloadProgress, downloadError } = useModelDownload(
    config.modelPath,
    options?.preventLoad,
  );
  const { model, error } = useModel(
    createStyleTransfer,
    localPath ? { ...config, modelPath: localPath } : null,
    [localPath],
  );

  return {
    isReady: !!model,
    error: downloadError || error,
    downloadProgress,
    localPath,
    transferStyle: model?.transferStyle,
    transferStyleWorklet: model?.transferStyleWorklet,
  };
}
