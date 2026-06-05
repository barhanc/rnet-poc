import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import {
  createDetector,
  type DetectorModel,
  type BoxFormat,
} from '../extensions/cv/tasks/detection';

export function useDetector<L, F extends BoxFormat>(
  config: DetectorModel<L, F>,
  options?: { preventLoad?: boolean },
) {
  const { localPath, downloadProgress, downloadError } = useModelDownload(
    config.modelPath,
    options?.preventLoad,
  );
  const { model, error } = useModel(
    createDetector<L, F>,
    localPath ? { ...config, modelPath: localPath } : null,
    [localPath],
  );

  return {
    isReady: !!model,
    error: downloadError || error,
    downloadProgress,
    localPath,
    labels: config.detectorOpts.labels,
    detect: model?.detect,
    detectWorklet: model?.detectWorklet,
  };
}
