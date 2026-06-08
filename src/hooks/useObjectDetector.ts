import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import {
  createObjectDetector,
  type ObjectDetectorModel,
  type BoxFormat,
} from '../extensions/cv/tasks/objectDetection';

export function useObjectDetector<L, F extends BoxFormat>(
  config: ObjectDetectorModel<L, F>,
  options?: { preventLoad?: boolean },
) {
  const { localPath, downloadProgress, downloadError } = useModelDownload(
    config.modelPath,
    options?.preventLoad,
  );
  const { model, error } = useModel(
    createObjectDetector<L, F>,
    localPath ? { ...config, modelPath: localPath } : null,
    [localPath],
  );

  return {
    isReady: !!model,
    error: downloadError || error,
    downloadProgress,
    localPath,
    labels: config.opts.labels,
    detectObjects: model?.detectObjects,
    detectObjectsWorklet: model?.detectObjectsWorklet,
  };
}
