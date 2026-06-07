import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import {
  createKeypointDetector,
  type KeypointDetectorModel,
  type BoxFormat,
} from '../extensions/cv/tasks/keypointDetection';

export function useKeypointDetector<F extends BoxFormat, L extends PropertyKey>(
  config: KeypointDetectorModel<F, L>,
  options?: { preventLoad?: boolean },
) {
  const { localPath, downloadProgress, downloadError } = useModelDownload(
    config.modelPath,
    options?.preventLoad,
  );
  const { model, error } = useModel(
    createKeypointDetector<F, L>,
    localPath ? { ...config, modelPath: localPath } : null,
    [localPath],
  );

  return {
    isReady: !!model,
    error: downloadError || error,
    downloadProgress,
    localPath,
    landmarks: config.opts.landmarks,
    detectKeypoints: model?.detectKeypoints,
    detectKeypointsWorklet: model?.detectKeypointsWorklet,
  };
}
