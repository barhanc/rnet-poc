import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import {
  createFaceDetector,
  type FaceDetectorModel,
} from '../extensions/cv/tasks/faceDetection';
import { type BoxFormat } from '../extensions/cv/ops/boxes';

export function useFaceDetector<F extends BoxFormat>(
  config: FaceDetectorModel<F>,
  options?: { preventLoad?: boolean },
) {
  const { localPath, downloadProgress, downloadError } = useModelDownload(
    config.modelPath,
    options?.preventLoad,
  );
  const { model, error } = useModel(
    createFaceDetector,
    localPath ? { ...config, modelPath: localPath } : null,
    [localPath],
  );

  return {
    isReady: !!model,
    error: downloadError || error,
    downloadProgress,
    localPath,
    detect: model?.detect,
    detectWorklet: model?.detectWorklet,
  };
}
