import { useModel } from './useModel';
import { useModelDownload } from './useModelDownload';
import {
  createSemanticSegmenter,
  type SemanticSegmentationModel,
} from '../extensions/cv/tasks/semanticSegmentation';

export function useSemanticSegmenter<L extends PropertyKey = string>(
  config: SemanticSegmentationModel<L>,
  options?: { preventLoad?: boolean },
) {
  const { localPath, downloadProgress, downloadError } = useModelDownload(
    config.modelPath,
    options?.preventLoad,
  );

  const { model, error } = useModel(
    createSemanticSegmenter<L>,
    localPath ? { ...config, modelPath: localPath } : null,
    [localPath],
  );

  return {
    isReady: !!model,
    error: downloadError || error,
    downloadProgress,
    localPath,
    segment: model?.segment,
    segmentAsync: model?.segmentAsync,
    labels: config.opts.labels,
  };
}
