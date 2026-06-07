import type { ClassifierModel } from './extensions/cv/tasks/classification';
import type { ObjectDetectorModel } from './extensions/cv/tasks/objectDetection';
import type { SemanticSegmentationModel } from './extensions/cv/tasks/semanticSegmentation';
import type { StyleTransferModel } from './extensions/cv/tasks/styleTransfer';
import type { KeypointDetectorModel } from './extensions/cv/tasks/keypointDetection';
import {
  COCO_CLASSES,
  IMAGENET1K_LABELS,
  PASCAL_VOC_LABELS,
  BLAZEFACE_LANDMARKS,
  COCO_LANDMARKS,
  type CocoClass,
  type ImageNet1KLabel,
  type PascalVocLabel,
  type BlazeFaceLandmark,
  type CocoLandmark,
} from './constants';

const BASE_URL = 'https://huggingface.co/software-mansion/react-native-executorch';
const VERSION_TAG = 'resolve/v0.9.0';

// ------------------------------------------------------------------------------------------------
// --- Classification models
// ------------------------------------------------------------------------------------------------

const EFFICIENTNET_V2_S_OPTS = {
  resizeMode: 'stretch' as const,
  interpolation: 'linear' as const,
  alpha: 1 / 255.0,
  beta: 0.0,
  labels: IMAGENET1K_LABELS,
};
const EFFICIENTNET_V2_S_XNNPACK_INT8: ClassifierModel<ImageNet1KLabel> = {
  modelPath: `${BASE_URL}-efficientnet-v2-s/${VERSION_TAG}/xnnpack/efficientnet_v2_s_xnnpack_int8.pte`,
  classifierOpts: EFFICIENTNET_V2_S_OPTS,
};
const EFFICIENTNET_V2_S_XNNPACK_FP32: ClassifierModel<ImageNet1KLabel> = {
  modelPath: `${BASE_URL}-efficientnet-v2-s/${VERSION_TAG}/xnnpack/efficientnet_v2_s_xnnpack_fp32.pte`,
  classifierOpts: EFFICIENTNET_V2_S_OPTS,
};
const EFFICIENTNET_V2_S_COREML_FP16: ClassifierModel<ImageNet1KLabel> = {
  modelPath: `${BASE_URL}-efficientnet-v2-s/${VERSION_TAG}/coreml/efficientnet_v2_s_coreml_fp16.pte`,
  classifierOpts: EFFICIENTNET_V2_S_OPTS,
};

// ------------------------------------------------------------------------------------------------
// --- Style Transfer models
// ------------------------------------------------------------------------------------------------

const STYLE_TRANSFER_OPTS = {
  resizeMode: 'stretch' as const,
  interpolation: 'linear' as const,
  alpha: 1 / 255.0,
  beta: 0.0,
  outAlpha: 255.0,
  outBeta: 0.0,
  outInterpolation: 'lanczos' as const,
};
const STYLE_TRANSFER_CANDY_XNNPACK_FP32: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-candy/${VERSION_TAG}/xnnpack/style_transfer_candy_xnnpack_fp32.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_CANDY_XNNPACK_INT8: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-candy/${VERSION_TAG}/xnnpack/style_transfer_candy_xnnpack_int8.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_CANDY_COREML_FP16: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-candy/${VERSION_TAG}/coreml/style_transfer_candy_coreml_fp16.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_CANDY_COREML_FP32: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-candy/${VERSION_TAG}/coreml/style_transfer_candy_coreml_fp32.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_MOSAIC_XNNPACK_FP32: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-mosaic/${VERSION_TAG}/xnnpack/style_transfer_mosaic_xnnpack_fp32.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_MOSAIC_XNNPACK_INT8: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-mosaic/${VERSION_TAG}/xnnpack/style_transfer_mosaic_xnnpack_int8.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_MOSAIC_COREML_FP16: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-mosaic/${VERSION_TAG}/coreml/style_transfer_mosaic_coreml_fp16.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_MOSAIC_COREML_FP32: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-mosaic/${VERSION_TAG}/coreml/style_transfer_mosaic_coreml_fp32.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_RAIN_PRINCESS_XNNPACK_FP32: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-rain-princess/${VERSION_TAG}/xnnpack/style_transfer_rain_princess_xnnpack_fp32.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_RAIN_PRINCESS_XNNPACK_INT8: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-rain-princess/${VERSION_TAG}/xnnpack/style_transfer_rain_princess_xnnpack_int8.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_RAIN_PRINCESS_COREML_FP16: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-rain-princess/${VERSION_TAG}/coreml/style_transfer_rain_princess_coreml_fp16.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_RAIN_PRINCESS_COREML_FP32: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-rain-princess/${VERSION_TAG}/coreml/style_transfer_rain_princess_coreml_fp32.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_UDNIE_XNNPACK_FP32: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-udnie/${VERSION_TAG}/xnnpack/style_transfer_udnie_xnnpack_fp32.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_UDNIE_XNNPACK_INT8: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-udnie/${VERSION_TAG}/xnnpack/style_transfer_udnie_xnnpack_int8.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_UDNIE_COREML_FP16: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-udnie/${VERSION_TAG}/coreml/style_transfer_udnie_coreml_fp16.pte`,
  opts: STYLE_TRANSFER_OPTS,
};
const STYLE_TRANSFER_UDNIE_COREML_FP32: StyleTransferModel = {
  modelPath: `${BASE_URL}-style-transfer-udnie/${VERSION_TAG}/coreml/style_transfer_udnie_coreml_fp32.pte`,
  opts: STYLE_TRANSFER_OPTS,
};

// ------------------------------------------------------------------------------------------------
// Semantic Segmentation
// ------------------------------------------------------------------------------------------------

const SELFIE_SEGMENTATION_XNNPACK_FP32: SemanticSegmentationModel<'background' | 'person'> = {
  modelPath: `${BASE_URL}-selfie-segmentation/${VERSION_TAG}/xnnpack/selfie_segmentation_xnnpack_fp32.pte`,
  opts: {
    labels: ['background', 'person'] as const,
    resizeMode: 'stretch',
    interpolation: 'linear',
    alpha: 1 / 255.0,
    beta: 0.0,
    outInterpolation: 'lanczos',
  },
};

const LRASPP_MOBILENET_V3_LARGE_OPTS = {
  labels: PASCAL_VOC_LABELS,
  resizeMode: 'stretch' as const,
  interpolation: 'linear' as const,
  alpha: [1 / (255.0 * 0.229), 1 / (255.0 * 0.224), 1 / (255.0 * 0.225)],
  beta: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
  outInterpolation: 'lanczos' as const,
};
const LRASPP_MOBILENET_V3_LARGE_XNNPACK_FP32: SemanticSegmentationModel<PascalVocLabel> = {
  modelPath: `${BASE_URL}-lraspp/${VERSION_TAG}/xnnpack/lraspp_mobilenet_v3_large_xnnpack_fp32.pte`,
  opts: LRASPP_MOBILENET_V3_LARGE_OPTS,
};
const LRASPP_MOBILENET_V3_LARGE_XNNPACK_INT8: SemanticSegmentationModel<PascalVocLabel> = {
  modelPath: `${BASE_URL}-lraspp/${VERSION_TAG}/xnnpack/lraspp_mobilenet_v3_large_xnnpack_int8.pte`,
  opts: LRASPP_MOBILENET_V3_LARGE_OPTS,
};

// ------------------------------------------------------------------------------------------------
// --- Object Detection models
// ------------------------------------------------------------------------------------------------

const SSDLITE320_MOBILENET_V3_LARGE_OPTS = {
  labels: COCO_CLASSES,
  boxFormat: 'xyxy' as const,
  resizeMode: 'stretch' as const,
  interpolation: 'linear' as const,
  alpha: 1 / 255.0,
  beta: 0.0,
  defaultConfidenceThreshold: 0.5,
  defaultIouThreshold: 0.55,
};
const SSDLITE320_MOBILENET_V3_LARGE_XNNPACK_FP32: ObjectDetectorModel<CocoClass, 'xyxy'> = {
  modelPath: `${BASE_URL}-ssdlite320-mobilenet-v3-large/${VERSION_TAG}/xnnpack/ssdlite320_mobilenet_v3_large_xnnpack_fp32.pte`,
  opts: SSDLITE320_MOBILENET_V3_LARGE_OPTS,
};
const SSDLITE320_MOBILENET_V3_LARGE_COREML_FP16: ObjectDetectorModel<CocoClass, 'xyxy'> = {
  modelPath: `${BASE_URL}-ssdlite320-mobilenet-v3-large/${VERSION_TAG}/coreml/ssdlite320_mobilenet_v3_large_coreml_fp16.pte`,
  opts: SSDLITE320_MOBILENET_V3_LARGE_OPTS,
};
const SSDLITE320_MOBILENET_V3_LARGE_COREML_FP32: ObjectDetectorModel<CocoClass, 'xyxy'> = {
  modelPath: `${BASE_URL}-ssdlite320-mobilenet-v3-large/${VERSION_TAG}/coreml/ssdlite320_mobilenet_v3_large_coreml_fp32.pte`,
  opts: SSDLITE320_MOBILENET_V3_LARGE_OPTS,
};

const RFDETR_NANO_DETECTOR_OPTS = {
  labels: COCO_CLASSES,
  boxFormat: 'xyxy' as const,
  resizeMode: 'stretch' as const,
  interpolation: 'linear' as const,
  alpha: [1 / (255.0 * 0.229), 1 / (255.0 * 0.224), 1 / (255.0 * 0.225)],
  beta: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
  defaultConfidenceThreshold: 0.5,
  defaultIouThreshold: 0.55,
};
const RFDETR_NANO_DETECTOR_XNNPACK_FP32: ObjectDetectorModel<CocoClass, 'xyxy'> = {
  modelPath: `${BASE_URL}-rfdetr-nano-detector/${VERSION_TAG}/xnnpack/rfdetr_nano_xnnpack_fp32.pte`,
  opts: RFDETR_NANO_DETECTOR_OPTS,
};
const RFDETR_NANO_DETECTOR_COREML_INT8: ObjectDetectorModel<CocoClass, 'xyxy'> = {
  modelPath: `${BASE_URL}-rfdetr-nano-detector/${VERSION_TAG}/coreml/rfdetr_nano_coreml_int8.pte`,
  opts: RFDETR_NANO_DETECTOR_OPTS,
};

// ------------------------------------------------------------------------------------------------
// --- Keypoint Detection models
// ------------------------------------------------------------------------------------------------

const BLAZEFACE_XNNPACK_FP32: KeypointDetectorModel<'xyxy', BlazeFaceLandmark> = {
  modelPath: `https://huggingface.co/bhanc/scratch/resolve/main/blazeface_xnnpack_fp32.pte`,
  opts: {
    boxFormat: 'xyxy',
    resizeMode: 'letterbox',
    interpolation: 'linear',
    alpha: 1 / 127.5,
    beta: -1.0,
    defaultIouThreshold: 0.3,
    defaultConfidenceThreshold: 0.75,
    landmarks: BLAZEFACE_LANDMARKS,
  },
};

const YOLOV8N_POSE_OPTS = {
  boxFormat: 'xyxy' as const,
  resizeMode: 'letterbox' as const,
  interpolation: 'linear' as const,
  alpha: 1 / 255.0,
  beta: 0.0,
  defaultIouThreshold: 0.7,
  defaultConfidenceThreshold: 0.25,
  landmarks: COCO_LANDMARKS,
};
const YOLOV8N_POSE_384_XNNPACK_FP32: KeypointDetectorModel<'xyxy', CocoLandmark> = {
  modelPath: `https://huggingface.co/bhanc/scratch/resolve/main/yolov8n_pose_384_xnnpack_fp32.pte`,
  opts: YOLOV8N_POSE_OPTS,
};
const YOLOV8N_POSE_512_XNNPACK_FP32: KeypointDetectorModel<'xyxy', CocoLandmark> = {
  modelPath: `https://huggingface.co/bhanc/scratch/resolve/main/yolov8n_pose_512_xnnpack_fp32.pte`,
  opts: YOLOV8N_POSE_OPTS,
};
const YOLOV8N_POSE_640_XNNPACK_FP32: KeypointDetectorModel<'xyxy', CocoLandmark> = {
  modelPath: `https://huggingface.co/bhanc/scratch/resolve/main/yolov8n_pose_640_xnnpack_fp32.pte`,
  opts: YOLOV8N_POSE_OPTS,
};

// ------------------------------------------------------------------------------------------------
// --- Exported models
// ------------------------------------------------------------------------------------------------

export const models = {
  classification: {
    EFFICIENTNET_V2_S: {
      ...EFFICIENTNET_V2_S_XNNPACK_INT8,
      XNNPACK_INT8: EFFICIENTNET_V2_S_XNNPACK_INT8,
      XNNPACK_FP32: EFFICIENTNET_V2_S_XNNPACK_FP32,
      COREML_FP16: EFFICIENTNET_V2_S_COREML_FP16,
    },
  },
  styleTransfer: {
    CANDY: {
      ...STYLE_TRANSFER_CANDY_XNNPACK_INT8,
      XNNPACK_FP32: STYLE_TRANSFER_CANDY_XNNPACK_FP32,
      XNNPACK_INT8: STYLE_TRANSFER_CANDY_XNNPACK_INT8,
      COREML_FP16: STYLE_TRANSFER_CANDY_COREML_FP16,
      COREML_FP32: STYLE_TRANSFER_CANDY_COREML_FP32,
    },
    MOSAIC: {
      ...STYLE_TRANSFER_MOSAIC_XNNPACK_INT8,
      XNNPACK_FP32: STYLE_TRANSFER_MOSAIC_XNNPACK_FP32,
      XNNPACK_INT8: STYLE_TRANSFER_MOSAIC_XNNPACK_INT8,
      COREML_FP16: STYLE_TRANSFER_MOSAIC_COREML_FP16,
      COREML_FP32: STYLE_TRANSFER_MOSAIC_COREML_FP32,
    },
    RAIN_PRINCESS: {
      ...STYLE_TRANSFER_RAIN_PRINCESS_XNNPACK_INT8,
      XNNPACK_FP32: STYLE_TRANSFER_RAIN_PRINCESS_XNNPACK_FP32,
      XNNPACK_INT8: STYLE_TRANSFER_RAIN_PRINCESS_XNNPACK_INT8,
      COREML_FP16: STYLE_TRANSFER_RAIN_PRINCESS_COREML_FP16,
      COREML_FP32: STYLE_TRANSFER_RAIN_PRINCESS_COREML_FP32,
    },
    UDNIE: {
      ...STYLE_TRANSFER_UDNIE_XNNPACK_INT8,
      XNNPACK_FP32: STYLE_TRANSFER_UDNIE_XNNPACK_FP32,
      XNNPACK_INT8: STYLE_TRANSFER_UDNIE_XNNPACK_INT8,
      COREML_FP16: STYLE_TRANSFER_UDNIE_COREML_FP16,
      COREML_FP32: STYLE_TRANSFER_UDNIE_COREML_FP32,
    },
  },
  semanticSegmentation: {
    SELFIE_SEGMENTATION: {
      ...SELFIE_SEGMENTATION_XNNPACK_FP32,
      XNNPACK_FP32: SELFIE_SEGMENTATION_XNNPACK_FP32,
    },
    LRASPP_MOBILENET_V3_LARGE: {
      ...LRASPP_MOBILENET_V3_LARGE_XNNPACK_INT8,
      XNNPACK_FP32: LRASPP_MOBILENET_V3_LARGE_XNNPACK_FP32,
      XNNPACK_INT8: LRASPP_MOBILENET_V3_LARGE_XNNPACK_INT8,
    },
  },
  objectDetection: {
    SSDLITE320_MOBILENET_V3_LARGE: {
      ...SSDLITE320_MOBILENET_V3_LARGE_XNNPACK_FP32,
      XNNPACK_FP32: SSDLITE320_MOBILENET_V3_LARGE_XNNPACK_FP32,
      COREML_FP16: SSDLITE320_MOBILENET_V3_LARGE_COREML_FP16,
      COREML_FP32: SSDLITE320_MOBILENET_V3_LARGE_COREML_FP32,
    },
    RFDETR_NANO: {
      ...RFDETR_NANO_DETECTOR_XNNPACK_FP32,
      XNNPACK_FP32: RFDETR_NANO_DETECTOR_XNNPACK_FP32,
      COREML_INT8: RFDETR_NANO_DETECTOR_COREML_INT8,
    },
  },
  keypointDetection: {
    BLAZEFACE: {
      ...BLAZEFACE_XNNPACK_FP32,
      XNNPACK_FP32: BLAZEFACE_XNNPACK_FP32,
    },
    YOLOV8N_POSE: {
      ...YOLOV8N_POSE_384_XNNPACK_FP32,
      SIZE_384: { XNNPACK_FP32: YOLOV8N_POSE_384_XNNPACK_FP32 },
      SIZE_512: { XNNPACK_FP32: YOLOV8N_POSE_512_XNNPACK_FP32 },
      SIZE_640: { XNNPACK_FP32: YOLOV8N_POSE_640_XNNPACK_FP32 },
    },
  },
};
