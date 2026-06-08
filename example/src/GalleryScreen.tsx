import { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Dimensions,
  ActivityIndicator,
  ScrollView,
  Alert,
  Platform,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import RNFS from 'react-native-fs';
import {
  Skia,
  ColorType,
  AlphaType,
  Canvas,
  Image as SkImage,
  type SkImage as SkiaImageType,
  Mask,
  ColorMatrix,
  Rect,
} from '@shopify/react-native-skia';
import {
  useClassifier,
  useObjectDetector,
  useStyleTransfer,
  useSemanticSegmenter,
  useKeypointDetector,
  models,
} from 'react-native-my-lib';
import { type ModelOption, ModelPicker } from './ModelPicker';

type TaskType = 'classification' | 'objectDetection' | 'styleTransfer' | 'segmentation' | 'keypointDetection';

const TASK_LABELS: Record<TaskType, string> = {
  classification: 'Classify',
  objectDetection: 'Detect',
  keypointDetection: 'Keypoints',
  styleTransfer: 'Style',
  segmentation: 'Segment',
};

const KEYPOINT_DETECTION_OPTIONS: ModelOption[] = [
  {
    label: 'BlazeFace (XNNPACK FP32)',
    value: models.keypointDetection.BLAZEFACE,
  },
  {
    label: 'Yolov8N Pose (XNNPACK FP32)',
    value: models.keypointDetection.YOLOV8N_POSE,
  },
];

// --- XNNPACK Model Registries ---
const CLASSIFICATION_OPTIONS: ModelOption[] = [
  {
    label: 'EfficientNet V2 S (XNNPACK FP32)',
    value: Platform.select({
      ios: models.classification.EFFICIENTNET_V2_S.XNNPACK_FP32,
      android: models.classification.EFFICIENTNET_V2_S.XNNPACK_FP32,
    }),
  },
  {
    label: `EfficientNet V2 S ${Platform.OS === 'ios' ? '(COREML FP16)' : '(XNNPACK INT8)'}`,
    value: Platform.select({
      ios: models.classification.EFFICIENTNET_V2_S.COREML_FP16,
      android: models.classification.EFFICIENTNET_V2_S.XNNPACK_INT8,
    }),
  },
];

const OBJECT_DETECTION_OPTIONS: ModelOption[] = [
  {
    label: `SSDLite MobileNet V3 ${Platform.OS === 'ios' ? '(COREML FP32)' : '(XNNPACK FP32)'}`,
    value: Platform.select({
      ios: models.objectDetection.SSDLITE320_MOBILENET_V3_LARGE.COREML_FP32,
      android: models.objectDetection.SSDLITE320_MOBILENET_V3_LARGE.XNNPACK_FP32,
    }),
    labels: models.objectDetection.SSDLITE320_MOBILENET_V3_LARGE.opts.labels,
  },
  {
    label: `RFDETR Nano ${Platform.OS === 'ios' ? '(COREML INT8)' : '(XNNPACK FP32)'}`,
    value: Platform.select({
      ios: models.objectDetection.RFDETR_NANO.COREML_INT8,
      android: models.objectDetection.RFDETR_NANO.XNNPACK_FP32,
    }),
    labels: models.objectDetection.RFDETR_NANO.opts.labels,
  },
];

const STYLE_OPTIONS: ModelOption[] = [
  {
    label: `Candy ${Platform.OS === 'ios' ? '(COREML FP32)' : '(XNNPACK FP32)'}`,
    value: Platform.select({
      ios: models.styleTransfer.CANDY.COREML_FP32,
      android: models.styleTransfer.CANDY.XNNPACK_FP32,
    }),
  },
  {
    label: `Candy ${Platform.OS === 'ios' ? '(COREML FP16)' : '(XNNPACK INT8)'}`,
    value: Platform.select({
      ios: models.styleTransfer.CANDY.COREML_FP16,
      android: models.styleTransfer.CANDY.XNNPACK_INT8,
    }),
  },
];

const SEGMENTATION_OPTIONS: ModelOption[] = [
  {
    label: 'Selfie Seg (FP32)',
    value: models.semanticSegmentation.SELFIE_SEGMENTATION.XNNPACK_FP32,
  },
  {
    label: 'LRASPP MobileNet V3 (FP32)',
    value: models.semanticSegmentation.LRASPP_MOBILENET_V3_LARGE.XNNPACK_FP32,
  },
  {
    label: 'LRASPP MobileNet V3 (INT8)',
    value: models.semanticSegmentation.LRASPP_MOBILENET_V3_LARGE.XNNPACK_INT8,
  },
];


// Color matrix to threshold the mask on the GPU with high contrast (non-inverted):
// out = 5.0 * maskVal - 2.0.
// If maskVal > 0.6 (person), out becomes >= 1.0 (fully opaque cutout).
// If maskVal < 0.4 (background), out becomes <= 0.0 (fully transparent background).
const THRESHOLD_MATRIX = [
  5.0, 0, 0, 0, -2.0,
  0, 5.0, 0, 0, -2.0,
  0, 0, 5.0, 0, -2.0,
  0, 0, 0, 1, 0,
];

export function GalleryScreen() {
  const [activeTask, setActiveTask] = useState<TaskType>('classification');
  const [skiaImage, setSkiaImage] = useState<SkiaImageType | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);

  // Dynamic Hooks initializations bound to the first option by default
  const [selectedClassifier, setSelectedClassifier] = useState(CLASSIFICATION_OPTIONS[0]!.value);
  const [selectedObjectDetector, setSelectedObjectDetector] = useState(OBJECT_DETECTION_OPTIONS[0]!.value);
  const [selectedStyle, setSelectedStyle] = useState(STYLE_OPTIONS[0]!.value);
  const [selectedSegmenter, setSelectedSegmenter] = useState(SEGMENTATION_OPTIONS[0]!.value);
  const [selectedKeypointDetector, setSelectedKeypointDetector] = useState(KEYPOINT_DETECTION_OPTIONS[0]!.value);

  // Task Results
  const [classificationResults, setClassificationResults] = useState<any[]>([]);
  const [objectDetectionResults, setObjectDetectionResults] = useState<any[]>([]);
  const [keypointDetectionResults, setKeypointDetectionResults] = useState<any[]>([]);
  const [styledImage, setStyledImage] = useState<SkiaImageType | null>(null);
  const [segmentationImage, setSegmentationImage] = useState<SkiaImageType | null>(null);

  // Model hooks
  const {
    isReady: isClassifierReady,
    downloadProgress: classifierProgress,
    classify,
    classifyWorklet,
  } = useClassifier(selectedClassifier, {
    preventLoad: activeTask !== 'classification',
  });

  const {
    isReady: isObjectDetectorReady,
    downloadProgress: objectDetectorProgress,
    detectObjects,
    detectObjectsWorklet,
  } = useObjectDetector(selectedObjectDetector, {
    preventLoad: activeTask !== 'objectDetection',
  });

  const {
    isReady: isStyleReady,
    downloadProgress: styleProgress,
    transferStyle,
    transferStyleWorklet,
  } = useStyleTransfer(selectedStyle, {
    preventLoad: activeTask !== 'styleTransfer',
  });

  const {
    isReady: isSegReady,
    downloadProgress: segProgress,
    segment,
    segmentWorklet,
  } = useSemanticSegmenter(selectedSegmenter, {
    preventLoad: activeTask !== 'segmentation',
  });

  const {
    isReady: isKeypointDetectorReady,
    downloadProgress: keypointDetectorProgress,
    detectKeypoints,
    detectKeypointsWorklet,
  } = useKeypointDetector(selectedKeypointDetector, {
    preventLoad: activeTask !== 'keypointDetection',
  });

  const modelReadyMap: Record<TaskType, boolean> = {
    classification: isClassifierReady,
    objectDetection: isObjectDetectorReady,
    styleTransfer: isStyleReady,
    segmentation: isSegReady,
    keypointDetection: isKeypointDetectorReady,
  };
  const isModelReady = modelReadyMap[activeTask];

  const progressMap: Record<TaskType, number> = {
    classification: classifierProgress,
    objectDetection: objectDetectorProgress,
    styleTransfer: styleProgress,
    segmentation: segProgress,
    keypointDetection: keypointDetectorProgress,
  };
  const downloadProgress = progressMap[activeTask];

  // Clear states when task or current model changes
  useEffect(() => {
    setLatency(null);
    setClassificationResults([]);
    setObjectDetectionResults([]);
    setKeypointDetectionResults([]);
    setStyledImage(null);
    setSegmentationImage(null);
  }, [activeTask, selectedClassifier, selectedObjectDetector, selectedStyle, selectedSegmenter, selectedKeypointDetector]);

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert('Permission Required', 'Permission to access camera roll is required!');
      return;
    }

    const pickerResult = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 1,
    });

    if (pickerResult.canceled || !pickerResult.assets[0]) return;

    setIsProcessing(true);
    setLatency(null);
    setClassificationResults([]);
    setObjectDetectionResults([]);
    setKeypointDetectionResults([]);
    setStyledImage(null);
    setSegmentationImage(null);

    const rawUri = pickerResult.assets[0].uri;

    try {
      const manipResult = await ImageManipulator.manipulateAsync(
        rawUri,
        [{ resize: { width: 800 } }],
        { format: ImageManipulator.SaveFormat.PNG },
      );

      const resizedUri = manipResult.uri;
      const cleanUri = resizedUri.replace('file://', '');
      const base64 = await RNFS.readFile(cleanUri, 'base64');
      const data = Skia.Data.fromBase64(base64);
      const img = Skia.Image.MakeImageFromEncoded(data);

      if (img) {
        setSkiaImage(img);
      } else {
        Alert.alert('Error', 'Failed to decode image in Skia.');
      }
    } catch (err: any) {
      console.error(err);
      Alert.alert('Error', 'Error loading image: ' + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const runModel = async () => {
    if (!skiaImage || !isModelReady) return;

    setIsProcessing(true);
    setLatency(null);
    setClassificationResults([]);
    setObjectDetectionResults([]);
    setKeypointDetectionResults([]);
    setStyledImage(null);
    setSegmentationImage(null);

    try {
      const pixels = skiaImage.readPixels() as Uint8Array;
      const width = skiaImage.width();
      const height = skiaImage.height();

      const inputBuffer = {
        data: pixels,
        width,
        height,
        format: 'rgba' as const,
        layout: 'hwc' as const,
      };

      const startTime = Date.now();

      if (activeTask === 'classification' && classify) {
        const results = await classify(inputBuffer, { topk: 2 });
        setClassificationResults(results);
      } else if (activeTask === 'objectDetection' && detectObjects) {
        const results = await detectObjects(inputBuffer);
        setObjectDetectionResults(results);
      } else if (activeTask === 'keypointDetection' && detectKeypoints) {
        const results = await detectKeypoints(inputBuffer);
        setKeypointDetectionResults(results);
      } else if (activeTask === 'styleTransfer' && transferStyle) {
        const results = await transferStyle(inputBuffer);
        const outData = Skia.Data.fromBytes(results.data);
        const info = {
          width: width,
          height: height,
          colorType: ColorType.RGBA_8888,
          alphaType: AlphaType.Premul,
        };
        const skiaStyled = Skia.Image.MakeImage(info, outData, width * 4);
        setStyledImage(skiaStyled);
      } else if (activeTask === 'segmentation' && segment) {
        const { buffer } = await segment(inputBuffer);
        const outData = Skia.Data.fromBytes(buffer.data);
        const info = {
          width: width,
          height: height,
          colorType: ColorType.RGBA_8888,
          alphaType: AlphaType.Premul,
        };
        const skiaSeg = Skia.Image.MakeImage(info, outData, width * 4);
        setSegmentationImage(skiaSeg);
      }

      const endTime = Date.now();
      setLatency(endTime - startTime);
    } catch (err: any) {
      console.error(err, err.message, String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  const runModelSync = () => {
    if (!skiaImage || !isModelReady) return;

    setLatency(null);
    setClassificationResults([]);
    setObjectDetectionResults([]);
    setKeypointDetectionResults([]);
    setStyledImage(null);
    setSegmentationImage(null);

    const pixels = skiaImage.readPixels() as Uint8Array;
    const width = skiaImage.width();
    const height = skiaImage.height();

    const inputBuffer = {
      data: pixels,
      width,
      height,
      format: 'rgba' as const,
      layout: 'hwc' as const,
    };

    const startTime = Date.now();

    try {
      if (activeTask === 'classification' && classifyWorklet) {
        const results = classifyWorklet(inputBuffer, { topk: 2 });
        setClassificationResults(results);
      } else if (activeTask === 'objectDetection' && detectObjectsWorklet) {
        const results = detectObjectsWorklet(inputBuffer);
        setObjectDetectionResults(results);
      } else if (activeTask === 'keypointDetection' && detectKeypointsWorklet) {
        const results = detectKeypointsWorklet(inputBuffer);
        setKeypointDetectionResults(results);
      } else if (activeTask === 'styleTransfer' && transferStyleWorklet) {
        const results = transferStyleWorklet(inputBuffer);
        const outData = Skia.Data.fromBytes(results.data);
        const info = {
          width,
          height,
          colorType: ColorType.RGBA_8888,
          alphaType: AlphaType.Premul,
        };
        setStyledImage(Skia.Image.MakeImage(info, outData, width * 4));
      } else if (activeTask === 'segmentation' && segmentWorklet) {
        const { buffer } = segmentWorklet(inputBuffer);
        const outData = Skia.Data.fromBytes(buffer.data);
        const info = {
          width,
          height,
          colorType: ColorType.RGBA_8888,
          alphaType: AlphaType.Premul,
        };
        setSegmentationImage(Skia.Image.MakeImage(info, outData, width * 4));
      }
    } catch (err: any) {
      console.error(err, err.message, String(err));
    }

    setLatency(Date.now() - startTime);
  };

  const screenWidth = Dimensions.get('window').width;
  const viewWidth = screenWidth - 32;
  const viewHeight = 350;

  let scaleX = 1;
  let scaleY = 1;
  let offsetX = 0;
  let offsetY = 0;

  if (skiaImage) {
    const imgW = skiaImage.width();
    const imgH = skiaImage.height();
    const scale = Math.min(viewWidth / imgW, viewHeight / imgH);
    const displayedW = imgW * scale;
    const displayedH = imgH * scale;
    offsetX = (viewWidth - displayedW) / 2;
    offsetY = (viewHeight - displayedH) / 2;
    scaleX = scale;
    scaleY = scale;
  }

  const renderActiveOutput = () => {
    if (!skiaImage) {
      return (
        <Pressable style={styles.placeholder} onPress={pickImage}>
          <Text style={styles.placeholderText}>Tap to select an image from gallery</Text>
        </Pressable>
      );
    }

    return (
      <View style={[styles.canvasWrapper, { width: viewWidth, height: viewHeight }]}>
        {activeTask === 'styleTransfer' && styledImage ? (
          <Canvas style={styles.canvas}>
            <SkImage
              image={styledImage}
              fit="contain"
              x={0}
              y={0}
              width={viewWidth}
              height={viewHeight}
            />
          </Canvas>
        ) : activeTask === 'segmentation' && segmentationImage ? (
          <Canvas style={styles.canvas}>
            {selectedSegmenter === models.semanticSegmentation.SELFIE_SEGMENTATION.XNNPACK_FP32 ? (
              <>
                {/* 1. Base layer: Render a solid red background over the active layout area */}
                <Rect
                  x={offsetX}
                  y={offsetY}
                  width={viewWidth - 2 * offsetX}
                  height={viewHeight - 2 * offsetY}
                  color="red"
                />
                {/* 2. Overlay layer: Mask the sharp image so only the person is drawn */}
                <Mask
                  mode="luminance"
                  mask={
                    <SkImage
                      image={segmentationImage}
                      fit="contain"
                      x={0}
                      y={0}
                      width={viewWidth}
                      height={viewHeight}
                    >
                      <ColorMatrix matrix={THRESHOLD_MATRIX} />
                    </SkImage>
                  }
                >
                  <SkImage
                    image={skiaImage}
                    fit="contain"
                    x={0}
                    y={0}
                    width={viewWidth}
                    height={viewHeight}
                  />
                </Mask>
              </>
            ) : (
              <>
                {/* 1. Base layer: Render the original background image */}
                <SkImage
                  image={skiaImage}
                  fit="contain"
                  x={0}
                  y={0}
                  width={viewWidth}
                  height={viewHeight}
                />
                {/* 2. Overlay layer: Render the segmentation mask on top with opacity */}
                <SkImage
                  image={segmentationImage}
                  fit="contain"
                  x={0}
                  y={0}
                  width={viewWidth}
                  height={viewHeight}
                  opacity={0.55} // Adjust this value (0.0 to 1.0) to change overlay intensity
                />
              </>
            )}
          </Canvas>
        ) : (
          <Canvas style={styles.canvas}>
            <SkImage
              image={skiaImage}
              fit="contain"
              x={0}
              y={0}
              width={viewWidth}
              height={viewHeight}
            />
          </Canvas>
        )}

        {/* Dynamic bounding boxes tied to active model configurations */}
        {activeTask === 'objectDetection' &&
          objectDetectionResults.map((det, index) => {
            const activeOpt = OBJECT_DETECTION_OPTIONS.find((opt) => opt.value === selectedObjectDetector);
            const labels = activeOpt?.labels || [];
            const labelIdx = labels.indexOf(det.label);

            const hue = (labelIdx * 137.5) % 360;
            const strokeColor = `hsl(${hue}, 95%, 50%)`;
            const bgColor = `hsla(${hue}, 95%, 50%, 0.15)`;

            const left = offsetX + det.box.xmin * scaleX;
            const top = offsetY + det.box.ymin * scaleY;
            const width = (det.box.xmax - det.box.xmin) * scaleX;
            const height = (det.box.ymax - det.box.ymin) * scaleY;

            return (
              <View
                key={index}
                style={[
                  styles.detectionBox,
                  { left, top, width, height, borderColor: strokeColor, backgroundColor: bgColor },
                ]}
              >
                <View style={[styles.boxLabelBadge, { backgroundColor: strokeColor }]}>
                  <Text style={styles.boxLabelText}>
                    {det.label} {Math.round(det.confidence * 100)}%
                  </Text>
                </View>
              </View>
            );
          })}

        {activeTask === 'keypointDetection' &&
          keypointDetectionResults.map((det, index) => {
            const strokeColor = '#00ff00';
            const bgColor = 'rgba(0, 255, 0, 0.15)';
            const landmarkColor = '#ff00ff';

            const left = offsetX + det.box.xmin * scaleX;
            const top = offsetY + det.box.ymin * scaleY;
            const width = (det.box.xmax - det.box.xmin) * scaleX;
            const height = (det.box.ymax - det.box.ymin) * scaleY;

            return (
              <View key={index} style={{ position: 'absolute', left: 0, top: 0, width: viewWidth, height: viewHeight }}>
                <View
                  style={[
                    styles.detectionBox,
                    { left, top, width, height, borderColor: strokeColor, backgroundColor: bgColor },
                  ]}
                >
                  <View style={[styles.boxLabelBadge, { backgroundColor: strokeColor }]}>
                    <Text style={styles.boxLabelText}>
                      Det {Math.round(det.confidence * 100)}%
                    </Text>
                  </View>
                </View>

                {Object.entries(det.landmarks).map(([key, point]: [string, any]) => {
                  const x = offsetX + point.x * scaleX;
                  const y = offsetY + point.y * scaleY;
                  return (
                    <View key={key} style={{ position: 'absolute', left: x - 50, top: y - 4, width: 100, alignItems: 'center' }}>
                      <View
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: landmarkColor,
                          borderWidth: 1,
                          borderColor: '#fff',
                        }}
                      />
                      <Text style={{ color: '#ff00ff', fontSize: 8, fontWeight: 'bold', textShadowColor: '#000', textShadowOffset: { width: 1, height: 1 }, textShadowRadius: 1, textAlign: 'center' }}>
                        {key}: {Math.round(point.confidence * 100)}%
                      </Text>
                    </View>
                  );
                })}
              </View>
            );
          })}
      </View>
    );
  };
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Navigation tabs for task selection */}
      <View style={styles.taskSelector}>
        {(['classification', 'objectDetection', 'keypointDetection', 'styleTransfer', 'segmentation'] as const).map((task) => (
          <Pressable
            key={task}
            style={[styles.taskTab, activeTask === task && styles.activeTaskTab]}
            onPress={() => setActiveTask(task)}
          >
            <Text style={[styles.taskTabText, activeTask === task && styles.activeTaskTabText]}>
              {TASK_LABELS[task]}
            </Text>
          </Pressable>
        ))}
      </View>

      {/* Conditional Model Selectors */}
      {activeTask === 'classification' && (
        <ModelPicker
          label="Model Precision"
          options={CLASSIFICATION_OPTIONS}
          selectedValue={selectedClassifier}
          onValueChange={setSelectedClassifier}
        />
      )}
      {activeTask === 'objectDetection' && (
        <ModelPicker
          label="Detection Model Architecture"
          options={OBJECT_DETECTION_OPTIONS}
          selectedValue={selectedObjectDetector}
          onValueChange={setSelectedObjectDetector}
        />
      )}
      {activeTask === 'keypointDetection' && (
        <ModelPicker
          label="Keypoint Detection Model Architecture"
          options={KEYPOINT_DETECTION_OPTIONS}
          selectedValue={selectedKeypointDetector}
          onValueChange={setSelectedKeypointDetector}
        />
      )}
      {activeTask === 'styleTransfer' && (
        <ModelPicker
          label="Style Quantization"
          options={STYLE_OPTIONS}
          selectedValue={selectedStyle}
          onValueChange={setSelectedStyle}
        />
      )}
      {activeTask === 'segmentation' && (
        <ModelPicker
          label="Segmentation Model Architecture"
          options={SEGMENTATION_OPTIONS}
          selectedValue={selectedSegmenter}
          onValueChange={setSelectedSegmenter}
        />
      )}

      {/* Main Image Display */}
      {renderActiveOutput()}

      {/* Control Buttons */}
      <View style={styles.buttonRow}>
        <Pressable style={styles.btnSecondary} onPress={pickImage} disabled={isProcessing}>
          <Text style={styles.btnTextSecondary}>Select Image</Text>
        </Pressable>

        <Pressable
          style={[
            styles.btnPrimary,
            (!skiaImage || !isModelReady || isProcessing) && styles.btnDisabled,
          ]}
          onPress={runModel}
          disabled={!skiaImage || !isModelReady || isProcessing}
        >
          {isProcessing ? (
            <ActivityIndicator color="#fff" size="small" />
          ) : (
            <Text style={styles.btnTextPrimary}>Run Async</Text>
          )}
        </Pressable>
      </View>

      <View style={styles.buttonRow}>
        <Pressable
          style={[
            styles.btnSync,
            (!skiaImage || !isModelReady || isProcessing) && styles.btnDisabled,
          ]}
          onPress={runModelSync}
          disabled={!skiaImage || !isModelReady || isProcessing}
        >
          <Text style={styles.btnTextPrimary}>Run Sync</Text>
        </Pressable>
      </View>

      {/* Model Download status */}
      {!isModelReady && (
        <View style={styles.statusBox}>
          <ActivityIndicator color="#000" size="small" style={{ marginRight: 8 }} />
          <Text style={styles.statusText}>
            Downloading model... {downloadProgress ? `${Math.round(downloadProgress)}%` : ''}
          </Text>
        </View>
      )}

      {/* Performance Latency Stats */}
      {latency !== null && (
        <View style={styles.perfBox}>
          <Text style={styles.perfLabel}>Total Pipeline Latency:</Text>
          <Text style={styles.perfValue}>{latency} ms</Text>
        </View>
      )}

      {/* Classification Results Details */}
      {activeTask === 'classification' && classificationResults.length > 0 && (
        <View style={styles.resultsCard}>
          <Text style={styles.resultsHeader}>Classification Matches</Text>
          {classificationResults.map((res, index) => (
            <View key={index} style={styles.resultItem}>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel} numberOfLines={1}>
                  {res.label}
                </Text>
                <Text style={styles.resultPercentage}>{Math.round(res.confidence * 100)}%</Text>
              </View>
              <View style={styles.progressBarBg}>
                <View style={[styles.progressBarFill, { width: `${res.confidence * 100}%` }]} />
              </View>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  contentContainer: { padding: 16, alignItems: 'center' },
  taskSelector: {
    flexDirection: 'row',
    backgroundColor: '#e0e0e0',
    borderRadius: 12,
    padding: 4,
    marginBottom: 16,
    width: '100%',
  },
  taskTab: { flex: 1, paddingVertical: 10, alignItems: 'center', borderRadius: 8 },
  activeTaskTab: {
    backgroundColor: '#fff',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
  },
  taskTabText: { fontSize: 12, fontWeight: '500', color: '#666' },
  activeTaskTabText: { color: '#000', fontWeight: '600' },
  placeholder: {
    width: '100%',
    height: 350,
    borderWidth: 2,
    borderColor: '#ccc',
    borderStyle: 'dashed',
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#eaeaea',
    marginBottom: 20,
  },
  placeholderText: { color: '#666', fontSize: 16, textAlign: 'center', paddingHorizontal: 32 },
  canvasWrapper: {
    position: 'relative',
    backgroundColor: '#000',
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 20,
  },
  canvas: { width: '100%', height: '100%' },
  detectionBox: { position: 'absolute', borderWidth: 2, borderRadius: 4 },
  boxLabelBadge: {
    position: 'absolute',
    top: -20,
    left: -2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  boxLabelText: { color: '#fff', fontSize: 10, fontWeight: 'bold' },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 16,
  },
  btnPrimary: {
    flex: 1,
    marginLeft: 8,
    backgroundColor: '#000',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnSecondary: {
    flex: 1,
    marginRight: 8,
    backgroundColor: '#fff',
    borderColor: '#000',
    borderWidth: 1.5,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnDisabled: { backgroundColor: '#aaa' },
  btnSync: {
    flex: 1,
    backgroundColor: '#1a73e8',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnTextPrimary: { color: '#fff', fontWeight: '600', fontSize: 15 },
  btnTextSecondary: { color: '#000', fontWeight: '600', fontSize: 15 },
  statusBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ffe8d6',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 10,
    marginBottom: 16,
    width: '100%',
  },
  statusText: { fontSize: 13, color: '#a0522d', fontWeight: '500' },
  perfBox: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#e8f0fe',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    marginBottom: 16,
    width: '100%',
    borderColor: '#c2d7fa',
    borderWidth: 1,
  },
  perfLabel: { fontSize: 14, fontWeight: '600', color: '#1a73e8' },
  perfValue: { fontSize: 14, fontWeight: '700', color: '#1a73e8' },
  resultsCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    width: '100%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 20,
  },
  resultsHeader: { fontSize: 16, fontWeight: '700', color: '#111', marginBottom: 12 },
  resultItem: { marginBottom: 12 },
  resultRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  resultLabel: { fontSize: 13, fontWeight: '500', color: '#444', flex: 1, marginRight: 8 },
  resultPercentage: { fontSize: 13, fontWeight: '600', color: '#000' },
  progressBarBg: { height: 6, backgroundColor: '#eee', borderRadius: 3, overflow: 'hidden' },
  progressBarFill: { height: '100%', backgroundColor: '#000', borderRadius: 3 },
});
