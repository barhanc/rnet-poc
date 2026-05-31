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
} from '@shopify/react-native-skia';
import {
  useClassifier,
  useDetector,
  useStyleTransfer,
  useSemanticSegmenter,
  models,
} from 'react-native-my-lib';

type TaskType = 'classification' | 'detection' | 'styleTransfer' | 'segmentation';

export function GalleryScreen() {
  const [activeTask, setActiveTask] = useState<TaskType>('classification');
  const [skiaImage, setSkiaImage] = useState<SkiaImageType | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);

  // Task Results
  const [classificationResults, setClassificationResults] = useState<any[]>([]);
  const [detectionResults, setDetectionResults] = useState<any[]>([]);
  const [styledImage, setStyledImage] = useState<SkiaImageType | null>(null);
  const [segmentationImage, setSegmentationImage] = useState<SkiaImageType | null>(null);

  // Model hooks
  const {
    isReady: isClassifierReady,
    downloadProgress: classifierProgress,
    classifyAsync,
  } = useClassifier(models.classification.EFFICIENTNET_V2_S.XNNPACK_FP32, {
    preventLoad: activeTask !== 'classification',
  });

  const {
    isReady: isDetectorReady,
    downloadProgress: detectorProgress,
    detectAsync,
  } = useDetector(models.objectDetection.SSDLITE320_MOBILENET_V3_LARGE.XNNPACK_FP32, {
    preventLoad: activeTask !== 'detection',
  });

  const {
    isReady: isStyleReady,
    downloadProgress: styleProgress,
    transferAsync,
  } = useStyleTransfer(models.styleTransfer.CANDY.XNNPACK_FP32, {
    preventLoad: activeTask !== 'styleTransfer',
  });

  const {
    isReady: isSegReady,
    downloadProgress: segProgress,
    segmentAsync,
  } = useSemanticSegmenter(models.semanticSegmentation.SELFIE_SEGMENTATION.XNNPACK_FP32, {
    preventLoad: activeTask !== 'segmentation',
  });

  const isModelReady =
    activeTask === 'classification'
      ? isClassifierReady
      : activeTask === 'detection'
        ? isDetectorReady
        : activeTask === 'styleTransfer'
          ? isStyleReady
          : isSegReady;

  const downloadProgress =
    activeTask === 'classification'
      ? classifierProgress
      : activeTask === 'detection'
        ? detectorProgress
        : activeTask === 'styleTransfer'
          ? styleProgress
          : segProgress;

  // Clear states when task changes
  useEffect(() => {
    setLatency(null);
    setClassificationResults([]);
    setDetectionResults([]);
    setStyledImage(null);
    setSegmentationImage(null);
  }, [activeTask]);

  // Pick image and scale it down to prevent massive JSI data transfers
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
    setDetectionResults([]);
    setStyledImage(null);
    setSegmentationImage(null);

    const rawUri = pickerResult.assets[0].uri;

    try {
      // Resize to max 800px width/height to keep memory transfer fast
      const manipResult = await ImageManipulator.manipulateAsync(
        rawUri,
        [{ resize: { width: 800 } }],
        { format: ImageManipulator.SaveFormat.PNG },
      );

      const resizedUri = manipResult.uri;

      // Load into Skia using robust base64 conversion
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

  // Run model on the selected image
  const runModel = async () => {
    if (!skiaImage || !isModelReady) return;

    setIsProcessing(true);
    setLatency(null);
    setClassificationResults([]);
    setDetectionResults([]);
    setStyledImage(null);
    setSegmentationImage(null);

    try {
      // Extract raw RGBA pixels from Skia
      const pixels = skiaImage.readPixels() as Uint8Array;
      const width = skiaImage.width();
      const height = skiaImage.height();

      const inputBuffer = {
        // Use slice() to account for potential non-zero byteOffset on the TypedArray
        buffer: pixels.buffer.slice(
          pixels.byteOffset,
          pixels.byteOffset + pixels.byteLength,
        ) as ArrayBuffer,
        width,
        height,
        format: 'rgba' as const,
        layout: 'hwc' as const,
      };

      const startTime = Date.now();

      if (activeTask === 'classification' && classifyAsync) {
        const results = await classifyAsync(inputBuffer);
        setClassificationResults(results.slice(0, 5));
      } else if (activeTask === 'detection' && detectAsync) {
        const results = await detectAsync(inputBuffer);
        setDetectionResults(results);
      } else if (activeTask === 'styleTransfer' && transferAsync) {
        const results = await transferAsync(inputBuffer);
        // Wrap ArrayBuffer in Uint8Array view for Skia (zero-copy)
        const outData = Skia.Data.fromBytes(new Uint8Array(results.buffer));
        const info = {
          width: results.width,
          height: results.height,
          colorType: ColorType.RGBA_8888,
          alphaType: AlphaType.Premul,
        };
        const skiaStyled = Skia.Image.MakeImage(info, outData, results.width * 4);
        setStyledImage(skiaStyled);
      } else if (activeTask === 'segmentation' && segmentAsync) {
        const results = await segmentAsync(inputBuffer);
        console.log(results.image.width, results.image.height, results.image.buffer.byteLength);
        // Wrap ArrayBuffer in Uint8Array view for Skia (zero-copy)
        const outData = Skia.Data.fromBytes(new Uint8Array(results.image.buffer));
        const info = {
          width: results.image.width,
          height: results.image.height,
          colorType: ColorType.RGBA_8888,
          alphaType: AlphaType.Premul,
        };
        const skiaSeg = Skia.Image.MakeImage(info, outData, results.image.width * 4);
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

  // Sizing and alignment
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
            <SkImage
              image={segmentationImage}
              fit="contain"
              x={0}
              y={0}
              width={viewWidth}
              height={viewHeight}
            />
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

        {/* Bounding box overlays for detection */}
        {activeTask === 'detection' &&
          detectionResults.map((det, index) => {
            const labelIdx =
              models.objectDetection.SSDLITE320_MOBILENET_V3_LARGE.detectorOpts.labels.indexOf(
                det.label,
              );
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
                  {
                    left,
                    top,
                    width,
                    height,
                    borderColor: strokeColor,
                    backgroundColor: bgColor,
                  },
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
      </View>
    );
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Navigation tabs for task selection */}
      <View style={styles.taskSelector}>
        {(['classification', 'detection', 'styleTransfer', 'segmentation'] as const).map((task) => (
          <Pressable
            key={task}
            style={[styles.taskTab, activeTask === task && styles.activeTaskTab]}
            onPress={() => setActiveTask(task)}
          >
            <Text style={[styles.taskTabText, activeTask === task && styles.activeTaskTabText]}>
              {task === 'classification'
                ? 'Classify'
                : task === 'detection'
                  ? 'Detect'
                  : task === 'styleTransfer'
                    ? 'Style'
                    : 'Segment'}
            </Text>
          </Pressable>
        ))}
      </View>

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
            <Text style={styles.btnTextPrimary}>Run Model</Text>
          )}
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
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  contentContainer: {
    padding: 16,
    alignItems: 'center',
  },
  taskSelector: {
    flexDirection: 'row',
    backgroundColor: '#e0e0e0',
    borderRadius: 12,
    padding: 4,
    marginBottom: 16,
    width: '100%',
  },
  taskTab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  activeTaskTab: {
    backgroundColor: '#fff',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
  },
  taskTabText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#666',
  },
  activeTaskTabText: {
    color: '#000',
    fontWeight: '600',
  },
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
  placeholderText: {
    color: '#666',
    fontSize: 16,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  canvasWrapper: {
    position: 'relative',
    backgroundColor: '#000',
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 20,
  },
  canvas: {
    width: '100%',
    height: '100%',
  },
  detectionBox: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 4,
  },
  boxLabelBadge: {
    position: 'absolute',
    top: -20,
    left: -2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  boxLabelText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
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
  btnDisabled: {
    backgroundColor: '#aaa',
  },
  btnTextPrimary: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 15,
  },
  btnTextSecondary: {
    color: '#000',
    fontWeight: '600',
    fontSize: 15,
  },
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
  statusText: {
    fontSize: 13,
    color: '#a0522d',
    fontWeight: '500',
  },
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
  perfLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1a73e8',
  },
  perfValue: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1a73e8',
  },
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
  resultsHeader: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111',
    marginBottom: 12,
  },
  resultItem: {
    marginBottom: 12,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  resultLabel: {
    fontSize: 13,
    fontWeight: '500',
    color: '#444',
    flex: 1,
    marginRight: 8,
  },
  resultPercentage: {
    fontSize: 13,
    fontWeight: '600',
    color: '#000',
  },
  progressBarBg: {
    height: 6,
    backgroundColor: '#eee',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#000',
    borderRadius: 3,
  },
});
