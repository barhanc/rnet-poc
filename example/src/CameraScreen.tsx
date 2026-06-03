import { useCallback, useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import { useResizer } from 'react-native-vision-camera-resizer';
import { scheduleOnRN } from 'react-native-worklets';
import { useClassifier, models } from 'react-native-my-lib';

export function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');

  const [prediction, setPrediction] = useState<{
    label: string;
    confidence: number;
  } | null>(null);

  const onPrediction = useCallback((label: string, confidence: number) => {
    setPrediction({ label, confidence });
  }, []);

  const [inputH, inputW] = [640, 640];
  const { resizer, state } = useResizer({
    width: inputW,
    height: inputH,
    channelOrder: 'rgb',
    dataType: 'uint8',
    scaleMode: 'cover',
    pixelLayout: 'interleaved',
  });

  const { classifyWorklet, isReady: isClassifierReady } = useClassifier(
    models.classification.EFFICIENTNET_V2_S,
  );

  useEffect(() => void requestPermission(), []);

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    onFrame(frame) {
      'worklet';
      if (!isClassifierReady || !classifyWorklet || !resizer || state !== 'ready') {
        frame.dispose();
        return;
      }
      const resized = resizer.resize(frame);
      try {
        const classificationResult = classifyWorklet(
          {
            data: new Uint8Array(resized.getPixelBuffer()),
            width: inputW,
            height: inputH,
            format: 'rgb',
            layout: 'hwc',
          },
          { topk: 1 },
        );

        const top = classificationResult[0];
        if (top) scheduleOnRN(onPrediction, top.label, top.confidence);
      } catch (e: any) {
        console.error(e.message);
      } finally {
        resized.dispose();
        frame.dispose();
      }
    },
  });

  if (!hasPermission) return <Text style={styles.centerText}>No permission</Text>;
  if (!device) return <Text style={styles.centerText}>No camera device</Text>;

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        outputs={[frameOutput]}
        orientationSource="device"
      />
      {prediction && (
        <View style={styles.overlay}>
          <Text style={styles.label}>{prediction.label}</Text>
          <Text style={styles.confidence}>{(prediction.confidence * 100).toFixed(1)}%</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  centerText: { color: '#000', marginTop: 100, textAlign: 'center' },
  overlay: {
    position: 'absolute',
    bottom: 48,
    left: 16,
    right: 16,
    padding: 16,
    borderRadius: 12,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  label: { color: '#fff', fontSize: 18, fontWeight: '600', flexShrink: 1 },
  confidence: { color: '#0f0', fontSize: 18, fontWeight: '700', marginLeft: 12 },
});
