import { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import { useResizer } from 'react-native-vision-camera-resizer';
import { useClassifier, models } from 'react-native-my-lib';

export function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');

  const { resizer, state } = useResizer({
    width: 1280,
    height: 720,
    channelOrder: 'rgb',
    dataType: 'uint8',
    scaleMode: 'cover',
    pixelLayout: 'interleaved',
  });
  const { classify, isReady } = useClassifier(models.classification.EFFICIENTNET_V2_S);

  useEffect(() => void requestPermission(), []);

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    onFrame(frame) {
      'worklet';
      if (!isReady || !classify || !resizer || state !== 'ready') {
        frame.dispose();
        return;
      }
      try {
        const resized = resizer.resize(frame);
        const result = classify({
          buffer: resized.getPixelBuffer() as ArrayBuffer,
          width: 1280,
          height: 720,
          format: 'rgb',
          layout: 'hwc',
        });
        if (result) console.log('Top Match:', result[0]?.label);
        resized.dispose();
      } catch (e: any) {
        console.error(e.message);
      } finally {
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
    </View>
  );
}

const styles = StyleSheet.create({
  centerText: { color: '#000', marginTop: 100, textAlign: 'center' },
});
