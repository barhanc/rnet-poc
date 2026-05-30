import { View, Text, StyleSheet, Pressable } from 'react-native';
import { cv } from 'react-native-my-lib';
import { useImage } from '@shopify/react-native-skia';
import { IMAGENET_CLASSES } from './imagenetClasses';

const modelPath = `/Users/bhanc/workspace/jsi-workshops/efficientnet_v2_s_xnnpack_int8.pte`;
const imageUrl =
  'https://raw.githubusercontent.com/yavuzceliker/sample-images/refs/heads/main/docs/image-1001.jpg';

export default function App() {
  const image = useImage(imageUrl);

  async function run() {
    const { classify, classifyAsync, dispose } = await cv.createClassifier({
      modelPath: modelPath,
      classifierOpts: {
        resizeMode: 'stretch',
        interpolation: 'linear',
        alpha: [1 / (255.0 * 0.229), 1 / (255.0 * 0.224), 1 / (255.0 * 0.225)],
        beta: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        labels: Object.values(IMAGENET_CLASSES),
      },
    });

    console.log('Running classification...');

    try {
      let time;
      let result: any;
      const repeat = 20;
      for (let i = 0; i < repeat; i++) {
        time = Date.now();
        result = classify({
          data: image!.readPixels() as Uint8Array,
          width: image!.width(),
          height: image!.height(),
          format: 'rgba',
          layout: 'hwc',
        });
        time = Date.now() - time;
        console.log(`Run ${i + 1}: ${time} ms`);
      }
      console.log('Classification result:', result.at(0));
    } finally {
      dispose();
    }
  }

  return (
    <View style={styles.container}>
      <Pressable style={styles.button} onPress={run}>
        <Text style={styles.buttonText}>Run Classifier</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  button: { paddingHorizontal: 20, paddingVertical: 12, borderRadius: 8, backgroundColor: '#111' },
  buttonText: { color: '#fff', fontSize: 16 },
});
