import { useState } from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { useClassifier } from 'react-native-my-lib';
import { useImage } from '@shopify/react-native-skia';
import { models } from 'react-native-my-lib';
import RNFS from 'react-native-fs';

const imageUrl =
  'https://raw.githubusercontent.com/yavuzceliker/sample-images/refs/heads/main/docs/image-1001.jpg';

export function HookScreen() {
  const image = useImage(imageUrl);
  const [result, setResult] = useState<string>('');
  const [isInferencing, setIsInferencing] = useState(false);
  const [preventLoad, setPreventLoad] = useState(true);

  const { isReady, error, downloadProgress, classify, localPath } = useClassifier(
    models.classification.EFFICIENTNET_V2_S.XNNPACK_FP32,
    { preventLoad },
  );

  async function run() {
    if (!classify || !image) return;

    setIsInferencing(true);
    setResult('Classifying...');

    try {
      const pixels = image.readPixels() as Uint8Array;
      const tStart = Date.now();

      const classifications = classify({
        data: pixels,
        width: image.width(),
        height: image.height(),
        format: 'rgba',
        layout: 'hwc',
      });

      const tEnd = Date.now();
      if (classifications.length > 0) {
        const top5 = classifications.slice(0, 5);
        const resultsString = top5
          .map((c) => `${c.label}: ${(c.confidence * 100).toFixed(1)}%`)
          .join('\n');
        
        setResult(`Top 5 Predictions:\n${resultsString}\n\nTime: ${tEnd - tStart}ms`);
      } else {
        setResult(`No classifications found.\nTime: ${tEnd - tStart}ms`);
      }
    } catch (e) {
      console.error('Error in run:', e);
      setResult('Error occurred.');
    } finally {
      setIsInferencing(false);
    }
  }

  async function deleteModel() {
    if (!localPath) return;
    try {
      await RNFS.unlink(localPath);
      setResult('Model deleted. You can reload the app to download again.');
    } catch (e) {
      console.error(e);
      setResult('Failed to delete model or model already deleted.');
    }
  }

  return (
    <View style={styles.container}>
      {error && <Text style={styles.errorText}>Error: {error.message}</Text>}

      {preventLoad && (
        <Pressable style={styles.button} onPress={() => setPreventLoad(false)}>
          <Text style={styles.buttonText}>Download & Load Model</Text>
        </Pressable>
      )}

      {!preventLoad && !isReady && !error && (
        <Text style={styles.loadingText}>Loading model... {Math.round(downloadProgress)}%</Text>
      )}

      {!preventLoad && (
        <Pressable
          style={[styles.button, (!isReady || isInferencing) && styles.buttonDisabled]}
          onPress={run}
          disabled={!isReady || isInferencing}
        >
          <Text style={styles.buttonText}>
            {isInferencing ? 'Running...' : 'Run Classifier (Hook)'}
          </Text>
        </Pressable>
      )}

      {localPath && (
        <Pressable style={[styles.button, { backgroundColor: '#c00' }]} onPress={deleteModel}>
          <Text style={styles.buttonText}>Delete Model</Text>
        </Pressable>
      )}

      {result !== '' && <Text style={styles.resultText}>{result}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: '#111',
    marginTop: 20,
  },
  buttonDisabled: { backgroundColor: '#888' },
  buttonText: { color: '#fff', fontSize: 16 },
  loadingText: { marginBottom: 10, fontSize: 16 },
  errorText: { color: 'red', marginBottom: 10, textAlign: 'center' },
  resultText: { marginTop: 20, fontSize: 18, textAlign: 'center', lineHeight: 24 },
});
