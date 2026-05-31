import { useState } from 'react';
import { View, Text, StyleSheet, Pressable, Dimensions } from 'react-native';
import { useStyleTransfer, models } from 'react-native-my-lib';
import {
  useImage,
  Canvas,
  Image as SkImage,
  Skia,
  AlphaType,
  ColorType,
} from '@shopify/react-native-skia';
import * as ImagePicker from 'expo-image-picker';
import type { SkImage as ISkImage } from '@shopify/react-native-skia';

const imageUrl =
  'https://raw.githubusercontent.com/yavuzceliker/sample-images/refs/heads/main/docs/image-1001.jpg';

export function HookScreen() {
  const [imageUri, setImageUri] = useState<string>(imageUrl);
  const image = useImage(imageUri);
  const [resultImage, setResultImage] = useState<ISkImage | null>(null);
  const [isInferencing, setIsInferencing] = useState(false);
  const [preventLoad, setPreventLoad] = useState(true);

  const { isReady, error, downloadProgress, transfer } = useStyleTransfer(
    models.styleTransfer.CANDY.XNNPACK_INT8,
    { preventLoad },
  );

  // inspectModel(
  //
  // );

  async function run() {
    if (!transfer || !image) return;

    setIsInferencing(true);

    try {
      const pixels = image.readPixels() as Uint8Array;

      let t = Date.now();
      const output = transfer({
        data: pixels,
        width: image.width(),
        height: image.height(),
        format: 'rgba',
        layout: 'hwc',
      });
      console.log('Inference time:', Date.now() - t, 'ms');

      const data = Skia.Data.fromBytes(output.data);
      const skImg = Skia.Image.MakeImage(
        {
          width: output.width,
          height: output.height,
          alphaType: AlphaType.Unpremul,
          colorType: ColorType.RGBA_8888,
        },
        data,
        output.width * 4,
      );

      setResultImage(skImg);
    } catch (e) {
      console.error('Error in run:', e);
    } finally {
      setIsInferencing(false);
    }
  }

  async function pickImage() {
    const pickerResult = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!pickerResult.canceled && pickerResult.assets && pickerResult.assets.length > 0) {
      setImageUri(pickerResult.assets[0]?.uri || '');
      setResultImage(null);
    }
  }

  return (
    <View style={styles.container}>
      {error && <Text style={styles.errorText}>Error: {error.message}</Text>}

      {preventLoad && (
        <Pressable style={styles.button} onPress={() => setPreventLoad(false)}>
          <Text style={styles.buttonText}>Download Candy Model</Text>
        </Pressable>
      )}

      {!preventLoad && !isReady && !error && (
        <Text style={styles.loadingText}>Loading model... {Math.round(downloadProgress)}%</Text>
      )}

      {!preventLoad && isReady && (
        <>
          <Pressable
            style={[styles.button, isInferencing && styles.buttonDisabled]}
            onPress={run}
            disabled={isInferencing}
          >
            <Text style={styles.buttonText}>
              {isInferencing ? 'Stylizing...' : 'Run Style Transfer'}
            </Text>
          </Pressable>

          <Pressable style={[styles.button, { backgroundColor: '#0055cc' }]} onPress={pickImage}>
            <Text style={styles.buttonText}>Select Image</Text>
          </Pressable>
        </>
      )}

      <Canvas style={styles.canvas}>
        {image && !resultImage && (
          <SkImage
            image={image}
            fit="contain"
            x={0}
            y={0}
            width={Dimensions.get('window').width - 40}
            height={300}
          />
        )}
        {resultImage && (
          <SkImage
            image={resultImage}
            fit="contain"
            x={0}
            y={0}
            width={Dimensions.get('window').width - 40}
            height={300}
          />
        )}
      </Canvas>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', padding: 20 },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: '#007AFF',
    borderRadius: 8,
    marginVertical: 10,
    width: '100%',
    alignItems: 'center',
  },
  buttonDisabled: { backgroundColor: '#A0CFFF' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  errorText: { color: 'red', marginVertical: 10, textAlign: 'center' },
  loadingText: { marginVertical: 10, fontSize: 16 },
  canvas: { width: Dimensions.get('window').width - 40, height: 300, marginTop: 20 },
});
