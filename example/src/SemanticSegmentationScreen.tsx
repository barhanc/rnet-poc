import { useState } from 'react';
import { View, Text, StyleSheet, Pressable, Dimensions, ScrollView } from 'react-native';
import { useSemanticSegmenter, models } from 'react-native-my-lib';
import {
  useImage,
  Canvas,
  Image as SkImage,
  Skia,
  Mask,
  AlphaType,
  ColorType,
} from '@shopify/react-native-skia';
import * as ImagePicker from 'expo-image-picker';
import type { SkImage as ISkImage } from '@shopify/react-native-skia';

const imageUrl =
  'https://raw.githubusercontent.com/yavuzceliker/sample-images/refs/heads/main/docs/image-1001.jpg';

const AVAILABLE_MODELS = [
  { name: 'Selfie FP32', model: models.semanticSegmentation.SELFIE_SEGMENTATION.XNNPACK_FP32 },
  {
    name: 'LRASPP INT8',
    model: models.semanticSegmentation.LRASPP_MOBILENET_V3_LARGE.XNNPACK_INT8,
  },
  {
    name: 'LRASPP FP32',
    model: models.semanticSegmentation.LRASPP_MOBILENET_V3_LARGE.XNNPACK_FP32,
  },
  {
    name: 'DeepLabV3 FP32',
    model: models.semanticSegmentation.DEEPLAB_V3_MOBILENET_V3_LARGE.XNNPACK_FP32,
  },
  {
    name: 'DeepLabV3 INT8',
    model: models.semanticSegmentation.DEEPLAB_V3_MOBILENET_V3_LARGE.XNNPACK_INT8,
  },
];

export function SemanticSegmentationScreen() {
  const [imageUri, setImageUri] = useState<string>(imageUrl);
  const image = useImage(imageUri);
  const [resultImage, setResultImage] = useState<ISkImage | null>(null);
  const [isInferencing, setIsInferencing] = useState(false);
  const [preventLoad, setPreventLoad] = useState(true);
  const [selectedModelIdx, setSelectedModelIdx] = useState(1);

  const { isReady, error, downloadProgress, segment, labels } = useSemanticSegmenter(
    AVAILABLE_MODELS[selectedModelIdx]!.model as any,
    { preventLoad },
  );

  const [lastColormap, setLastColormap] = useState<Record<
    string,
    [number, number, number, number]
  > | null>(null);

  async function run() {
    if (!segment || !image) return;

    setIsInferencing(true);

    try {
      const pixels = image.readPixels() as Uint8Array;

      let t = Date.now();
      const result = segment({
        data: pixels,
        width: image.width(),
        height: image.height(),
        format: 'rgba',
        layout: 'hwc',
      });
      console.log('Inference time:', Date.now() - t, 'ms');

      const output = result.buffer;
      setLastColormap(result.colormap as Record<string, [number, number, number, number]>);

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

  const [isolateForeground, setIsolateForeground] = useState(false);

  const isSelfie = AVAILABLE_MODELS[selectedModelIdx]!.name.includes('Selfie');

  return (
    <View style={styles.container}>
      <View style={styles.pickerContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {AVAILABLE_MODELS.map((m, idx) => (
            <Pressable
              key={m.name}
              style={[styles.modelTab, selectedModelIdx === idx && styles.modelTabActive]}
              onPress={() => {
                setSelectedModelIdx(idx);
                setPreventLoad(true);
                setResultImage(null);
                setIsolateForeground(false); // Reset masking mode
              }}
            >
              <Text
                style={[styles.modelTabText, selectedModelIdx === idx && styles.modelTabTextActive]}
              >
                {m.name}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      </View>

      {error && <Text style={styles.errorText}>Error: {error.message}</Text>}

      {preventLoad && (
        <Pressable style={styles.button} onPress={() => setPreventLoad(false)}>
          <Text style={styles.buttonText}>Download {AVAILABLE_MODELS[selectedModelIdx]!.name}</Text>
        </Pressable>
      )}

      {!preventLoad && !isReady && !error && (
        <Text style={styles.loadingText}>Loading model... {Math.round(downloadProgress)}%</Text>
      )}

      {!preventLoad && isReady && (
        <>
          <View style={{ flexDirection: 'row', width: '100%', justifyContent: 'space-between' }}>
            <Pressable
              style={[
                styles.button,
                isInferencing && styles.buttonDisabled,
                { flex: 1, marginRight: 5 },
              ]}
              onPress={run}
              disabled={isInferencing}
            >
              <Text style={styles.buttonText}>
                {isInferencing ? 'Segmenting...' : 'Run Segmentation'}
              </Text>
            </Pressable>

            <Pressable
              style={[styles.button, { backgroundColor: '#0055cc', flex: 1, marginLeft: 5 }]}
              onPress={pickImage}
            >
              <Text style={styles.buttonText}>Select Image</Text>
            </Pressable>
          </View>

          {isSelfie && resultImage && (
            <Pressable
              style={[styles.button, { backgroundColor: '#34C759' }]}
              onPress={() => setIsolateForeground(!isolateForeground)}
            >
              <Text style={styles.buttonText}>
                {isolateForeground ? 'Show Mask Overlay' : 'Isolate Foreground'}
              </Text>
            </Pressable>
          )}

          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.labelsContainer}
          >
            {labels?.map((l) => {
              const rgba = lastColormap?.[l as string] || [255, 255, 255, 255];
              const color = `rgb(${rgba[0]}, ${rgba[1]}, ${rgba[2]})`;

              return (
                <View key={String(l)} style={styles.labelBadge}>
                  <View
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: 6,
                      backgroundColor: color,
                      marginRight: 6,
                    }}
                  />
                  <Text style={styles.labelText}>{String(l)}</Text>
                </View>
              );
            })}
          </ScrollView>
        </>
      )}

      <Canvas style={styles.canvas}>
        {/* Draw base image always */}
        {image && !isolateForeground && (
          <SkImage
            image={image}
            fit="contain"
            x={0}
            y={0}
            width={Dimensions.get('window').width - 40}
            height={300}
          />
        )}

        {/* If isolating foreground, use the Mask component with luminance mode */}
        {image && resultImage && isolateForeground && (
          <Mask
            mode="luminance"
            mask={
              <SkImage
                image={resultImage}
                fit="contain"
                x={0}
                y={0}
                width={Dimensions.get('window').width - 40}
                height={300}
              />
            }
          >
            <SkImage
              image={image}
              fit="contain"
              x={0}
              y={0}
              width={Dimensions.get('window').width - 40}
              height={300}
            />
          </Mask>
        )}

        {/* If overlaying, just draw the resultImage with opacity */}
        {resultImage && !isolateForeground && (
          <SkImage
            image={resultImage}
            fit="contain"
            x={0}
            y={0}
            width={Dimensions.get('window').width - 40}
            height={300}
            opacity={0.6}
          />
        )}
      </Canvas>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', padding: 20 },
  pickerContainer: { height: 50, marginBottom: 10, width: '100%' },
  modelTab: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#eee',
    marginRight: 10,
    justifyContent: 'center',
  },
  modelTabActive: { backgroundColor: '#111' },
  modelTabText: { color: '#666', fontWeight: '500' },
  modelTabTextActive: { color: '#fff' },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: '#007AFF',
    borderRadius: 8,
    marginVertical: 5,
    width: '100%',
    alignItems: 'center',
  },
  buttonDisabled: { backgroundColor: '#A0CFFF' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  errorText: { color: 'red', marginVertical: 10, textAlign: 'center' },
  loadingText: { marginVertical: 10, fontSize: 16 },
  labelsContainer: { marginTop: 10, maxHeight: 40, width: '100%' },
  labelBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: '#e0e0e0',
    borderRadius: 12,
    marginRight: 8,
    justifyContent: 'center',
  },
  labelText: { color: '#333', fontSize: 12, fontWeight: '500' },
  canvas: { width: Dimensions.get('window').width - 40, height: 300, marginTop: 20 },
});
