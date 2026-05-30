import { useEffect, useState } from "react";
import { View, Text, StyleSheet, Pressable, ScrollView, ActivityIndicator } from "react-native";
import { Model, Tensor, cv, type ModelOutput, getRegisteredBackends } from "react-native-my-lib";
import {
  useImage,
  Image,
  Canvas,
  Skia,
  AlphaType,
  ColorType,
  type SkImage,
} from "@shopify/react-native-skia";
import { IMAGENET_CLASSES } from "./imagenetClasses";

const MODEL_PATH = "/Users/bhanc/workspace/jsi-workshops/efficientnet_v2_s_xnnpack_int8.pte";
const IMAGE_URI = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg";

export default function App() {
  const [jsTicks, setJsTicks] = useState(0);
  const [cvStatus, setCvStatus] = useState<string>("");
  const [processedImage, setProcessedImage] = useState<SkImage | null>(null);

  const image = useImage(IMAGE_URI);

  useEffect(() => {
    let frameId: number;
    const tick = () => {
      setJsTicks((t) => (t + 1) % 1000);
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, []);

  async function run() {
    console.log("[DEMO] Starting background inference test...");
    let model: Model | null = null;
    let input: Tensor | null = null;
    let outputs: ModelOutput[] | null = null;

    try {
      model = await Model.loadAsync(MODEL_PATH);
      console.log(
        `[DEMO] Available Backends: ${getRegisteredBackends().join(", ")}\n`,
        `[DEMO] Model loaded successfully from ${model.path}\n`,
        `[DEMO] Model method names: ${model.methodNames.join(", ")}\n`,
        `[DEMO] Method Meta: ${JSON.stringify(model.getMethodMeta("forward"), null, 2)}\n`,
      );

      if (!image) throw new Error("Image not loaded yet");

      let t = Date.now();

      input = Tensor.fromTypedArray(
        image.readPixels() as Uint8Array,
        [image.height(), image.width(), 4],
        "uint8",
      )
        .through({ selfDispose: true }, cv.resize, { width: 384, height: 384, mode: "stretch" })
        .through({ selfDispose: true }, cv.cvtColor, "RGBA2RGB")
        .through({ selfDispose: true }, cv.toChannelsFirst)
        .through({ selfDispose: true }, cv.normalize, {
          alpha: [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)],
          beta: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        })
        .reshape([1, 3, 384, 384]);

      outputs = await model.executeAsync("forward", [input]);

      t = Date.now() - t;

      console.log(`[DEMO] Inference success! Elapsed: ${t}ms`);

      const output = outputs[0] as Tensor;
      const logits = output.toTypedArray({ selfDispose: true });
      const indices = Array.from({ length: logits.length }, (_, i) => i);
      indices.sort((a, b) => logits[b]! - logits[a]!);

      console.log(`[DEMO] Top-5 Classes (Inference Time: ${t}ms)`);
      for (let k = 0; k < 5; k++) {
        const idx = indices[k]!;
        const name = IMAGENET_CLASSES[idx] ?? `Class ${idx}`;
        console.log(`  ${k + 1}. ${name}`);
      }
    } catch (e: any) {
      console.error("[DEMO] Inference loop failed:", e.message);
    } finally {
      if (input) input.dispose();
      if (model) model.dispose();
      console.log("[DEMO] Cleanup finished.");
    }
  }

  async function runCV() {
    if (!image) {
      setCvStatus("Error: Image not loaded yet");
      return;
    }

    setCvStatus("Running CV pipeline...");

    try {
      console.log(
        `[CV DEMO] Read pixels from Skia Image, height: ${image.height()}, width: ${image.width()}`,
      );

      const normalizeOpts: cv.NormalizeOptions = {
        alpha: [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)],
        beta: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
      };

      let t = Date.now();
      const out = Tensor.fromTypedArray(image.readPixels() as Uint8Array, [
        image.height(),
        image.width(),
        4,
      ])
        .through({ selfDispose: true }, cv.resize, { width: 300, height: 300, mode: "letterbox" })
        .through({ selfDispose: true }, cv.cvtColor, "RGBA2RGB")
        .through({ selfDispose: true }, cv.toChannelsFirst)
        .through({ selfDispose: true }, cv.normalize, normalizeOpts)
        .through({ selfDispose: true }, cv.toChannelsLast)
        .through({ selfDispose: true }, cv.cvtColor, "RGB2RGBA")
        .toTypedArray({ selfDispose: true }) as Float32Array;
      t = Date.now() - t;

      const data = Skia.Data.fromBytes(new Uint8Array(out.buffer, out.byteOffset, out.byteLength));
      const outImg = Skia.Image.MakeImage(
        {
          width: 300,
          height: 300,
          alphaType: AlphaType.Opaque,
          colorType: ColorType.RGBA_F32,
        },
        data,
        300 * 4 * 4, // rowBytes: width * channels * bytesPerFloat
      );

      if (!outImg) {
        throw new Error("Failed to construct Skia Image from float32 pixels");
      }

      setProcessedImage(outImg);
      setCvStatus(`Pipeline succeeded in ${t}ms!`);
      console.log(`[CV DEMO] Pipeline success in ${t}ms`);
    } catch (e: any) {
      console.error("[CV DEMO] Pipeline failed:", e.message);
      setCvStatus(`Failed: ${e.message}`);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>JSI Computer Vision Demo</Text>
      <Text style={styles.tickText}>JS Thread Active (Ticks: {jsTicks})</Text>

      <View style={styles.buttonRow}>
        <Pressable style={styles.button} onPress={run}>
          <Text style={styles.buttonText}>Run Inference</Text>
        </Pressable>
        <Pressable
          style={[styles.button, styles.cvButton, !image && styles.buttonDisabled]}
          onPress={runCV}
          disabled={!image}
        >
          <Text style={styles.buttonText}>Run CV Pipeline</Text>
        </Pressable>
      </View>

      {cvStatus !== "" && <Text style={styles.statusText}>{cvStatus}</Text>}

      <View style={styles.gallery}>
        <View style={styles.card}>
          <Text style={styles.cardHeader}>Source Image</Text>
          {image ? (
            <Canvas style={styles.canvas}>
              <Image image={image} x={0} y={0} width={280} height={200} fit="contain" />
            </Canvas>
          ) : (
            <View style={styles.placeholder}>
              <ActivityIndicator size="small" color="#6366F1" />
              <Text style={styles.placeholderText}>Downloading image...</Text>
            </View>
          )}
        </View>

        {processedImage && (
          <View style={styles.card}>
            <Text style={styles.cardHeader}>Processed (300x300 Normalized)</Text>
            <Canvas style={styles.canvas}>
              <Image image={processedImage} x={0} y={0} width={280} height={200} fit="contain" />
            </Canvas>
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    justifyContent: "flex-start",
    alignItems: "center",
    backgroundColor: "#0F172A",
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#F8FAFC",
    marginBottom: 8,
  },
  tickText: {
    fontSize: 14,
    color: "#94A3B8",
    marginBottom: 24,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 20,
  },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: "#334155",
    elevation: 2,
  },
  cvButton: {
    backgroundColor: "#6366F1",
  },
  buttonDisabled: {
    backgroundColor: "#475569",
    opacity: 0.6,
  },
  buttonText: {
    color: "#F8FAFC",
    fontSize: 16,
    fontWeight: "600",
  },
  statusText: {
    fontSize: 15,
    color: "#38BDF8",
    marginBottom: 24,
    fontWeight: "500",
    textAlign: "center",
  },
  gallery: {
    width: "100%",
    gap: 20,
    alignItems: "center",
  },
  card: {
    width: 320,
    backgroundColor: "#1E293B",
    borderRadius: 12,
    padding: 16,
    alignItems: "center",
    borderColor: "#334155",
    borderWidth: 1,
  },
  cardHeader: {
    fontSize: 16,
    fontWeight: "600",
    color: "#F1F5F9",
    marginBottom: 12,
  },
  canvas: {
    width: 280,
    height: 200,
    backgroundColor: "#0F172A",
    borderRadius: 8,
  },
  placeholder: {
    width: 280,
    height: 200,
    backgroundColor: "#0F172A",
    borderRadius: 8,
    justifyContent: "center",
    alignItems: "center",
    gap: 8,
  },
  placeholderText: {
    color: "#94A3B8",
    fontSize: 14,
  },
});
