import { View, Text, StyleSheet, Pressable } from "react-native";
import { useState } from "react";
import {
  getRegisteredBackends,
  loadModel,
  tensor,
  cv,
  math,
  type Tensor,
} from "react-native-my-lib";
import { runOnRuntimeAsync, createWorkletRuntime } from "react-native-worklets";
import ReactNativeBlobUtil from "react-native-blob-util";
import { useImage } from "@shopify/react-native-skia";
import { IMAGENET_CLASSES } from "./imagenetClasses";

const urlPrefix = "https://huggingface.co/software-mansion/react-native-executorch";
const modelUrl = `${urlPrefix}-efficientnet-v2-s/resolve/main/xnnpack/efficientnet_v2_s_xnnpack_int8.pte`;
const workletRuntime = createWorkletRuntime({ name: "InferenceWorklet" });

export default function App() {
  const [downloadProgress, setDownloadProgress] = useState<number>(0);
  const [isDownloading, setIsDownloading] = useState<boolean>(false);
  const [modelPath, setModelPath] = useState<string | null>(null);
  const image = useImage("https://upload.wikimedia.org/wikipedia/commons/d/d6/Thai-Ridgeback.jpg");

  async function run() {
    try {
      let t: number;

      if (!image) throw new Error("Image is still loading");
      const pixels = image.readPixels();
      if (!pixels) throw new Error("Failed to read image pixels");

      setIsDownloading(true);

      const res = await ReactNativeBlobUtil.config({ appendExt: "pte", fileCache: true })
        .fetch("GET", modelUrl)
        .progress((received, total) => setDownloadProgress(Number(received) / Number(total)));

      const path = res.path();
      setModelPath(path);
      setIsDownloading(false);

      t = Date.now();
      const result = await runOnRuntimeAsync(
        workletRuntime,
        (p: string) => {
          "worklet";
          try {
            return { ok: true, value: loadModel(p) };
          } catch (e: any) {
            return { ok: false, error: e?.message ?? String(e) };
          }
        },
        path,
      );

      if (!result.ok) throw new Error(result.error);
      const model = result.value!;
      t = Date.now() - t;

      console.log(
        `[DEMO] Model loaded in worklet successfully! Elapsed: ${t}ms\n`,
        "[DEMO] Starting background inference test...\n",
        `[DEMO] Available Backends: ${getRegisteredBackends().join(", ")}\n`,
        `[DEMO] Model loaded successfully from ${model.path}\n`,
        `[DEMO] Model method names: ${model.getMethodNames().join(", ")}\n`,
        `[DEMO] Method Meta: ${JSON.stringify(model.getMethodMeta("forward"), null, 2)}\n`,
      );

      t = Date.now();
      const src = tensor("uint8", [image.height(), image.width(), 4], pixels as Uint8Array);
      const out = tensor("float32", [1, 1000]);
      const prb = tensor("float32", [1, 1000]);
      const aux: [Tensor, Tensor, Tensor, Tensor, Tensor] = [
        tensor("uint8", [384, 384, 4]),
        tensor("uint8", [384, 384, 3]),
        tensor("uint8", [3, 384, 384]),
        tensor("float32", [3, 384, 384]),
        tensor("float32", [1, 3, 384, 384]),
      ];

      t = Date.now() - t;
      console.log(`[DEMO] Tensor allocation success! Elapsed: ${t}ms`);

      const error = await runOnRuntimeAsync(workletRuntime, () => {
        "worklet";
        try {
          for (let i = 0; i < 10; i++) {
            t = Date.now();
            const input = src
              .through(cv.resize, aux[0], { mode: "stretch", interpolation: "linear" })
              .through(cv.cvtColor, aux[1], "RGBA2RGB")
              .through(cv.toChannelsFirst, aux[2])
              .through(cv.normalize, aux[3])
              .reshape(aux[4]);

            model.execute("forward", [input], [out]);
            out.through(math.softmax, prb);
            t = Date.now() - t;

            console.log(`[DEMO] Inference success! Elapsed: ${t}ms`);
          }
        } catch (e: any) {
          return e?.message ?? String(e);
        }
      });

      if (error) throw new Error(error);

      console.log(
        Array.from(prb.getData(new Float32Array(prb.numel)))
          .map((value, index) => ({ index, value }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 5)
          .map((x) => `#${IMAGENET_CLASSES[x.index]} (${(x.value * 100).toFixed(6)}%)`),
      );

      model.dispose();
      src.dispose();
      out.dispose();
      prb.dispose();
      aux.forEach((t) => t.dispose());
    } catch (e: any) {
      console.error("[DEMO] Inference loop failed:", e.message);
      setIsDownloading(false);
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.tickText}>
        {isDownloading
          ? `Downloading… ${Math.round(downloadProgress * 100)}%`
          : modelPath
            ? `Model ready`
            : `Model not downloaded`}
      </Text>
      <Pressable style={styles.button} onPress={run}>
        <Text style={styles.buttonText}>Run</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", alignItems: "center" },
  tickText: { fontSize: 18, marginBottom: 12 },
  button: { paddingHorizontal: 20, paddingVertical: 12, borderRadius: 8, backgroundColor: "#111" },
  buttonText: { color: "#fff", fontSize: 16 },
});
