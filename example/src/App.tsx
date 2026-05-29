import { useEffect, useState } from "react";
import { View, Text, StyleSheet, Pressable } from "react-native";
import { MyLib as RNET } from "react-native-my-lib";
import { IMAGENET_CLASSES } from "./imagenetClasses";

const MODEL_PATH =
  "/Users/bhanc/workspace/jsi-workshops/efficientnet_v2_s_xnnpack_int8.pte";
const IMAGE_SHAPE = [1, 3, 384, 384];
const INPUT_SIZE = 1 * 3 * 384 * 384;

export default function App() {
  const [jsTicks, setJsTicks] = useState(0);

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

    let model: any = null;
    let input: any = null;
    let outputs: any[] = [];

    try {
      model = await RNET.loadModel(MODEL_PATH);

      console.log(
        `[DEMO] Model loaded successfully from ${model.path}\n`,
        `[DEMO] Model method names: ${RNET.getModelMethodNames(model).join(", ")}\n`,
        `[DEMO] Method Meta: ${JSON.stringify(RNET.getModelMethodMeta(model, "forward"), null, 2)}\n`,
      );

      input = RNET.createTensor(IMAGE_SHAPE, "float32");
      RNET.setTensorFromTypedArray(
        input,
        new Float32Array(INPUT_SIZE).fill(0.0),
      );

      console.log(
        "[DEMO] Allocated input tensor\n",
        "[DEMO] Copied data to input tensor\n",
      );

      let t = Date.now();
      outputs = await RNET.executeModelMethod(model, "forward", input);
      t = Date.now() - t;

      console.log(`[DEMO] Inference success! Elapsed: ${t}ms`);

      const logits = new Float32Array(1000);
      RNET.setTypedArrayFromTensor(logits, outputs[0]);

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
      RNET.disposeTensor(input);
      RNET.disposeTensor(outputs[0]);
      RNET.disposeModel(model);
      console.log("[DEMO] Cleanup finished.");
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.tickText}>JS Ticks: {jsTicks}</Text>
      <Pressable style={styles.button} onPress={run}>
        <Text style={styles.buttonText}>Run</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", alignItems: "center" },
  tickText: { fontSize: 18, marginBottom: 12 },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: "#111",
  },
  buttonText: { color: "#fff", fontSize: 16 },
});
