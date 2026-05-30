import { View, Text, StyleSheet, Pressable, Platform } from "react-native";
import { getRegisteredBackends, loadModel, tensor, cv, math } from "react-native-my-lib";
import { runOnRuntimeAsync, createWorkletRuntime } from "react-native-worklets";

const MODEL_PATH = Platform.select({
  ios: "/Users/bhanc/workspace/jsi-workshops/efficientnet_v2_s_xnnpack_int8.pte",
  /* adb push efficientnet_v2_s_xnnpack_int8.pte /data/local/tmp/efficientnet_v2_s_xnnpack_int8.pte */
  android: "/data/local/tmp/efficientnet_v2_s_xnnpack_int8.pte",
})!;

const pixels = new Uint8Array(1920 * 1080 * 4);
const workletRuntime = createWorkletRuntime({ name: "InferenceWorklet" });

export default function App() {
  async function run() {
    try {
      let t: number;

      t = Date.now();
      const result = await runOnRuntimeAsync(workletRuntime, () => {
        "worklet";
        try {
          return { ok: true, value: loadModel(MODEL_PATH) };
        } catch (e: any) {
          return { ok: false, error: e instanceof Error ? e.message : String(e) };
        }
      });
      if (!result.ok) throw new Error(result.error);
      const model = result.value!;
      t = Date.now() - t;
      console.log(`[DEMO] Model loaded in worklet successfully! Elapsed: ${t}ms`);

      console.log(
        "[DEMO] Starting background inference test...\n",
        `[DEMO] Available Backends: ${getRegisteredBackends().join(", ")}\n`,
        `[DEMO] Model loaded successfully from ${model.path}\n`,
        `[DEMO] Model method names: ${model.getMethodNames().join(", ")}\n`,
        `[DEMO] Method Meta: ${JSON.stringify(model.getMethodMeta("forward"), null, 2)}\n`,
      );

      t = Date.now();
      const src = tensor("uint8", [1080, 1920, 4], pixels);
      const tmp1 = tensor("uint8", [384, 384, 4]);
      const tmp2 = tensor("uint8", [384, 384, 3]);
      const tmp3 = tensor("uint8", [3, 384, 384]);
      const tmp4 = tensor("float32", [3, 384, 384]);
      const tmp5 = tensor("float32", [1, 3, 384, 384]);

      const probabilities = tensor("float32", [1, 1000]);
      const outputTensors = model
        .getMethodMeta("forward")
        .outputTensorMeta.map((m) => tensor(m.dtype, m.shape));

      t = Date.now() - t;

      console.log(`[DEMO] Tensor allocation success! Elapsed: ${t}ms`);

      for (let i = 0; i < 20; i++) {
        t = Date.now();

        const result = await runOnRuntimeAsync(workletRuntime, () => {
          "worklet";
          try {
            const input = src
              .through(cv.resize, tmp1, { mode: "stretch" })
              .through(cv.cvtColor, tmp2, "RGBA2RGB")
              .through(cv.toChannelsFirst, tmp3)
              .through(cv.normalize, tmp4, { alpha: 1 / 255.0 })
              .reshape(tmp5);

            model.execute("forward", [input], outputTensors);
            const logits = outputTensors[0];
            if (!logits) throw new Error("forward did not return logits");

            logits.through(math.softmax, probabilities);

            return { ok: true, value: { probabilities } };
          } catch (e: any) {
            return { ok: false, error: e.message };
          }
        });

        if (!result.ok) throw new Error(result.error);

        t = Date.now() - t;

        console.log(`[DEMO] Inference success! Elapsed: ${t}ms`);
      }

      console.log(probabilities.getData(new Float32Array(1000)).slice(0, 10));

      src.dispose();
      tmp1.dispose();
      tmp2.dispose();
      tmp3.dispose();
      tmp4.dispose();
      tmp5.dispose();
      probabilities.dispose();

      for (const t of outputTensors) t.dispose();
    } catch (e: any) {
      console.error("[DEMO] Inference loop failed:", e.message);
    }
  }

  return (
    <View style={styles.container}>
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
