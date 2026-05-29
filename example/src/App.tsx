import { useEffect, useState } from "react";
import {
  View,
  Text,
  Button,
  StyleSheet,
  ActivityIndicator,
  Platform,
} from "react-native";
import { MyLib, type NativeModel } from "react-native-my-lib";
import RNFS from "react-native-fs";

const modelPaths = {
  int8:
    Platform.OS === "ios"
      ? "/Users/bhanc/workspace/jsi-workshops/efficientnet_v2_s_xnnpack_int8.pte"
      : RNFS.ExternalDirectoryPath + "/efficientnet_v2_s_xnnpack_int8.pte",
  fp32:
    Platform.OS === "ios"
      ? "/Users/bhanc/workspace/jsi-workshops/efficientnet_v2_s_xnnpack_fp32.pte"
      : RNFS.ExternalDirectoryPath + "/efficientnet_v2_s_xnnpack_fp32.pte",
};
const IMAGE_SHAPE = [1, 3, 384, 384];
const NUM_PIXELS = 1 * 3 * 384 * 384;

export default function App() {
  const [model, setModel] = useState<NativeModel | null>(null);
  const [modelVariant, setModelVariant] = useState<"int8" | "fp32">("fp32");
  const [isInferencing, setIsInferencing] = useState(false);
  const [isDiagnosing, setIsDiagnosing] = useState(false);
  const [resultText, setResultText] = useState("Model not loaded");

  // A simple counter to prove the main JS thread is not frozen
  const [tick, setTick] = useState(0);

  useEffect(() => {
    // Ticks every 50ms. If the main thread blocks, this stops updating.
    const interval = setInterval(() => setTick((t) => t + 1), 50);
    return () => clearInterval(interval);
  }, []);

  const handleLoadModel = () => {
    try {
      const modelPath = modelPaths[modelVariant];

      if (model) {
        MyLib.disposeModel(model);
      }

      const loadedModel = MyLib.loadModel(modelPath);
      setModel(loadedModel);
      setResultText(
        `Model loaded successfully!\nVariant: ${modelVariant}\nPath: ${modelPath}`,
      );
    } catch (e: any) {
      console.error("Load model error:", e);
      setResultText(`Failed to load: ${e.message}`);
    }
  };

  const handleRunDiagnostics = async () => {
    if (!model) return;

    setIsDiagnosing(true);
    setResultText("Running diagnostics...");

    try {
      const methodNames = MyLib.getModelMethodNames(model);
      const methodName = methodNames.includes("forward")
        ? "forward"
        : (methodNames[0] ?? "forward");

      const methodMeta = MyLib.getModelMethodMeta(model, methodName);

      console.log("=== DIAGNOSTICS START ===");
      console.log("Model variant:", modelVariant);
      console.log("Available methods:", methodNames);
      console.log("Selected method:", methodName);
      console.log("Method metadata:", JSON.stringify(methodMeta, null, 2));

      const inputTensor = MyLib.createTensor(IMAGE_SHAPE, "float32");
      const dummyImageData = new Float32Array(NUM_PIXELS);
      MyLib.setTensorFromTypedArray(inputTensor, dummyImageData);

      let executionSummary = "Execution: not attempted";
      try {
        console.log("Attempting execution with input shape:", IMAGE_SHAPE);
        MyLib.executeModelMethod(model, methodName, inputTensor);
        executionSummary = "Execution: success on main thread";
        console.log("✓ Execution successful!");
      } catch (execError: any) {
        console.error("✗ Diagnostic execution error:", execError);
        console.error("Error message:", execError?.message);
        console.error("Full error:", execError);
        executionSummary = `Execution failed: ${execError?.message ?? String(execError)}`;
      }

      console.log("=== DIAGNOSTICS END ===");

      setResultText(
        [
          `Diagnostics for ${modelVariant}`,
          `Methods: ${JSON.stringify(methodNames)}`,
          `Selected method: ${methodName}`,
          `Method meta: ${JSON.stringify(methodMeta)}`,
          executionSummary,
        ].join("\n"),
      );
    } catch (e: any) {
      console.error("Diagnostics error:", e);
      setResultText(`Diagnostic failed: ${e.message}`);
    } finally {
      setIsDiagnosing(false);
    }
  };

  const handleRunInference = async () => {
    if (!model) return;

    setIsInferencing(true);
    setResultText("Running inference...");

    try {
      console.log("\n=== INFERENCE START ===");
      // 1. Create and populate the input tensor
      const inputTensor = MyLib.createTensor(IMAGE_SHAPE, "float32");
      const dummyImageData = new Float32Array(NUM_PIXELS);
      MyLib.setTensorFromTypedArray(inputTensor, dummyImageData);
      console.log("✓ Input tensor created:", IMAGE_SHAPE);

      const startTime = Date.now();

      // 2. Hand off to the background Worklet thread
      console.log("Calling runInferenceAsync...");
      const outputs = await MyLib.runInferenceAsync(
        model,
        "forward",
        inputTensor,
      );
      console.log("✓ runInferenceAsync completed");

      const endTime = Date.now();

      // 3. Extract the output data back into a JS array
      const outputTensor = outputs[0];
      if (!outputTensor) {
        throw new Error("Model returned no outputs");
      }
      const outputArray = MyLib.getTypedArrayFromTensor(
        outputTensor,
      ) as Float32Array;

      if (outputArray.length === 0) {
        throw new Error("Output tensor is empty");
      }

      // 4. Simple ArgMax to find the predicted class
      let maxIdx = 0;
      let maxVal = outputArray[0] as number;
      for (let i = 1; i < outputArray.length; i++) {
        const value = outputArray[i] as number;
        if (value > maxVal) {
          maxVal = value;
          maxIdx = i;
        }
      }

      console.log("✓ Inference successful!");
      console.log("  Prediction class:", maxIdx);
      console.log("  Confidence:", maxVal.toFixed(4));
      console.log("  Time:", (endTime - startTime).toFixed(2) + "ms");
      console.log("=== INFERENCE END ===\n");

      setResultText(
        `Prediction: Class ${maxIdx}\n` +
          `Confidence: ${maxVal.toFixed(4)}\n` +
          `Time: ${(endTime - startTime).toFixed(2)}ms`,
      );
    } catch (e: any) {
      console.error("Inference error:", e.message);
      setResultText(`Inference failed: ${e.message}`);
    } finally {
      setIsInferencing(false);
    }
  };

  return (
    <View style={styles.container}>
      {/* Thread unblock proof */}
      <View style={styles.tickerContainer}>
        <Text style={styles.tickerText}>UI Thread Ticker: {tick}</Text>
        <Text style={styles.subtitle}>This should never freeze!</Text>
      </View>

      <View style={styles.controls}>
        <Button
          title={`0. Toggle Model Variant (${modelVariant.toUpperCase()})`}
          onPress={() =>
            setModelVariant((current) => (current === "fp32" ? "int8" : "fp32"))
          }
          disabled={isInferencing || isDiagnosing}
        />

        <Button
          title="1. Load Model"
          onPress={handleLoadModel}
          disabled={isInferencing || isDiagnosing}
        />

        <Button
          title={isInferencing ? "Processing..." : "2. Run Inference"}
          onPress={handleRunInference}
          disabled={!model || isInferencing || isDiagnosing}
        />

        <Button
          title={isDiagnosing ? "Diagnosing..." : "3. Run Diagnostics"}
          onPress={handleRunDiagnostics}
          disabled={!model || isInferencing || isDiagnosing}
        />
      </View>

      {isInferencing && (
        <ActivityIndicator
          size="large"
          color="#0000ff"
          style={{ marginBottom: 20 }}
        />
      )}

      <Text style={styles.result}>{resultText}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#F5FCFF",
  },
  tickerContainer: {
    padding: 20,
    backgroundColor: "#e0e0e0",
    borderRadius: 8,
    marginBottom: 40,
    alignItems: "center",
  },
  tickerText: {
    fontSize: 24,
    fontWeight: "bold",
    fontVariant: ["tabular-nums"], // Keeps the numbers from jittering left/right
  },
  subtitle: {
    fontSize: 12,
    color: "#666",
    marginTop: 5,
  },
  controls: {
    gap: 15,
    marginBottom: 30,
    width: "100%",
    maxWidth: 300,
  },
  result: {
    fontSize: 16,
    textAlign: "center",
    color: "#333",
    lineHeight: 24,
  },
});
