import React, { useState, useEffect, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Animated,
  Easing,
} from "react-native";
import * as ET from "react-native-my-lib";

export default function ProTestScreen() {
  const [model, setModel] = useState<any>(null);
  const [methods, setMethods] = useState<string[]>([]);
  const [status, setStatus] = useState("Idle");
  const [logs, setLogs] = useState<{ t: string; m: string }[]>([]);

  // UI Thread Heartbeat
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const [uiCounter, setUiCounter] = useState(0);

  // 1. Start a continuous counter to show UI thread health
  useEffect(() => {
    const timer = setInterval(() => setUiCounter((c) => c + 1), 100);
    return () => clearInterval(timer);
  }, []);

  // 2. Start a continuous animation pulse
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.5,
          duration: 800,
          useNativeDriver: true,
          easing: Easing.inOut(Easing.ease),
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
          easing: Easing.inOut(Easing.ease),
        }),
      ]),
    ).start();
  }, [pulseAnim]);

  const addLog = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString().split(" ")[0];
    setLogs((prev) => [{ t: timestamp, m: msg }, ...prev.slice(0, 15)]);
  };

  const runLoad = async () => {
    setStatus("LOADING...");
    try {
      const m = await ET.loadModel(
        "/Users/bhanc/workspace/jsi-workshops/qwen2_5_1_5b_8da4w.pte",
      );
      setModel(m);
      // Sync call for metadata
      const names = ET.getModelMethodNames(m);
      setMethods(names);
      addLog("Model Loaded (Background)");
      setStatus("READY");
    } catch (e: any) {
      addLog(`Error: ${e.message}`);
      setStatus("ERROR");
    }
  };

  const runExec = async (name: string) => {
    setStatus(`RUNNING ${name}...`);
    try {
      addLog(`Started ${name} (off-thread)`);
      const result = await ET.executeModel(model, name);
      addLog(`Result: ${result}`);
      setStatus("READY");
    } catch (e: any) {
      addLog(`Exec Error: ${e}`);
      setStatus("ERROR");
    }
  };

  return (
    <View style={styles.container}>
      {/* UI Thread Monitor Section */}
      <View style={styles.monitorCard}>
        <Text style={styles.monitorLabel}>MAIN THREAD HEARTBEAT</Text>
        <View style={styles.heartbeatRow}>
          <Animated.View
            style={[styles.pulse, { transform: [{ scale: pulseAnim }] }]}
          />
          <Text style={styles.counterText}>Counter: {uiCounter}</Text>
        </View>
        <Text style={styles.hint}>
          If this circle stops pulsing or the counter freezes, the UI thread is
          blocked.
        </Text>
      </View>

      <Text style={styles.statusText}>
        STATUS: <Text style={styles.statusValue}>{status}</Text>
      </Text>

      {/* Control Section */}
      {!model ? (
        <TouchableOpacity style={styles.primaryBtn} onPress={runLoad}>
          <Text style={styles.btnText}>LOAD 8GB EXECUTORCH MODEL</Text>
        </TouchableOpacity>
      ) : (
        <View style={styles.controls}>
          <Text style={styles.sectionTitle}>Methods (Sync Fetched)</Text>
          <View style={styles.methodGrid}>
            {methods.map((name) => (
              <TouchableOpacity
                key={name}
                style={styles.methodBtn}
                onPress={() => runExec(name)}
              >
                <Text style={styles.methodBtnText}>{name}</Text>
              </TouchableOpacity>
            ))}
          </View>
          <TouchableOpacity
            style={styles.disposeBtn}
            onPress={async () => {
              await ET.disposeModel(model);
              setModel(null);
              setMethods([]);
              addLog("Memory Freed");
              setStatus("IDLE");
            }}
          >
            <Text style={styles.btnText}>DISPOSE & FREE RAM</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Log Terminal */}
      <View style={styles.terminal}>
        <ScrollView inverted contentContainerStyle={{ paddingBottom: 10 }}>
          {logs.map((log, i) => (
            <Text key={i} style={styles.logText}>
              <Text style={styles.logTime}>[{log.t}]</Text> {log.m}
            </Text>
          ))}
        </ScrollView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#121212",
    padding: 20,
    paddingTop: 60,
  },
  monitorCard: {
    backgroundColor: "#1e1e1e",
    padding: 15,
    borderRadius: 12,
    marginBottom: 20,
    borderLeftWidth: 4,
    borderLeftColor: "#00e676",
  },
  monitorLabel: {
    color: "#888",
    fontSize: 10,
    fontWeight: "bold",
    letterSpacing: 1,
  },
  heartbeatRow: {
    flexDirection: "row",
    alignItems: "center",
    marginVertical: 10,
  },
  pulse: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: "#00e676",
    marginRight: 15,
  },
  counterText: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "300",
    fontFamily: "monospace",
  },
  hint: { color: "#555", fontSize: 11 },
  statusText: {
    color: "#fff",
    fontSize: 14,
    marginBottom: 15,
    fontWeight: "600",
  },
  statusValue: { color: "#2196F3" },
  primaryBtn: {
    backgroundColor: "#2196F3",
    padding: 18,
    borderRadius: 12,
    alignItems: "center",
  },
  controls: { backgroundColor: "#1e1e1e", padding: 15, borderRadius: 12 },
  sectionTitle: { color: "#888", fontSize: 12, marginBottom: 10 },
  methodGrid: { flexDirection: "row", flexWrap: "wrap", marginBottom: 20 },
  methodBtn: {
    backgroundColor: "#333",
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 8,
    margin: 4,
    borderWidth: 1,
    borderColor: "#444",
  },
  methodBtnText: { color: "#fff", fontSize: 14 },
  disposeBtn: {
    backgroundColor: "#cf6679",
    padding: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  btnText: { color: "#fff", fontWeight: "bold", letterSpacing: 1 },
  terminal: {
    flex: 1,
    marginTop: 20,
    backgroundColor: "#000",
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: "#333",
  },
  logText: {
    color: "#e0e0e0",
    fontSize: 12,
    fontFamily: "monospace",
    marginBottom: 4,
  },
  logTime: { color: "#555" },
});
