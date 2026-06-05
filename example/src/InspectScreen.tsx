import { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Alert,
  Platform,
} from 'react-native';
import { inspectModel, models } from 'react-native-my-lib';

const EXAMPLE_URLS = [
  {
    name: 'EfficientNet V2 S',
    url: models.classification.EFFICIENTNET_V2_S.XNNPACK_FP32.modelPath,
  },
  {
    name: 'SSDLite MobileNet V3',
    url: Platform.select({
      ios: models.objectDetection.SSDLITE320_MOBILENET_V3_LARGE.COREML_FP32.modelPath,
      android: models.objectDetection.SSDLITE320_MOBILENET_V3_LARGE.XNNPACK_FP32.modelPath,
    }) || models.objectDetection.SSDLITE320_MOBILENET_V3_LARGE.XNNPACK_FP32.modelPath,
  },
  {
    name: 'Candy Style Transfer',
    url: models.styleTransfer.CANDY.XNNPACK_FP32.modelPath,
  },
];

export function InspectScreen() {
  const [url, setUrl] = useState(EXAMPLE_URLS[0]?.url || '');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleInspect = async (targetUrl: string) => {
    if (!targetUrl.trim()) {
      Alert.alert('Error', 'Please enter a valid URL');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const inspectRes = await inspectModel(targetUrl.trim());
      setResult(inspectRes);
    } catch (error: any) {
      console.error(error);
      Alert.alert('Error', error?.message || 'Failed to inspect model. Make sure it is a valid PTE file.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Model URL Inspector</Text>
        <Text style={styles.cardDescription}>
          Paste a URL to an ExecuTorch (.pte) model below to download and inspect its metadata and methods.
        </Text>

        <TextInput
          style={styles.input}
          placeholder="https://example.com/model.pte"
          placeholderTextColor="#999"
          value={url}
          onChangeText={setUrl}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          multiline
        />

        <TouchableOpacity
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={() => handleInspect(url)}
          disabled={loading}
          activeOpacity={0.8}
        >
          {loading ? (
            <ActivityIndicator color="#fff" size="small" />
          ) : (
            <Text style={styles.buttonText}>Inspect Model</Text>
          )}
        </TouchableOpacity>

        <Text style={styles.presetsTitle}>Or select a preset:</Text>
        <View style={styles.presetsContainer}>
          {EXAMPLE_URLS.map((preset, idx) => (
            <TouchableOpacity
              key={idx}
              style={styles.presetChip}
              onPress={() => {
                setUrl(preset.url);
                handleInspect(preset.url);
              }}
              disabled={loading}
            >
              <Text style={styles.presetChipText}>{preset.name}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {result && (
        <View style={styles.resultsCard}>
          <Text style={styles.resultsHeader}>Inspection Results</Text>
          <Text style={styles.sourceLabel}>Source URL:</Text>
          <Text style={styles.sourceValue}>{result.source}</Text>

          <Text style={styles.methodsTitle}>Methods ({result.methods.length})</Text>

          {result.methods.map((method: any, mIdx: number) => (
            <View key={mIdx} style={styles.methodContainer}>
              <View style={styles.methodHeader}>
                <Text style={styles.methodName}>{method.name}</Text>
                <View style={styles.badge}>
                  <Text style={styles.badgeText}>Method</Text>
                </View>
              </View>

              <View style={styles.statsRow}>
                <View style={styles.statBox}>
                  <Text style={styles.statVal}>{method.meta.numInputs}</Text>
                  <Text style={styles.statLabel}>Inputs</Text>
                </View>
                <View style={styles.statBox}>
                  <Text style={styles.statVal}>{method.meta.numOutputs}</Text>
                  <Text style={styles.statLabel}>Outputs</Text>
                </View>
              </View>

              {method.meta.usesBackend && Object.keys(method.meta.usesBackend).length > 0 && (
                <View style={styles.metaSection}>
                  <Text style={styles.metaSectionTitle}>Backends Used:</Text>
                  <View style={styles.tagRow}>
                    {Object.entries(method.meta.usesBackend).map(([backend, used]) => (
                      <View key={backend} style={[styles.tag, used ? styles.tagActive : styles.tagInactive]}>
                        <Text style={used ? styles.tagActiveText : styles.tagInactiveText}>
                          {backend}: {used ? 'Yes' : 'No'}
                        </Text>
                      </View>
                    ))}
                  </View>
                </View>
              )}

              {/* Input Tensors */}
              {method.meta.inputTensorMeta && method.meta.inputTensorMeta.length > 0 && (
                <View style={styles.tensorsSection}>
                  <Text style={styles.tensorsSectionTitle}>Input Tensors</Text>
                  {method.meta.inputTensorMeta.map((tensor: any, tIdx: number) => (
                    <View key={tIdx} style={styles.tensorCard}>
                      <View style={styles.tensorHeader}>
                        <Text style={styles.tensorName}>{tensor.name || `Input #${tIdx}`}</Text>
                        <Text style={styles.tensorDtype}>{tensor.dtype}</Text>
                      </View>
                      <View style={styles.tensorDetails}>
                        <Text style={styles.tensorDetailText}>
                          Shape: <Text style={styles.tensorDetailValue}>[{tensor.shape.join(', ')}]</Text>
                        </Text>
                        <Text style={styles.tensorDetailText}>
                          Bytes: <Text style={styles.tensorDetailValue}>{tensor.nbytes}</Text>
                        </Text>
                      </View>
                    </View>
                  ))}
                </View>
              )}

              {/* Output Tensors */}
              {method.meta.outputTensorMeta && method.meta.outputTensorMeta.length > 0 && (
                <View style={styles.tensorsSection}>
                  <Text style={styles.tensorsSectionTitle}>Output Tensors</Text>
                  {method.meta.outputTensorMeta.map((tensor: any, tIdx: number) => (
                    <View key={tIdx} style={styles.tensorCard}>
                      <View style={styles.tensorHeader}>
                        <Text style={styles.tensorName}>{tensor.name || `Output #${tIdx}`}</Text>
                        <Text style={styles.tensorDtype}>{tensor.dtype}</Text>
                      </View>
                      <View style={styles.tensorDetails}>
                        <Text style={styles.tensorDetailText}>
                          Shape: <Text style={styles.tensorDetailValue}>[{tensor.shape.join(', ')}]</Text>
                        </Text>
                        <Text style={styles.tensorDetailText}>
                          Bytes: <Text style={styles.tensorDetailValue}>{tensor.nbytes}</Text>
                        </Text>
                      </View>
                    </View>
                  ))}
                </View>
              )}
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 40,
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
    marginBottom: 20,
  },
  cardTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#1a1a1a',
    marginBottom: 8,
  },
  cardDescription: {
    fontSize: 14,
    color: '#666666',
    lineHeight: 20,
    marginBottom: 20,
  },
  input: {
    backgroundColor: '#f1f3f5',
    borderRadius: 10,
    padding: 12,
    fontSize: 14,
    color: '#212529',
    minHeight: 60,
    textAlignVertical: 'top',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  button: {
    backgroundColor: '#0070f3',
    borderRadius: 10,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#0070f3',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 3,
  },
  buttonDisabled: {
    backgroundColor: '#a3cdff',
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  presetsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#495057',
    marginTop: 20,
    marginBottom: 10,
  },
  presetsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  presetChip: {
    backgroundColor: '#e9ecef',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  presetChipText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#495057',
  },
  resultsCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  resultsHeader: {
    fontSize: 18,
    fontWeight: '700',
    color: '#212529',
    marginBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f3f5',
    paddingBottom: 10,
  },
  sourceLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#868e96',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  sourceValue: {
    fontSize: 13,
    color: '#495057',
    backgroundColor: '#f8f9fa',
    padding: 8,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#e9ecef',
    marginBottom: 20,
  },
  methodsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#212529',
    marginBottom: 12,
  },
  methodContainer: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  methodHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  methodName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0070f3',
    flex: 1,
    marginRight: 8,
  },
  badge: {
    backgroundColor: '#e7f5ff',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#228be6',
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#ffffff',
    borderRadius: 8,
    padding: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  statVal: {
    fontSize: 18,
    fontWeight: '700',
    color: '#212529',
  },
  statLabel: {
    fontSize: 11,
    color: '#868e96',
    fontWeight: '500',
    marginTop: 2,
  },
  metaSection: {
    marginTop: 10,
    marginBottom: 10,
  },
  metaSectionTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#495057',
    marginBottom: 6,
  },
  tagRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  tag: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    borderWidth: 1,
  },
  tagActive: {
    backgroundColor: '#ebfbee',
    borderColor: '#b2f2bb',
  },
  tagInactive: {
    backgroundColor: '#f1f3f5',
    borderColor: '#e9ecef',
  },
  tagActiveText: {
    fontSize: 11,
    color: '#2b8a3e',
    fontWeight: '600',
  },
  tagInactiveText: {
    fontSize: 11,
    color: '#868e96',
  },
  tensorsSection: {
    marginTop: 14,
  },
  tensorsSectionTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#495057',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  tensorCard: {
    backgroundColor: '#ffffff',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  tensorHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  tensorName: {
    fontSize: 13,
    fontWeight: '600',
    color: '#212529',
    flex: 1,
    marginRight: 8,
  },
  tensorDtype: {
    fontSize: 11,
    fontWeight: '700',
    color: '#fd7e14',
    backgroundColor: '#fff9db',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  tensorDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  tensorDetailText: {
    fontSize: 11,
    color: '#868e96',
  },
  tensorDetailValue: {
    color: '#495057',
    fontWeight: '600',
  },
});
