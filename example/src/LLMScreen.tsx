import { useEffect, useRef, useState, type ComponentRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import {
  useLLMChatSession,
  models,
  type ChatMessage,
  type GenerationStats,
} from 'react-native-my-lib';
import { NestedModelPicker } from './ModelPicker';

const SYSTEM_PROMPT =
  "You are a pirate. You must start every response with 'Ahoy matey!' and speak like a pirate.";
const INITIAL_MESSAGES: ChatMessage[] = [{ role: 'system', content: SYSTEM_PROMPT }];
const GENERATION_CONFIG = { temperature: 0.7, maxNewTokens: 512, echo: false };

type Turn = { role: 'user' | 'assistant'; content: string; stats?: GenerationStats };

function formatStats(stats: GenerationStats): string {
  const decodeMs = stats.inferenceEndMs - stats.firstTokenMs;
  const tokensPerSec = (stats.numGeneratedTokens / decodeMs) * 1000;
  const totalMs = stats.inferenceEndMs - stats.inferenceStartMs;
  const ttftMs = stats.firstTokenMs - stats.inferenceStartMs;
  return (
    `gen ${stats.numGeneratedTokens} toks · ` +
    `${tokensPerSec.toFixed(1)} tok/s · ` +
    `${ttftMs.toFixed(0)}ms ttft · ` +
    `${(totalMs / 1000).toFixed(2)}s`
  );
}

function getFirstLeafModel(node: any): any {
  if (!node || typeof node !== 'object') return null;
  if (typeof node.modelPath === 'string') return node;
  for (const key of Object.keys(node)) {
    const leaf = getFirstLeafModel(node[key]);
    if (leaf) return leaf;
  }
  return null;
}

export function LLMScreen() {
  const [activeModel, setActiveModel] = useState<any>(getFirstLeafModel(models.llm));

  const { isReady, downloadProgress, error, sendMessage, stop } = useLLMChatSession(activeModel, {
    initialMessages: INITIAL_MESSAGES,
    generationConfig: GENERATION_CONFIG,
  });

  const [input, setInput] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streamingResponse, setStreamingResponse] = useState<string | null>(null);

  const scrollRef = useRef<ComponentRef<typeof ScrollView>>(null);
  const isGenerating = streamingResponse !== null;

  // Reset chat turns when model changes
  useEffect(() => {
    setTurns([]);
    setStreamingResponse(null);
    setInput('');
  }, [activeModel]);

  const handleSend = async () => {
    const message = input.trim();
    if (!message || !sendMessage || isGenerating) return;

    setInput('');
    setStreamingResponse('');
    setTurns((prev) => [...prev, { role: 'user', content: message }]);

    try {
      const { response, stats } = await sendMessage(message, (token) => {
        setStreamingResponse((prev) => (prev !== null ? prev + token : token));
      });
      setTurns((prev) => [...prev, { role: 'assistant', content: response, stats }]);
    } catch (e: any) {
      setTurns((prev) => [...prev, { role: 'assistant', content: `[Error] ${e?.message}` }]);
    } finally {
      setStreamingResponse(null);
    }
  };

  if (error)
    return (
      <View style={styles.centered}>
        <Text style={styles.errorTitle}>Failed to load model</Text>
        <Text style={styles.errorBody}>{error.message}</Text>
      </View>
    );

  if (!isReady)
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#0070f3" />
        <Text style={styles.loadingText}>
          {downloadProgress < 100
            ? `Downloading model… ${downloadProgress.toFixed(0)}%`
            : 'Loading model into memory…'}
        </Text>
        <Text style={styles.loadingSub}>{activeModel.modelPath}</Text>
      </View>
    );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      {/* Model Selector Header */}
      <View style={styles.header}>
        <NestedModelPicker
          labelPrefix="Model"
          registry={models.llm}
          selectedValue={activeModel}
          onValueChange={setActiveModel}
        />
      </View>

      {/* Screen Content */}
      <View style={styles.content}>
        <ScrollView
          ref={scrollRef}
          style={styles.messages}
          contentContainerStyle={styles.messagesContent}
          onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: false })}
        >
          {turns.length === 0 && streamingResponse === null && (
            <Text style={styles.placeholder}>Ask the on-device model anything to get started.</Text>
          )}
          {turns.map((turn, idx) => (
            <View key={idx} style={styles.turn}>
              <View
                style={[
                  styles.bubble,
                  turn.role === 'user' ? styles.userBubble : styles.assistantBubble,
                ]}
              >
                <Text style={turn.role === 'user' ? styles.userText : styles.assistantText}>
                  {turn.content || '…'}
                </Text>
              </View>
              {turn.stats && (
                <Text
                  style={styles.statsLine}
                  numberOfLines={1}
                  adjustsFontSizeToFit
                  minimumFontScale={0.7}
                >
                  {formatStats(turn.stats)}
                </Text>
              )}
            </View>
          ))}
          {streamingResponse !== null && (
            <View style={styles.turn}>
              <View style={[styles.bubble, styles.assistantBubble]}>
                <Text style={styles.assistantText}>{streamingResponse || '…'}</Text>
              </View>
            </View>
          )}
        </ScrollView>

        <View style={styles.inputRow}>
          <TextInput
            style={styles.input}
            placeholder="Message"
            placeholderTextColor="#999"
            value={input}
            onChangeText={setInput}
            multiline
            editable={!isGenerating}
          />
          {isGenerating ? (
            <TouchableOpacity
              style={[styles.sendButton, styles.stopButton]}
              onPress={() => stop?.()}
            >
              <Text style={styles.sendButtonText}>Stop</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={[styles.sendButton, !input.trim() && styles.sendButtonDisabled]}
              onPress={handleSend}
              disabled={!input.trim()}
            >
              <Text style={styles.sendButtonText}>Send</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8f9fa' },
  header: {
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 4,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
  },
  content: {
    flex: 1,
  },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  loadingText: { marginTop: 16, fontSize: 15, color: '#495057', fontWeight: '600' },
  loadingSub: { marginTop: 4, fontSize: 13, color: '#868e96', textAlign: 'center' },
  errorTitle: { fontSize: 16, fontWeight: '700', color: '#e03131', marginBottom: 8 },
  errorBody: { fontSize: 13, color: '#868e96', textAlign: 'center' },
  messages: { flex: 1 },
  messagesContent: { padding: 16, paddingBottom: 8 },
  placeholder: { textAlign: 'center', color: '#adb5bd', marginTop: 40, fontSize: 14 },
  turn: { marginBottom: 12 },
  bubble: {
    maxWidth: '85%',
    borderRadius: 16,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  statsLine: {
    alignSelf: 'flex-start',
    marginTop: 5,
    marginLeft: 4,
    fontSize: 11,
    color: '#adb5bd',
    fontVariant: ['tabular-nums'],
  },
  userBubble: { alignSelf: 'flex-end', backgroundColor: '#0070f3' },
  assistantBubble: {
    alignSelf: 'flex-start',
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  userText: { color: '#fff', fontSize: 15, lineHeight: 21 },
  assistantText: { color: '#212529', fontSize: 15, lineHeight: 21 },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: 12,
    gap: 8,
    borderTopWidth: 1,
    borderTopColor: '#e9ecef',
    backgroundColor: '#fff',
  },
  input: {
    flex: 1,
    backgroundColor: '#f1f3f5',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 15,
    color: '#212529',
    maxHeight: 120,
  },
  sendButton: {
    backgroundColor: '#0070f3',
    borderRadius: 20,
    paddingHorizontal: 18,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: { backgroundColor: '#a3cdff' },
  stopButton: { backgroundColor: '#e03131' },
  sendButtonText: { color: '#fff', fontSize: 15, fontWeight: '600' },
});
