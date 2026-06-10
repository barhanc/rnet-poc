import { useRef, useState, type ComponentRef } from 'react';
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

const MODEL = models.nlp.LFM2_5_1_2B.XNNPACK_8DA4W;

const SYSTEM_PROMPT = 'You are a helpful, concise assistant running on-device.';
const INITIAL_MESSAGES: ChatMessage[] = [{ role: 'system', content: SYSTEM_PROMPT }];
const GENERATION_CONFIG = { temperature: 0.7, maxNewTokens: 512 };

type Turn = { role: 'user' | 'assistant'; content: string; stats?: GenerationStats };

// One-line summary. All timestamps are ms (ExecuTorch stats scale to 1000 units/sec).
function formatStats(stats: GenerationStats): string {
  const decodeSeconds = (stats.inferenceEndMs - stats.firstTokenMs) / 1000;
  const tokensPerSec = decodeSeconds > 0 ? stats.numGeneratedTokens / decodeSeconds : 0;
  const ttftMs = stats.firstTokenMs - stats.inferenceStartMs;
  const totalSeconds = (stats.inferenceEndMs - stats.inferenceStartMs) / 1000;
  return (
    `gen ${stats.numGeneratedTokens} tok · ` +
    `${tokensPerSec.toFixed(1)} tok/s · ${ttftMs.toFixed(0)} ms to first · ${totalSeconds.toFixed(2)} s`
  );
}

export function LLMScreen() {
  // The chat template + special tokens are derived from the model's
  // tokenizer_config.json automatically — no hand-written prompt format.
  const { isReady, downloadProgress, error, sendMessage, stop } = useLLMChatSession({
    ...MODEL,
    initialMessages: INITIAL_MESSAGES,
    generationConfig: GENERATION_CONFIG,
  });

  const [input, setInput] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const scrollRef = useRef<ComponentRef<typeof ScrollView>>(null);

  const handleSend = async () => {
    const message = input.trim();
    if (!message || !sendMessage || isGenerating) return;

    setInput('');
    setIsGenerating(true);
    // Optimistically render the user turn plus an empty assistant turn we stream into.
    setTurns((prev) => [
      ...prev,
      { role: 'user', content: message },
      { role: 'assistant', content: '' },
    ]);

    try {
      const { stats } = await sendMessage(message, undefined, (token) => {
        setTurns((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === 'assistant') {
            next[next.length - 1] = { ...last, content: last.content + token };
          }
          return next;
        });
        requestAnimationFrame(() => scrollRef.current?.scrollToEnd({ animated: true }));
      });
      // Attach the generation report once the full response is in.
      setTurns((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === 'assistant') {
          next[next.length - 1] = { ...last, stats };
        }
        return next;
      });
    } catch (e: any) {
      setTurns((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === 'assistant' && last.content.length === 0) {
          next[next.length - 1] = {
            role: 'assistant',
            content: `[error] ${e?.message ?? String(e)}`,
          };
        }
        return next;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  if (error) {
    return (
      <View style={styles.centered}>
        <Text style={styles.errorTitle}>Failed to load model</Text>
        <Text style={styles.errorBody}>{error.message}</Text>
      </View>
    );
  }

  if (!isReady) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#0070f3" />
        <Text style={styles.loadingText}>
          {downloadProgress < 100
            ? `Downloading model… ${downloadProgress.toFixed(0)}%`
            : 'Loading model into memory…'}
        </Text>
        <Text style={styles.loadingSub}>LFM2.5 1.2B (8da4w)</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <ScrollView
        ref={scrollRef}
        style={styles.messages}
        contentContainerStyle={styles.messagesContent}
        onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
      >
        {turns.length === 0 && (
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
          <TouchableOpacity style={[styles.sendButton, styles.stopButton]} onPress={() => stop?.()}>
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
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8f9fa' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  loadingText: { marginTop: 16, fontSize: 15, color: '#495057', fontWeight: '600' },
  loadingSub: { marginTop: 4, fontSize: 13, color: '#868e96' },
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
