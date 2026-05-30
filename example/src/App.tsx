import { useState } from 'react';
import { View, Text, StyleSheet, Pressable, SafeAreaView } from 'react-native';
import { ManualScreen } from './ManualScreen';
import { HookScreen } from './HookScreen';

export default function App() {
  const [screen, setScreen] = useState<'manual' | 'hook'>('hook');

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.tabs}>
        <Pressable 
          style={[styles.tab, screen === 'manual' && styles.activeTab]} 
          onPress={() => setScreen('manual')}
        >
          <Text style={[styles.tabText, screen === 'manual' && styles.activeTabText]}>Manual API</Text>
        </Pressable>
        <Pressable 
          style={[styles.tab, screen === 'hook' && styles.activeTab]} 
          onPress={() => setScreen('hook')}
        >
          <Text style={[styles.tabText, screen === 'hook' && styles.activeTabText]}>Hooks API</Text>
        </Pressable>
      </View>
      <View style={styles.content}>
        {screen === 'manual' ? <ManualScreen /> : <HookScreen />}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  tabs: { flexDirection: 'row', borderBottomWidth: 1, borderBottomColor: '#ddd', backgroundColor: '#fff' },
  tab: { flex: 1, paddingVertical: 15, alignItems: 'center' },
  activeTab: { borderBottomWidth: 2, borderBottomColor: '#111' },
  tabText: { fontSize: 16, color: '#888' },
  activeTabText: { color: '#111', fontWeight: '600' },
  content: { flex: 1 }
});
