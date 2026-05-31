import { useState } from 'react';
import { View, Text, StyleSheet, Pressable, SafeAreaView } from 'react-native';
import { CameraScreen } from './CameraScreen';
import { GalleryScreen } from './GalleryScreen';

export default function App() {
  const [screen, setScreen] = useState<'camera' | 'gallery'>('camera');

  return (
    <SafeAreaView style={styles.container}>
      {/* Navigation Tabs */}
      <View style={styles.tabs}>
        <Pressable 
          style={[styles.tab, screen === 'camera' && styles.activeTab]} 
          onPress={() => setScreen('camera')}
        >
          <Text style={[styles.tabText, screen === 'camera' && styles.activeTabText]}>Live Camera</Text>
        </Pressable>
        <Pressable 
          style={[styles.tab, screen === 'gallery' && styles.activeTab]} 
          onPress={() => setScreen('gallery')}
        >
          <Text style={[styles.tabText, screen === 'gallery' && styles.activeTabText]}>Gallery Analysis</Text>
        </Pressable>
      </View>

      {/* Screen Render */}
      <View style={styles.content}>
        {screen === 'camera' && <CameraScreen />}
        {screen === 'gallery' && <GalleryScreen />}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  tabs: { 
    flexDirection: 'row', 
    borderBottomWidth: 1, 
    borderBottomColor: '#ddd', 
    backgroundColor: '#fff',
    zIndex: 5,
    elevation: 5,
  },
  tab: { flex: 1, paddingVertical: 15, alignItems: 'center' },
  activeTab: { borderBottomWidth: 2, borderBottomColor: '#111' },
  tabText: { fontSize: 14, color: '#888', fontWeight: '500' },
  activeTabText: { color: '#111', fontWeight: '600' },
  content: { flex: 1 }
});


