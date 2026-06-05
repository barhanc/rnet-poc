import { useState } from 'react';
import { View, StyleSheet, Text, TouchableOpacity } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { GalleryScreen } from './GalleryScreen';
import { CameraScreen } from './CameraScreen';
import { InspectScreen } from './InspectScreen';

type Tab = 'camera' | 'gallery' | 'inspect';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('gallery');

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        {/* Navigation Tabs */}
        <View style={styles.tabs}>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'camera' && styles.activeTab]}
            onPress={() => setActiveTab('camera')}
            activeOpacity={0.7}
          >
            <Text style={[styles.tabText, activeTab === 'camera' && styles.activeTabText]}>
              Camera
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.tab, activeTab === 'gallery' && styles.activeTab]}
            onPress={() => setActiveTab('gallery')}
            activeOpacity={0.7}
          >
            <Text style={[styles.tabText, activeTab === 'gallery' && styles.activeTabText]}>
              Gallery
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.tab, activeTab === 'inspect' && styles.activeTab]}
            onPress={() => setActiveTab('inspect')}
            activeOpacity={0.7}
          >
            <Text style={[styles.tabText, activeTab === 'inspect' && styles.activeTabText]}>
              Inspect
            </Text>
          </TouchableOpacity>
        </View>

        {/* Dynamic Screen Content */}
        <View style={styles.content}>
          {activeTab === 'camera' ? (
            <CameraScreen />
          ) : activeTab === 'gallery' ? (
            <GalleryScreen />
          ) : (
            <InspectScreen />
          )}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
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
  content: { flex: 1 },
});
