import { View, StyleSheet } from 'react-native';
// 1. Import from the new library instead of 'react-native'
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { GalleryScreen } from './GalleryScreen';
import { CameraScreen } from './CameraScreen';

export default function App() {
  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <View style={styles.content}>
          <CameraScreen />
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
