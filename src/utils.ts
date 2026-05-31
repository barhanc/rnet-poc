import { mylibJsi } from './native/bridge';
import { loadModel } from './core/model';
import RNFS from 'react-native-fs';

export function getRegisteredBackends(): string[] {
  return mylibJsi.getExecuTorchRegisteredBackends();
}

export async function inspectModel(source: string) {
  let localPath = source;
  let downloaded = false;

  if (source.startsWith('http')) {
    localPath = `${RNFS.TemporaryDirectoryPath}/inspect_model_${Date.now()}.pte`;
    await RNFS.downloadFile({ fromUrl: source, toFile: localPath }).promise;
    downloaded = true;
  }

  try {
    const model = loadModel(localPath);
    const methodNames = model.getMethodNames();

    console.log(`\n=== MODEL INSPECTION ===`);
    console.log(`Source:  ${source}`);
    console.log(`Methods: ${methodNames.join(', ')}`);

    for (const method of methodNames) {
      const meta = model.getMethodMeta(method);
      console.log(`\n[Method: ${method}]`);
      console.log(JSON.stringify(meta, null, 2));
    }
    console.log(`========================\n`);

    model.dispose();
  } catch (e) {
    console.error(`Failed to inspect model at ${source}:`, e);
  } finally {
    if (downloaded) {
      await RNFS.unlink(localPath).catch(() => {});
    }
  }
}
