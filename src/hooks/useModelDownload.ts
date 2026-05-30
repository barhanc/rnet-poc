import { useState, useEffect } from 'react';
import RNFS from 'react-native-fs';

export function useModelDownload(source?: string, preventLoad?: boolean) {
  const [localPath, setLocalPath] = useState<string>();
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadError, setDownloadError] = useState<Error | null>(null);

  useEffect(() => {
    if (preventLoad) return;
    if (!source) return void setDownloadProgress(100);
    if (!source.startsWith('http')) {
      setLocalPath(source);
      return void setDownloadProgress(100);
    }

    let isMounted = true;
    const dest = `${RNFS.CachesDirectoryPath}/${source.split('/').pop()?.split('?')[0] || 'model.ptl'}`;

    RNFS.exists(dest).then((exists) => {
      if (!isMounted) return;
      if (exists) {
        setLocalPath(dest);
        return void setDownloadProgress(100);
      }

      if (preventLoad) {
        return void setDownloadError(
          new Error(`Model not found in cache and preventLoad is true: ${source}`),
        );
      }

      RNFS.downloadFile({
        fromUrl: source,
        toFile: dest,
        progress: (r: any) =>
          isMounted && setDownloadProgress((r.bytesWritten / r.contentLength) * 100),
      })
        .promise.then(() => isMounted && (setLocalPath(dest), setDownloadProgress(100)))
        .catch((e) => isMounted && setDownloadError(e instanceof Error ? e : new Error(String(e))));
    });

    return () => {
      isMounted = false;
    };
  }, [source, preventLoad]);

  return { localPath, downloadProgress, downloadError };
}
