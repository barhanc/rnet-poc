import { useState, useEffect } from 'react';
import RNFS from 'react-native-fs';

export function useModelDownload(source?: string, preventLoad?: boolean) {
  const [localPath, setLocalPath] = useState<string>();
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadError, setDownloadError] = useState<Error | null>(null);

  useEffect(() => {
    setLocalPath(undefined);
    setDownloadProgress(0);
    setDownloadError(null);

    if (preventLoad) return;

    if (!source) {
      setDownloadProgress(100);
      return;
    }

    if (!source.startsWith('http')) {
      setLocalPath(source);
      setDownloadProgress(100);
      return;
    }

    let isMounted = true;
    const dest = `${RNFS.CachesDirectoryPath}/${encodeURIComponent(source.split('?')[0]!)}`;

    RNFS.exists(dest).then((exists) => {
      if (!isMounted) return;

      if (exists) {
        setLocalPath(dest);
        setDownloadProgress(100);
        return;
      }

      RNFS.downloadFile({
        fromUrl: source,
        toFile: dest,
        progress: (r: any) => {
          if (isMounted) setDownloadProgress((r.bytesWritten / r.contentLength) * 100);
        },
      }).promise
        .then(() => {
          if (isMounted) {
            setLocalPath(dest);
            setDownloadProgress(100);
          }
        })
        .catch((e) => {
          if (isMounted) setDownloadError(e instanceof Error ? e : new Error(String(e)));
        });
    });

    return () => { isMounted = false; };
  }, [source, preventLoad]);

  return { localPath, downloadProgress, downloadError };
}
