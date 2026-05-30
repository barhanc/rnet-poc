import { useEffect, useState } from 'react';

export function useModel<TConfig, TModel extends { dispose: () => void }>(
  createModel: (config: TConfig) => Promise<TModel>,
  config: TConfig | null,
  deps: React.DependencyList,
) {
  const [model, setModel] = useState<TModel | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!config) return void setModel(null);

    let isMounted = true;
    let instance: TModel | null = null;
    setError(null);

    createModel(config)
      .then((m) => (isMounted ? setModel((instance = m)) : m.dispose()))
      .catch((e) => isMounted && setError(e instanceof Error ? e : new Error(String(e))));

    return () => {
      isMounted = false;
      instance?.dispose();
    };
  }, deps);

  return { model, error };
}
