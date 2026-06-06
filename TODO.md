# TODO

- [ ] Add a `downloadModel(ModelType): ModelType` function to download the model
  from url and return the same model with `modelPath` switched to the local
  path, so that the user can use pre-exported models defined in `models.ts` with
  `create<Task>` API like this:

  ```ts
  const model = await downloadModel(models.classification.EFFICIENTNET_V2_S);
  { classify, dispose } = await createClassifier(model);
  ```

- [ ] Add loading all methods during `loadModel`, so that e.g. CoreML doesn't
  require long startup time on first inference.
