declare module 'pngjs/browser' {
  export class PNG {
    static sync: {
      read(buffer: Buffer): { data: Uint8Array; width: number; height: number };
    };
  }
}
