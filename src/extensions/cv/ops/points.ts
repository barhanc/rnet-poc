import { type ResizeMode } from './image';

export type Point = {
  readonly x: number;
  readonly y: number;
};

export function scalePoint(
  point: Point,
  opts: {
    from: { readonly width: number; readonly height: number };
    to: { readonly width: number; readonly height: number };
    readonly resizeMode: Exclude<ResizeMode, 'crop'>;
  },
): Point {
  'worklet';
  const { from, to, resizeMode } = opts;
  switch (resizeMode) {
    case 'letterbox': {
      const scale = Math.min(from.width / to.width, from.height / to.height);
      const offsetX = (from.width - to.width * scale) / 2.0;
      const offsetY = (from.height - to.height * scale) / 2.0;
      return { x: (point.x - offsetX) / scale, y: (point.y - offsetY) / scale };
    }
    case 'stretch': {
      const scaleX = from.width / to.width;
      const scaleY = from.height / to.height;
      return { x: point.x / scaleX, y: point.y / scaleY };
    }
  }
}
