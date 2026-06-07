import { mylibJsi } from '../../../native/bridge';
import { type Tensor } from '../../../core/tensor';
import { type ResizeMode } from './image';
import { scalePoint } from './points';

export type BoxMap = {
  xyxy: { xmin: number; ymin: number; xmax: number; ymax: number };
  xywh: { xmin: number; ymin: number; w: number; h: number };
  cxcywh: { cx: number; cy: number; w: number; h: number };
};
export type BoxFormat = keyof BoxMap;
export type BoundingBox<F extends BoxFormat> = F extends any
  ? { readonly format: F } & Readonly<BoxMap[F]>
  : never;

export function decodeBox<F extends BoxFormat>(
  tuple: [number, number, number, number],
  format: F,
): BoundingBox<F> {
  'worklet';
  const [a, b, c, d] = tuple;
  switch (format) {
    case 'xyxy':
      return { format: 'xyxy', xmin: a, ymin: b, xmax: c, ymax: d } as BoundingBox<F>;
    case 'xywh':
      return { format: 'xywh', xmin: a, ymin: b, w: c, h: d } as BoundingBox<F>;
    case 'cxcywh':
      return { format: 'cxcywh', cx: a, cy: b, w: c, h: d } as BoundingBox<F>;
  }
}

export function scaleBox<F extends BoxFormat>(
  box: BoundingBox<F>,
  opts: {
    from: { width: number; height: number };
    to: { width: number; height: number };
    readonly resizeMode: Exclude<ResizeMode, 'crop'>;
  },
): BoundingBox<F> {
  'worklet';
  const { from, to, resizeMode } = opts;

  let scaleX: number;
  let scaleY: number;
  switch (resizeMode) {
    case 'letterbox': {
      const scale = Math.min(from.width / to.width, from.height / to.height);
      scaleX = scale;
      scaleY = scale;
      break;
    }
    case 'stretch':
      scaleX = from.width / to.width;
      scaleY = from.height / to.height;
      break;
  }

  switch (box.format) {
    case 'xyxy': {
      const pMin = scalePoint({ x: box.xmin, y: box.ymin }, opts);
      const pMax = scalePoint({ x: box.xmax, y: box.ymax }, opts);
      return {
        format: 'xyxy',
        xmin: pMin.x,
        ymin: pMin.y,
        xmax: pMax.x,
        ymax: pMax.y,
      } as BoundingBox<F>;
    }
    case 'xywh': {
      const pMin = scalePoint({ x: box.xmin, y: box.ymin }, opts);
      return {
        format: 'xywh',
        xmin: pMin.x,
        ymin: pMin.y,
        w: box.w / scaleX,
        h: box.h / scaleY,
      } as BoundingBox<F>;
    }
    case 'cxcywh': {
      const pCenter = scalePoint({ x: box.cx, y: box.cy }, opts);
      return {
        format: 'cxcywh',
        cx: pCenter.x,
        cy: pCenter.y,
        w: box.w / scaleX,
        h: box.h / scaleY,
      } as BoundingBox<F>;
    }
  }
}

export type NmsOptions = {
  readonly boxFormat: BoxFormat;
  readonly iouThreshold: number;
  readonly confidenceThreshold: number;
  readonly nmsType: 'standard' | 'weighted';
};

export function nms(
  boxes: Tensor,
  scores: Tensor,
  opts: NmsOptions & { readonly nmsType: 'standard' },
): number[];
export function nms(
  boxes: Tensor,
  scores: Tensor,
  opts: NmsOptions & { readonly nmsType: 'weighted' },
): number[][];
export function nms(
  boxes: Tensor,
  scores: Tensor,
  opts: NmsOptions,
): number[] | number[][] {
  'worklet';
  return mylibJsi.cv.nms(boxes, scores, opts);
}

