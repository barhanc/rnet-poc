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

export type NmsOptions = {
  readonly boxFormat: BoxFormat;
  readonly iouThreshold?: number;
  readonly scoreThreshold?: number;
};

export function nms(boxes: Tensor, scores: Tensor, opts: NmsOptions): number[] {
  'worklet';
  const defaultNmsOptions = { iouThreshold: 0.5, scoreThreshold: 0.5 } as const;
  return mylibJsi.cv.nms(boxes, scores, { ...defaultNmsOptions, ...opts });
}

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
  from: { width: number; height: number },
  to: { width: number; height: number },
  options: { readonly resizeMode: Exclude<ResizeMode, 'crop'> },
): BoundingBox<F> {
  'worklet';
  let scaleX: number;
  let scaleY: number;
  switch (options.resizeMode) {
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
      const pMin = scalePoint({ x: box.xmin, y: box.ymin }, from, to, options);
      const pMax = scalePoint({ x: box.xmax, y: box.ymax }, from, to, options);
      return {
        format: 'xyxy',
        xmin: pMin.x,
        ymin: pMin.y,
        xmax: pMax.x,
        ymax: pMax.y,
      } as BoundingBox<F>;
    }
    case 'xywh': {
      const pMin = scalePoint({ x: box.xmin, y: box.ymin }, from, to, options);
      return {
        format: 'xywh',
        xmin: pMin.x,
        ymin: pMin.y,
        w: box.w / scaleX,
        h: box.h / scaleY,
      } as BoundingBox<F>;
    }
    case 'cxcywh': {
      const pCenter = scalePoint({ x: box.cx, y: box.cy }, from, to, options);
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

export type WeightedNmsOptions = {
  readonly boxFormat: BoxFormat;
  readonly scoreThreshold: number;
  readonly suppressionThreshold?: number;
};

export function weightedNms(
  boxes: Tensor,
  scores: Tensor,
  opts: WeightedNmsOptions,
): number[][] {
  'worklet';
  const defaultOpts = { suppressionThreshold: 0.3 } as const;
  return mylibJsi.cv.weightedNms(boxes, scores, { ...defaultOpts, ...opts });
}
