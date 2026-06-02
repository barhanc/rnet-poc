import { mylibJsi } from '../../../native/bridge';
import { type Tensor } from '../../../core/tensor';

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
): BoundingBox<F> {
  'worklet';
  const scaleX = to.width / from.width;
  const scaleY = to.height / from.height;

  switch (box.format) {
    case 'xyxy':
      return {
        format: 'xyxy',
        xmin: box.xmin * scaleX,
        ymin: box.ymin * scaleY,
        xmax: box.xmax * scaleX,
        ymax: box.ymax * scaleY,
      } as BoundingBox<F>;
    case 'xywh':
      return {
        format: 'xywh',
        xmin: box.xmin * scaleX,
        ymin: box.ymin * scaleY,
        w: box.w * scaleX,
        h: box.h * scaleY,
      } as BoundingBox<F>;
    case 'cxcywh':
      return {
        format: 'cxcywh',
        cx: box.cx * scaleX,
        cy: box.cy * scaleY,
        w: box.w * scaleX,
        h: box.h * scaleY,
      } as BoundingBox<F>;
  }
}
