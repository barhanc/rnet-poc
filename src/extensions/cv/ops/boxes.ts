import { mylibJsi } from '../../../native/bridge';
import { type Tensor } from '../../../core/tensor';

export type BoxFormat = 'xyxy' | 'xywh' | 'cxcywh';

export type NmsOptions = {
  readonly iouThreshold?: number;
  readonly scoreThreshold?: number;
};

export function nms(boxes: Tensor, scores: Tensor, opts?: NmsOptions): number[] {
  'worklet';
  const defaultNmsOptions = {
    iouThreshold: 0.5,
    scoreThreshold: 0.5,
  } as const;
  return mylibJsi.cv.nms(boxes, scores, { ...defaultNmsOptions, ...opts });
}

export function decodeBoxes(
  src: Tensor,
  dst: Tensor,
  opts: { from: BoxFormat; to: BoxFormat },
): Tensor {
  'worklet';
  return mylibJsi.cv.decodeBoxes(src, dst, opts);
}

export function scaleBoxes(
  src: Tensor,
  dst: Tensor,
  opts: { from: [number, number]; to: [number, number] },
): Tensor {
  'worklet';
  return mylibJsi.cv.scaleBoxes(src, dst, opts);
}
