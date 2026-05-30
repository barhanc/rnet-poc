import { mylibJsi } from '../native/bridge';
import { type Tensor } from '../core/tensor';

export function sigmoid(src: Tensor, dst: Tensor): Tensor {
  'worklet';
  return mylibJsi.math.sigmoid(src, dst);
}

export function softmax(src: Tensor, dst: Tensor, axis: number = -1): Tensor {
  'worklet';
  return mylibJsi.math.softmax(src, dst, axis);
}
