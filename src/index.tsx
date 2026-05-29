import MyLib from './NativeMyLib';

export function multiply(a: number, b: number): number {
  return MyLib.multiply(a, b);
}
