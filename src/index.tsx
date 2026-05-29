const MyLib = require("./NativeMyLib").default;

if (!global.__myModule__) MyLib.install();
const module = global.__myModule__;

export const answerToTheUltimateQuestionOfLifeTheUniverseAndEverything =
  module.answerToTheUltimateQuestionOfLifeTheUniverseAndEverything;

export const isWednesday = module.isWednesday;

export const myAwesomeArray = module.myAwesomeArray;

export function giveMeFive(): number {
  return module.giveMeFive();
}

export function sumMeThis(a: number, b: number): number {
  return module.sumMeThis(a, b);
}

export function divideMeThis(a: number, b: number): number {
  return module.divideMeThis(a, b);
}

export function reverseMeThis(str: string): string {
  return module.reverseMeThis(str);
}

export function sumMeThisObject(obj: { firstNum: number; lastNum: number }): {
  result: number;
} {
  return module.sumMeThisObject(obj);
}

export function sumMeThisArray(arr: number[]): number {
  return module.sumMeThisArray(arr);
}

export function nativeMap(arr: number[], fn: (_: number) => number): number[] {
  return module.nativeMap(arr, fn);
}

export function runJsFunction() {
  return module.runJsFunction();
}

export function getDateObject() {
  return module.getDateObject();
}

export function getInfinityObject() {
  return module.getInfinityObject();
}

export function checkExecuTorch(): string {
  return module.checkExecuTorch();
}
