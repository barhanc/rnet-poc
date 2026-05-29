const MyLib = require("./NativeMyLib").default;

if (!global.__myModule__) {
  MyLib.install();
}

const module = global.__myModule__;

export function multiply(a: number, b: number): number {
  return module.multiply(a, b);
}
