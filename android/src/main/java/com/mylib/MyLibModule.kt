package com.mylib

import com.facebook.react.bridge.ReactApplicationContext

class MyLibModule(reactContext: ReactApplicationContext) :
  NativeMyLibSpec(reactContext) {

  override fun multiply(a: Double, b: Double): Double {
    return a * b
  }

  companion object {
    const val NAME = NativeMyLibSpec.NAME
  }
}
