package com.mylib

import com.facebook.react.bridge.ReactApplicationContext

abstract class MyLibSpec(reactContext: ReactApplicationContext) : NativeMyLibSpec(reactContext) {
  abstract override fun getName(): String
}
