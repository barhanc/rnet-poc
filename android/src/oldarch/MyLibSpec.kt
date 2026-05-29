package com.mylib

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule

abstract class MyLibSpec(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
  abstract fun install(): Boolean
}
