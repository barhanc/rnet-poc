package com.mylib

import com.facebook.react.bridge.JavaScriptContextHolder
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactMethod

class MyLibModule(reactContext: ReactApplicationContext) : MyLibSpec(reactContext) {

  override fun getName(): String = NAME

  override fun install(): Boolean {
    val contextHolder: JavaScriptContextHolder =
      reactApplicationContext.javaScriptContextHolder ?: return false
    nativeInstall(contextHolder.get())
    return true
  }

  companion object {
    const val NAME = "MyLib"

    init {
      System.loadLibrary("react-native-my-lib")
    }

    @JvmStatic
    external fun nativeInstall(jsi: Long)
  }
}
