package com.mylib

import com.facebook.react.bridge.JavaScriptContextHolder
import com.facebook.react.bridge.ReactApplicationContext

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
      System.loadLibrary("executorch")
      System.loadLibrary("MyLib")
    }

    @JvmStatic
    external fun nativeInstall(jsi: Long)
  }
}
