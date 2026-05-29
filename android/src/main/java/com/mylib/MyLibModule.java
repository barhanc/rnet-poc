package com.mylib;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.JavaScriptContextHolder;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;

public class MyLibModule extends MyLibSpec {
  public static final String NAME = "MyLib";

  MyLibModule(ReactApplicationContext context) {
    super(context);
  }

  @NonNull
  @Override
  public String getName() {
    return NAME;
  }

  static {
    System.loadLibrary("react-native-my-lib");
  }

  public static native void nativeInstall(long jsi);

  // Example method
  // See https://reactnative.dev/docs/native-modules-android
  @ReactMethod(isBlockingSynchronousMethod = true)
  public boolean install() {
    JavaScriptContextHolder contextHolder = getReactApplicationContext().getJavaScriptContextHolder();
    if(contextHolder != null) {
    nativeInstall(contextHolder.get());
    return true;
    }
    return false;
  }
}
