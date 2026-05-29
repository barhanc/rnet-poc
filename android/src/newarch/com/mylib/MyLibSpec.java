package com.mylib;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.ReactApplicationContext;

abstract class MyLibSpec extends NativeMyLibSpec {
  MyLibSpec(ReactApplicationContext context) {
    super(context);
  }

  @NonNull
  public abstract String getName();
}
