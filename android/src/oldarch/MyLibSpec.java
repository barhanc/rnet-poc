package com.mylib;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.Promise;

abstract class MyLibSpec extends ReactContextBaseJavaModule {
  MyLibSpec(ReactApplicationContext context) {
    super(context);
  }

  public abstract boolean install();
}
