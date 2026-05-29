package com.mylib

import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider

class MyLibPackage : BaseReactPackage() {

  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? =
    if (name == MyLibModule.NAME) MyLibModule(reactContext) else null

  override fun getReactModuleInfoProvider(): ReactModuleInfoProvider =
    ReactModuleInfoProvider {
      mapOf(
        MyLibModule.NAME to ReactModuleInfo(
          MyLibModule.NAME,
          MyLibModule.NAME,
          false, // canOverrideExistingModule
          false, // needsEagerInit
          true,  // hasConstants
          false, // isCxxModule
          BuildConfig.IS_NEW_ARCHITECTURE_ENABLED // isTurboModule
        )
      )
    }
}
