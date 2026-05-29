#include "MyLib.h"
#include <jni.h>
#include <jsi/jsi.h>

extern "C" JNIEXPORT void JNICALL Java_com_mylib_MyLibModule_nativeInstall(JNIEnv *env, jclass clazz, jlong jsi)
{
    jsi::Runtime *runtime = reinterpret_cast<jsi::Runtime *>(jsi);
    mylib::install(*runtime);
}
