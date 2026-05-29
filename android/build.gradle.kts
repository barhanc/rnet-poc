buildscript {
  repositories {
    google()
    mavenCentral()
  }

  dependencies {
    classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.0.21")
  }
}

fun reactNativeArchitectures(): List<String> {
  val value = rootProject.properties["reactNativeArchitectures"] as? String
  return value?.split(",") ?: listOf("armeabi-v7a", "x86", "x86_64", "arm64-v8a")
}

plugins {
  id("com.android.library")
  id("org.jetbrains.kotlin.android")
}

apply(plugin = "com.facebook.react")

fun getExtOrDefault(name: String): Any =
  if (rootProject.ext.has(name)) {
    rootProject.ext.get(name)!!
  } else {
    project.properties["MyLib_$name"]!!
  }

fun getExtOrIntegerDefault(name: String): Int =
  if (rootProject.ext.has(name)) {
    rootProject.ext.get(name) as Int
  } else {
    (project.properties["MyLib_$name"] as String).toInt()
  }

android {
  namespace = "com.mylib"

  ndkVersion = getExtOrDefault("ndkVersion") as String
  compileSdk = getExtOrIntegerDefault("compileSdkVersion")

  defaultConfig {
    minSdk = getExtOrIntegerDefault("minSdkVersion")
    targetSdk = getExtOrIntegerDefault("targetSdkVersion")

    externalNativeBuild {
      cmake {
        arguments("-DANDROID_STL=c++_shared")
        cppFlags("-O2 -frtti -fexceptions -Wall -fstack-protector-all")
        abiFilters(*reactNativeArchitectures().toTypedArray())
      }
    }
  }

  externalNativeBuild {
    cmake {
      path("CMakeLists.txt")
    }
  }

  buildFeatures {
    buildConfig = true
    prefab = true
  }

  packaging {
    resources.excludes.add("**/libjsi.so")
  }

  buildTypes {
    release {
      isMinifyEnabled = false
    }
  }

  lint {
    disable += "GradleCompatible"
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  kotlinOptions {
    jvmTarget = "17"
  }

  sourceSets {
    named("main") {
      java.srcDirs("${project.buildDir}/generated/source/codegen/java")
    }
  }
}

repositories {
  mavenCentral()
  google()
}

dependencies {
  //noinspection GradleDynamicVersion
  implementation("com.facebook.react:react-android:+")
  implementation("androidx.core:core-ktx:1.17.0")
}

extensions.configure<com.facebook.react.ReactExtension>("react") {
  jsRootDir = file("../src/")
  libraryName = "MyLib"
  codegenJavaPackageName = "com.mylib"
}
