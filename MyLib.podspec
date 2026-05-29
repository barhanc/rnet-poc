require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "MyLib"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => min_ios_version_supported } 
  s.source       = { :git => "https://github.com/barhanc/react-native-my-lib.git", :tag => "#{s.version}" }

  s.source_files = "ios/**/*.{h,m,mm,swift,cpp}", "cpp/**/*.{hpp,cpp,c,h}"
  s.private_header_files = "ios/**/*.h"

  s.vendored_frameworks = [
    "third-party/ios/Frameworks/executorch.xcframework",
    "third-party/ios/Frameworks/backend_coreml.xcframework",
    "third-party/ios/Frameworks/backend_mps.xcframework",
    "third-party/ios/Frameworks/backend_xnnpack.xcframework",
    "third-party/ios/Frameworks/kernels_optimized.xcframework",
    "third-party/ios/Frameworks/kernels_quantized.xcframework",
    "third-party/ios/Frameworks/threadpool.xcframework",
    "third-party/ios/Frameworks/executorch_llm.xcframework",
    "third-party/ios/Frameworks/kernels_llm.xcframework"
  ]
  s.frameworks = "CoreML", "Metal", "MetalPerformanceShaders", "Accelerate"
  s.library = "sqlite3"

s.pod_target_xcconfig = {
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
    "OTHER_LDFLAGS" => "-all_load",
    "HEADER_SEARCH_PATHS" => [
      "\"$(PODS_TARGET_SRCROOT)/third-party\"", 
      "\"$(PODS_TARGET_SRCROOT)/third-party/executorch\"",
      "\"$(PODS_TARGET_SRCROOT)/third-party/ios/Frameworks\""
    ].join(' '),
    "LIBRARY_SEARCH_PATHS" => [
      "\"$(inherited)\"",
      "\"$(PODS_CONFIGURATION_BUILD_DIR)/XCFrameworkIntermediates/MyLib/**\""
    ].join(' ')
  }

  install_modules_dependencies(s)
end

