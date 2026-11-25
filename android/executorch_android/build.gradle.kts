// ExecuTorch v1.6.0 Android library, built from source with the BoardCheck
// thread patch (EXECUTORCH_NUM_THREADS). Java/Kotlin sources copied unchanged from
// executorch/extension/android/executorch_android; libexecutorch.so built with
// executorch/scripts/build_android_library.sh settings for arm64-v8a.
plugins { id("com.android.library") }

android {
    namespace = "org.pytorch.executorch"
    compileSdk = 36
    defaultConfig { minSdk = 26; consumerProguardFiles("consumer-proguard-rules.pro") }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
}

dependencies {
    implementation("com.facebook.fbjni:fbjni:0.7.0")
    implementation("com.facebook.soloader:nativeloader:0.10.5")
    implementation("androidx.core:core-ktx:1.13.1")
}
