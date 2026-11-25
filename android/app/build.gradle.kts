plugins {
    id("com.android.application")
}

android {
    namespace = "com.boardcheck.app"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.boardcheck.app"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    buildTypes {
        release { isMinifyEnabled = false; signingConfig = signingConfigs.getByName("debug") }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    implementation(project(":executorch_android"))
    implementation("com.facebook.soloader:soloader:0.10.5")
    implementation("com.facebook.fbjni:fbjni:0.7.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    // QR codes (board serial sticker, laptop pairing). Bundled model: works offline.
    implementation("com.google.mlkit:barcode-scanning:17.3.0")
}
