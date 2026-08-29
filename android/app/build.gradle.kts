plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.pokoin.dslocalscan"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.pokoin.dslocalscan"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk {
            // Dual-ABI debug APK prefers x86_64 on the emulator and then never
            // loads arm64 libopencvNative.so. Pass -PonlyArm64 so PackageManager
            // installs the ARM64 libs and the emulator's ndk_translation runs them.
            if (project.hasProperty("onlyArm64")) {
                abiFilters += "arm64-v8a"
            } else {
                abiFilters += listOf("arm64-v8a", "x86_64")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
    packaging {
        jniLibs {
            val drop = mutableSetOf("**/armeabi-v7a/**", "**/x86/**")
            if (project.hasProperty("onlyArm64")) {
                drop += "**/x86_64/**"
            }
            excludes += drop
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")
    implementation("org.opencv:opencv:4.11.0")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
}
