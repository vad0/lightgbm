plugins {
    id("java")
    id("me.champeau.jmh") version "0.7.2"
    id("checkstyle")
    id("maven-publish")
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "vad0"
version = "1.0-SNAPSHOT"

repositories {
    mavenLocal()
    mavenCentral()
}

dependencies {
    implementation("org.agrona:agrona:1.21.2")
    implementation("org.apache.logging.log4j:log4j-core:3.0.0-beta2")
    implementation("org.apache.logging.log4j:log4j-slf4j-impl:3.0.0-beta2")
    implementation("io.github.metarank:lightgbm4j:4.1.0-2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

java {
    val javaVersion = JavaVersion.VERSION_21
    sourceCompatibility = javaVersion
    targetCompatibility = javaVersion
}

tasks.test {
    useJUnitPlatform()
}

jmh {
    jvmArgs.add("-Djmh.ignoreLock=true")
//    includes.set(listOf(".Unsafe"))
    fork = 1
    warmupIterations = 3
    iterations = 5
    benchmarkMode.set(listOf("avgt"))
//    benchmarkMode.set(listOf("all"))
    timeUnit = "us"
    failOnError = true
    duplicateClassesStrategy = DuplicatesStrategy.INCLUDE
    val argsMap = mapOf(
        "libPath" to "/home/vadim/Documents/soft/async-profiler-3.0-linux-x64/lib/libasyncProfiler.so",
        "output" to "flamegraph",
        "dir" to "build"
    )
    val argsString = argsMap.entries
        .map { it.key + "=" + it.value }
        .joinToString(";")
    profilers.set(listOf("async:$argsString"))
}

publishing {
    publications {
        // This mavenJava can be filled in randomly, it's just a task name
        // MavenPublication must have, this is the task class to call
        create<MavenPublication>("maven") {
            // The header here is the artifacts configuration information, do not fill in the default
            groupId = "lightgbm"
            artifactId = "library"
            version = "1.1"

            from(components["java"])
        }
    }
}
