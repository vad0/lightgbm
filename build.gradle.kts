plugins {
    id("java")
    id("me.champeau.jmh") version "0.7.2"
}

group = "vad0"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.github.metarank:lightgbm4j:4.1.0-2")
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
//    includes.set(listOf("Write*"))
    fork = 1
    warmupIterations = 5
    iterations = 7
    benchmarkMode.set(listOf("avgt"))
//    benchmarkMode.set(listOf("all"))
    timeUnit = "us"
    failOnError = true
    duplicateClassesStrategy = DuplicatesStrategy.INCLUDE
    profilers.set(listOf("async:libPath=/home/vadim/Documents/soft/async-profiler-3.0-linux-x64/lib/libasyncProfiler.so;output=flamegraph;dir=profile-results"))
}

//jmhJar {
//    duplicatesStrategy(DuplicatesStrategy.INCLUDE)
//}
