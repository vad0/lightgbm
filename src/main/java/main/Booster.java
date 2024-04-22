package main;

import com.microsoft.ml.lightgbm.*;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

import static java.util.Objects.requireNonNull;

public class Booster
    implements AutoCloseable {
    private static final Logger LOGGER = LogManager.getLogger(Booster.class);
    private static volatile boolean nativeLoaded = false;

    static {
        try {
            loadNative();
        } catch (IOException exception) {
            LOGGER.info("Cannot load native library for your platform");
        }
    }

    private final SWIGTYPE_p_long_long outLength = lightgbmlib.new_int64_tp();
    private final MyDoublePointer outputBuffer;
    private final SWIGTYPE_p_void dataVoidPtr;
    private final UnsafeBuffer inBuffer;
    private final UnsafeBuffer outBuffer;
    private final SWIGTYPE_p_p_void handle;
    private final SWIGTYPE_p_void fastConfigHandle;

    private volatile boolean isClosed = false;

    Booster(int iterations, SWIGTYPE_p_p_void handle) {
        this.handle = handle;
        int features = numFeatures();
        assert features > 0 : features;
        SWIGTYPE_p_double inputBuffer = lightgbmlib.new_doubleArray(features);
        SWIGTYPE_p_void handlePtr = lightgbmlib.voidpp_value(handle);
        dataVoidPtr = lightgbmlib.double_to_voidp_ptr(inputBuffer);
        inBuffer = new UnsafeBuffer(SWIGTYPE_p_void.getCPtr(dataVoidPtr), features * Double.BYTES);
        long outAddress = lightgbmlibJNI.new_doubleArray(1);
        outputBuffer = new MyDoublePointer(outAddress);
        outBuffer = new UnsafeBuffer(outAddress, Double.BYTES);
        var outFastConfig = lightgbmlib.voidpp_handle();
        int result = lightgbmlib.LGBM_BoosterPredictForMatSingleRowFastInit(
            handlePtr,
            PredictionType.C_API_PREDICT_NORMAL.getType(),
            0,
            iterations,
            lightgbmlib.C_API_DTYPE_FLOAT64,
            features,
            "",
            outFastConfig);
        if (result < 0) {
            throw new RuntimeException(lightgbmlib.LGBM_GetLastError());
        }
        fastConfigHandle = lightgbmlib.voidpp_value(outFastConfig);
    }

    public static synchronized void loadNative() throws IOException {
        if (!nativeLoaded) {
            String os = System.getProperty("os.name");
            String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH);
            UnsatisfiedLinkError err;
            String message;
            if (!os.startsWith("Linux") && !os.startsWith("LINUX")) {
                if (os.startsWith("Mac")) {
                    try {
                        if (!arch.startsWith("amd64") && !arch.startsWith("x86_64")) {
                            if (!arch.startsWith("aarch64") && !arch.startsWith("arm64")) {
                                LOGGER.warn("arch " + arch + " is not supported");
                                throw new UnsatisfiedLinkError("no native lightgbm library found for your OS " + os);
                            }

                            loadNative("lightgbm4j/osx/aarch64/", "lib_lightgbm.dylib");
                            loadNative("lightgbm4j/osx/aarch64/", "lib_lightgbm_swig.dylib");
                            nativeLoaded = true;
                        } else {
                            loadNative("lightgbm4j/osx/x86_64/", "lib_lightgbm.dylib");
                            loadNative("lightgbm4j/osx/x86_64/", "lib_lightgbm_swig.dylib");
                            nativeLoaded = true;
                        }
                    } catch (UnsatisfiedLinkError var4) {
                        err = var4;
                        message = err.getMessage();
                        if (message.contains("libomp.dylib")) {
                            LOGGER.warn("\n\n\n");
                            LOGGER.warn("****************************************************");
                            LOGGER.warn("Your MacOS system probably has no 'libomp' library installed!");
                            LOGGER.warn("Please double-check the lightgbm4j install instructions:");
                            LOGGER.warn("- https://github.com/metarank/lightgbm4j/");
                            LOGGER.warn("- or just do 'brew install libomp'");
                            LOGGER.warn("****************************************************");
                            LOGGER.warn("\n\n\n");
                        }

                        throw err;
                    }
                } else if (os.startsWith("Windows")) {
                    loadNative("lightgbm4j/windows/x86_64/", "lib_lightgbm.dll");
                    loadNative("lightgbm4j/windows/x86_64/", "lib_lightgbm_swig.dll");
                    nativeLoaded = true;
                } else {
                    LOGGER.error("Only Linux@x86_64, Windows@x86_64, Mac@x86_64 and Mac@aarch are supported");
                }
            } else {
                try {
                    if (!arch.startsWith("amd64") && !arch.startsWith("x86_64")) {
                        if (arch.startsWith("aarch64") || arch.startsWith("arm64")) {
                            loadNative("lightgbm4j/linux/aarch64/", "lib_lightgbm.so");
                            loadNative("lightgbm4j/linux/aarch64/", "lib_lightgbm_swig.so");
                            nativeLoaded = true;
                        }
                    } else {
                        loadNative("lightgbm4j/linux/x86_64/", "lib_lightgbm.so");
                        loadNative("lightgbm4j/linux/x86_64/", "lib_lightgbm_swig.so");
                        nativeLoaded = true;
                    }
                } catch (UnsatisfiedLinkError var5) {
                    err = var5;
                    message = err.getMessage();
                    if (message.contains("libgomp")) {
                        LOGGER.warn("\n\n\n");
                        LOGGER.warn("****************************************************");
                        LOGGER.warn("Your Linux system probably has no 'libgomp' library installed!");
                        LOGGER.warn("Please double-check the lightgbm4j install instructions:");
                        LOGGER.warn("- https://github.com/metarank/lightgbm4j/");
                        LOGGER.warn("- or just install the libgomp with your package manager");
                        LOGGER.warn("****************************************************");
                        LOGGER.warn("\n\n\n");
                    }
                }
            }
        }
    }

    private static void loadNative(String path, String name) throws IOException, UnsatisfiedLinkError {
        String nativePathOverride = System.getenv("LIGHTGBM_NATIVE_LIB_PATH");
        String libPath;
        if (nativePathOverride != null) {
            if (!nativePathOverride.endsWith("/")) {
                nativePathOverride = nativePathOverride + "/";
            }

            libPath = nativePathOverride + name;
            LOGGER.info("LIGHTGBM_NATIVE_LIB_PATH is set: loading {}", libPath);

            try {
                System.load(libPath);
            } catch (UnsatisfiedLinkError var7) {
                LOGGER.error("Cannot load library:{}", var7.getMessage(), var7);
                throw var7;
            }
        } else {
            LOGGER.info("Loading native lib from resource " + path + "/" + name);
            libPath = System.getProperty("java.io.tmpdir");
            File libFile = new File(libPath + File.separator + name);
            if (libFile.exists()) {
                LOGGER.info(libFile + " already exists");
            } else {
                extractResource(path + name, name, libFile);
            }

            LOGGER.info("Extracted file: exists=" + libFile.exists() + " path=" + libFile);

            try {
                System.load(libFile.toString());
            } catch (UnsatisfiedLinkError var6) {
                LOGGER.error("Cannot load library:" + var6.getMessage(), var6);
                throw var6;
            }
        }
    }

    private static void extractResource(String path, String name, File dest) throws IOException {
        LOGGER.info("Extracting native lib {}", dest);
        var libByteStream = new ByteArrayOutputStream();
        try (var libStream = requireNonNull(Booster.class.getClassLoader().getResourceAsStream(path))) {
            copyStream(libStream, libByteStream);
        }
        var md5ByteStream = new ByteArrayOutputStream();
        try (var md5Stream = requireNonNull(Booster.class.getClassLoader().getResourceAsStream(path + ".md5"))) {
            copyStream(md5Stream, md5ByteStream);
        }
        String expectedDigest = md5ByteStream.toString();

        try {
            byte[] digest = MessageDigest.getInstance("MD5").digest(libByteStream.toByteArray());
            String checksum = (new BigInteger(1, digest)).toString(16);
            if (!checksum.equals(expectedDigest)) {
                LOGGER.warn("\n\n\n");
                LOGGER.warn("****************************************************");
                LOGGER.warn("Hash mismatch between expected and real LightGBM native library in classpath!");
                LOGGER.warn("Your JVM classpath has {} with md5={} and we expect {}", name, checksum, expectedDigest);
                LOGGER.warn("This usually means that you have another LightGBM wrapper in classpath");
                LOGGER.warn("- MMLSpark/SynapseML is the main suspect");
                LOGGER.warn("****************************************************");
                LOGGER.warn("\n\n\n");
                throw new IOException("hash mismatch");
            } else {
                ByteArrayInputStream source = new ByteArrayInputStream(libByteStream.toByteArray());
                OutputStream fileStream = new FileOutputStream(dest);
                copyStream(source, fileStream);
                source.close();
                fileStream.close();
            }
        } catch (NoSuchAlgorithmException var12) {
            throw new IOException("md5 algorithm not supported, cannot check digest");
        }
    }

    private static void copyStream(InputStream source, OutputStream target) throws IOException {
        byte[] buf = new byte[8192];

        int length;
        int bytesCopied;
        for (bytesCopied = 0; (length = source.read(buf)) > 0; bytesCopied += length) {
            target.write(buf, 0, length);
        }

        LOGGER.info("Copied {} bytes", bytesCopied);
    }

    public static Booster createFromModelFile(String file) {
        var handle = lightgbmlib.new_voidpp();
        var outIterations = lightgbmlib.new_intp();
        int result = lightgbmlib.LGBM_BoosterCreateFromModelfile(file, outIterations, handle);
        if (result < 0) {
            throw new RuntimeException(lightgbmlib.LGBM_GetLastError());
        }
        int iterations = lightgbmlib.intp_value(outIterations);
        lightgbmlib.delete_intp(outIterations);
        return new Booster(iterations, handle);
    }

    public void close() {
        if (isClosed) {
            return;
        }
        isClosed = true;
        // TODO: close all stuff
        int result = lightgbmlib.LGBM_BoosterFree(lightgbmlib.voidpp_value(handle));
        if (result < 0) {
            throw new RuntimeException(lightgbmlib.LGBM_GetLastError());
        }
    }

    public String[] featureNames() {
        SWIGTYPE_p_void buffer = lightgbmlib.LGBM_BoosterGetFeatureNamesSWIG(lightgbmlib.voidpp_value(this.handle));
        String[] result = lightgbmlib.StringArrayHandle_get_strings(buffer);
        lightgbmlib.StringArrayHandle_free(buffer);
        return result;
    }

    public int numFeatures() {
        var outNum = lightgbmlib.new_int32_tp();
        int result = lightgbmlib.LGBM_BoosterGetNumFeature(lightgbmlib.voidpp_value(handle), outNum);
        if (result < 0) {
            lightgbmlib.delete_intp(outNum);
            throw new RuntimeException(lightgbmlib.LGBM_GetLastError());
        }
        int num = lightgbmlib.intp_value(outNum);
        lightgbmlib.delete_intp(outNum);
        return num;
    }

    public double predict(double[] data) {
        for (int i = 0; i < data.length; ++i) {
            setFeatureValue(i, data[i]);
        }
        return predict();
    }

    /**
     * Make sure to set feature values via {@link #setFeatureValue} before calling this method.
     */
    public double predict() {
        int result = lightgbmlib.LGBM_BoosterPredictForMatSingleRowFast(
            fastConfigHandle,
            dataVoidPtr,
            outLength,
            outputBuffer);
        if (result < 0) {
            throw new RuntimeException(lightgbmlib.LGBM_GetLastError());
        } else {
            return outBuffer.getDouble(0);
        }
    }

    public void setFeatureValue(int index, double value) {
        inBuffer.putDouble(Double.BYTES * index, value);
    }
}
