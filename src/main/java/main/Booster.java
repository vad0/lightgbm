package main;

import com.microsoft.ml.lightgbm.*;
import io.github.metarank.lightgbm4j.LGBMException;
import org.agrona.concurrent.UnsafeBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

public class Booster
    implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(Booster.class);
    private static volatile boolean nativeLoaded = false;

    static {
        try {
            loadNative();
        } catch (IOException var1) {
            LOGGER.info("Cannot load native library for your platform");
        }
    }

    private final SWIGTYPE_p_long_long outLength = lightgbmlib.new_int64_tp();
    private final MyDoublePointer outputBuffer;
    private final SWIGTYPE_p_void handlePtr;
    private final SWIGTYPE_p_void dataVoidPtr;
    private final SWIGTYPE_p_p_void out_fastConfig = lightgbmlib.voidpp_handle();
    private final UnsafeBuffer inBuffer;
    private final UnsafeBuffer outBuffer;
    private final int iterations;
    private final SWIGTYPE_p_p_void handle;
    private final int features;
    private final SWIGTYPE_p_double inputBuffer;
    private SWIGTYPE_p_void fastConfigHandle;

    private volatile boolean isClosed = false;

    Booster(int iterations, SWIGTYPE_p_p_void handle, int features) {
        assert features > 0 : features;
        this.iterations = iterations;
        this.handle = handle;
        this.features = features;
        inputBuffer = lightgbmlib.new_doubleArray(features);
        handlePtr = lightgbmlib.voidpp_value(handle);
        dataVoidPtr = lightgbmlib.double_to_voidp_ptr(inputBuffer);
        inBuffer = new UnsafeBuffer(SWIGTYPE_p_void.getCPtr(dataVoidPtr), features * Double.BYTES);
        long outAddress = lightgbmlibJNI.new_doubleArray(1);
        outputBuffer = new MyDoublePointer(outAddress);
        outBuffer = new UnsafeBuffer(outAddress, Double.BYTES);
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
        InputStream libStream = Booster.class.getClassLoader().getResourceAsStream(path);
        ByteArrayOutputStream libByteStream = new ByteArrayOutputStream();
        copyStream(libStream, libByteStream);
        libStream.close();
        InputStream md5Stream = Booster.class.getClassLoader().getResourceAsStream(path + ".md5");
        ByteArrayOutputStream md5ByteStream = new ByteArrayOutputStream();
        copyStream(md5Stream, md5ByteStream);
        md5Stream.close();
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

        LOGGER.info("Copied " + bytesCopied + " bytes");
    }

    public static Booster createFromModelFile(String file, int features) throws LGBMException {
        SWIGTYPE_p_p_void handle = lightgbmlib.new_voidpp();
        SWIGTYPE_p_int outIterations = lightgbmlib.new_intp();
        int result = lightgbmlib.LGBM_BoosterCreateFromModelfile(file, outIterations, handle);
        if (result < 0) {
            throw new LGBMException(lightgbmlib.LGBM_GetLastError());
        } else {
            int iterations = lightgbmlib.intp_value(outIterations);
            lightgbmlib.delete_intp(outIterations);
            return new Booster(iterations, handle, features);
        }
    }

    public void close() throws LGBMException {
        if (!this.isClosed) {
            this.isClosed = true;
            int result = lightgbmlib.LGBM_BoosterFree(lightgbmlib.voidpp_value(this.handle));
            if (result < 0) {
                throw new LGBMException(lightgbmlib.LGBM_GetLastError());
            }
        }
    }

    public double[] predictForMat(
        double[] input, int rows, int cols, boolean isRowMajor,
        PredictionType predictionType, String parameter
    ) throws LGBMException {
        SWIGTYPE_p_double dataBuffer = lightgbmlib.new_doubleArray(input.length);

        for (int i = 0; i < input.length; ++i) {
            lightgbmlib.doubleArray_setitem(dataBuffer, i, input[i]);
        }

        SWIGTYPE_p_long_long outLength = lightgbmlib.new_int64_tp();
        long outSize = this.outBufferSize(rows, cols, predictionType);
        SWIGTYPE_p_double outBuffer = lightgbmlib.new_doubleArray(outSize);
        int result = lightgbmlib.LGBM_BoosterPredictForMat(
            lightgbmlib.voidpp_value(this.handle),
            lightgbmlib.double_to_voidp_ptr(dataBuffer),
            lightgbmlib.C_API_DTYPE_FLOAT64,
            rows,
            cols,
            isRowMajor ? 1 : 0,
            predictionType.getType(),
            0,
            this.iterations,
            parameter,
            outLength,
            outBuffer);
        if (result < 0) {
            lightgbmlib.delete_doubleArray(dataBuffer);
            lightgbmlib.delete_int64_tp(outLength);
            lightgbmlib.delete_doubleArray(outBuffer);
            throw new LGBMException(lightgbmlib.LGBM_GetLastError());
        } else {
            long length = lightgbmlib.int64_tp_value(outLength);
            double[] values = new double[(int)length];

            for (int i = 0; (long)i < length; ++i) {
                values[i] = lightgbmlib.doubleArray_getitem(outBuffer, i);
            }

            lightgbmlib.delete_doubleArray(dataBuffer);
            lightgbmlib.delete_int64_tp(outLength);
            lightgbmlib.delete_doubleArray(outBuffer);
            return values;
        }
    }

    public String[] getFeatureNames() {
        SWIGTYPE_p_void buffer = lightgbmlib.LGBM_BoosterGetFeatureNamesSWIG(lightgbmlib.voidpp_value(this.handle));
        String[] result = lightgbmlib.StringArrayHandle_get_strings(buffer);
        lightgbmlib.StringArrayHandle_free(buffer);
        return result;
    }

    public int getNumFeature() throws LGBMException {
        SWIGTYPE_p_int outNum = lightgbmlib.new_int32_tp();
        int result = lightgbmlib.LGBM_BoosterGetNumFeature(lightgbmlib.voidpp_value(this.handle), outNum);
        if (result < 0) {
            lightgbmlib.delete_intp(outNum);
            throw new LGBMException(lightgbmlib.LGBM_GetLastError());
        } else {
            int num = lightgbmlib.intp_value(outNum);
            lightgbmlib.delete_intp(outNum);
            return num;
        }
    }

    public double predictForMatSingleRow(double[] data, PredictionType predictionType) throws LGBMException {
        for (int i = 0; i < data.length; ++i) {
            lightgbmlib.doubleArray_setitem(inputBuffer, i, data[i]);
        }
        int result = lightgbmlib.LGBM_BoosterPredictForMatSingleRow(
            handlePtr,
            dataVoidPtr,
            lightgbmlib.C_API_DTYPE_FLOAT64,
            data.length,
            1,
            predictionType.getType(),
            0,
            iterations,
            "",
            outLength,
            outputBuffer);
        if (result < 0) {
            throw new LGBMException(lightgbmlib.LGBM_GetLastError());
        } else {
            return lightgbmlib.doubleArray_getitem(outputBuffer, 0);
        }
    }

    public void preparePredict() {
        int result = lightgbmlib.LGBM_BoosterPredictForMatSingleRowFastInit(
            handlePtr,
            lightgbmlib.C_API_DTYPE_FLOAT64,
            0,
            iterations,
            PredictionType.C_API_PREDICT_NORMAL.getType(),
            features,
            "",
            out_fastConfig);
        if (result < 0) {
            throw new RuntimeException(lightgbmlib.LGBM_GetLastError());
        }
        fastConfigHandle = lightgbmlib.voidpp_value(out_fastConfig);
    }

    public double predictForMatSingleRowFast(double[] data) {
        for (int i = 0; i < data.length; ++i) {
            lightgbmlib.doubleArray_setitem(inputBuffer, i, data[i]);
        }
        int result = lightgbmlib.LGBM_BoosterPredictForMatSingleRowFast(
            fastConfigHandle,
            dataVoidPtr,
            outLength,
            outputBuffer);
        if (result < 0) {
            throw new RuntimeException(lightgbmlib.LGBM_GetLastError());
        } else {
            return lightgbmlib.doubleArray_getitem(outputBuffer, 0);
        }
    }

    public double predictForMatSingleRowUnsafe(double[] data) {
        for (int i = 0; i < data.length; ++i) {
            inBuffer.putDouble(Double.BYTES * i, data[i]);
        }
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

    private long outBufferSize(int rows, int cols, PredictionType predictionType) {
        long defaultSize = 2L * (long)rows;
        if (PredictionType.C_API_PREDICT_CONTRIB.equals(predictionType)) {
            return defaultSize * (long)(cols + 1);
        } else {
            return PredictionType.C_API_PREDICT_LEAF_INDEX.equals(predictionType) ?
                defaultSize * (long)this.iterations : defaultSize;
        }
    }
}
