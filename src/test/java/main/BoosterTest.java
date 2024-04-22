package main;

import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMException;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BoosterTest {
    public static final int FEATURES = 15;

    private static double[] getInput(int numFeatures) {
        final double[] input = new double[numFeatures];
        for (int i = 0; i < input.length; i++) {
            input[i] = ThreadLocalRandom.current().nextDouble();
        }
        return input;
    }

    private static double getExpectedPrediction(double[] input) {
        try (var lgbmBooster = LGBMBooster.createFromModelfile(Main.PATH)) {
            return Main.predictSingleRow(lgbmBooster, input);
        } catch (LGBMException e) {
            throw new RuntimeException(e);
        }
    }

    private static double getExpectedPredictionMat(double[] input) {
        try (var lgbmBooster = LGBMBooster.createFromModelfile(Main.PATH)) {
            return Main.predictDefault(lgbmBooster, input)[0];
        } catch (LGBMException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testMatchesDefaultImplementation() {
        try (var booster = Booster.createFromModelFile(Main.PATH)) {
            final double[] input = new double[booster.numFeatures()];
            for (int i = 0; i < input.length; i++) {
                input[i] = i++;
            }
            final double expected = getExpectedPrediction(input);
            double actual = booster.predict(input);
            assertEquals(expected, actual);
        }
    }

    @Test
    public void testMatchesDefaultImplementationRandom() {
        try (var booster = Booster.createFromModelFile(Main.PATH)) {
            for (int j = 0; j < 1; j++) {
                final double[] input = getInput(booster.numFeatures());
                final double expected = getExpectedPrediction(input);
                for (int i = 0; i < 10; i++) {
                    double actual = booster.predict(input);
                    assertEquals(expected, actual);
                }
            }
        }
    }

    @Test
    public void testRowAndMatrixAreSame() {
        for (int j = 0; j < 10; j++) {
            final double[] input = getInput(FEATURES);
            final double prediction = getExpectedPrediction(input);
            final double predictionMat = getExpectedPredictionMat(input);
            assertEquals(prediction, predictionMat);
        }
    }

    @Test
    public void featureNames() {
        try (var booster = Booster.createFromModelFile(Main.PATH)) {
            assertEquals(FEATURES, booster.numFeatures());
            var names = Arrays.stream(booster.featureNames()).collect(Collectors.toSet());
            assertEquals(FEATURES, names.size());
            assertTrue(names.contains("Ret-BNB-OKX_SPOT"));
        }
    }
}