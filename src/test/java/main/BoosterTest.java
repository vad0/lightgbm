package main;

import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMException;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BoosterTest {
    private static double[] getInput() {
        final double[] input = new double[Main.FEATURES];
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
        final double[] input = new double[Main.FEATURES];
        for (int i = 0; i < input.length; i++) {
            input[i] = i++;
        }
        final double expected = getExpectedPrediction(input);
        try (var booster = Booster.createFromModelFile(Main.PATH, Main.FEATURES)) {
            double actual = booster.predictForMatSingleRowFast(input);
            assertEquals(expected, actual);
        }
    }

    @Test
    public void testMatchesDefaultImplementationRandom() {
        try (var booster = Booster.createFromModelFile(Main.PATH, Main.FEATURES)) {
            for (int j = 0; j < 1; j++) {
                final double[] input = getInput();
                final double expected = getExpectedPrediction(input);
                for (int i = 0; i < 10; i++) {
                    double actual = booster.predictForMatSingleRowUnsafe(input);
                    assertEquals(expected, actual);
                }
            }
        }
    }

    @Test
    public void testRowAndMatrixAreSame() {
        for (int j = 0; j < 10; j++) {
            final double[] input = getInput();
            final double prediction = getExpectedPrediction(input);
            final double predictionMat = getExpectedPredictionMat(input);
            assertEquals(prediction, predictionMat);
        }
    }
}