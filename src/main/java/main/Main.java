package main;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMException;

import java.util.Arrays;

public class Main {
    public static final String PATH = "src/test/resources/model.txt";
    public static final int FEATURES = 15;

    public static void main(String[] args) throws LGBMException {
        try (var booster = Booster.createFromModelFile(PATH, FEATURES)) {
            double[] input = new double[FEATURES];

            booster.preparePredict();
            double pred3 = booster.predictForMatSingleRowFast(input);
            System.out.println(pred3);

            double pred4 = booster.predictForMatSingleRowUnsafe(input);
            System.out.println(pred4);

            final double pred2 = predictNoAllocation(booster, input);
            System.out.println(pred2);
            final double[] pred1 = predictSingleThread(booster, input);
            System.out.println(Arrays.toString(pred1));
        }
        System.out.println("Hello world!");
    }

    public static double[] predictDefault(LGBMBooster booster, double[] input) throws LGBMException {
        return booster.predictForMat(input, 1, input.length, true, PredictionType.C_API_PREDICT_NORMAL);
    }

    public static double[] predictSingleThread(Booster booster, double[] input) throws LGBMException {
        return booster.predictForMat(
                input,
                1,
                input.length,
                true,
                PredictionType.C_API_PREDICT_NORMAL,
                "num_threads=1 device=cpu");
    }

    public static double predictSingleRow(LGBMBooster booster, double[] input) throws LGBMException {
        return booster.predictForMatSingleRow(input, PredictionType.C_API_PREDICT_NORMAL);
    }

    public static double predictNoAllocation(Booster booster, double[] input) throws LGBMException {
        return booster.predictForMatSingleRow(input, PredictionType.C_API_PREDICT_NORMAL);
    }
}