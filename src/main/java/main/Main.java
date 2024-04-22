package main;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMException;

public class Main {
    public static final String PATH = "src/test/resources/model.txt";

    public static double[] predictDefault(LGBMBooster booster, double[] input) {
        try {
            return booster.predictForMat(input, 1, input.length, true, PredictionType.C_API_PREDICT_NORMAL);
        } catch (LGBMException e) {
            throw new RuntimeException(e);
        }
    }

    public static double predictSingleRow(LGBMBooster booster, double[] input) {
        try {
            return booster.predictForMatSingleRow(input, PredictionType.C_API_PREDICT_NORMAL);
        } catch (LGBMException e) {
            throw new RuntimeException(e);
        }
    }
}