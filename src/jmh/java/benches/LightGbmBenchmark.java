package benches;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMException;
import org.openjdk.jmh.annotations.*;
import vad0.Main;

import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
public class LightGbmBenchmark {
    private final LGBMBooster booster;
    private final double[] input = new double[]{15};

    public LightGbmBenchmark() {
        try {
            this.booster = LGBMBooster.createFromModelfile(Main.PATH);
        } catch (LGBMException e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double[] lightgbm4j() throws LGBMException {
        return booster.predictForMat(input, 1, 15, true, PredictionType.C_API_PREDICT_NORMAL,"device_type");
    }
}
