package benches;

import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMException;
import main.Booster;
import main.Main;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
public class LightGbmBenchmark {
    private final LGBMBooster lgbmBooster;
    private final Booster booster;
    private final double[] input;

    public LightGbmBenchmark() {
        try {
            lgbmBooster = LGBMBooster.createFromModelfile(Main.PATH);
            booster = Booster.createFromModelFile(Main.PATH);
            input = new double[booster.numFeatures()];
        } catch (LGBMException e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double[] predictDefault() {
        return Main.predictDefault(lgbmBooster, input);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double predictSingleRow() {
        return Main.predictSingleRow(lgbmBooster, input);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double predictSingleRowUnsafe() {
        return booster.predict(input);
    }
}
