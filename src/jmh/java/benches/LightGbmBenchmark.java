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
    private final double[] input = new double[15];

    public LightGbmBenchmark() {
        try {
            this.lgbmBooster = LGBMBooster.createFromModelfile(Main.PATH);
            this.booster = Booster.createFromModelFile(Main.PATH, Main.FEATURES);
            booster.preparePredict();
        } catch (LGBMException e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double[] predictDefault() throws LGBMException {
        return Main.predictDefault(lgbmBooster, input);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double[] predictSingleThread() throws LGBMException {
        return Main.predictSingleThread(booster, input);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double predictSingleRow() throws LGBMException {
        return Main.predictSingleRow(lgbmBooster, input);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double predictSingleRowNoAllocation() throws LGBMException {
        return Main.predictNoAllocation(booster, input);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double predictSingleRowFast() {
        return booster.predictForMatSingleRowFast(input);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double predictSingleRowUnsafe() {
        return booster.predictForMatSingleRowUnsafe(input);
    }
}
