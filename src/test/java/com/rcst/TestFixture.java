package com.rcst;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import com.rcst.common.RMSNorm;
import com.rcst.layers.BitLinear;
import com.rcst.layers.Embedder;
import com.rcst.utils.ModelConfig;

/**
 * Shared test fixture. Single NDManager and Embedder for the entire test suite.
 *
 * Usage:
 *   public static Test suite() {
 *       return new TestSetup(new TestSuite(MyTest.class)) {
 *           protected void setUp()    throws Exception { TestFixture.init(); }
 *           protected void tearDown() throws Exception { TestFixture.destroy(); }
 *       };
 *   }
 *
 * BATCH_SIZE and BLOCK_SIZE are clamped small so tests stay fast regardless
 * of model-config.yaml. Always call freshPs() per test — sharing a
 * ParameterStore across Batch scopes causes "Native resource already released".
 */
public final class TestFixture {

    public static int VOCAB_SIZE;
    public static int D_MODEL;
    public static int BATCH_SIZE;
    public static int BLOCK_SIZE;
    public static double TRAIN_RATIO;
    public static long SEED;

    public static NDManager manager;
    public static Embedder embedder;

    private TestFixture() {}

    public static void init() throws Exception {
        ModelConfig cfg = ModelConfig.get();

        VOCAB_SIZE = cfg.vocabSize;
        D_MODEL = cfg.dModel;
        TRAIN_RATIO = cfg.trainRatio;
        SEED = cfg.seed;
        BATCH_SIZE = Math.min(cfg.batchSize, 4);
        BLOCK_SIZE = Math.min(cfg.blockSize, 8);

        manager = NDManager.newBaseManager();

        embedder = new Embedder(cfg.vocabSize, cfg.dModel);
        embedder.initialize(
            manager,
            DataType.FLOAT32,
            new Shape(BATCH_SIZE, BLOCK_SIZE)
        );
        embedder.initLoader(manager, BLOCK_SIZE, BATCH_SIZE, TRAIN_RATIO, SEED);
    }

    public static void destroy() throws Exception {
        if (manager != null) manager.close();
    }

    public static ParameterStore freshPs() {
        return new ParameterStore(manager, false);
    }

    public static NDArray embed(Batch batch, ParameterStore ps) {
        return embedder
            .forward(ps, new NDList(batch.getData().head()), false)
            .singletonOrThrow();
    }

    public static RMSNorm buildRMSNorm() {
        RMSNorm norm = new RMSNorm(D_MODEL, ModelConfig.get().eps);
        norm.initialize(
            manager,
            DataType.FLOAT32,
            new Shape(BATCH_SIZE, BLOCK_SIZE, D_MODEL)
        );
        return norm;
    }

    public static BitLinear buildBitLinear(int inFeat, int outFeat) {
        BitLinear bl = new BitLinear(
            inFeat,
            outFeat,
            ModelConfig.get().quantEps
        );
        bl.initialize(
            manager,
            DataType.FLOAT32,
            new Shape(BATCH_SIZE, BLOCK_SIZE, inFeat)
        );
        return bl;
    }
}
