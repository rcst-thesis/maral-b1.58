package com.rcst;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import com.rcst.layers.BitLinear;
import com.rcst.layers.RMSNorm;
import com.rcst.utils.ModelConfig;
import com.rcst.utils.TensorEncoder;
import com.rcst.utils.TokenEmbeddingTable;
import com.rcst.utils.Tokenizer;
import com.rcst.utils.TrainingDataLoader;
import java.util.List;

/**
 * Shared test fixture — single NDManager, Tokenizer, Loader and EmbeddingTable
 * for the entire test suite. Eliminates duplicated setUp/tearDown across classes.
 *
 * Usage
 *
 *   public static Test suite() {
 *       return new TestSetup(new TestSuite(MyTest.class)) {
 *           protected void setUp()    throws Exception { TestFixture.init(); }
 *           protected void tearDown() throws Exception { TestFixture.destroy(); }
 *       };
 *   }
 *
 *   // inside a test method:
 *   ParameterStore ps    = TestFixture.freshPs();
 *   Batch          batch = TestFixture.loader.sampleTrain();
 *   NDArray        emb   = TestFixture.embed(batch, ps);
 *
 * Notes
 *  - BATCH_SIZE and BLOCK_SIZE are clamped to small values so tests stay fast,
 *    regardless of what is set in model-config.yaml.
 *  - Always call freshPs() per test — never share one ParameterStore across
 *    Batch scopes, or you will hit "Native resource already released".
 */
public final class TestFixture {

    // Dimensions (derived from config, clamped for speed)
    public static int VOCAB_SIZE;
    public static int D_MODEL;
    public static int BATCH_SIZE;
    public static int BLOCK_SIZE;
    public static double TRAIN_RATIO;
    public static long SEED;

    // Shared objects
    public static NDManager manager;
    public static Tokenizer tokenizer;
    public static TrainingDataLoader loader;
    public static TokenEmbeddingTable embeddingTable;

    // Bilingual sample sentences (mirrors en-tl corpus style)
    public static final List<String> SENTENCES = List.of(
        "Good morning, how are you?",
        "Magandang umaga, kumusta ka?",
        "I am going to the market.",
        "Pupunta ako sa palengke.",
        "The weather is nice today.",
        "Maganda ang panahon ngayon.",
        "Can you help me please?",
        "Maaari mo ba akong tulungan?",
        "I love learning new languages.",
        "Mahilig akong matuto ng bagong wika."
    );

    private TestFixture() {}

    // Lifecycle

    public static void init() throws Exception {
        ModelConfig cfg = ModelConfig.get();

        VOCAB_SIZE = cfg.vocabSize;
        D_MODEL = cfg.dModel;
        TRAIN_RATIO = cfg.trainRatio;
        SEED = cfg.seed;
        // Clamp to small values — tests must be fast regardless of prod config
        BATCH_SIZE = Math.min(cfg.batchSize, 4);
        BLOCK_SIZE = Math.min(cfg.blockSize, 8);

        manager = NDManager.newBaseManager();
        tokenizer = new Tokenizer();

        TensorEncoder encoder = new TensorEncoder(tokenizer, manager);
        NDArray[] tensors = encoder.encodeBatch(SENTENCES);

        loader = new TrainingDataLoader(
            tensors,
            manager,
            BLOCK_SIZE,
            BATCH_SIZE,
            TRAIN_RATIO,
            SEED
        );

        embeddingTable = new TokenEmbeddingTable(VOCAB_SIZE, D_MODEL);
        embeddingTable.initialize(
            manager,
            DataType.FLOAT32,
            new Shape(BATCH_SIZE, BLOCK_SIZE)
        );
    }

    public static void destroy() throws Exception {
        if (tokenizer != null) tokenizer.close();
        if (manager != null) manager.close();
    }

    // Factory helpers

    /**
     * Fresh ParameterStore per test — prevents stale weight cache across
     * Batch scopes. Call this at the top of every test method.
     */
    public static ParameterStore freshPs() {
        return new ParameterStore(manager, false);
    }

    /**
     * Token embedding lookup from a batch's data tensor.
     * Returns shape: (BATCH_SIZE, BLOCK_SIZE, D_MODEL).
     */
    public static NDArray embed(Batch batch, ParameterStore ps) {
        return embeddingTable
            .forward(ps, new NDList(batch.getData().head()), false)
            .singletonOrThrow();
    }

    /**
     * Build and initialise a RMSNorm sized to D_MODEL with eps from config.
     */
    public static RMSNorm buildRMSNorm() {
        RMSNorm norm = new RMSNorm(D_MODEL, ModelConfig.get().eps);
        norm.initialize(
            manager,
            DataType.FLOAT32,
            new Shape(BATCH_SIZE, BLOCK_SIZE, D_MODEL)
        );
        return norm;
    }

    /**
     * Build and initialise a BitLinear(inFeat → outFeat) with eps from config.
     */
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
