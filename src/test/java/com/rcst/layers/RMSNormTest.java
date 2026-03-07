package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import com.rcst.TensorEncoder;
import com.rcst.TokenEmbeddingTable;
import com.rcst.Tokenizer;
import com.rcst.TrainingDataLoader;
import java.util.List;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class RMSNormTest extends TestCase {

    private static final int BLOCK_SIZE = 8;
    private static final int BATCH_SIZE = 4;
    private static final int N_EMBED = 32;
    private static final double TRAIN_RATIO = 0.8;
    private static final float EPS = 1e-6f;

    private static final List<String> SENTENCES = List.of(
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

    // Shared across all tests — initialised once, torn down once
    private static NDManager manager;
    private static Tokenizer tokenizer;
    private static TrainingDataLoader loader;
    private static TokenEmbeddingTable embeddingTable;
    private static RMSNorm rmsNorm;

    public static Test suite() {
        return new TestSetup(new TestSuite(RMSNormTest.class)) {
            @Override
            protected void setUp() throws Exception {
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
                    42L
                );

                List<Integer> sample = tokenizer.encodeWithBosEos(
                    SENTENCES.get(0)
                );
                int vocabSize =
                    sample
                        .stream()
                        .mapToInt(Integer::intValue)
                        .max()
                        .getAsInt() +
                    1000;

                embeddingTable = new TokenEmbeddingTable(vocabSize, N_EMBED);
                embeddingTable.initialize(
                    manager,
                    DataType.FLOAT32,
                    new Shape(BATCH_SIZE, BLOCK_SIZE)
                );

                rmsNorm = new RMSNorm(N_EMBED, EPS);
                rmsNorm.initialize(
                    manager,
                    DataType.FLOAT32,
                    new Shape(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
                );
            }

            @Override
            protected void tearDown() throws Exception {
                tokenizer.close();
                manager.close();
            }
        };
    }

    /** Fresh ParameterStore per test — prevents stale cache across Batch scopes. */
    private ParameterStore freshPs() {
        return new ParameterStore(manager, false);
    }

    /**
     * Run embedding → RMSNorm end-to-end from a real batch.
     */
    private NDArray pipeline(Batch batch, ParameterStore ps) {
        NDArray tokenIds = batch.getData().head(); // (B, T)
        NDArray embedded = embeddingTable
            .forward(ps, new NDList(tokenIds), false)
            .singletonOrThrow(); // (B, T, N_EMBED)
        return rmsNorm
            .forward(ps, new NDList(embedded), false)
            .singletonOrThrow(); // (B, T, N_EMBED)
    }

    // -------------------------------------------------------------------------
    // 1. Output shape equals input shape — RMSNorm is shape-preserving
    // -------------------------------------------------------------------------
    public void testOutputShapeMatchesInput() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray out = pipeline(batch, ps);

            assertEquals(BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(N_EMBED, (int) out.getShape().get(2));

            System.out.printf("RMSNorm output shape: %s%n", out.getShape());
        }
    }

    // -------------------------------------------------------------------------
    // 2. getOutputShapes returns same shape as input
    // -------------------------------------------------------------------------
    public void testGetOutputShapes() {
        Shape input = new Shape(BATCH_SIZE, BLOCK_SIZE, N_EMBED);
        Shape[] out = rmsNorm.getOutputShapes(new Shape[] { input });

        assertEquals(1, out.length);
        assertEquals(input, out[0]);

        System.out.printf("getOutputShapes: %s%n", out[0]);
    }

    // -------------------------------------------------------------------------
    // 3. Output dtype is FLOAT32
    // -------------------------------------------------------------------------
    public void testOutputIsFloat32() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray out = pipeline(batch, ps);

            assertEquals(DataType.FLOAT32, out.getDataType());

            System.out.printf("output dtype: %s%n", out.getDataType());
        }
    }

    // -------------------------------------------------------------------------
    // 4. Each token vector has RMS ≈ 1 after normalisation
    //    (γ is initialised to 1, so output RMS should equal γ = 1)
    // -------------------------------------------------------------------------
    public void testRmsOfEachTokenIsApproxOne() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray out = pipeline(batch, ps); // (B, T, N_EMBED)

            float[] vals = out.toFloatArray();
            int total = BATCH_SIZE * BLOCK_SIZE;

            for (int bt = 0; bt < total; bt++) {
                float sumSq = 0f;
                for (int d = 0; d < N_EMBED; d++) {
                    float v = vals[bt * N_EMBED + d];
                    sumSq += v * v;
                }
                float rms = (float) Math.sqrt(sumSq / N_EMBED);
                assertEquals(
                    "RMS of token " + bt + " should be ≈ 1.0 but was " + rms,
                    1.0f,
                    rms,
                    0.05f // 5 % tolerance
                );
            }

            System.out.println("All token RMS values ≈ 1.0 ✓");
        }
    }

    // -------------------------------------------------------------------------
    // 5. Zero input produces zero output (RMSNorm(0) = 0)
    // -------------------------------------------------------------------------
    public void testZeroInputGivesZeroOutput() {
        ParameterStore ps = freshPs();
        NDArray zeros = manager.zeros(
            new Shape(BATCH_SIZE, BLOCK_SIZE, N_EMBED),
            DataType.FLOAT32
        );

        NDArray out = rmsNorm
            .forward(ps, new NDList(zeros), false)
            .singletonOrThrow();

        for (float v : out.toFloatArray()) {
            assertEquals("RMSNorm(0) must be 0", 0f, v, 1e-6f);
        }

        System.out.println("RMSNorm(0) = 0 ✓");
    }

    // -------------------------------------------------------------------------
    // 6. Gamma (γ) parameter is initialised to ones
    // -------------------------------------------------------------------------
    public void testGammaInitialisedToOnes() {
        NDArray g = rmsNorm.getParameters().get("gamma").getArray();

        for (float v : g.toFloatArray()) {
            assertEquals("γ must be initialised to 1.0", 1.0f, v, 1e-6f);
        }

        System.out.printf("gamma shape: %s  all ones ✓%n", g.getShape());
    }

    // -------------------------------------------------------------------------
    // 7. Output shape is stable across multiple batches
    // -------------------------------------------------------------------------
    public void testShapeIsConsistentAcrossBatches() {
        for (int i = 0; i < 3; i++) {
            ParameterStore ps = freshPs();
            try (Batch batch = loader.sampleTrain()) {
                NDArray out = pipeline(batch, ps);

                assertEquals(BATCH_SIZE, (int) out.getShape().get(0));
                assertEquals(BLOCK_SIZE, (int) out.getShape().get(1));
                assertEquals(N_EMBED, (int) out.getShape().get(2));

                System.out.printf("call %d shape: %s%n", i, out.getShape());
            }
        }
    }

    // -------------------------------------------------------------------------
    // 8. Validation batch passes through with correct shape
    // -------------------------------------------------------------------------
    public void testValBatchShape() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleValidation()) {
            NDArray out = pipeline(batch, ps);

            assertEquals(BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(N_EMBED, (int) out.getShape().get(2));

            System.out.printf("val output shape: %s%n", out.getShape());
        }
    }
}
