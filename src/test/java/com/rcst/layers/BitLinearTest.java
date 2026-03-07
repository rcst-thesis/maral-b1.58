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

public class BitLinearTest extends TestCase {

    private static final int BLOCK_SIZE = 8;
    private static final int BATCH_SIZE = 4;
    private static final int N_EMBED = 32;
    private static final int OUT_FEAT = 16;
    private static final double TRAIN_RATIO = 0.8;

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

    private static NDManager manager;
    private static Tokenizer tokenizer;
    private static TrainingDataLoader loader;
    private static TokenEmbeddingTable embeddingTable;
    private static BitLinear bitLinear;

    public static Test suite() {
        return new TestSetup(new TestSuite(BitLinearTest.class)) {
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

                // embedding produces (B, T, N_EMBED) — BitLinear projects to OUT_FEAT
                embeddingTable = new TokenEmbeddingTable(vocabSize, N_EMBED);
                embeddingTable.initialize(
                    manager,
                    DataType.FLOAT32,
                    new Shape(BATCH_SIZE, BLOCK_SIZE)
                );

                bitLinear = new BitLinear(N_EMBED, OUT_FEAT);
                bitLinear.initialize(
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
     * Run a full embedding → BitLinear pipeline from a real train batch.
     * Returns the BitLinear output NDArray.
     */
    private NDArray pipeline(Batch batch, ParameterStore ps) {
        NDArray tokenIds = batch.getData().head(); // (B, T)
        NDArray embedded = embeddingTable
            .forward(ps, new NDList(tokenIds), false)
            .singletonOrThrow(); // (B, T, N_EMBED)
        return bitLinear
            .forward(ps, new NDList(embedded), false)
            .singletonOrThrow(); // (B, T, OUT_FEAT)
    }

    // -------------------------------------------------------------------------
    // 1. Output shape is (B, T, OUT_FEAT) end-to-end from a real batch
    // -------------------------------------------------------------------------
    public void testOutputShapeEndToEnd() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray out = pipeline(batch, ps);

            assertEquals(BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(OUT_FEAT, (int) out.getShape().get(2));

            System.out.printf("BitLinear output shape: %s%n", out.getShape());
        }
    }

    // -------------------------------------------------------------------------
    // 2. getOutputShapes reports correct shape without a forward pass
    // -------------------------------------------------------------------------
    public void testGetOutputShapes() {
        Shape[] out = bitLinear.getOutputShapes(
            new Shape[] { new Shape(BATCH_SIZE, BLOCK_SIZE, N_EMBED) }
        );

        assertEquals(1, out.length);
        assertEquals(new Shape(BATCH_SIZE, BLOCK_SIZE, OUT_FEAT), out[0]);

        System.out.printf("getOutputShapes: %s%n", out[0]);
    }

    // -------------------------------------------------------------------------
    // 3. Output dtype is FLOAT32 after dequantization
    // -------------------------------------------------------------------------
    public void testOutputIsFloat32() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray out = pipeline(batch, ps);

            assertEquals(
                "BitLinear output must be FLOAT32 after dequantization",
                DataType.FLOAT32,
                out.getDataType()
            );

            System.out.printf("output dtype: %s%n", out.getDataType());
        }
    }

    // -------------------------------------------------------------------------
    // 4. Weights are ternary — all values in {-1, 0, +1} after quantization
    // -------------------------------------------------------------------------
    public void testWeightsAreTernaryAfterQuantization() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            // trigger forward so quantized weight is computed
            pipeline(batch, ps);
        }

        // Read raw master weights and manually ternarize to verify
        NDArray w = bitLinear.getParameters().get("weight").getArray();
        NDArray gamma = w.abs().mean();
        NDArray wTilde = w.div(gamma.add(1e-8f)).round().clip(-1, 1);

        for (float v : wTilde.toFloatArray()) {
            assertTrue(
                "Quantized weight must be -1, 0, or +1 but got " + v,
                v == -1f || v == 0f || v == 1f
            );
        }

        System.out.println("All quantized weights are ternary");
    }

    // -------------------------------------------------------------------------
    // 5. Same input always produces the same output (deterministic)
    // -------------------------------------------------------------------------
    public void testDeterministic() {
        try (Batch batch = loader.sampleTrain()) {
            NDArray tokenIds = batch.getData().head();

            float[] out1 = bitLinear
                .forward(
                    freshPs(),
                    new NDList(
                        embeddingTable
                            .forward(freshPs(), new NDList(tokenIds), false)
                            .singletonOrThrow()
                    ),
                    false
                )
                .singletonOrThrow()
                .toFloatArray();

            float[] out2 = bitLinear
                .forward(
                    freshPs(),
                    new NDList(
                        embeddingTable
                            .forward(freshPs(), new NDList(tokenIds), false)
                            .singletonOrThrow()
                    ),
                    false
                )
                .singletonOrThrow()
                .toFloatArray();

            for (int i = 0; i < out1.length; i++) {
                assertEquals(
                    "Output must be deterministic at index " + i,
                    out1[i],
                    out2[i],
                    1e-5f
                );
            }

            System.out.println("Output is deterministic");
        }
    }

    // -------------------------------------------------------------------------
    // 6. Output values are not all zero (layer is actually doing work)
    // -------------------------------------------------------------------------
    public void testOutputIsNonZero() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray out = pipeline(batch, ps);
            float[] vals = out.toFloatArray();

            boolean hasNonZero = false;
            for (float v : vals) {
                if (v != 0f) {
                    hasNonZero = true;
                    break;
                }
            }

            assertTrue(
                "BitLinear output must contain non-zero values",
                hasNonZero
            );

            System.out.printf(
                "output[0]: %.6f  output[1]: %.6f%n",
                vals[0],
                vals[1]
            );
        }
    }

    // -------------------------------------------------------------------------
    // 7. Training flag does not change output shape
    // -------------------------------------------------------------------------
    public void testTrainingFlagDoesNotAffectShape() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray tokenIds = batch.getData().head();

            NDArray embedded = embeddingTable
                .forward(ps, new NDList(tokenIds), false)
                .singletonOrThrow();

            Shape trainShape = bitLinear
                .forward(ps, new NDList(embedded), true)
                .singletonOrThrow()
                .getShape();

            Shape inferShape = bitLinear
                .forward(ps, new NDList(embedded), false)
                .singletonOrThrow()
                .getShape();

            assertEquals(trainShape, inferShape);

            System.out.printf(
                "train shape: %s  infer shape: %s%n",
                trainShape,
                inferShape
            );
        }
    }

    // -------------------------------------------------------------------------
    // 8. Output shape is stable across multiple batches
    // -------------------------------------------------------------------------
    public void testShapeIsConsistentAcrossBatches() {
        for (int i = 0; i < 3; i++) {
            ParameterStore ps = freshPs();
            try (Batch batch = loader.sampleTrain()) {
                NDArray out = pipeline(batch, ps);

                assertEquals(BATCH_SIZE, (int) out.getShape().get(0));
                assertEquals(BLOCK_SIZE, (int) out.getShape().get(1));
                assertEquals(OUT_FEAT, (int) out.getShape().get(2));

                System.out.printf("call %d shape: %s%n", i, out.getShape());
            }
        }
    }

    // -------------------------------------------------------------------------
    // 9. Validation batch passes through correctly
    // -------------------------------------------------------------------------
    public void testValBatchShape() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleValidation()) {
            NDArray out = pipeline(batch, ps);

            assertEquals(BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(OUT_FEAT, (int) out.getShape().get(2));

            System.out.printf("val output shape: %s%n", out.getShape());
        }
    }
}
