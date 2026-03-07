package com.rcst;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import java.util.List;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class TokenEmbeddingTableTest extends TestCase {

    private static final int BLOCK_SIZE = 8;
    private static final int BATCH_SIZE = 4;
    private static final int N_EMBED = 32;
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

    // Shared across all tests — initialised once, torn down once
    private static NDManager manager;
    private static Tokenizer tokenizer;
    private static TrainingDataLoader loader;
    private static TokenEmbeddingTable embeddingTable;

    public static Test suite() {
        return new TestSetup(new TestSuite(TokenEmbeddingTableTest.class)) {
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
            }

            @Override
            protected void tearDown() throws Exception {
                tokenizer.close();
                manager.close();
            }
        };
    }

    private ParameterStore freshPs() {
        return new ParameterStore(manager, false);
    }

    // Output shape is (B, T, n_embed) given a real batch from the loader
    public void testOutputShapeFromRealBatch() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray x = batch.getData().head();
            NDArray embedded = embeddingTable
                .forward(ps, new NDList(x), false)
                .singletonOrThrow();

            assertEquals(BATCH_SIZE, (int) embedded.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) embedded.getShape().get(1));
            assertEquals(N_EMBED, (int) embedded.getShape().get(2));

            System.out.printf("embedded shape: %s%n", embedded.getShape());
        }
    }

    // getOutputShapes returns correct shape without a forward pass
    public void testGetOutputShapes() {
        Shape[] out = embeddingTable.getOutputShapes(
            new Shape[] { new Shape(BATCH_SIZE, BLOCK_SIZE) }
        );

        assertEquals(1, out.length);
        assertEquals(new Shape(BATCH_SIZE, BLOCK_SIZE, N_EMBED), out[0]);

        System.out.printf("getOutputShapes: %s%n", out[0]);
    }

    // Output dtype is FLOAT32 — embedding stays full precision, not quantized
    public void testOutputIsFullPrecision() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray x = batch.getData().head();
            NDArray embedded = embeddingTable
                .forward(ps, new NDList(x), false)
                .singletonOrThrow();

            assertEquals(
                "Embedding output must be FLOAT32 (not quantized)",
                DataType.FLOAT32,
                embedded.getDataType()
            );

            System.out.printf("output dtype: %s%n", embedded.getDataType());
        }
    }

    // Same token id always returns the same embedding row
    public void testSameTokenReturnsSameEmbedding() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            long tokenId = batch.getData().head().getLong(0, 0);

            NDArray ids1 = manager.create(
                new long[] { tokenId },
                new Shape(1, 1)
            );
            NDArray ids2 = manager.create(
                new long[] { tokenId },
                new Shape(1, 1)
            );

            float[] v1 = embeddingTable
                .forward(ps, new NDList(ids1), false)
                .singletonOrThrow()
                .reshape(N_EMBED)
                .toFloatArray();

            float[] v2 = embeddingTable
                .forward(ps, new NDList(ids2), false)
                .singletonOrThrow()
                .reshape(N_EMBED)
                .toFloatArray();

            for (int i = 0; i < N_EMBED; i++) {
                assertEquals(
                    "Same token must always return the same embedding at index " +
                        i,
                    v1[i],
                    v2[i],
                    0f
                );
            }

            System.out.printf("token %d → consistent embedding ✓%n", tokenId);
        }
    }

    // x and y (shifted by 1) produce different embedding outputs
    public void testXAndYEmbeddingsDiffer() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray xBatch = batch.getData().head();
            NDArray yBatch = batch.getLabels().head();

            float[] xVals = embeddingTable
                .forward(ps, new NDList(xBatch), false)
                .singletonOrThrow()
                .toFloatArray();

            float[] yVals = embeddingTable
                .forward(ps, new NDList(yBatch), false)
                .singletonOrThrow()
                .toFloatArray();

            boolean differs = false;
            for (int i = 0; i < xVals.length; i++) {
                if (xVals[i] != yVals[i]) {
                    differs = true;
                    break;
                }
            }

            assertTrue(
                "x and y (shifted by 1) should produce different embeddings",
                differs
            );

            System.out.printf(
                "x-embed[0]: %.4f  y-embed[0]: %.4f%n",
                xVals[0],
                yVals[0]
            );
        }
    }

    // Output shape is stable across multiple sample() calls
    public void testShapeIsConsistentAcrossBatches() {
        for (int i = 0; i < 3; i++) {
            ParameterStore ps = freshPs();
            try (Batch batch = loader.sampleTrain()) {
                NDArray x = batch.getData().head();
                NDArray embedded = embeddingTable
                    .forward(ps, new NDList(x), false)
                    .singletonOrThrow();

                assertEquals(BATCH_SIZE, (int) embedded.getShape().get(0));
                assertEquals(BLOCK_SIZE, (int) embedded.getShape().get(1));
                assertEquals(N_EMBED, (int) embedded.getShape().get(2));

                System.out.printf(
                    "call %d shape: %s%n",
                    i,
                    embedded.getShape()
                );
            }
        }
    }

    // Training flag does not change output shape
    public void testTrainingFlagDoesNotAffectShape() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleTrain()) {
            NDArray x = batch.getData().head();

            Shape trainShape = embeddingTable
                .forward(ps, new NDList(x), true)
                .singletonOrThrow()
                .getShape();

            Shape inferShape = embeddingTable
                .forward(ps, new NDList(x), false)
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

    // Validation batch also embeds with the correct shape
    public void testValBatchEmbeddingShape() {
        ParameterStore ps = freshPs();
        try (Batch batch = loader.sampleValidation()) {
            NDArray x = batch.getData().head();
            NDArray embedded = embeddingTable
                .forward(ps, new NDList(x), false)
                .singletonOrThrow();

            assertEquals(BATCH_SIZE, (int) embedded.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) embedded.getShape().get(1));
            assertEquals(N_EMBED, (int) embedded.getShape().get(2));

            System.out.printf("val embedded shape: %s%n", embedded.getShape());
        }
    }
}
