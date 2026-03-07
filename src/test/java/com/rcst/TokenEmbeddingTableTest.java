package com.rcst;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class TokenEmbeddingTableTest extends TestCase {

    public static Test suite() {
        return new TestSetup(new TestSuite(TokenEmbeddingTableTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    public void testOutputShape() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.loader.sampleTrain()) {
            NDArray out = TestFixture.embed(batch, ps);
            assertEquals(TestFixture.BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(TestFixture.D_MODEL, (int) out.getShape().get(2));
            System.out.printf("embedded shape: %s%n", out.getShape());
        }
    }

    public void testGetOutputShapes() {
        Shape[] out = TestFixture.embeddingTable.getOutputShapes(
            new Shape[] {
                new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE),
            }
        );
        assertEquals(1, out.length);
        assertEquals(
            new Shape(
                TestFixture.BATCH_SIZE,
                TestFixture.BLOCK_SIZE,
                TestFixture.D_MODEL
            ),
            out[0]
        );
    }

    public void testOutputIsFullPrecision() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.loader.sampleTrain()) {
            assertEquals(
                DataType.FLOAT32,
                TestFixture.embed(batch, ps).getDataType()
            );
        }
    }

    public void testSameTokenReturnsSameEmbedding() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.loader.sampleTrain()) {
            long id = batch.getData().head().getLong(0, 0);

            NDArray row1 = TestFixture.manager.create(
                new long[] { id },
                new Shape(1, 1)
            );
            NDArray row2 = TestFixture.manager.create(
                new long[] { id },
                new Shape(1, 1)
            );

            float[] v1 = TestFixture.embeddingTable
                .forward(ps, new NDList(row1), false)
                .singletonOrThrow()
                .reshape(TestFixture.D_MODEL)
                .toFloatArray();
            float[] v2 = TestFixture.embeddingTable
                .forward(ps, new NDList(row2), false)
                .singletonOrThrow()
                .reshape(TestFixture.D_MODEL)
                .toFloatArray();

            for (int i = 0; i < TestFixture.D_MODEL; i++) {
                assertEquals(
                    "same token → same embedding at " + i,
                    v1[i],
                    v2[i],
                    0f
                );
            }
            System.out.printf("token %d → consistent embedding ✓%n", id);
        }
    }

    public void testXAndYEmbeddingsDiffer() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.loader.sampleTrain()) {
            float[] xVals = TestFixture.embeddingTable
                .forward(ps, new NDList(batch.getData().head()), false)
                .singletonOrThrow()
                .toFloatArray();
            float[] yVals = TestFixture.embeddingTable
                .forward(ps, new NDList(batch.getLabels().head()), false)
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
                "x and y embeddings should differ (shifted by 1)",
                differs
            );
        }
    }

    public void testShapeIsConsistentAcrossBatches() {
        for (int i = 0; i < 3; i++) {
            ParameterStore ps = TestFixture.freshPs();
            try (Batch batch = TestFixture.loader.sampleTrain()) {
                NDArray out = TestFixture.embed(batch, ps);
                assertEquals(
                    TestFixture.BATCH_SIZE,
                    (int) out.getShape().get(0)
                );
                assertEquals(
                    TestFixture.BLOCK_SIZE,
                    (int) out.getShape().get(1)
                );
                assertEquals(TestFixture.D_MODEL, (int) out.getShape().get(2));
            }
        }
    }

    public void testTrainingFlagDoesNotChangeShape() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.loader.sampleTrain()) {
            NDArray x = batch.getData().head();
            Shape train = TestFixture.embeddingTable
                .forward(ps, new NDList(x), true)
                .singletonOrThrow()
                .getShape();
            Shape infer = TestFixture.embeddingTable
                .forward(ps, new NDList(x), false)
                .singletonOrThrow()
                .getShape();
            assertEquals(train, infer);
        }
    }

    public void testValBatchShape() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.loader.sampleValidation()) {
            NDArray out = TestFixture.embed(batch, ps);
            assertEquals(TestFixture.BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(TestFixture.D_MODEL, (int) out.getShape().get(2));
        }
    }
}
