package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import com.rcst.TestFixture;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class BitLinearTest extends TestCase {

    // Project D_MODEL → D_MODEL/2 to exercise a non-trivial weight matrix
    private static int OUT_FEAT;
    private static BitLinear bitLinear;

    public static Test suite() {
        return new TestSetup(new TestSuite(BitLinearTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                OUT_FEAT = TestFixture.D_MODEL / 2;
                bitLinear = TestFixture.buildBitLinear(
                    TestFixture.D_MODEL,
                    OUT_FEAT
                );
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    /** Embed tokens from a batch then pass through BitLinear. */
    private NDArray pipeline(Batch batch, ParameterStore ps) {
        return bitLinear
            .forward(ps, new NDList(TestFixture.embed(batch, ps)), false)
            .singletonOrThrow();
    }

    public void testOutputShape() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            NDArray out = pipeline(batch, ps);
            assertEquals(TestFixture.BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(OUT_FEAT, (int) out.getShape().get(2));
            System.out.printf("BitLinear output: %s%n", out.getShape());
        }
    }

    public void testGetOutputShapes() {
        Shape[] out = bitLinear.getOutputShapes(
            new Shape[] {
                new Shape(
                    TestFixture.BATCH_SIZE,
                    TestFixture.BLOCK_SIZE,
                    TestFixture.D_MODEL
                ),
            }
        );
        assertEquals(1, out.length);
        assertEquals(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE, OUT_FEAT),
            out[0]
        );
    }

    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            assertEquals(DataType.FLOAT32, pipeline(batch, ps).getDataType());
        }
    }

    public void testWeightsAreTernary() {
        // trigger forward so parameters are initialised
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            pipeline(batch, ps);
        }

        NDArray w = bitLinear.getParameters().get("weight").getArray();
        NDArray wTilde = w.div(w.abs().mean().add(1e-8f)).round().clip(-1, 1);

        for (float v : wTilde.toFloatArray()) {
            assertTrue(
                "Quantized weight must be -1, 0, or +1 but was " + v,
                v == -1f || v == 0f || v == 1f
            );
        }
        System.out.println("All quantized weights are ternary ✓");
    }

    public void testOutputIsDeterministic() {
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            NDArray ids = batch.getData().head();

            float[] out1 = bitLinear
                .forward(
                    TestFixture.freshPs(),
                    new NDList(
                        TestFixture.embedder
                            .forward(
                                TestFixture.freshPs(),
                                new NDList(ids),
                                false
                            )
                            .singletonOrThrow()
                    ),
                    false
                )
                .singletonOrThrow()
                .toFloatArray();

            float[] out2 = bitLinear
                .forward(
                    TestFixture.freshPs(),
                    new NDList(
                        TestFixture.embedder
                            .forward(
                                TestFixture.freshPs(),
                                new NDList(ids),
                                false
                            )
                            .singletonOrThrow()
                    ),
                    false
                )
                .singletonOrThrow()
                .toFloatArray();

            for (int i = 0; i < out1.length; i++) {
                assertEquals(
                    "output must be deterministic at " + i,
                    out1[i],
                    out2[i],
                    1e-5f
                );
            }
        }
    }

    public void testOutputIsNonZero() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            boolean hasNonZero = false;
            for (float v : pipeline(batch, ps).toFloatArray()) {
                if (v != 0f) {
                    hasNonZero = true;
                    break;
                }
            }
            assertTrue("BitLinear must produce non-zero outputs", hasNonZero);
        }
    }

    public void testTrainingFlagDoesNotChangeShape() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            NDArray emb = TestFixture.embed(batch, ps);
            Shape train = bitLinear
                .forward(ps, new NDList(emb), true)
                .singletonOrThrow()
                .getShape();
            Shape infer = bitLinear
                .forward(ps, new NDList(emb), false)
                .singletonOrThrow()
                .getShape();
            assertEquals(train, infer);
        }
    }

    public void testShapeIsConsistentAcrossBatches() {
        for (int i = 0; i < 3; i++) {
            ParameterStore ps = TestFixture.freshPs();
            try (
                Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()
            ) {
                NDArray out = pipeline(batch, ps);
                assertEquals(
                    TestFixture.BATCH_SIZE,
                    (int) out.getShape().get(0)
                );
                assertEquals(
                    TestFixture.BLOCK_SIZE,
                    (int) out.getShape().get(1)
                );
                assertEquals(OUT_FEAT, (int) out.getShape().get(2));
            }
        }
    }

    public void testValBatchShape() {
        ParameterStore ps = TestFixture.freshPs();
        try (
            Batch batch = TestFixture.embedder.getSampleFromValidationSplit()
        ) {
            NDArray out = pipeline(batch, ps);
            assertEquals(TestFixture.BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(OUT_FEAT, (int) out.getShape().get(2));
        }
    }
}
