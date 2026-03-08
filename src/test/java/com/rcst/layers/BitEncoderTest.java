package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.rcst.TestFixture;
import com.rcst.utils.ModelConfig;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class BitEncoderTest extends TestCase {

    private static BitEncoder encoder;
    private static Shape SEQ_SHAPE; // (B, T, dModel)

    public static Test suite() {
        return new TestSetup(new TestSuite(BitEncoderTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                ModelConfig cfg = ModelConfig.get();
                SEQ_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    TestFixture.BLOCK_SIZE,
                    TestFixture.D_MODEL
                );
                encoder = new BitEncoder(
                    cfg.nEncoderLayers,
                    cfg.dModel,
                    cfg.nHeads,
                    cfg.dFfn,
                    cfg.ropeBase,
                    cfg.maxSeqLen,
                    cfg.eps,
                    cfg.quantEps
                );
                encoder.initialize(
                    TestFixture.manager,
                    DataType.FLOAT32,
                    SEQ_SHAPE
                );
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    private NDArray rand() {
        return TestFixture.manager.randomNormal(SEQ_SHAPE, DataType.FLOAT32);
    }

    public void testOutputShapeMatchesInput() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray out = encoder
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow();
        assertEquals(SEQ_SHAPE, out.getShape());
        System.out.printf("BitEncoder output: %s%n", out.getShape());
    }

    public void testGetOutputShapes() {
        Shape[] out = encoder.getOutputShapes(new Shape[] { SEQ_SHAPE });
        assertEquals(1, out.length);
        assertEquals(SEQ_SHAPE, out[0]);
    }

    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        assertEquals(
            DataType.FLOAT32,
            encoder
                .forward(ps, new NDList(rand()), false)
                .singletonOrThrow()
                .getDataType()
        );
    }

    public void testOutputIsNonZero() {
        ParameterStore ps = TestFixture.freshPs();
        boolean hasNonZero = false;
        for (float v : encoder
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow()
            .toFloatArray()) {
            if (v != 0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("encoder output must not be all zeros", hasNonZero);
    }

    /**
     * Two different inputs must produce different encoder outputs —
     * confirms the stack is not collapsed to a constant function.
     */
    public void testDifferentInputsProduceDifferentOutputs() {
        NDArray out1 = encoder
            .forward(TestFixture.freshPs(), new NDList(rand()), false)
            .singletonOrThrow();
        NDArray out2 = encoder
            .forward(TestFixture.freshPs(), new NDList(rand()), false)
            .singletonOrThrow();

        boolean differs = false;
        float[] a = out1.toFloatArray(),
            b = out2.toFloatArray();
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        assertTrue("different inputs must produce different outputs", differs);
        System.out.println("encoder is input-sensitive ✓");
    }

    /**
     * An all-zero padding mask must not change the output.
     */
    public void testAllZeroPaddingMaskHasNoEffect() {
        NDArray x = rand();
        NDArray mask = TestFixture.manager.zeros(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE),
            DataType.FLOAT32
        );

        float[] plain = encoder
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] masked = encoder
            .forward(TestFixture.freshPs(), new NDList(x, mask), false)
            .singletonOrThrow()
            .toFloatArray();

        for (int i = 0; i < plain.length; i++) {
            assertEquals(
                "zero mask must not affect output at " + i,
                plain[i],
                masked[i],
                1e-4f
            );
        }
        System.out.println("zero padding mask has no effect ✓");
    }

    public void testTrainingFlagDoesNotChangeShape() {
        NDArray x = rand();
        Shape train = encoder
            .forward(TestFixture.freshPs(), new NDList(x), true)
            .singletonOrThrow()
            .getShape();
        Shape infer = encoder
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .getShape();
        assertEquals(train, infer);
    }

    public void testToString() {
        String s = encoder.toString();
        assertTrue(s.contains("BitEncoder"));
        System.out.println(s);
    }
}
