package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.rcst.ModelConfig;
import com.rcst.TestFixture;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class BitEncoderBlockTest extends TestCase {

    private static BitEncoderBlock block;
    private static Shape SEQ_SHAPE; // (B, T, dModel)

    public static Test suite() {
        return new TestSetup(new TestSuite(BitEncoderBlockTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                ModelConfig cfg = ModelConfig.get();
                SEQ_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    TestFixture.BLOCK_SIZE,
                    TestFixture.D_MODEL
                );
                block = new BitEncoderBlock(
                    cfg.dModel,
                    cfg.nHeads,
                    cfg.dFfn,
                    cfg.ropeBase,
                    cfg.maxSeqLen,
                    cfg.eps,
                    cfg.quantEps
                );
                block.initialize(
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
        NDArray out = block
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow();
        assertEquals(SEQ_SHAPE, out.getShape());
        System.out.printf("BitEncoderBlock output: %s%n", out.getShape());
    }

    public void testGetOutputShapes() {
        Shape[] out = block.getOutputShapes(new Shape[] { SEQ_SHAPE });
        assertEquals(1, out.length);
        assertEquals(SEQ_SHAPE, out[0]);
    }

    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        assertEquals(
            DataType.FLOAT32,
            block
                .forward(ps, new NDList(rand()), false)
                .singletonOrThrow()
                .getDataType()
        );
    }

    public void testOutputIsNonZero() {
        ParameterStore ps = TestFixture.freshPs();
        boolean hasNonZero = false;
        for (float v : block
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow()
            .toFloatArray()) {
            if (v != 0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("encoder block output must not be all zeros", hasNonZero);
    }

    /**
     * The residual connection means output and input must differ
     * (the sublayer adds something non-trivial to x).
     */
    public void testResidualChangesOutput() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray x = rand();
        NDArray out = block
            .forward(ps, new NDList(x), false)
            .singletonOrThrow();

        float[] inArr = x.toFloatArray();
        float[] outArr = out.toFloatArray();
        boolean differs = false;
        for (int i = 0; i < inArr.length; i++) {
            if (Math.abs(inArr[i] - outArr[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        assertTrue("residual must modify the input", differs);
        System.out.println("residual connection changes output ✓");
    }

    /**
     * With a key-padding mask of all zeros (no padding), output must equal
     * the no-mask output — the mask should have no effect when inactive.
     */
    public void testAllZeroPaddingMaskHasNoEffect() {
        NDArray x = rand();
        NDArray mask = TestFixture.manager.zeros(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE),
            DataType.FLOAT32
        );

        float[] withMask = block
            .forward(TestFixture.freshPs(), new NDList(x, mask), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] noMask = block
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .toFloatArray();

        for (int i = 0; i < withMask.length; i++) {
            assertEquals(
                "zero mask must not affect output at " + i,
                noMask[i],
                withMask[i],
                1e-4f
            );
        }
        System.out.println("zero padding mask has no effect ✓");
    }

    public void testTrainingFlagDoesNotChangeShape() {
        NDArray x = rand();
        Shape train = block
            .forward(TestFixture.freshPs(), new NDList(x), true)
            .singletonOrThrow()
            .getShape();
        Shape infer = block
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .getShape();
        assertEquals(train, infer);
    }

    public void testToString() {
        String s = block.toString();
        assertTrue(s.contains("BitEncoderBlock"));
        System.out.println(s);
    }
}
