package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.rcst.TestFixture;
import com.rcst.utils.ModelConfig;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class DecoderBlockTest extends TestCase {

    private static DecoderBlock block;
    private static Shape TGT_SHAPE;
    private static Shape SRC_SHAPE; // S intentionally != T to catch shape confusion

    public static Test suite() {
        return new TestSetup(new TestSuite(DecoderBlockTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                ModelConfig cfg = ModelConfig.get();

                TGT_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    TestFixture.BLOCK_SIZE,
                    TestFixture.D_MODEL
                );
                // Source length is deliberately different from target length
                int srcLen = Math.max(TestFixture.BLOCK_SIZE + 2, 4);
                SRC_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    srcLen,
                    TestFixture.D_MODEL
                );

                block = new DecoderBlock(
                    cfg.dModel,
                    cfg.nHeads,
                    cfg.dFfn,
                    cfg.ropeBase,
                    cfg.maxSeqLen,
                    cfg.quantEps
                );
                block.initialize(
                    TestFixture.manager,
                    DataType.FLOAT32,
                    TGT_SHAPE,
                    SRC_SHAPE
                );
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    private NDArray rand(Shape shape) {
        return TestFixture.manager.randomNormal(shape, DataType.FLOAT32);
    }

    private NDArray zeros(Shape shape) {
        return TestFixture.manager.zeros(shape, DataType.FLOAT32);
    }

    private NDArray ones(Shape shape) {
        return TestFixture.manager.ones(shape, DataType.FLOAT32);
    }

    public void testOutputShapeMatchesTarget() {
        NDArray out = block
            .forward(
                TestFixture.freshPs(),
                new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)),
                false
            )
            .singletonOrThrow();
        assertEquals(TGT_SHAPE, out.getShape());
        System.out.printf("DecoderBlock output: %s%n", out.getShape());
    }

    public void testGetOutputShapes() {
        Shape[] out = block.getOutputShapes(
            new Shape[] { TGT_SHAPE, SRC_SHAPE }
        );
        assertEquals(1, out.length);
        assertEquals(TGT_SHAPE, out[0]);
    }

    public void testOutputIsFloat32() {
        NDArray out = block
            .forward(
                TestFixture.freshPs(),
                new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)),
                false
            )
            .singletonOrThrow();
        assertEquals(DataType.FLOAT32, out.getDataType());
    }

    public void testOutputIsNonZero() {
        float[] values = block
            .forward(
                TestFixture.freshPs(),
                new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)),
                false
            )
            .singletonOrThrow()
            .toFloatArray();

        boolean hasNonZero = false;
        for (float v : values) {
            if (v != 0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("decoder block output must not be all zeros", hasNonZero);
    }

    public void testResidualChangesOutput() {
        NDArray x = rand(TGT_SHAPE);
        NDArray out = block
            .forward(
                TestFixture.freshPs(),
                new NDList(x, rand(SRC_SHAPE)),
                false
            )
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
        assertTrue("residual connections must modify the input", differs);
        System.out.println("residual modifies input ✓");
    }

    public void testDifferentInputsProduceDifferentOutputs() {
        float[] a = block
            .forward(
                TestFixture.freshPs(),
                new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)),
                false
            )
            .singletonOrThrow()
            .toFloatArray();
        float[] b = block
            .forward(
                TestFixture.freshPs(),
                new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)),
                false
            )
            .singletonOrThrow()
            .toFloatArray();

        boolean differs = false;
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        assertTrue("different inputs must produce different outputs", differs);
    }

    /*
     * Fix the target input and vary only the encoder memory.
     * If cross-attention is correctly wired, the output must change.
     */
    public void testDifferentMemoryProducesDifferentOutput() {
        NDArray x = rand(TGT_SHAPE);
        NDArray m1 = rand(SRC_SHAPE);
        NDArray m2 = rand(SRC_SHAPE);

        float[] out1 = block
            .forward(TestFixture.freshPs(), new NDList(x, m1), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] out2 = block
            .forward(TestFixture.freshPs(), new NDList(x, m2), false)
            .singletonOrThrow()
            .toFloatArray();

        boolean differs = false;
        for (int i = 0; i < out1.length; i++) {
            if (Math.abs(out1[i] - out2[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        assertTrue(
            "different encoder memory must change decoder output",
            differs
        );
        System.out.println("cross-attention consumes memory ✓");
    }

    /*
     * Position 0 can only attend to itself; the last position can attend to
     * all prior tokens. With the same weights, those contexts differ, so the
     * output vectors at position 0 and position T-1 must differ.
     */
    public void testCausalMaskProducesPositionDependentOutput() {
        NDArray out = block
            .forward(
                TestFixture.freshPs(),
                new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)),
                false
            )
            .singletonOrThrow();

        float[] first = out.get("0, 0, :").toFloatArray();
        float[] last = out
            .get("0, " + (TestFixture.BLOCK_SIZE - 1) + ", :")
            .toFloatArray();

        boolean differs = false;
        for (int i = 0; i < first.length; i++) {
            if (Math.abs(first[i] - last[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        assertTrue(
            "causal mask must produce position-dependent outputs",
            differs
        );
        System.out.println("causal mask is position-sensitive ✓");
    }

    public void testAllZeroTgtMaskHasNoEffect() {
        NDArray x = rand(TGT_SHAPE);
        NDArray mem = rand(SRC_SHAPE);
        NDArray zero = zeros(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE)
        );

        float[] plain = block
            .forward(TestFixture.freshPs(), new NDList(x, mem), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] masked = block
            .forward(TestFixture.freshPs(), new NDList(x, mem, zero), false)
            .singletonOrThrow()
            .toFloatArray();

        for (int i = 0; i < plain.length; i++) {
            assertEquals(
                "zero tgt mask must not affect output at " + i,
                plain[i],
                masked[i],
                1e-4f
            );
        }
        System.out.println("zero tgt mask has no effect ✓");
    }

    public void testAllZeroSrcMaskHasNoEffect() {
        NDArray x = rand(TGT_SHAPE);
        NDArray mem = rand(SRC_SHAPE);
        NDArray zero = zeros(
            new Shape(TestFixture.BATCH_SIZE, SRC_SHAPE.get(1))
        );

        // Pass a zero tgt mask as a placeholder so srcMask lands at index 3
        NDArray zeroTgt = zeros(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE)
        );

        float[] plain = block
            .forward(TestFixture.freshPs(), new NDList(x, mem), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] masked = block
            .forward(
                TestFixture.freshPs(),
                new NDList(x, mem, zeroTgt, zero),
                false
            )
            .singletonOrThrow()
            .toFloatArray();

        for (int i = 0; i < plain.length; i++) {
            assertEquals(
                "zero src mask must not affect output at " + i,
                plain[i],
                masked[i],
                1e-4f
            );
        }
        System.out.println("zero src mask has no effect ✓");
    }

    public void testActiveSrcMaskChangesOutput() {
        NDArray x = rand(TGT_SHAPE);
        NDArray mem = rand(SRC_SHAPE);
        NDArray fullSrcMask = ones(
            new Shape(TestFixture.BATCH_SIZE, SRC_SHAPE.get(1))
        );
        NDArray zeroTgt = zeros(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE)
        );

        float[] plain = block
            .forward(TestFixture.freshPs(), new NDList(x, mem), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] masked = block
            .forward(
                TestFixture.freshPs(),
                new NDList(x, mem, zeroTgt, fullSrcMask),
                false
            )
            .singletonOrThrow()
            .toFloatArray();

        boolean differs = false;
        for (int i = 0; i < plain.length; i++) {
            if (Math.abs(plain[i] - masked[i]) > 1e-4f) {
                differs = true;
                break;
            }
        }
        assertTrue("fully-active src mask must change the output", differs);
        System.out.println("active src mask changes output ✓");
    }

    public void testTrainingFlagDoesNotChangeShape() {
        NDArray x = rand(TGT_SHAPE);
        NDArray mem = rand(SRC_SHAPE);

        Shape train = block
            .forward(TestFixture.freshPs(), new NDList(x, mem), true)
            .singletonOrThrow()
            .getShape();
        Shape infer = block
            .forward(TestFixture.freshPs(), new NDList(x, mem), false)
            .singletonOrThrow()
            .getShape();

        assertEquals(train, infer);
    }

    public void testToString() {
        String s = block.toString();
        assertTrue(s.contains("DecoderBlock"));
        assertTrue(s.contains("maskedSelfAttn"));
        assertTrue(s.contains("crossAttn"));
        assertTrue(s.contains("ffn"));
        System.out.println(s);
    }
}
