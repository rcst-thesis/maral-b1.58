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

public class BitDecoderBlockTest extends TestCase {

    private static BitDecoderBlock block;
    private static Shape TGT_SHAPE; // (B, T, dModel)
    private static Shape SRC_SHAPE; // (B, S, dModel)  S != T

    public static Test suite() {
        return new TestSetup(new TestSuite(BitDecoderBlockTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                ModelConfig cfg = ModelConfig.get();
                TGT_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    TestFixture.BLOCK_SIZE,
                    TestFixture.D_MODEL
                );
                int SRC_LEN = Math.max(TestFixture.BLOCK_SIZE + 2, 4);
                SRC_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    SRC_LEN,
                    TestFixture.D_MODEL
                );
                block = new BitDecoderBlock(
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

    public void testOutputShapeMatchesTarget() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray out = block
            .forward(ps, new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)), false)
            .singletonOrThrow();
        assertEquals(TGT_SHAPE, out.getShape());
        System.out.printf("BitDecoderBlock output: %s%n", out.getShape());
    }

    public void testGetOutputShapes() {
        Shape[] out = block.getOutputShapes(
            new Shape[] { TGT_SHAPE, SRC_SHAPE }
        );
        assertEquals(1, out.length);
        assertEquals(TGT_SHAPE, out[0]);
    }

    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        assertEquals(
            DataType.FLOAT32,
            block
                .forward(
                    ps,
                    new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)),
                    false
                )
                .singletonOrThrow()
                .getDataType()
        );
    }

    public void testOutputIsNonZero() {
        ParameterStore ps = TestFixture.freshPs();
        boolean hasNonZero = false;
        for (float v : block
            .forward(ps, new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)), false)
            .singletonOrThrow()
            .toFloatArray()) {
            if (v != 0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("decoder block output must not be all zeros", hasNonZero);
    }

    public void testResidualChangesOutput() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray x = rand(TGT_SHAPE);
        NDArray out = block
            .forward(ps, new NDList(x, rand(SRC_SHAPE)), false)
            .singletonOrThrow();

        boolean differs = false;
        float[] inArr = x.toFloatArray(),
            outArr = out.toFloatArray();
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
     * Different encoder memory must produce different decoder output —
     * verifies the cross-attention sublayer is actually consuming memory.
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
        assertTrue("different memory must produce different output", differs);
        System.out.println("cross-attention consumes memory ✓");
    }

    /**
     * All-zero padding masks (no padding) must have no effect on output.
     */
    public void testAllZeroPaddingMasksHaveNoEffect() {
        NDArray x = rand(TGT_SHAPE);
        NDArray memory = rand(SRC_SHAPE);
        NDArray tgtMask = zeros(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE)
        );
        NDArray srcMask = zeros(
            new Shape(TestFixture.BATCH_SIZE, SRC_SHAPE.get(1))
        );

        float[] masked = block
            .forward(
                TestFixture.freshPs(),
                new NDList(x, memory, tgtMask, srcMask),
                false
            )
            .singletonOrThrow()
            .toFloatArray();
        float[] plain = block
            .forward(TestFixture.freshPs(), new NDList(x, memory), false)
            .singletonOrThrow()
            .toFloatArray();

        for (int i = 0; i < masked.length; i++) {
            assertEquals(
                "zero masks must not affect output at " + i,
                plain[i],
                masked[i],
                1e-4f
            );
        }
        System.out.println("zero padding masks have no effect ✓");
    }

    public void testTrainingFlagDoesNotChangeShape() {
        NDArray x = rand(TGT_SHAPE),
            m = rand(SRC_SHAPE);
        Shape train = block
            .forward(TestFixture.freshPs(), new NDList(x, m), true)
            .singletonOrThrow()
            .getShape();
        Shape infer = block
            .forward(TestFixture.freshPs(), new NDList(x, m), false)
            .singletonOrThrow()
            .getShape();
        assertEquals(train, infer);
    }

    public void testToString() {
        String s = block.toString();
        assertTrue(s.contains("BitDecoderBlock"));
        System.out.println(s);
    }
}
