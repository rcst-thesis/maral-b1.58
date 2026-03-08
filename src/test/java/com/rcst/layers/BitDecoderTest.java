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

public class BitDecoderTest extends TestCase {

    private static BitDecoder decoder;
    private static Shape TGT_SHAPE; // (B, T, dModel)
    private static Shape SRC_SHAPE; // (B, S, dModel)  S != T

    public static Test suite() {
        return new TestSetup(new TestSuite(BitDecoderTest.class)) {
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
                decoder = new BitDecoder(
                    cfg.nDecoderLayers,
                    cfg.dModel,
                    cfg.nHeads,
                    cfg.dFfn,
                    cfg.ropeBase,
                    cfg.maxSeqLen,
                    cfg.eps,
                    cfg.quantEps
                );
                decoder.initialize(
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
        NDArray out = decoder
            .forward(ps, new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)), false)
            .singletonOrThrow();
        assertEquals(TGT_SHAPE, out.getShape());
        System.out.printf("BitDecoder output: %s%n", out.getShape());
    }

    public void testGetOutputShapes() {
        Shape[] out = decoder.getOutputShapes(
            new Shape[] { TGT_SHAPE, SRC_SHAPE }
        );
        assertEquals(1, out.length);
        assertEquals(TGT_SHAPE, out[0]);
    }

    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        assertEquals(
            DataType.FLOAT32,
            decoder
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
        for (float v : decoder
            .forward(ps, new NDList(rand(TGT_SHAPE), rand(SRC_SHAPE)), false)
            .singletonOrThrow()
            .toFloatArray()) {
            if (v != 0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("decoder output must not be all zeros", hasNonZero);
    }

    /**
     * Different encoder memory must produce different decoder output —
     * confirms cross-attention is active through the full stack.
     */
    public void testDifferentMemoryProducesDifferentOutput() {
        NDArray x = rand(TGT_SHAPE);
        float[] o1 = decoder
            .forward(
                TestFixture.freshPs(),
                new NDList(x, rand(SRC_SHAPE)),
                false
            )
            .singletonOrThrow()
            .toFloatArray();
        float[] o2 = decoder
            .forward(
                TestFixture.freshPs(),
                new NDList(x, rand(SRC_SHAPE)),
                false
            )
            .singletonOrThrow()
            .toFloatArray();

        boolean differs = false;
        for (int i = 0; i < o1.length; i++) {
            if (Math.abs(o1[i] - o2[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        assertTrue("different memory must produce different output", differs);
        System.out.println("cross-attention active through decoder stack ✓");
    }

    /**
     * All-zero padding masks must not change the output.
     */
    public void testAllZeroPaddingMasksHaveNoEffect() {
        NDArray x = rand(TGT_SHAPE);
        NDArray memory = rand(SRC_SHAPE);
        NDArray tgtMask = zeros(
            new Shape(TestFixture.BATCH_SIZE, TGT_SHAPE.get(1))
        );
        NDArray srcMask = zeros(
            new Shape(TestFixture.BATCH_SIZE, SRC_SHAPE.get(1))
        );

        float[] plain = decoder
            .forward(TestFixture.freshPs(), new NDList(x, memory), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] masked = decoder
            .forward(
                TestFixture.freshPs(),
                new NDList(x, memory, tgtMask, srcMask),
                false
            )
            .singletonOrThrow()
            .toFloatArray();

        for (int i = 0; i < plain.length; i++) {
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
        Shape train = decoder
            .forward(TestFixture.freshPs(), new NDList(x, m), true)
            .singletonOrThrow()
            .getShape();
        Shape infer = decoder
            .forward(TestFixture.freshPs(), new NDList(x, m), false)
            .singletonOrThrow()
            .getShape();
        assertEquals(train, infer);
    }

    public void testToString() {
        String s = decoder.toString();
        assertTrue(s.contains("BitDecoder"));
        System.out.println(s);
    }
}
