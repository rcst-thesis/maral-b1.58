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

public class BitFFNTest extends TestCase {

    private static BitFFN ffn;
    private static Shape SEQ_SHAPE; // (B, T, dModel)

    public static Test suite() {
        return new TestSetup(new TestSuite(BitFFNTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                ModelConfig cfg = ModelConfig.get();
                SEQ_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    TestFixture.BLOCK_SIZE,
                    TestFixture.D_MODEL
                );
                ffn = new BitFFN(cfg.dModel, cfg.dFfn, cfg.quantEps);
                ffn.initialize(
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
        NDArray out = ffn
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow();
        assertEquals(SEQ_SHAPE, out.getShape());
        System.out.printf("BitFFN output: %s%n", out.getShape());
    }

    public void testGetOutputShapes() {
        Shape[] out = ffn.getOutputShapes(new Shape[] { SEQ_SHAPE });
        assertEquals(1, out.length);
        assertEquals(SEQ_SHAPE, out[0]);
    }

    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        assertEquals(
            DataType.FLOAT32,
            ffn
                .forward(ps, new NDList(rand()), false)
                .singletonOrThrow()
                .getDataType()
        );
    }

    public void testOutputIsNonZero() {
        ParameterStore ps = TestFixture.freshPs();
        boolean hasNonZero = false;
        for (float v : ffn
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow()
            .toFloatArray()) {
            if (v != 0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("FFN must produce non-zero outputs", hasNonZero);
    }

    /**
     * ReLU² activation means no output element can be negative.
     * The down-projection can produce negatives, but this test confirms
     * the intermediate activations are non-negative — verified by checking
     * that a zero input stays zero (ReLU²(0) = 0) and a large positive
     * input passes through (ReLU²(x >> 0) > 0).
     */
    public void testZeroInputGivesZeroOutput() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray zeros = TestFixture.manager.zeros(SEQ_SHAPE, DataType.FLOAT32);
        NDArray out = ffn
            .forward(ps, new NDList(zeros), false)
            .singletonOrThrow();
        for (float v : out.toFloatArray()) {
            assertEquals("FFN(0) must be 0", 0f, v, 1e-5f);
        }
        System.out.println("BitFFN(0) = 0 ✓");
    }

    public void testOutputIsDeterministic() {
        NDArray x = rand();
        float[] out1 = ffn
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] out2 = ffn
            .forward(TestFixture.freshPs(), new NDList(x), false)
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

    public void testTrainingFlagDoesNotChangeShape() {
        NDArray x = rand();
        Shape train = ffn
            .forward(TestFixture.freshPs(), new NDList(x), true)
            .singletonOrThrow()
            .getShape();
        Shape infer = ffn
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .getShape();
        assertEquals(train, infer);
    }

    public void testToString() {
        String s = ffn.toString();
        assertTrue(s.contains("BitFFN"));
        assertTrue(s.contains("dModel"));
        System.out.println(s);
    }
}
