package com.rcst.common;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.rcst.TestFixture;
import com.rcst.utils.ModelConfig;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class RoPETest extends TestCase {

    private static RoPE rope;
    private static int HEAD_DIM;
    private static int N_HEADS;
    private static int MAX_SEQ_LEN;

    public static Test suite() {
        return new TestSetup(new TestSuite(RoPETest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                ModelConfig cfg = ModelConfig.get();
                N_HEADS = cfg.nHeads;
                HEAD_DIM = cfg.dModel / cfg.nHeads;
                MAX_SEQ_LEN = cfg.maxSeqLen;
                rope = new RoPE(
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    cfg.ropeBase,
                    TestFixture.manager
                );
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    /** Random float tensor of given shape. */
    private NDArray rand(long... dims) {
        return TestFixture.manager.randomNormal(
            new Shape(dims),
            DataType.FLOAT32
        );
    }

    public void testOutputShapeMatchesInput() {
        NDArray x = rand(
            TestFixture.BATCH_SIZE,
            TestFixture.BLOCK_SIZE,
            N_HEADS,
            HEAD_DIM
        );
        NDArray out = rope.apply(x);
        assertEquals(x.getShape(), out.getShape());
        System.out.printf("RoPE output shape: %s%n", out.getShape());
    }

    public void testOutputIsFloat32() {
        NDArray x = rand(2, 4, N_HEADS, HEAD_DIM);
        assertEquals(DataType.FLOAT32, rope.apply(x).getDataType());
    }

    /**
     * RoPE is norm-preserving per token-position: ‖rotate(x)‖ = ‖x‖.
     * (cos² + sin² = 1 for every pair, so vector length is unchanged.)
     */
    public void testNormPreservation() {
        NDArray x = rand(2, 8, N_HEADS, HEAD_DIM);
        NDArray out = rope.apply(x);

        // Compute L2 norm along last axis for input and output
        NDArray normIn = x.pow(2).sum(new int[] { 3 }).sqrt(); // (B, T, H)
        NDArray normOut = out.pow(2).sum(new int[] { 3 }).sqrt(); // (B, T, H)

        float[] ni = normIn.toFloatArray();
        float[] no = normOut.toFloatArray();
        for (int i = 0; i < ni.length; i++) {
            assertEquals(
                "Norm must be preserved at index " +
                    i +
                    " (in=" +
                    ni[i] +
                    ", out=" +
                    no[i] +
                    ")",
                ni[i],
                no[i],
                1e-4f
            );
        }
        System.out.println("RoPE norm-preservation ✓");
    }

    /**
     * Position 0 rotation should produce cos(0)=1, sin(0)=0 → output == input.
     */
    public void testPositionZeroIsIdentity() {
        NDArray x = rand(1, 1, N_HEADS, HEAD_DIM);
        NDArray out = rope.apply(x, 0); // seqLen=1, posOffset=0

        float[] in = x.toFloatArray();
        float[] o = out.toFloatArray();
        for (int i = 0; i < in.length; i++) {
            assertEquals("pos-0 must be identity at " + i, in[i], o[i], 1e-5f);
        }
        System.out.println("RoPE position-0 identity ✓");
    }

    /**
     * Different positions must produce different rotations (non-trivial encoding).
     */
    public void testDifferentPositionsProduceDifferentOutput() {
        // Same input vector, different position offsets
        NDArray x = rand(1, 1, N_HEADS, HEAD_DIM);
        NDArray o1 =
            rope.apply(x, 0).toFloatArray() != null ? rope.apply(x, 0) : x;
        NDArray o5 = rope.apply(x, 5);

        boolean differs = false;
        float[] a = o1.toFloatArray(),
            b = o5.toFloatArray();
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 1e-6f) {
                differs = true;
                break;
            }
        }
        assertTrue("pos-0 and pos-5 must differ", differs);
        System.out.println("RoPE produces position-dependent output ✓");
    }

    /**
     * posOffset slicing: applying RoPE with offset should equal taking a
     * contiguous slice of the full-range application.
     *
     * Concretely: apply([x_pos3, x_pos4], offset=3)
     *          == apply([x_pos0, x_pos1, x_pos2, x_pos3, x_pos4])[3:5]
     */
    public void testPosOffsetConsistency() {
        int T = 5;
        NDArray xFull = rand(1, T, N_HEADS, HEAD_DIM);

        // Full forward
        NDArray outFull = rope.apply(xFull, 0);
        NDArray outSliced = outFull.get(":, 3:, :, :"); // positions 3 and 4

        // Sub-sequence with offset
        NDArray xSub = xFull.get(":, 3:, :, :"); // same vectors, positions 3-4
        NDArray outOffset = rope.apply(xSub, 3);

        float[] expected = outSliced.toFloatArray();
        float[] actual = outOffset.toFloatArray();
        assertEquals("length mismatch", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(
                "offset mismatch at " + i,
                expected[i],
                actual[i],
                1e-4f
            );
        }
        System.out.println("RoPE posOffset consistency ✓");
    }

    public void testToString() {
        String s = rope.toString();
        assertTrue(s.contains("RoPE"));
        assertTrue(s.contains(String.valueOf(HEAD_DIM)));
        System.out.println("toString: " + s);
    }

    public void testOddHeadDimThrows() {
        try {
            new RoPE(65, 64, 10000, TestFixture.manager);
            fail("Expected IllegalArgumentException for odd headDim");
        } catch (IllegalArgumentException e) {
            System.out.println(
                "Odd headDim correctly rejected: " + e.getMessage()
            );
        }
    }
}
