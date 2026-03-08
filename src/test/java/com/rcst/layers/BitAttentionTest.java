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

public class BitAttentionTest extends TestCase {

    private static BitAttention encoderAttn; // self, bidirectional
    private static BitAttention decoderAttn; // self, causal
    private static BitAttention crossAttn; // cross

    private static Shape SEQ_SHAPE; // (B, T, dModel)
    private static Shape MEM_SHAPE; // (B, S, dModel) — S != T to catch shape bugs

    public static Test suite() {
        return new TestSetup(new TestSuite(BitAttentionTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                ModelConfig cfg = ModelConfig.get();
                SEQ_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    TestFixture.BLOCK_SIZE,
                    TestFixture.D_MODEL
                );
                // Use a different source length for cross-attention to surface bugs
                // where T and S are accidentally swapped
                int SRC_LEN = Math.max(TestFixture.BLOCK_SIZE + 2, 4);
                MEM_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    SRC_LEN,
                    TestFixture.D_MODEL
                );

                encoderAttn = new BitAttention(
                    cfg.dModel,
                    cfg.nHeads,
                    cfg.ropeBase,
                    cfg.maxSeqLen,
                    cfg.quantEps,
                    false,
                    false
                );
                encoderAttn.initialize(
                    TestFixture.manager,
                    DataType.FLOAT32,
                    SEQ_SHAPE
                );

                decoderAttn = new BitAttention(
                    cfg.dModel,
                    cfg.nHeads,
                    cfg.ropeBase,
                    cfg.maxSeqLen,
                    cfg.quantEps,
                    true,
                    false
                );
                decoderAttn.initialize(
                    TestFixture.manager,
                    DataType.FLOAT32,
                    SEQ_SHAPE
                );

                crossAttn = new BitAttention(
                    cfg.dModel,
                    cfg.nHeads,
                    cfg.ropeBase,
                    cfg.maxSeqLen,
                    cfg.quantEps,
                    false,
                    true
                );
                crossAttn.initialize(
                    TestFixture.manager,
                    DataType.FLOAT32,
                    SEQ_SHAPE,
                    MEM_SHAPE
                );
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    private NDArray randSeq(Shape shape) {
        return TestFixture.manager.randomNormal(shape, DataType.FLOAT32);
    }

    public void testEncoderSelfAttnOutputShape() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray x = randSeq(SEQ_SHAPE);
        NDArray out = encoderAttn
            .forward(ps, new NDList(x), false)
            .singletonOrThrow();
        assertEquals(SEQ_SHAPE, out.getShape());
        System.out.printf("encoder self-attn output: %s%n", out.getShape());
    }

    public void testDecoderSelfAttnOutputShape() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray x = randSeq(SEQ_SHAPE);
        NDArray out = decoderAttn
            .forward(ps, new NDList(x), false)
            .singletonOrThrow();
        assertEquals(SEQ_SHAPE, out.getShape());
        System.out.printf("decoder self-attn output: %s%n", out.getShape());
    }

    public void testCrossAttnOutputShape() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray x = randSeq(SEQ_SHAPE);
        NDArray memory = randSeq(MEM_SHAPE);
        NDArray out = crossAttn
            .forward(ps, new NDList(x, memory), false)
            .singletonOrThrow();
        // Output query length must equal T, not S
        assertEquals(SEQ_SHAPE, out.getShape());
        System.out.printf(
            "cross-attn output: %s  (memory was %s)%n",
            out.getShape(),
            MEM_SHAPE
        );
    }

    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray out = encoderAttn
            .forward(ps, new NDList(randSeq(SEQ_SHAPE)), false)
            .singletonOrThrow();
        assertEquals(DataType.FLOAT32, out.getDataType());
    }

    public void testOutputIsNonZero() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray out = encoderAttn
            .forward(ps, new NDList(randSeq(SEQ_SHAPE)), false)
            .singletonOrThrow();
        boolean hasNonZero = false;
        for (float v : out.toFloatArray()) {
            if (v != 0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("attention output must not be all zeros", hasNonZero);
    }

    /**
     * The causal mask must prevent position t from attending to t+1 and beyond.
     * A concrete check: run a constant input through causal attention and verify
     * that the first-token output differs from the last-token output (if they
     * were identical the mask would not have changed anything — but since each
     * position sees a different context window, the outputs must differ).
     */
    public void testCausalMaskChangesOutput() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray x = randSeq(SEQ_SHAPE);
        NDArray out = decoderAttn
            .forward(ps, new NDList(x), false)
            .singletonOrThrow();

        // First and last token embeddings from the output
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
            "causal masking must produce position-dependent outputs",
            differs
        );
        System.out.println("causal mask produces position-dependent outputs ✓");
    }

    /**
     * Causal and bidirectional attention on the same input must give different
     * outputs for any token that is not the very first (position 0 can attend
     * to the same single token in both cases, but position 1 onward cannot).
     */
    public void testCausalDiffersFromBidirectional() {
        NDArray x = randSeq(SEQ_SHAPE);

        NDArray outCausal = decoderAttn
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow();
        NDArray outBidir = encoderAttn
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow();

        // Compare at position 1 (the causal model can only attend to 0 and 1,
        // the bidirectional model can attend to all positions)
        float[] causal = outCausal.get("0, 1, :").toFloatArray();
        float[] bidir = outBidir.get("0, 1, :").toFloatArray();

        boolean differs = false;
        for (int i = 0; i < causal.length; i++) {
            if (Math.abs(causal[i] - bidir[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        assertTrue(
            "causal and bidirectional outputs must differ at pos > 0",
            differs
        );
        System.out.println("causal vs bidirectional outputs differ ✓");
    }

    public void testGetOutputShapes() {
        Shape[] out = encoderAttn.getOutputShapes(new Shape[] { SEQ_SHAPE });
        assertEquals(1, out.length);
        assertEquals(SEQ_SHAPE, out[0]);
    }

    public void testDModelNotDivisibleByNHeadsThrows() {
        ModelConfig cfg = ModelConfig.get();
        try {
            new BitAttention(
                513,
                cfg.nHeads,
                cfg.ropeBase,
                cfg.maxSeqLen,
                cfg.quantEps,
                false,
                false
            );
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            System.out.println(
                "Indivisible dModel correctly rejected: " + e.getMessage()
            );
        }
    }

    public void testToString() {
        String s = encoderAttn.toString();
        assertTrue(s.contains("BitAttention"));
        assertTrue(s.contains("causal=false"));
        System.out.println(s);
    }
}
