package com.rcst.encoder;

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

/**
 * Unit tests for {@link EncoderBlock}.
 *
 * Covers:
 *   - Output shape preservation  (B, T, dModel) → (B, T, dModel)
 *   - getOutputShapes static contract
 *   - Data type (float32)
 *   - Non-zero output
 *   - Residual connection modifies the input
 *   - All-zero padding mask has no effect on the output
 *   - Active padding mask changes the output
 *   - Different inputs produce different outputs (non-constant function)
 *   - Training vs inference flag does not change the output shape
 *   - toString contains expected tokens
 */
public class EncoderBlockTest extends TestCase {

    private static EncoderBlock block;
    private static Shape SEQ_SHAPE; // (B, T, dModel)

    // ── Suite wiring ──────────────────────────────────────────────────────────

    public static Test suite() {
        return new TestSetup(new TestSuite(EncoderBlockTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                ModelConfig cfg = ModelConfig.get();

                SEQ_SHAPE = new Shape(
                    TestFixture.BATCH_SIZE,
                    TestFixture.BLOCK_SIZE,
                    TestFixture.D_MODEL
                );

                block = new EncoderBlock(
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
                    SEQ_SHAPE
                );
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** Random normal tensor of the standard sequence shape. */
    private NDArray rand() {
        return TestFixture.manager.randomNormal(SEQ_SHAPE, DataType.FLOAT32);
    }

    /** All-zeros mask of shape (B, T) — no padding active. */
    private NDArray zeroPaddingMask() {
        return TestFixture.manager.zeros(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE),
            DataType.FLOAT32
        );
    }

    /** All-ones mask of shape (B, T) — every position marked as padding. */
    private NDArray fullPaddingMask() {
        return TestFixture.manager.ones(
            new Shape(TestFixture.BATCH_SIZE, TestFixture.BLOCK_SIZE),
            DataType.FLOAT32
        );
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /**
     * The encoder block must preserve the spatial shape of its input:
     * (B, T, dModel) in → (B, T, dModel) out.
     */
    public void testOutputShapeMatchesInput() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray out = block
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow();

        assertEquals(SEQ_SHAPE, out.getShape());
        System.out.printf("EncoderBlock output shape: %s%n", out.getShape());
    }

    /**
     * getOutputShapes must echo the input shape without a forward pass,
     * so callers can plan memory before building the graph.
     */
    public void testGetOutputShapes() {
        Shape[] out = block.getOutputShapes(new Shape[] { SEQ_SHAPE });
        assertEquals(1, out.length);
        assertEquals(SEQ_SHAPE, out[0]);
    }

    /** Output tensor must stay in float32; BitLinear must not demote dtype. */
    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        DataType dtype = block
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow()
            .getDataType();

        assertEquals(DataType.FLOAT32, dtype);
    }

    /**
     * A freshly initialised block must not collapse every input to zero.
     * Even with random ternary weights the residual addition keeps the
     * output non-trivially non-zero.
     */
    public void testOutputIsNonZero() {
        ParameterStore ps = TestFixture.freshPs();
        float[] values = block
            .forward(ps, new NDList(rand()), false)
            .singletonOrThrow()
            .toFloatArray();

        boolean hasNonZero = false;
        for (float v : values) {
            if (v != 0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("EncoderBlock output must not be all zeros", hasNonZero);
    }

    /**
     * The two residual connections (attention + FFN) must change the input.
     * If out == in element-wise the sublayers are no-ops, which indicates
     * a wiring bug.
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
        assertTrue("residual connection must modify the input tensor", differs);
        System.out.println("residual connection modifies input ✓");
    }

    /**
     * An all-zero padding mask (no tokens masked) must produce the same
     * output as passing no mask at all — additive mask values are 0×MASK_VAL = 0.
     */
    public void testAllZeroPaddingMaskHasNoEffect() {
        NDArray x = rand();
        NDArray mask = zeroPaddingMask();

        float[] noMask = block
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] withMask = block
            .forward(TestFixture.freshPs(), new NDList(x, mask), false)
            .singletonOrThrow()
            .toFloatArray();

        for (int i = 0; i < noMask.length; i++) {
            assertEquals(
                "zero padding mask must not affect output at index " + i,
                noMask[i],
                withMask[i],
                1e-4f
            );
        }
        System.out.println("all-zero padding mask has no effect ✓");
    }

    /**
     * A fully-active padding mask (all ones) must produce a different output
     * than the no-mask baseline — every key position receives MASK_VAL before
     * softmax, so attention weights collapse and the output changes.
     *
     * Note: the residual path still adds x, so the output is not identically
     * zero; we only assert that it differs from the un-masked result.
     */
    public void testActivePaddingMaskChangesOutput() {
        NDArray x = rand();
        NDArray mask = fullPaddingMask();

        float[] noMask = block
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .toFloatArray();
        float[] withMask = block
            .forward(TestFixture.freshPs(), new NDList(x, mask), false)
            .singletonOrThrow()
            .toFloatArray();

        boolean differs = false;
        for (int i = 0; i < noMask.length; i++) {
            if (Math.abs(noMask[i] - withMask[i]) > 1e-4f) {
                differs = true;
                break;
            }
        }
        assertTrue(
            "a fully-active padding mask must change the output",
            differs
        );
        System.out.println("active padding mask changes output ✓");
    }

    /**
     * Two independently sampled random inputs must produce different outputs.
     * If they are identical the block is a constant function — a degenerate
     * initialisation or wiring bug.
     */
    public void testDifferentInputsProduceDifferentOutputs() {
        NDArray out1 = block
            .forward(TestFixture.freshPs(), new NDList(rand()), false)
            .singletonOrThrow();
        NDArray out2 = block
            .forward(TestFixture.freshPs(), new NDList(rand()), false)
            .singletonOrThrow();

        float[] a = out1.toFloatArray();
        float[] b = out2.toFloatArray();

        boolean differs = false;
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        assertTrue("different inputs must produce different outputs", differs);
        System.out.println("block is input-sensitive ✓");
    }

    /**
     * Toggling the training flag must not change the output shape.
     * (No dropout is applied in the current architecture, so values may
     * also be identical, but shape stability is the hard contract.)
     */
    public void testTrainingFlagDoesNotChangeShape() {
        NDArray x = rand();

        Shape trainShape = block
            .forward(TestFixture.freshPs(), new NDList(x), true)
            .singletonOrThrow()
            .getShape();
        Shape inferShape = block
            .forward(TestFixture.freshPs(), new NDList(x), false)
            .singletonOrThrow()
            .getShape();

        assertEquals(trainShape, inferShape);
    }

    /**
     * toString must at minimum mention the class and its two main sublayers
     * so log output is human-readable.
     */
    public void testToString() {
        String s = block.toString();
        assertTrue(
            "toString must mention EncoderBlock",
            s.contains("EncoderBlock")
        );
        assertTrue("toString must mention attn", s.contains("attn"));
        assertTrue("toString must mention ffn", s.contains("ffn"));
        System.out.println(s);
    }
}
