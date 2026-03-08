package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Rotary Position Embedding (RoPE).
 *
 * Encodes absolute position into query and key vectors by rotating
 * consecutive dimension pairs by position-dependent angles.
 *
 * ── Device safety ────────────────────────────────────────────────────
 * Previous version stored cosTable / sinTable as NDArrays attached to a
 * fixed manager passed at construction time. This caused a device-mismatch
 * crash whenever the forward pass ran on a different device than the one
 * used during construction (e.g. validation on CPU after training on GPU).
 *
 * Fix: store the tables as plain Java float[] arrays and create NDArrays
 * inside apply() using x.getManager(). The result:
 *   - Tables are always created on the same device as the input.
 *   - Lifetime matches the step manager — freed automatically when the
 *     step sub-manager closes, so no VRAM accumulates across epochs.
 *
 * ── Memory note ──────────────────────────────────────────────────────
 * Creating NDArrays every forward call is cheap: for headDim=32,
 * maxSeqLen=32 the tables are 32×32×4 bytes = 4 KiB. The overhead is
 * negligible compared to the attention computation itself.
 *
 * Input  shape: (batch, seqLen, nHeads, headDim)
 * Output shape: same
 */
public class RoPE {

    private final int headDim;
    private final int maxSeqLen;

    /**
     * Pre-computed tables stored as Java float arrays — device-agnostic.
     * Shape (logically): (maxSeqLen, headDim)
     * Stride: cosData[pos * headDim + d]
     */
    private final float[] cosData;
    private final float[] sinData;

    /**
     * @param headDim    per-head feature dimension  (d_model / n_heads, must be even)
     * @param maxSeqLen  maximum sequence length to pre-compute
     * @param ropeBase   base frequency θ₀ (typically 10 000)
     * @param manager    only used here to temporarily compute the table;
     *                   the result is immediately pulled back to float[] and
     *                   the NDArrays are released before the constructor returns
     */
    public RoPE(int headDim, int maxSeqLen, int ropeBase, NDManager manager) {
        if (headDim % 2 != 0) {
            throw new IllegalArgumentException(
                "RoPE requires an even headDim, got " + headDim
            );
        }
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;

        // ── Build tables using a temporary sub-manager ────────────────────
        // All NDArrays used in construction are freed when this scope closes.
        try (NDManager tmp = manager.newSubManager()) {
            int half = headDim / 2;

            // θᵢ = ropeBase^(-2i / headDim)
            float[] invFreqArr = new float[half];
            for (int i = 0; i < half; i++) {
                invFreqArr[i] = (float) Math.pow(
                    ropeBase,
                    (-2.0 * i) / headDim
                );
            }
            NDArray invFreq = tmp.create(invFreqArr); // (half,)

            // Position indices 0 … maxSeqLen-1
            float[] posArr = new float[maxSeqLen];
            for (int m = 0; m < maxSeqLen; m++) posArr[m] = m;
            NDArray positions = tmp.create(posArr); // (maxSeqLen,)

            // Outer product → angles (maxSeqLen, half)
            NDArray angles = positions
                .reshape(maxSeqLen, 1)
                .mul(invFreq.reshape(1, half));

            // Duplicate along last axis → (maxSeqLen, headDim)
            NDArray anglesFull = angles.concat(angles, 1);

            // Pull to Java arrays — from here on the tables are device-agnostic
            this.cosData = anglesFull.cos().toFloatArray();
            this.sinData = anglesFull.sin().toFloatArray();
        } // tmp closes → all NDArrays freed, no persistent CUDA memory
    }

    /**
     * Apply RoPE to a query or key tensor.
     *
     * NDArrays are created using x.getManager() so they land on the same
     * device as the input and are freed when that manager's scope closes.
     *
     * @param x         shape (batch, seqLen, nHeads, headDim)
     * @param posOffset starting position (0 for encoder / full-seq training;
     *                  = KV-cache length during incremental decoding)
     * @return          rotated tensor, same shape as x
     */
    public NDArray apply(NDArray x, int posOffset) {
        NDManager mgr = x.getManager();
        long seqLen = x.getShape().get(1);

        // Create tables on the fly in the input's manager — correct device, correct lifetime
        NDArray fullCos = mgr.create(
            cosData,
            new ai.djl.ndarray.types.Shape(maxSeqLen, headDim)
        );
        NDArray fullSin = mgr.create(
            sinData,
            new ai.djl.ndarray.types.Shape(maxSeqLen, headDim)
        );

        // Slice positions [posOffset, posOffset + seqLen)
        NDArray cos = fullCos.get(posOffset + ":" + (posOffset + seqLen)); // (seqLen, headDim)
        NDArray sin = fullSin.get(posOffset + ":" + (posOffset + seqLen));

        // Reshape to (1, seqLen, 1, headDim) for broadcasting over batch and heads
        long[] broadShape = { 1, seqLen, 1, headDim };
        cos = cos.reshape(broadShape);
        sin = sin.reshape(broadShape);

        // x_rot = x ⊙ cos + rotate_half(x) ⊙ sin
        return x.mul(cos).add(rotateHalf(x).mul(sin));
    }

    /** Convenience overload — posOffset defaults to 0. */
    public NDArray apply(NDArray x) {
        return apply(x, 0);
    }

    // ── Private ───────────────────────────────────────────────────────────────

    /**
     * rotate_half: given x (..., D), split into [x1 | x2] and return [-x2 | x1].
     */
    private NDArray rotateHalf(NDArray x) {
        int lastAxis = x.getShape().dimension() - 1;
        long half = x.getShape().get(lastAxis) / 2;

        NDArray x1 = x.get(":, :, :, :" + half);
        NDArray x2 = x.get(":, :, :, " + half + ":");

        return x2.neg().concat(x1, lastAxis);
    }

    @Override
    public String toString() {
        return String.format(
            "RoPE(headDim=%d, maxSeqLen=%d)",
            headDim,
            maxSeqLen
        );
    }
}
