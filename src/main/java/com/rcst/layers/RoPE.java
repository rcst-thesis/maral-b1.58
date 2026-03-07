package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Rotary Position Embedding (RoPE).
 *
 * Encodes absolute position into query and key vectors by rotating
 * consecutive dimension pairs by position-dependent angles. Because
 * the rotation is applied to both Q and K before the dot-product,
 * the attention score implicitly captures the *relative* distance
 * between any two tokens — without adding a separate position vector
 * to the embeddings.
 *
 * Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
 *            Position Embedding" (2021).
 *
 * ── Maths ────────────────────────────────────────────────────────────
 *
 * For head dimension D (must be even) and base frequency θ₀:
 *
 *   θᵢ = θ₀^( -2i / D )   for i = 0, 1, …, D/2 - 1
 *
 * For token at position m, dimension pair (2i, 2i+1):
 *
 *   [ x₂ᵢ  ]   [ cos(m·θᵢ)  -sin(m·θᵢ) ] [ x₂ᵢ   ]
 *   [ x₂ᵢ₊₁] = [ sin(m·θᵢ)   cos(m·θᵢ) ] [ x₂ᵢ₊₁ ]
 *
 * Equivalently, using the "rotate-half" trick that avoids an explicit
 * interleave/de-interleave shuffle:
 *
 *   x_rot = x ⊙ cos + rotate_half(x) ⊙ sin
 *
 *   rotate_half(x): split x into [x₁ | x₂] along last dim,
 *                   return [-x₂ | x₁].
 *
 * ── Usage ─────────────────────────────────────────────────────────────
 *
 *   RoPE rope = new RoPE(headDim, maxSeqLen, ropeBase, manager);
 *
 *   // Inside BitAttention.forward(), after splitting heads:
 *   // q, k shape: (batch, seqLen, nHeads, headDim)
 *   NDArray q = rope.apply(q, posOffset);
 *   NDArray k = rope.apply(k, posOffset);
 *
 * posOffset is 0 for the encoder and for full-sequence decoder training;
 * it equals the number of already-generated tokens during autoregressive
 * inference (KV-cache step).
 *
 * ── Caching ───────────────────────────────────────────────────────────
 *
 * cos / sin tables are pre-computed once in the constructor for all
 * positions 0 … maxSeqLen-1, shape (maxSeqLen, headDim).
 * The tables live in the supplied NDManager; close that manager to free them.
 *
 * Input  shape: (batch, seqLen, nHeads, headDim)
 * Output shape: (batch, seqLen, nHeads, headDim)   — same as input
 */
public class RoPE {

    private final int headDim;
    private final int maxSeqLen;

    /** Pre-computed cosine table, shape: (maxSeqLen, headDim). */
    private final NDArray cosTable; // broadcast-ready over batch & head dims

    /** Pre-computed sine table, shape: (maxSeqLen, headDim). */
    private final NDArray sinTable;

    /**
     * @param headDim    per-head feature dimension  (d_model / n_heads)
     * @param maxSeqLen  maximum sequence length to pre-compute
     * @param ropeBase   base frequency θ₀ (typically 10 000)
     * @param manager    NDManager that will own the cached tables
     */
    public RoPE(int headDim, int maxSeqLen, int ropeBase, NDManager manager) {
        if (headDim % 2 != 0) {
            throw new IllegalArgumentException(
                "RoPE requires an even headDim, got " + headDim
            );
        }
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;

        // ── 1. Inverse frequencies: θᵢ = ropeBase^(-2i / headDim) ─────
        // shape: (headDim / 2,)
        int half = headDim / 2;
        float[] invFreqArr = new float[half];
        for (int i = 0; i < half; i++) {
            invFreqArr[i] = (float) Math.pow(ropeBase, (-2.0 * i) / headDim);
        }
        NDArray invFreq = manager.create(invFreqArr); // (half,)

        // ── 2. Position indices: 0, 1, …, maxSeqLen-1 ──────────────────
        // shape: (maxSeqLen,)
        float[] posArr = new float[maxSeqLen];
        for (int m = 0; m < maxSeqLen; m++) posArr[m] = m;
        NDArray positions = manager.create(posArr); // (maxSeqLen,)

        // ── 3. Outer product → angles, shape: (maxSeqLen, half) ─────────
        // angles[m][i] = m * θᵢ
        NDArray angles = positions
            .reshape(maxSeqLen, 1)
            .mul(invFreq.reshape(1, half)); // (maxSeqLen, half)

        // ── 4. Duplicate each angle for the pair dims: (maxSeqLen, headDim)
        // The "rotate-half" trick needs cos/sin broadcast over both dim 2i
        // AND dim 2i+1. We tile along the last axis: [θ₀,θ₁,…,θ₀,θ₁,…].
        // NDArray.concat over the last axis achieves [angles | angles].
        NDArray anglesFull = angles.concat(angles, 1); // (maxSeqLen, headDim)

        // ── 5. Cache cos / sin tables ────────────────────────────────────
        this.cosTable = anglesFull.cos(); // (maxSeqLen, headDim)
        this.sinTable = anglesFull.sin(); // (maxSeqLen, headDim)
    }

    /**
     * Apply RoPE to a query or key tensor.
     *
     * @param x         shape (batch, seqLen, nHeads, headDim)
     * @param posOffset starting position index (0 for encoder / training;
     *                  = cache length during incremental decoding)
     * @return          rotated tensor, same shape as x
     */
    public NDArray apply(NDArray x, int posOffset) {
        long seqLen = x.getShape().get(1);

        // Slice the pre-computed tables for positions [posOffset, posOffset+seqLen)
        // Shape after slice: (seqLen, headDim)
        NDArray cos = cosTable.get(posOffset + ":" + (posOffset + seqLen));
        NDArray sin = sinTable.get(posOffset + ":" + (posOffset + seqLen));

        // Reshape to (1, seqLen, 1, headDim) for broadcasting over batch & heads
        long[] broadShape = { 1, seqLen, 1, headDim };
        cos = cos.reshape(broadShape); // (1, T, 1, D)
        sin = sin.reshape(broadShape); // (1, T, 1, D)

        // rotate_half(x):
        //   split x into first half and second half along last dim,
        //   return concat([-x2, x1]).
        NDArray rotated = rotateHalf(x);

        // x_rot = x ⊙ cos + rotate_half(x) ⊙ sin
        return x.mul(cos).add(rotated.mul(sin));
    }

    /**
     * Convenience overload — posOffset defaults to 0 (training / encoder).
     */
    public NDArray apply(NDArray x) {
        return apply(x, 0);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * rotate_half — negate the second half of the last dimension and swap halves.
     *
     * Given x of shape (..., D):
     *   x₁ = x[..., :D/2]
     *   x₂ = x[..., D/2:]
     *   return concat([-x₂, x₁], dim=-1)
     *
     * This implements the pair-wise rotation without explicitly interleaving
     * even/odd indices, which would require a reshape + transpose shuffle.
     */
    private NDArray rotateHalf(NDArray x) {
        int lastAxis = x.getShape().dimension() - 1;
        long D = x.getShape().get(lastAxis);
        long half = D / 2;

        NDArray x1 = x.get(":, :, :, :" + half); // (..., D/2)
        NDArray x2 = x.get(":, :, :, " + half + ":"); // (..., D/2)

        return x2.neg().concat(x1, lastAxis); // (..., D)
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    @Override
    public String toString() {
        return String.format(
            "RoPE(headDim=%d, maxSeqLen=%d)",
            headDim,
            maxSeqLen
        );
    }
}
