package com.rcst.common;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.rcst.layers.BitLinear;

/**
 * Multi-Head Attention with BitLinear projections and RoPE.
 *
 * MEMORY FIX: Don't cache causal mask - recreate each forward pass
 * to prevent memory accumulation in long training runs.
 */
public class MultiHeadAttention extends AbstractBlock {

    private static final float MASK_VAL = -1e9f;

    private final int dModel;
    private final int nHeads;
    private final int headDim;
    private final boolean isCausal;
    private final boolean isCross;

    // Held only to initialize RoPE in initializeChildBlocks
    private final int ropeBase;
    private final int maxSeqLen;

    private final BitLinear wq, wk, wv, wo;

    // Null for cross-attention; set in initializeChildBlocks
    private RoPE rope;

    // REMOVED: Cached causal mask causes memory leak
    // private NDArray cachedMask;
    // private long cachedMaskLen = -1;

    public MultiHeadAttention(
        int dModel,
        int nHeads,
        int ropeBase,
        int maxSeqLen,
        float quantEps,
        boolean isCausal,
        boolean isCross
    ) {
        if (dModel % nHeads != 0) throw new IllegalArgumentException(
            "dModel " + dModel + " must be divisible by nHeads " + nHeads
        );

        this.dModel = dModel;
        this.nHeads = nHeads;
        this.headDim = dModel / nHeads;
        this.isCausal = isCausal;
        this.isCross = isCross;
        this.ropeBase = ropeBase;
        this.maxSeqLen = maxSeqLen;

        this.wq = addChildBlock("wq", new BitLinear(dModel, dModel, quantEps));
        this.wk = addChildBlock("wk", new BitLinear(dModel, dModel, quantEps));
        this.wv = addChildBlock("wv", new BitLinear(dModel, dModel, quantEps));
        this.wo = addChildBlock("wo", new BitLinear(dModel, dModel, quantEps));
    }

    @Override
    public void initializeChildBlocks(
        NDManager manager,
        DataType dataType,
        Shape... inputShapes
    ) {
        Shape qShape = inputShapes[0];
        Shape kvShape = (isCross && inputShapes.length > 1)
            ? inputShapes[1]
            : qShape;

        wq.initialize(manager, dataType, qShape);
        wk.initialize(manager, dataType, kvShape);
        wv.initialize(manager, dataType, kvShape);
        wo.initialize(manager, dataType, qShape);

        if (!isCross) {
            rope = new RoPE(headDim, maxSeqLen, ropeBase, manager);
        }
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore ps,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray x = inputs.get(0);
        long B = x.getShape().get(0);
        long T = x.getShape().get(1);

        NDArray kvSrc = isCross ? inputs.get(1) : x;
        NDArray paddingMask = isCross
            ? (inputs.size() > 2 ? inputs.get(2) : null)
            : (inputs.size() > 1 ? inputs.get(1) : null);
        long S = kvSrc.getShape().get(1);

        // (B, T/S, nHeads, headDim)
        NDArray q = projectHeads(ps, wq, x, B, T, training);
        NDArray k = projectHeads(ps, wk, kvSrc, B, S, training);
        NDArray v = projectHeads(ps, wv, kvSrc, B, S, training);

        // RoPE expects (B, seqLen, nHeads, headDim)
        if (!isCross) {
            q = rope.apply(q);
            k = rope.apply(k);
        }

        // Transpose once to (B, nHeads, T/S, headDim) right before matmul
        q = q.transpose(0, 2, 1, 3);
        k = k.transpose(0, 2, 1, 3);
        v = v.transpose(0, 2, 1, 3);

        // Scaled dot-product → (B, nHeads, T, S)
        float scale = (float) (1.0 / Math.sqrt(headDim));
        NDArray scores = q.matMul(k.transpose(0, 1, 3, 2)).mul(scale);

        // FIX: Create mask fresh each time - no caching!
        if (isCausal) {
            NDArray mask = createCausalMask(x.getManager(), T);
            scores = scores.add(mask);
            mask.close(); // Free immediately after use
        }

        if (paddingMask != null) {
            scores = scores.add(paddingMask.reshape(B, 1, 1, S).mul(MASK_VAL));
        }

        // Weighted sum → merge heads → (B, T, dModel)
        NDArray out = scores
            .softmax(-1) // (B, H, T, S)
            .matMul(v) // (B, H, T, headDim)
            .transpose(0, 2, 1, 3) // (B, T, H, headDim)
            .reshape(B, T, dModel); // (B, T, dModel)

        return new NDList(
            wo.forward(ps, new NDList(out), training).singletonOrThrow()
        );
    }

    /** Project → reshape  →  (B, seqLen, nHeads, headDim) */
    private NDArray projectHeads(
        ParameterStore ps,
        BitLinear proj,
        NDArray src,
        long B,
        long seqLen,
        boolean training
    ) {
        return proj
            .forward(ps, new NDList(src), training)
            .singletonOrThrow()
            .reshape(B, seqLen, nHeads, headDim);
    }

    /**
     * Create causal mask fresh each forward pass.
     * NO CACHING - prevents memory leak over long training runs.
     */
    private NDArray createCausalMask(NDManager mgr, long T) {
        int n = (int) T;
        float[] data = new float[n * n];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                data[row * n + col] = (col <= row) ? 0f : MASK_VAL;
            }
        }
        return mgr.create(data, new Shape(1, 1, T, T));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] { inputShapes[0] };
    }

    @Override
    public String toString() {
        return String.format(
            "BitAttention(dModel=%d, nHeads=%d, headDim=%d, causal=%b, cross=%b)",
            dModel,
            nHeads,
            headDim,
            isCausal,
            isCross
        );
    }
}
