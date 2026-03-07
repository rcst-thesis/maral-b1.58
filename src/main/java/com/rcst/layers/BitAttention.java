package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * BitNet b1.58 Multi-Head Attention.
 *
 * Handles three configurations selected at construction time:
 *
 *   isCross=false, isCausal=false  encoder self-attention (bidirectional)
 *   isCross=false, isCausal=true   decoder masked self-attention
 *   isCross=true,  isCausal=false  decoder cross-attention
 *
 * All Q/K/V/O projections use BitLinear (ternary weights, 8-bit activations).
 * RoPE is applied to Q and K in self-attention only; cross-attention receives
 * encoder memory whose positions were already encoded by the encoder.
 * Pre-norm (RMSNorm) lives in the enclosing BitEncoderBlock / BitDecoderBlock,
 * not here, so this class stays focused on attention arithmetic.
 *
 * NDList input contract:
 *   self-attention : [x]               or [x, keyPaddingMask]
 *   cross-attention: [x, memory]       or [x, memory, keyPaddingMask]
 *
 *   x              shape: (B, T, dModel)   query source
 *   memory         shape: (B, S, dModel)   encoder output (cross only)
 *   keyPaddingMask shape: (B, S)           1 at pad positions, 0 elsewhere
 *
 * Output: NDList with a single tensor of shape (B, T, dModel).
 */
public class BitAttention extends AbstractBlock {

    private static final float MASK_VAL = -1e9f;

    private final int dModel;
    private final int nHeads;
    private final int headDim;
    private final boolean isCausal;
    private final boolean isCross;

    // Stored only to construct RoPE inside initializeChildBlocks
    private final int ropeBase;
    private final int maxSeqLen;

    private final BitLinear wq;
    private final BitLinear wk;
    private final BitLinear wv;
    private final BitLinear wo;

    // Populated in initializeChildBlocks; null for cross-attention
    private RoPE rope;

    /**
     * @param dModel    model (embedding) dimension
     * @param nHeads    number of attention heads  (dModel % nHeads == 0)
     * @param ropeBase  RoPE base frequency (10 000)
     * @param maxSeqLen maximum sequence length for RoPE table
     * @param quantEps  epsilon for BitLinear weight quantization
     * @param isCausal  mask future positions (decoder self-attention)
     * @param isCross   keys/values come from a second input (cross-attention)
     */
    public BitAttention(
        int dModel,
        int nHeads,
        int ropeBase,
        int maxSeqLen,
        float quantEps,
        boolean isCausal,
        boolean isCross
    ) {
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException(
                "dModel " + dModel + " must be divisible by nHeads " + nHeads
            );
        }
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
        // Cross-attention caller passes [qShape, memoryShape];
        // self-attention Q and K/V share the same shape.
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
        NDManager mgr = x.getManager();
        long B = x.getShape().get(0);
        long T = x.getShape().get(1);

        NDArray kvSrc;
        NDArray paddingMask = null;

        if (isCross) {
            kvSrc = inputs.get(1);
            if (inputs.size() > 2) paddingMask = inputs.get(2);
        } else {
            kvSrc = x;
            if (inputs.size() > 1) paddingMask = inputs.get(1);
        }
        long S = kvSrc.getShape().get(1);

        // Project Q, K, V
        NDArray q = wq.forward(ps, new NDList(x), training).singletonOrThrow();
        NDArray k = wk
            .forward(ps, new NDList(kvSrc), training)
            .singletonOrThrow();
        NDArray v = wv
            .forward(ps, new NDList(kvSrc), training)
            .singletonOrThrow();

        // Split heads  →  (B, T or S, nHeads, headDim)
        q = q.reshape(B, T, nHeads, headDim);
        k = k.reshape(B, S, nHeads, headDim);
        v = v.reshape(B, S, nHeads, headDim);

        // RoPE encodes position into Q and K for self-attention
        if (!isCross) {
            q = rope.apply(q);
            k = rope.apply(k);
        }

        // Transpose to (B, nHeads, T or S, headDim)
        q = q.transpose(0, 2, 1, 3);
        k = k.transpose(0, 2, 1, 3);
        v = v.transpose(0, 2, 1, 3);

        // Scaled dot-product scores  →  (B, H, T, S)
        float scale = (float) (1.0 / Math.sqrt(headDim));
        NDArray scores = q.matMul(k.transpose(0, 1, 3, 2)).mul(scale);

        // Additive causal mask: future positions receive MASK_VAL before softmax
        if (isCausal) {
            scores = scores.add(causalMask(mgr, T));
        }

        // Additive key-padding mask: pad positions receive MASK_VAL
        if (paddingMask != null) {
            scores = scores.add(paddingMask.reshape(B, 1, 1, S).mul(MASK_VAL));
        }

        NDArray attn = scores.softmax(-1); // (B, H, T, S)
        NDArray out = attn.matMul(v); // (B, H, T, headDim)

        // Merge heads  →  (B, T, dModel)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, dModel);

        return new NDList(
            wo.forward(ps, new NDList(out), training).singletonOrThrow()
        );
    }

    /**
     * Additive causal mask, shape (1, 1, T, T).
     * 0 where a query may attend (lower triangle + diagonal),
     * MASK_VAL everywhere else, so softmax zeroes out future positions.
     *
     * DJL has no tril(); we build the mask directly as a float array:
     * mask[row][col] = 0 if col <= row, else MASK_VAL.
     */
    private NDArray causalMask(NDManager mgr, long T) {
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
