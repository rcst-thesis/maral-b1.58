package com.rcst.encoder;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * BitNet b1.58 Encoder Block.
 *
 * Pre-norm architecture — RMSNorm applied before each sublayer,
 * residual connection after:
 *
 *   x = x + Attention(RMSNorm(x))
 *   x = x + FFN(RMSNorm(x))
 *
 * Input  shape: (B, T, dModel)
 * Output shape: (B, T, dModel)
 */
public class EncoderBlock extends AbstractBlock {

    private final RMSNorm norm1;
    private final MultiHeadAttention attn;
    private final RMSNorm norm2;
    private final BitFFN ffn;

    /**
     * @param dModel    model dimension
     * @param nHeads    number of attention heads (dModel % nHeads == 0)
     * @param ffnDim    hidden dimension of the feed-forward network
     * @param ropeBase  RoPE base frequency (typically 10 000)
     * @param maxSeqLen maximum sequence length for RoPE table
     * @param quantEps  epsilon for BitLinear quantization
     */
    public EncoderBlock(
        int dModel,
        int nHeads,
        int ffnDim,
        int ropeBase,
        int maxSeqLen,
        float quantEps
    ) {
        this.norm1 = addChildBlock("norm1", new RMSNorm(dModel));
        this.attn = addChildBlock(
            "attn",
            new MultiHeadAttention(
                dModel,
                nHeads,
                ropeBase,
                maxSeqLen,
                quantEps,
                false, // encoder is bidirectional
                false // self-attention only
            )
        );

        this.norm2 = addChildBlock("norm2", new RMSNorm(dModel));
        this.ffn = addChildBlock("ffn", new BitFFN(dModel, ffnDim, quantEps));
    }

    @Override
    public void initializeChildBlocks(
        NDManager manager,
        DataType dataType,
        Shape... inputShapes
    ) {
        norm1.initialize(manager, dataType, inputShapes);
        attn.initialize(manager, dataType, inputShapes);
        norm2.initialize(manager, dataType, inputShapes);
        ffn.initialize(manager, dataType, inputShapes);
    }

    /**
     * NDList input contract:
     *   [x]                 — no padding mask
     *   [x, keyPaddingMask] — (B, T) mask, 1 at pad positions
     *
     * input  shape: (B, T, dModel)
     * output shape: (B, T, dModel)
     */
    @Override
    protected NDList forwardInternal(
        ParameterStore ps,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray x = inputs.get(0);
        NDArray paddingMask = inputs.size() > 1 ? inputs.get(1) : null;

        // Attention sublayer
        NDArray normed1 = norm1
            .forward(ps, new NDList(x), training)
            .singletonOrThrow();
        NDList attnIn =
            paddingMask != null
                ? new NDList(normed1, paddingMask)
                : new NDList(normed1);
        x = x.add(attn.forward(ps, attnIn, training).singletonOrThrow());

        // FFN sublayer
        NDArray normed2 = norm2
            .forward(ps, new NDList(x), training)
            .singletonOrThrow();
        x = x.add(
            ffn.forward(ps, new NDList(normed2), training).singletonOrThrow()
        );

        return new NDList(x);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] { inputShapes[0] };
    }

    @Override
    public String toString() {
        return String.format("EncoderBlock(attn=%s, ffn=%s)", attn, ffn);
    }
}
