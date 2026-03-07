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
 * BitNet b1.58 Encoder Block.
 *
 * One full encoder layer following the Pre-Norm (Pre-LN) layout:
 *
 *   x = x + SelfAttn( RMSNorm(x) )
 *   x = x + FFN(      RMSNorm(x) )
 *
 * Pre-LN places normalisation before the sublayer rather than after
 * (Post-LN, the original "Attention Is All You Need" arrangement).
 * This stabilises gradients in deep networks and is standard in modern
 * transformer variants including BitNet b1.58.
 *
 * NDList input contract:
 *   [x]                    — no padding mask
 *   [x, keyPaddingMask]    — 1 at pad positions, 0 elsewhere; shape (B, T)
 *
 * Output: NDList with a single tensor of shape (B, T, dModel).
 */
public class BitEncoderBlock extends AbstractBlock {

    private final RMSNorm norm1;
    private final BitAttention selfAttn;
    private final RMSNorm norm2;
    private final BitFFN ffn;

    /**
     * @param dModel    model dimension
     * @param nHeads    attention heads
     * @param dFfn      FFN hidden dimension
     * @param ropeBase  RoPE base frequency
     * @param maxSeqLen maximum sequence length for RoPE table
     * @param eps       RMSNorm epsilon
     * @param quantEps  BitLinear quantization epsilon
     */
    public BitEncoderBlock(
        int dModel,
        int nHeads,
        int dFfn,
        int ropeBase,
        int maxSeqLen,
        float eps,
        float quantEps
    ) {
        this.norm1 = addChildBlock("norm1", new RMSNorm(dModel, eps));
        this.selfAttn = addChildBlock(
            "selfAttn",
            new BitAttention(
                dModel,
                nHeads,
                ropeBase,
                maxSeqLen,
                quantEps,
                false,
                false
            )
        );
        this.norm2 = addChildBlock("norm2", new RMSNorm(dModel, eps));
        this.ffn = addChildBlock("ffn", new BitFFN(dModel, dFfn, quantEps));
    }

    @Override
    public void initializeChildBlocks(
        NDManager manager,
        DataType dataType,
        Shape... inputShapes
    ) {
        Shape xShape = inputShapes[0]; // (B, T, dModel)
        norm1.initialize(manager, dataType, xShape);
        selfAttn.initialize(manager, dataType, xShape);
        norm2.initialize(manager, dataType, xShape);
        ffn.initialize(manager, dataType, xShape);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore ps,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray x = inputs.get(0);

        // Self-attention sublayer with optional padding mask
        NDArray normed1 = norm1
            .forward(ps, new NDList(x), training)
            .singletonOrThrow();
        NDList attnInputs = (inputs.size() > 1)
            ? new NDList(normed1, inputs.get(1))
            : new NDList(normed1);
        x = x.add(
            selfAttn.forward(ps, attnInputs, training).singletonOrThrow()
        );

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
        return String.format("BitEncoderBlock(%s, %s)", selfAttn, ffn);
    }
}
