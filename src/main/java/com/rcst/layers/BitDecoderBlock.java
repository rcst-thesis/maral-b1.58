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
 * BitNet b1.58 Decoder Block.
 *
 * One full decoder layer following the Pre-Norm layout:
 *
 *   x = x + MaskedSelfAttn( RMSNorm(x) )
 *   x = x + CrossAttn(      RMSNorm(x), memory )
 *   x = x + FFN(            RMSNorm(x) )
 *
 * NDList input contract:
 *   [x, memory]
 *   [x, memory, tgtPaddingMask]
 *   [x, memory, tgtPaddingMask, srcPaddingMask]
 *
 *   x              shape: (B, T, dModel)   target sequence (decoder side)
 *   memory         shape: (B, S, dModel)   encoder output
 *   tgtPaddingMask shape: (B, T)           1 at pad positions in target
 *   srcPaddingMask shape: (B, S)           1 at pad positions in source
 *
 * Output: NDList with a single tensor of shape (B, T, dModel).
 */
public class BitDecoderBlock extends AbstractBlock {

    private final RMSNorm norm1;
    private final BitAttention maskedSelfAttn;
    private final RMSNorm norm2;
    private final BitAttention crossAttn;
    private final RMSNorm norm3;
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
    public BitDecoderBlock(
        int dModel,
        int nHeads,
        int dFfn,
        int ropeBase,
        int maxSeqLen,
        float eps,
        float quantEps
    ) {
        this.norm1 = addChildBlock("norm1", new RMSNorm(dModel, eps));
        this.maskedSelfAttn = addChildBlock(
            "maskedSelfAttn",
            new BitAttention(
                dModel,
                nHeads,
                ropeBase,
                maxSeqLen,
                quantEps,
                true,
                false
            )
        );
        this.norm2 = addChildBlock("norm2", new RMSNorm(dModel, eps));
        this.crossAttn = addChildBlock(
            "crossAttn",
            new BitAttention(
                dModel,
                nHeads,
                ropeBase,
                maxSeqLen,
                quantEps,
                false,
                true
            )
        );
        this.norm3 = addChildBlock("norm3", new RMSNorm(dModel, eps));
        this.ffn = addChildBlock("ffn", new BitFFN(dModel, dFfn, quantEps));
    }

    @Override
    public void initializeChildBlocks(
        NDManager manager,
        DataType dataType,
        Shape... inputShapes
    ) {
        Shape xShape = inputShapes[0]; // (B, T, dModel)
        Shape memShape = inputShapes[1]; // (B, S, dModel)

        norm1.initialize(manager, dataType, xShape);
        maskedSelfAttn.initialize(manager, dataType, xShape);
        norm2.initialize(manager, dataType, xShape);
        crossAttn.initialize(manager, dataType, xShape, memShape);
        norm3.initialize(manager, dataType, xShape);
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
        NDArray memory = inputs.get(1);
        NDArray tgtMask = inputs.size() > 2 ? inputs.get(2) : null;
        NDArray srcMask = inputs.size() > 3 ? inputs.get(3) : null;

        // Masked self-attention sublayer
        NDArray normed1 = norm1
            .forward(ps, new NDList(x), training)
            .singletonOrThrow();
        NDList selfAttnInputs = (tgtMask != null)
            ? new NDList(normed1, tgtMask)
            : new NDList(normed1);
        x = x.add(
            maskedSelfAttn
                .forward(ps, selfAttnInputs, training)
                .singletonOrThrow()
        );

        // Cross-attention sublayer
        NDArray normed2 = norm2
            .forward(ps, new NDList(x), training)
            .singletonOrThrow();
        NDList crossAttnInputs = (srcMask != null)
            ? new NDList(normed2, memory, srcMask)
            : new NDList(normed2, memory);
        x = x.add(
            crossAttn.forward(ps, crossAttnInputs, training).singletonOrThrow()
        );

        // FFN sublayer
        NDArray normed3 = norm3
            .forward(ps, new NDList(x), training)
            .singletonOrThrow();
        x = x.add(
            ffn.forward(ps, new NDList(normed3), training).singletonOrThrow()
        );

        return new NDList(x);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] { inputShapes[0] };
    }

    @Override
    public String toString() {
        return String.format("BitDecoderBlock(%s, %s)", maskedSelfAttn, ffn);
    }
}
