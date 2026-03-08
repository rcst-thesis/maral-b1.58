package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.rcst.common.BitFFN;
import com.rcst.common.MultiHeadAttention;
import com.rcst.common.RMSNorm;

/**
 * Three sublayers in Pre-Norm (Pre-LN) order:
 *
 *   x = x + MaskedSelfAttn( RMSNorm(x) )           ← causal self-attention
 *   x = x + CrossAttn(      RMSNorm(x), memory )    ← cross-attention over encoder output
 *   x = x + FFN(            RMSNorm(x) )            ← feed-forward
 *
 * The causal mask in masked self-attention prevents each target position
 * from attending to future positions — required for autoregressive generation.
 * Cross-attention lets every decoder position freely attend all encoder
 * positions (subject to the optional source-padding mask).
 *
 * NDList input contract
 *   [x, memory]
 *   [x, memory, tgtPaddingMask]
 *   [x, memory, tgtPaddingMask, srcPaddingMask]
 *
 *   x              shape: (B, T, dModel)   target (decoder) sequence
 *   memory         shape: (B, S, dModel)   encoder output
 *   tgtPaddingMask shape: (B, T)           1 at pad positions in target, 0 elsewhere
 *   srcPaddingMask shape: (B, S)           1 at pad positions in source, 0 elsewhere
 *
 * Both masks are optional and independent — pass null / omit either when
 * there is no padding on that side.
 *
 * Output: NDList with a single tensor of shape (B, T, dModel).
 */
public class DecoderBlock extends AbstractBlock {

    private final RMSNorm norm1;
    private final MultiHeadAttention maskedSelfAttn;

    private final RMSNorm norm2;
    private final MultiHeadAttention crossAttn;

    private final RMSNorm norm3;
    private final BitFFN ffn;

    /**
     * @param dModel    model dimension
     * @param nHeads    number of attention heads  (dModel % nHeads == 0)
     * @param ffnDim    hidden dimension of the feed-forward network
     * @param ropeBase  RoPE base frequency (typically 10 000)
     * @param maxSeqLen maximum sequence length for RoPE table
     * @param quantEps  epsilon for BitLinear quantization
     */
    public DecoderBlock(
        int dModel,
        int nHeads,
        int ffnDim,
        int ropeBase,
        int maxSeqLen,
        float quantEps
    ) {
        // causal masked self-attention
        this.norm1 = addChildBlock("norm1", new RMSNorm(dModel));
        this.maskedSelfAttn = addChildBlock(
            "maskedSelfAttn",
            new MultiHeadAttention(
                dModel,
                nHeads,
                ropeBase,
                maxSeqLen,
                quantEps,
                true, // mask future target positions
                false // queries and keys both come from x
            )
        );

        // cross-attention over encoder memory
        this.norm2 = addChildBlock("norm2", new RMSNorm(dModel));
        this.crossAttn = addChildBlock(
            "crossAttn",
            new MultiHeadAttention(
                dModel,
                nHeads,
                ropeBase,
                maxSeqLen,
                quantEps,
                false, // cross-attention is not causal
                true // keys/values come from encoder memory
            )
        );

        // position-wise feed-forward
        this.norm3 = addChildBlock("norm3", new RMSNorm(dModel));
        this.ffn = addChildBlock("ffn", new BitFFN(dModel, ffnDim, quantEps));
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
        // Cross-attention needs both query shape and key/value (memory) shape
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

        // masked self-attention
        NDArray normed1 = norm1
            .forward(ps, new NDList(x), training)
            .singletonOrThrow();
        NDList selfAttnIn =
            tgtMask != null
                ? new NDList(normed1, tgtMask)
                : new NDList(normed1);
        x = x.add(
            maskedSelfAttn.forward(ps, selfAttnIn, training).singletonOrThrow()
        );

        // cross-attention
        NDArray normed2 = norm2
            .forward(ps, new NDList(x), training)
            .singletonOrThrow();
        NDList crossAttnIn =
            srcMask != null
                ? new NDList(normed2, memory, srcMask)
                : new NDList(normed2, memory);
        x = x.add(
            crossAttn.forward(ps, crossAttnIn, training).singletonOrThrow()
        );

        // feed-forward
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
        // Output shape matches the target (query) sequence shape
        return new Shape[] { inputShapes[0] };
    }

    @Override
    public String toString() {
        return String.format(
            "DecoderBlock(maskedSelfAttn=%s, crossAttn=%s, ffn=%s)",
            maskedSelfAttn,
            crossAttn,
            ffn
        );
    }
}
