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
 * BitNet b1.58 Decoder.
 *
 * A stack of N identical BitDecoderBlocks followed by a final RMSNorm.
 *
 * NDList input contract:
 *   [x, memory]
 *   [x, memory, tgtPaddingMask]
 *   [x, memory, tgtPaddingMask, srcPaddingMask]
 *
 *   x              shape: (B, T, dModel)   target sequence
 *   memory         shape: (B, S, dModel)   encoder output
 *   tgtPaddingMask shape: (B, T)           1 at pad positions in target
 *   srcPaddingMask shape: (B, S)           1 at pad positions in source
 *
 * Output: NDList with a single tensor of shape (B, T, dModel).
 */
public class BitDecoder extends AbstractBlock {

    private final BitDecoderBlock[] blocks;
    private final RMSNorm finalNorm;

    /**
     * @param nLayers   number of decoder blocks
     * @param dModel    model dimension
     * @param nHeads    attention heads
     * @param dFfn      FFN hidden dimension
     * @param ropeBase  RoPE base frequency
     * @param maxSeqLen maximum sequence length for RoPE table
     * @param eps       RMSNorm epsilon
     * @param quantEps  BitLinear quantization epsilon
     */
    public BitDecoder(
        int nLayers,
        int dModel,
        int nHeads,
        int dFfn,
        int ropeBase,
        int maxSeqLen,
        float eps,
        float quantEps
    ) {
        this.blocks = new BitDecoderBlock[nLayers];
        for (int i = 0; i < nLayers; i++) {
            blocks[i] = addChildBlock(
                "block" + i,
                new BitDecoderBlock(
                    dModel,
                    nHeads,
                    dFfn,
                    ropeBase,
                    maxSeqLen,
                    eps,
                    quantEps
                )
            );
        }
        this.finalNorm = addChildBlock("finalNorm", new RMSNorm(dModel, eps));
    }

    @Override
    public void initializeChildBlocks(
        NDManager manager,
        DataType dataType,
        Shape... inputShapes
    ) {
        Shape xShape = inputShapes[0]; // (B, T, dModel)
        Shape memShape = inputShapes[1]; // (B, S, dModel)
        for (BitDecoderBlock block : blocks) {
            block.initialize(manager, dataType, xShape, memShape);
        }
        finalNorm.initialize(manager, dataType, xShape);
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

        for (BitDecoderBlock block : blocks) {
            x = block
                .forward(
                    ps,
                    buildBlockInputs(x, memory, tgtMask, srcMask),
                    training
                )
                .singletonOrThrow();
        }

        return new NDList(
            finalNorm.forward(ps, new NDList(x), training).singletonOrThrow()
        );
    }

    private static NDList buildBlockInputs(
        NDArray x,
        NDArray memory,
        NDArray tgtMask,
        NDArray srcMask
    ) {
        if (srcMask != null) return new NDList(x, memory, tgtMask, srcMask);
        if (tgtMask != null) return new NDList(x, memory, tgtMask);
        return new NDList(x, memory);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] { inputShapes[0] };
    }

    @Override
    public String toString() {
        return String.format(
            "BitDecoder(nLayers=%d, %s)",
            blocks.length,
            finalNorm
        );
    }
}
