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
 * BitNet b1.58 Encoder.
 *
 * A stack of N identical BitEncoderBlocks followed by a final RMSNorm.
 * The trailing norm stabilises the representation handed to the decoder's
 * cross-attention and is standard practice in Pre-LN transformers.
 *
 * NDList input contract:
 *   [x]                  — no padding mask
 *   [x, keyPaddingMask]  — shape (B, T), 1 at pad positions
 *
 * Output: NDList with a single tensor of shape (B, T, dModel).
 */
public class BitEncoder extends AbstractBlock {

    private final BitEncoderBlock[] blocks;
    private final RMSNorm finalNorm;

    /**
     * @param nLayers   number of encoder blocks
     * @param dModel    model dimension
     * @param nHeads    attention heads
     * @param dFfn      FFN hidden dimension
     * @param ropeBase  RoPE base frequency
     * @param maxSeqLen maximum sequence length for RoPE table
     * @param eps       RMSNorm epsilon
     * @param quantEps  BitLinear quantization epsilon
     */
    public BitEncoder(
        int nLayers,
        int dModel,
        int nHeads,
        int dFfn,
        int ropeBase,
        int maxSeqLen,
        float eps,
        float quantEps
    ) {
        this.blocks = new BitEncoderBlock[nLayers];
        for (int i = 0; i < nLayers; i++) {
            blocks[i] = addChildBlock(
                "block" + i,
                new BitEncoderBlock(
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
        Shape xShape = inputShapes[0];
        for (BitEncoderBlock block : blocks) {
            block.initialize(manager, dataType, xShape);
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

        // Pass the optional padding mask through every block unchanged
        NDList blockInputs = (inputs.size() > 1)
            ? new NDList(x, inputs.get(1))
            : new NDList(x);

        for (BitEncoderBlock block : blocks) {
            blockInputs = block.forward(ps, blockInputs, training);
            // Carry the mask forward with the updated x
            if (inputs.size() > 1) {
                blockInputs = new NDList(blockInputs.get(0), inputs.get(1));
            }
        }

        x = finalNorm
            .forward(ps, new NDList(blockInputs.get(0)), training)
            .singletonOrThrow();

        return new NDList(x);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] { inputShapes[0] };
    }

    @Override
    public String toString() {
        return String.format(
            "BitEncoder(nLayers=%d, %s)",
            blocks.length,
            finalNorm
        );
    }
}
