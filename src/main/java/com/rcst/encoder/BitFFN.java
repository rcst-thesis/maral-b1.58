package com.rcst.encoder;

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
 * BitNet b1.58 Feed-Forward Network (FFN).
 *
 * Two-layer MLP with squared ReLU activation (ReLU²), as used in the
 * BitNet b1.58 paper. Both linear projections use BitLinear.
 *
 *   gate(x) = ReLU(W_up · x)²
 *   FFN(x)  = W_down · gate(x)
 *
 * Pre-norm (RMSNorm) and the residual connection live in the enclosing
 * BitEncoderBlock / BitDecoderBlock, not here.
 *
 * Why ReLU²?
 * Squared ReLU (Primer, So et al. 2021) increases sparsity in the
 * intermediate activations relative to plain ReLU, which aligns well
 * with the ternary-weight regime: sparse activations reduce the effective
 * number of non-zero multiply-accumulate operations at inference time.
 *
 * Input  shape: (B, T, dModel)
 * Output shape: (B, T, dModel)
 */
public class BitFFN extends AbstractBlock {

    private final int dModel;
    private final int dFfn;
    private final BitLinear wUp;
    private final BitLinear wDown;

    /**
     * @param dModel   model dimension (input and output)
     * @param dFfn     hidden (intermediate) dimension
     * @param quantEps epsilon for BitLinear weight quantization
     */
    public BitFFN(int dModel, int dFfn, float quantEps) {
        this.dModel = dModel;
        this.dFfn = dFfn;
        this.wUp = addChildBlock("wUp", new BitLinear(dModel, dFfn, quantEps));
        this.wDown = addChildBlock(
            "wDown",
            new BitLinear(dFfn, dModel, quantEps)
        );
    }

    @Override
    public void initializeChildBlocks(
        NDManager manager,
        DataType dataType,
        Shape... inputShapes
    ) {
        Shape inShape = inputShapes[0]; // (B, T, dModel)
        wUp.initialize(manager, dataType, inShape);

        long[] hiddenDims = inShape.getShape().clone();
        hiddenDims[hiddenDims.length - 1] = wUp
            .getOutputShapes(new Shape[] { inShape })[0].get(
                inShape.dimension() - 1
            );
        wDown.initialize(manager, dataType, new Shape(hiddenDims)); // (B, T, dFfn)
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore ps,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray x = inputs.singletonOrThrow();

        // up-projection then squared ReLU
        NDArray h = wUp.forward(ps, new NDList(x), training).singletonOrThrow();
        NDArray activated = relu(h).square(); // ReLU²

        // down-projection back to dModel
        NDArray out = wDown
            .forward(ps, new NDList(activated), training)
            .singletonOrThrow();

        return new NDList(out);
    }

    /** Element-wise ReLU: max(0, x). */
    private static NDArray relu(NDArray x) {
        return x.maximum(0f);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] { inputShapes[0] };
    }

    @Override
    public String toString() {
        return String.format("BitFFN(dModel=%d, dFfn=%d)", dModel, dFfn);
    }
}
