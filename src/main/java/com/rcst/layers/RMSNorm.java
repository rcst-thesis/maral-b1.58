package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Root Mean Square Layer Normalization (RMSNorm).
 *
 * Used in BitNet b1.58 in place of LayerNorm — no mean subtraction,
 * no bias term, just RMS scaling with a learnable gain γ.
 *
 * Formula (Zhang & Sennrich, 2019):
 *   RMS(x)       = sqrt( mean(x²) + ε )
 *   RMSNorm(x)   = (x / RMS(x)) * γ
 *
 * γ is a per-feature learnable scale vector, initialised to ones.
 * ε is a small constant for numerical stability (default 1e-6).
 *
 * Input  shape: (..., dModel)
 * Output shape: (..., dModel)   — same shape as input
 */
public class RMSNorm extends AbstractBlock {

    private static final float DEFAULT_EPS = 1e-6f;

    private final int dModel;
    private final float eps;

    // Learnable scale γ — shape: [dModel], initialised to 1
    private final Parameter gamma;

    public RMSNorm(int dModel) {
        this(dModel, DEFAULT_EPS);
    }

    public RMSNorm(int dModel, float eps) {
        this.dModel = dModel;
        this.eps = eps;
        this.gamma = addParameter(
            Parameter.builder()
                .setName("gamma")
                .setType(Parameter.Type.GAMMA)
                .optShape(new Shape(dModel))
                .build()
        );
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray x = inputs.singletonOrThrow();
        NDManager mgr = x.getManager();
        NDArray g = parameterStore.getValue(gamma, mgr.getDevice(), training);

        // RMS(x) = sqrt( mean(x²) + ε )  — reduce over last dimension only
        int lastAxis = x.getShape().dimension() - 1;
        NDArray rms = x
            .square()
            .mean(new int[] { lastAxis }, true) // keep dim for broadcast
            .add(eps)
            .sqrt(); // (..., 1)

        // Normalise then scale by γ
        NDArray normed = x.div(rms).mul(g); // (..., dModel)

        return new NDList(normed);
    }

    // ── Shape inference ───────────────────────────────────────────────────────

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] { inputShapes[0] }; // same shape as input
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    @Override
    public String toString() {
        return String.format("RMSNorm(%d, eps=%.0e)", dModel, eps);
    }
}
