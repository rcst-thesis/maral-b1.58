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
 * BitLinear — core building block of BitNet b1.58.
 *
 * Replaces nn.Linear for all projection layers in the transformer.
 * No bias term (consistent with BitNet b1.58 spec).
 *
 * Forward pass (Ma et al. 2024, "The Era of 1-bit LLMs"):
 *
 *   1. LayerNorm  (internal, non-parametric RMS norm):
 *        x̂ = x / RMS(x)
 *        Stabilises activation magnitudes before quantization so that
 *        a single large value does not dominate the absmax scale.
 *
 *   2. Weight quantization (absmean):
 *        β   = mean(|W|)
 *        W̃   = RoundClip(W / (β + ε), -1, 1)   ∈ {-1, 0, +1}
 *
 *   3. Activation quantization (absmax per-token, 8-bit):
 *        γ_i = max(|x̂_i|)  for each token row i
 *        x̃_i = Clip(round(x̂_i · 127 / γ_i), -128, 127)
 *
 *   4. Linear projection:
 *        y = x̃ @ W̃ᵀ
 *
 *   5. De-quantize:
 *        y_out = y · (β · γ_i / 127)   per token row
 *
 * Input  shape: (..., inFeatures)
 * Output shape: (..., outFeatures)
 */
public class BitLinear extends AbstractBlock {

    private static final float DEFAULT_EPS = 1e-8f;

    private final int inFeatures;
    private final int outFeatures;
    private final float eps;
    private final Parameter weight;

    public BitLinear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, DEFAULT_EPS);
    }

    public BitLinear(int inFeatures, int outFeatures, float eps) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.eps = eps;
        this.weight = addParameter(
            Parameter.builder()
                .setName("weight")
                .setType(Parameter.Type.WEIGHT)
                .optShape(new Shape(outFeatures, inFeatures))
                .build()
        );
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray x = inputs.singletonOrThrow();
        NDManager mgr = x.getManager();
        NDArray w = parameterStore.getValue(weight, mgr.getDevice(), training);

        NDArray yOut;
        try (NDManager wScope = mgr.newSubManager()) {
            // Move both x and w into wScope so ALL intermediates are freed
            NDArray xScoped = x.duplicate();
            xScoped.attach(wScope);

            NDArray wScoped = w.duplicate();
            wScoped.attach(wScope);

            // 1. Internal RMSNorm (non-parametric)
            NDArray xNorm = rmsNorm(xScoped, wScope);

            // 2. Weight quantization (absmean → ternary)
            NDArray beta = wScoped.abs().mean();
            NDArray wTilde = wScoped.div(beta.add(eps)).round().clip(-1, 1);

            // 3. Activation quantization (absmax per-token, 8-bit)
            int lastDim = inFeatures;
            long tokens = xNorm.getShape().size() / lastDim;

            NDArray xFlat = xNorm.reshape(tokens, lastDim);
            NDArray gamma = xFlat.abs().max(new int[] { 1 }, true);
            NDArray xQ = xFlat
                .mul(127f)
                .div(gamma.add(eps))
                .round()
                .clip(-128, 127);

            // 4. Linear projection
            NDArray y = xQ.matMul(wTilde.transpose());

            // 5. De-quantize
            NDArray deqScale = gamma.mul(beta).div(127f);
            NDArray yFlat = y.mul(deqScale);

            // 6. Restore shape
            long[] outShape = xScoped.getShape().getShape().clone();
            outShape[outShape.length - 1] = outFeatures;
            NDArray yReshaped = yFlat.reshape(outShape);

            // Move result back to parent manager before scope closes
            yOut = yReshaped.duplicate();
            yOut.attach(mgr);

            // Explicitly close scoped arrays to free memory immediately
            xScoped.close();
            wScoped.close();
        }
        // wScope closes → all intermediates freed

        return new NDList(yOut);
    }

    private NDArray rmsNorm(NDArray x, NDManager scope) {
        int lastAxis = x.getShape().dimension() - 1;
        NDArray rms = x
            .square()
            .mean(new int[] { lastAxis }, true)
            .add(eps)
            .sqrt();
        rms.attach(scope);

        NDArray result = x.div(rms);
        result.attach(scope);
        return result;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        long[] in = inputShapes[0].getShape();
        long[] out = in.clone();
        out[out.length - 1] = outFeatures;
        return new Shape[] { new Shape(out) };
    }

    @Override
    public String toString() {
        return String.format("BitLinear(%d → %d)", inFeatures, outFeatures);
    }
}
