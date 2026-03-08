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
 * ── Memory management (DJL issue #2210) ──────────────────────────────────
 * parameterStore.getValue(weight, ...) returns an NDArray owned by the
 * long-lived model manager. Every intermediate derived from that array
 * (w.abs(), w.div(), ...) is also attached to the model manager and is
 * NEVER freed by the per-step sub-manager — VRAM grows each step.
 *
 * Fix: inside forwardInternal, all weight-derived work happens inside a
 * short-lived wScope sub-manager. The final output is re-attached to the
 * input's manager (mgr) before wScope closes, so it remains valid for
 * the caller. wScope then closes and all intermediates are freed.
 *
 * Input  shape: (..., inFeatures)
 * Output shape: (..., outFeatures)
 */
public class BitLinear extends AbstractBlock {

    private static final float DEFAULT_EPS = 1e-8f;

    private final int inFeatures;
    private final int outFeatures;
    private final float eps;

    // Weight matrix shape: [outFeatures × inFeatures]
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

        // w is owned by the long-lived model manager — do NOT operate on it directly.
        NDArray w = parameterStore.getValue(weight, mgr.getDevice(), training);

        NDArray yOut;
        try (NDManager wScope = mgr.newSubManager()) {
            // Internal LayerNorm (non-parametric RMS)
            // Normalise over the feature dimension so that absmax quantization
            // is not dominated by outlier activation values.
            NDArray xNorm = rmsNorm(x, wScope); // (..., inFeatures)

            // Weight quantization (absmean → ternary)
            // Duplicate w into wScope so all w-derived intermediates live here
            // and are freed when the scope closes.
            NDArray wCopy = w.duplicate();
            wCopy.attach(wScope);

            NDArray beta = wCopy.abs().mean(); // scalar β
            NDArray wTilde = wCopy.div(beta.add(eps)).round().clip(-1, 1); // W̃ ∈ {-1, 0, +1}

            // Activation quantization (absmax per-token, 8-bit)
            int lastDim = (int) xNorm
                .getShape()
                .get(xNorm.getShape().dimension() - 1);
            long tokens = xNorm.getShape().size() / lastDim;

            NDArray xFlat = xNorm.reshape(tokens, lastDim); // [T, inFeatures]
            NDArray gamma = xFlat.abs().max(new int[] { 1 }, true); // [T, 1]  per-token γ
            NDArray xQ = xFlat
                .mul(127f)
                .div(gamma.add(eps))
                .round()
                .clip(-128, 127); // [T, inFeatures]

            // Linear projection
            NDArray y = xQ.matMul(wTilde.transpose()); // [T, outFeatures]

            // De-quantize: y_out = y · (β · γ / 127)
            NDArray deqScale = gamma.mul(beta).div(127f); // [T, 1]
            NDArray yFlat = y.mul(deqScale); // [T, outFeatures]

            // Restore original leading shape: (..., outFeatures)
            long[] outShape = x.getShape().getShape().clone();
            outShape[outShape.length - 1] = outFeatures;
            yOut = yFlat.reshape(outShape);

            // Move result out of wScope before it closes
            yOut = yOut.duplicate();
            yOut.attach(mgr);
        }
        // wScope closes → wCopy, beta, wTilde, gamma, xQ, y, deqScale freed

        return new NDList(yOut);
    }

    /**
     * Non-parametric RMS normalisation over the last axis.
     * No learnable γ — this is purely for stabilising activations before
     * absmax quantization, not a substitute for the block-level RMSNorm.
     *
     * RMS(x) = sqrt( mean(x²) + ε )
     * x̂      = x / RMS(x)
     */
    private NDArray rmsNorm(NDArray x, NDManager scope) {
        int lastAxis = x.getShape().dimension() - 1;
        NDArray rms = x
            .square()
            .mean(new int[] { lastAxis }, true) // keep dim for broadcast
            .add(eps)
            .sqrt();

        // Attach intermediate to scope so it is freed with the scope
        rms.attach(scope);
        return x.div(rms);
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
