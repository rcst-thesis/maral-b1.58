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
 * Forward pass (Ma et al. 2024):
 *   1. Weight quantization (absmean):
 *        γ   = mean(|W|)
 *        W̃   = RoundClip(W / (γ + ε), -1, 1)   ∈ {-1, 0, +1}
 *   2. Activation quantization (absmax per-token, 8-bit):
 *        η_i = max(|x_i|)  for each token row i
 *        x̃_i = Clip(x_i · (127 / η_i), -128, 127)
 *   3. Linear projection:
 *        y = x̃ @ W̃ᵀ
 *   4. De-quantize:
 *        y_out = y · (γ · η_i / 127)   per token row
 *
 * ── Memory fix (DJL issue #2210) ──────────────────────────────────────
 * `parameterStore.getValue(weight, ...)` returns an NDArray owned by the
 * long-lived model manager. Every computation derived from that array
 * (w.abs(), w.div(), etc.) is also attached to the model manager and is
 * NEVER freed by the per-step sub-manager, causing VRAM to grow each epoch.
 *
 * Fix: inside forwardInternal, duplicate `w` into a short-lived scope
 * manager (`wScope`) so all weight-derived intermediates are freed at the
 * end of the forward pass. The final result `yOut` is re-attached to the
 * input's manager before the scope closes so it remains valid for the
 * caller.
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
        NDManager mgr = x.getManager(); // step sub-manager — correct device + lifetime

        // `w` is owned by the long-lived model manager.
        NDArray w = parameterStore.getValue(weight, mgr.getDevice(), training);

        // ── Open a scope so every w-derived intermediate is freed here ─────
        // yOut is re-attached to mgr before the scope closes.
        NDArray yOut;
        try (NDManager wScope = mgr.newSubManager()) {
            // Duplicate w into wScope — all subsequent ops on wCopy live here
            NDArray wCopy = w.duplicate();
            wCopy.attach(wScope);

            // ── 1. Weight quantization (absmean) ──────────────────────────
            NDArray gamma = wCopy.abs().mean(); // scalar γ
            float gScale = gamma.getFloat();
            NDArray wTilde = wCopy
                .div(gamma.add(eps)) // W / (γ + ε)
                .round() // nearest int
                .clip(-1, 1); // {-1, 0, +1}

            // ── 2. Activation quantization (absmax per-token, 8-bit) ───────
            int lastDim = (int) x.getShape().get(x.getShape().dimension() - 1);
            long tokens = x.getShape().size() / lastDim;

            NDArray xFlat = x.reshape(tokens, lastDim); // [T, in]
            NDArray eta = xFlat.abs().max(new int[] { 1 }, true); // [T, 1]
            NDArray xQ = xFlat
                .mul(127f)
                .div(eta.add(eps))
                .round()
                .clip(-128, 127); // [T, in]

            // ── 3. Linear projection ───────────────────────────────────────
            NDArray y = xQ.matMul(wTilde.transpose()); // [T, out]

            // ── 4. De-quantize ─────────────────────────────────────────────
            NDArray deqScale = eta.mul(gScale).div(127f); // [T, 1]
            NDArray yFlat = y.mul(deqScale); // [T, out]

            // Restore leading shape: (..., outFeatures)
            long[] inShape = x.getShape().getShape();
            long[] outShape = inShape.clone();
            outShape[outShape.length - 1] = outFeatures;
            yOut = yFlat.reshape(outShape);

            // Move result out of wScope before it closes
            yOut.attach(mgr);
        } // wScope closes → wCopy, gamma, wTilde, eta, xQ, y, deqScale freed

        return new NDList(yOut);
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
