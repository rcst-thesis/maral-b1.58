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
 *   1. Weight quantization (absmean):
 *        γ   = mean(|W|)
 *        W̃   = RoundClip(W / (γ + ε), -1, 1)   ∈ {-1, 0, +1}
 *
 *   2. Activation quantization (absmax per-token, 8-bit):
 *        η_i = max(|x_i|)  for each token row i
 *        x̃_i = Clip(x_i · (127 / η_i), -128, 127)
 *
 *   3. Linear projection:
 *        y = x̃ @ W̃ᵀ
 *
 *   4. De-quantize:
 *        y_out = y · (γ · η_i / 127)   per token row
 *
 * Gradient uses the Straight-Through Estimator (STE): quantization is
 * treated as identity during the backward pass by DJL autograd.
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

    /**
     * Weight quantization (absmean)
     *
     * W̃ = RoundClip(W / (γ + ε), -1, 1)
     * Returns a ternary {-1, 0, +1} matrix; γ returned via wScale[0].
     */
    private NDArray ternarize(NDArray w, float[] wScale) {
        NDArray gamma = w.abs().mean(); // scalar γ
        wScale[0] = gamma.getFloat();
        NDArray wTilde = w
            .div(gamma.add(eps)) // W / (γ + ε)
            .round() // nearest int
            .clip(-1, 1); // clamp → {-1,0,+1}
        return wTilde;
    }

    /**
     * Activation quantization (absmax per-token, 8-bit)
     * x̃_i = Clip(round(x_i · 127 / η_i), -128, 127)
     * η_i  = max(|x_i|) for each token row.
     * Returns quantized activations; per-token η values returned via etaOut.
     */
    private NDArray quantizeActivations(NDArray x, NDArray[] etaOut) {
        // x shape: [*, inFeatures] — treat all leading dims as tokens
        int lastDim = (int) x.getShape().get(x.getShape().dimension() - 1);
        long tokens = x.getShape().size() / lastDim;

        NDArray xFlat = x.reshape(tokens, lastDim); // [T, inFeatures]
        NDArray absX = xFlat.abs();
        NDArray eta = absX.max(new int[] { 1 }, true); // [T, 1] per-token max
        etaOut[0] = eta;

        NDArray xScaled = xFlat.mul(127f).div(eta.add(eps));
        NDArray xQ = xScaled.round().clip(-128, 127); // [T, inFeatures]
        return xQ;
    }

    // Forward
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

        // Quantize weights → W̃  (ternary {-1, 0, +1})
        float[] wScale = new float[1];
        NDArray wTilde = ternarize(w, wScale); // [outFeatures, inFeatures]

        // Quantize activations → x̃  (8-bit per token)
        NDArray[] etaOut = new NDArray[1];
        NDArray xQ = quantizeActivations(x, etaOut); // [T, inFeatures]
        NDArray eta = etaOut[0]; // [T, 1]

        // y = x̃ @ W̃ᵀ  → [T, outFeatures]
        NDArray y = xQ.matMul(wTilde.transpose());

        // De-quantize: y_out = y · (γ · η / 127)  per token
        NDArray deqScale = eta.mul(wScale[0]).div(127f); // [T, 1]
        NDArray yOut = y.mul(deqScale); // broadcast over outFeatures

        // Restore original leading shape: (..., outFeatures)
        long[] inShape = x.getShape().getShape();
        long[] outShape = inShape.clone();
        outShape[outShape.length - 1] = outFeatures;
        yOut = yOut.reshape(outShape);

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
