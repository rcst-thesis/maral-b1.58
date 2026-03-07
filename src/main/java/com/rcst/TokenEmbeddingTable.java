package com.rcst;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * BitNet b1.58 Token Embedding Table
 *
 * Per the b1.58 spec, the embedding table is kept in FULL PRECISION (BF16/FP16).
 * Ternary quantization applies only to BitLinear (nn.Linear replacement) layers —
 * NOT to the embedding lookup. The memory proportion of the embedding shrinks
 * relative to total model size as the model scales, so keeping it full precision
 * is both correct and intentional.
 *
 * Weight matrix shape : (vocab_size, n_embed)
 * Output shape        : (batch_size, block_size, n_embed)
 */
public class TokenEmbeddingTable extends AbstractBlock {

    private final int nEmbed;
    private final Parameter weight;

    public TokenEmbeddingTable(int vocabSize, int nEmbed) {
        this.nEmbed = nEmbed;
        this.weight = addParameter(
            Parameter.builder()
                .setName("weight")
                .setType(Parameter.Type.WEIGHT)
                .optShape(new Shape(vocabSize, nEmbed))
                .build()
        );
    }

    /**
     * Forward pass:
     *   Index-select rows from the full-precision embedding matrix by token ids.
     *   No quantization is applied here — the embedding table stays full precision.
     *
     * input  shape: (batch_size, block_size)         — token ids (int64)
     * output shape: (batch_size, block_size, n_embed) — embedding vectors
     */
    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray x = inputs.singletonOrThrow();
        NDManager manager = x.getManager();

        // Full-precision weight matrix — no ternarization
        NDArray w = parameterStore.getValue(
            weight,
            manager.getDevice(),
            training
        );

        // Row-index into the embedding table by token id
        NDArray embedded = w.get(x.toType(DataType.INT64, false)); // (B, T, n_embed)

        return new NDList(embedded);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape in = inputShapes[0]; // (B, T)
        return new Shape[] { new Shape(in.get(0), in.get(1), nEmbed) };
    }
}
