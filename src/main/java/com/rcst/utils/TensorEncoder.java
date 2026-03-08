package com.rcst.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import java.util.List;

public class TensorEncoder {

    private final Tokenizer tokenizer;
    private final NDManager manager;

    public TensorEncoder(Tokenizer tokenizer, NDManager manager) {
        this.tokenizer = tokenizer;
        this.manager = manager;
    }

    public NDArray encode(String text) {
        List<Integer> ids = tokenizer.encodeWithBosEos(text);
        long[] arr = ids.stream().mapToLong(Integer::longValue).toArray();
        return manager.create(arr);
    }

    // Single sentence, 2-D tensor shape: (1, seq_len)
    // --> [tok1, tok2, ...]    // shape: [1][seq_len]
    public NDArray encodeBatched(String text) {
        return encode(text).reshape(1, -1);
    }

    // Batch of sentences, array of 1-D tensors, one per sentence
    // [0] [t1, t2, ...]    // shape: [len0]
    // [1] [t1, t2, ...]    // shape: [len1]
    // [2] [t1, t2, ...]    // shape: [len2]
    public NDArray[] encodeBatch(List<String> sentences) {
        return sentences.stream().map(this::encode).toArray(NDArray[]::new);
    }
}
