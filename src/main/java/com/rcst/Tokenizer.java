package com.rcst;

import com.sentencepiece.Model;
import com.sentencepiece.Scoring;
import com.sentencepiece.SentencePieceAlgorithm;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Tokenizer implements AutoCloseable {

    private static final int BOS_ID = 2;
    private static final int EOS_ID = 3;
    private static final String DEFAULT_MODEL =
        "src/main/resources/models/maral.bpe.model";

    private final Model model;
    private final SentencePieceAlgorithm algorithm;

    public Tokenizer() throws IOException {
        this(Paths.get(DEFAULT_MODEL));
    }

    // Override for different model name
    public Tokenizer(Path modelPath) throws IOException {
        this.model = Model.parseFrom(modelPath);
        this.algorithm = new SentencePieceAlgorithm(
            true,
            Scoring.HIGHEST_SCORE
        );
    }

    public List<Integer> encode(String text) {
        return model.encodeNormalized(text, algorithm);
    }

    public List<Integer> encodeWithBosEos(String text) {
        List<Integer> ids = new java.util.ArrayList<>();
        ids.add(BOS_ID);
        ids.addAll(model.encodeNormalized(text, algorithm));
        ids.add(EOS_ID);
        return ids;
    }

    public String decode(List<Integer> ids) {
        return model.decodeSmart(ids);
    }

    public List<List<Integer>> encodeBatch(List<String> sentences) {
        return sentences
            .stream()
            .map(this::encode)
            .collect(java.util.stream.Collectors.toList());
    }

    @Override
    public void close() {
        // nothing to see here bro
    }
}
