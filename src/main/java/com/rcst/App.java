package com.rcst;

import com.sentencepiece.Model;
import com.sentencepiece.Scoring;
import com.sentencepiece.SentencePieceAlgorithm;
import java.nio.file.Paths;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App {

    public static void main(String[] args) throws Exception {
        Model model = Model.parseFrom(
            Paths.get("src/main/resources/models/maral.bpe.model")
        );
        SentencePieceAlgorithm algorithm = new SentencePieceAlgorithm(
            true,
            Scoring.HIGHEST_SCORE
        );

        String raw = "Good morning, how are you?";
        List<Integer> ids = model.encodeNormalized(raw, algorithm);
        System.out.println("Token IDs: " + ids);
        System.out.println("Decoded text: " + model.decodeSmart(ids));
    }
}
