package com.rcst;

import com.sentencepiece.Model;
import com.sentencepiece.Scoring;
import com.sentencepiece.SentencePieceAlgorithm;
import java.nio.file.Paths;
import java.util.List;
import junit.framework.TestCase;

// If case-preserving is needed for round-trips then
// retrain with `normalization_rule_name=identity` in train_spm.py
public class SpmModelTest extends TestCase {

    private static final String MODEL_PATH =
        "src/main/resources/models/maral.bpe.model";

    private void assertRoundTrip(String raw) throws Exception {
        Model model = Model.parseFrom(Paths.get(MODEL_PATH));
        SentencePieceAlgorithm algorithm = new SentencePieceAlgorithm(
            true,
            Scoring.HIGHEST_SCORE
        );

        List<Integer> ids = model.encodeNormalized(raw, algorithm);
        String decoded = model.decodeSmart(ids);
        System.out.println("Token IDs  : " + ids);
        System.out.println("Decoded    : " + decoded);
        assertEquals(raw.toLowerCase(), decoded.toLowerCase());
    }

    public void testEnglish() throws Exception {
        assertRoundTrip("Good morning, how are you?");
    }

    public void testTagalog() throws Exception {
        assertRoundTrip("Magandang umaga, kumusta ka?");
    }
}
