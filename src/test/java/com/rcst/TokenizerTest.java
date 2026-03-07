package com.rcst;

import java.nio.file.Paths;
import java.util.List;
import junit.framework.TestCase;

public class TokenizerTest extends TestCase {

    private static final String MODEL_PATH =
        "src/main/resources/models/maral.bpe.model";

    private Tokenizer tokenizer;

    @Override
    protected void setUp() throws Exception {
        tokenizer = new Tokenizer(Paths.get(MODEL_PATH));
    }

    @Override
    protected void tearDown() throws Exception {
        tokenizer.close();
    }

    public void testEncodeReturnsIds() {
        List<Integer> ids = tokenizer.encode("Good morning");
        assertNotNull(ids);
        assertFalse(ids.isEmpty());
    }

    public void testEncodeWithBosEosHasBoundaryTokens() {
        List<Integer> ids = tokenizer.encodeWithBosEos("Good morning");
        assertEquals(2, (int) ids.get(0)); // BOS
        assertEquals(3, (int) ids.get(ids.size() - 1)); // EOS
    }

    public void testDecodeRoundTrip() {
        String raw = "magandang umaga";
        List<Integer> ids = tokenizer.encode(raw);
        assertEquals(raw, tokenizer.decode(ids).toLowerCase());
    }

    public void testEncodeBatchSize() {
        List<String> sentences = List.of("Hello", "Kumusta", "Good evening");
        List<List<Integer>> batch = tokenizer.encodeBatch(sentences);
        assertEquals(3, batch.size());
    }
}
