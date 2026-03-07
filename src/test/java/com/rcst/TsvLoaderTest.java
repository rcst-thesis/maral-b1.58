package com.rcst;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import junit.framework.TestCase;

public class TsvLoaderTest extends TestCase {

    private Path tsvPath() {
        URL url = getClass().getClassLoader().getResource("data/en-tl.tsv");
        assertNotNull("data/en-tl.tsv not found in resources", url);
        return Paths.get(url.getPath());
    }

    // 10 rows × 2 columns = 20 sentences
    public void testRowCount() throws IOException {
        List<String> sentences = new TsvLoader().load(tsvPath());
        assertEquals(20, sentences.size());
    }

    // cols interleaved per row: even index = en, odd index = tl
    public void testEnFirstRow() throws IOException {
        List<String> sentences = new TsvLoader().load(tsvPath());
        assertEquals("Good morning, how are you?", sentences.get(0));
    }

    public void testTlFirstRow() throws IOException {
        List<String> sentences = new TsvLoader().load(tsvPath());
        assertEquals("Magandang umaga, kumusta ka?", sentences.get(1));
    }

    public void testEnLastRow() throws IOException {
        List<String> sentences = new TsvLoader().load(tsvPath());
        assertEquals(
            "We will leave early tomorrow morning.",
            sentences.get(18)
        );
    }

    public void testTlLastRow() throws IOException {
        List<String> sentences = new TsvLoader().load(tsvPath());
        assertEquals(
            "Aalis tayo nang maaga bukas ng umaga.",
            sentences.get(19)
        );
    }

    public void testNoBlankSentences() throws IOException {
        List<String> sentences = new TsvLoader().load(tsvPath());
        for (String s : sentences) {
            assertFalse("Blank sentence found", s.isBlank());
        }
    }
}
