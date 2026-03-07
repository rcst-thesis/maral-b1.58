package com.rcst;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class TsvLoader {

    /**
     * Loads a two-column TSV (source at col 0, target at col 1) and returns
     * all sentences in a single list, ready for SentencePiece training.
     */
    public List<String> load(Path tsvPath) throws IOException {
        List<String> sentences = new ArrayList<>();

        try (
            BufferedReader br = Files.newBufferedReader(
                tsvPath,
                StandardCharsets.UTF_8
            )
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] cols = line.split("\t", -1);
                if (cols.length < 2) continue;

                for (int i = 0; i < 2; i++) {
                    String text = cols[i].trim().replaceAll("\\s+", " ");
                    if (!text.isEmpty()) sentences.add(text);
                }
            }
        }

        return sentences;
    }
}
