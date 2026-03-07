package com.rcst;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class TsvLoader {

    private final int textColumnIndex;
    private final boolean hasHeader;

    public TsvLoader(int textColumnIndex, boolean hasHeader) {
        this.textColumnIndex = textColumnIndex;
        this.hasHeader = hasHeader;
    }

    public List<String> load(Path tsvPath) throws IOException {
        List<String> sentences = new ArrayList<>();
        boolean firstLine = true;

        try (
            BufferedReader br = Files.newBufferedReader(
                tsvPath,
                StandardCharsets.UTF_8
            )
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                if (firstLine && hasHeader) {
                    firstLine = false;
                    continue;
                }
                firstLine = false;

                String[] cols = line.split("\t", -1);
                if (textColumnIndex >= cols.length) continue;

                String text = cols[textColumnIndex].trim().replaceAll(
                    "\\s+",
                    " "
                );
                if (!text.isEmpty()) {
                    sentences.add(text);
                }
            }
        }

        return sentences;
    }
}
