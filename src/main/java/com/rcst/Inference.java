package com.rcst;

import ai.djl.MalformedModelException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Greedy inference for maral-b1.58 using the Model class.
 *
 * Checkpoint format follows Model.save() layout:
 *   checkpoints/best/
 *     src-embed-0000.params
 *     tgt-embed-0000.params
 *     encoder-0-0000.params, encoder-1-0000.params, ...
 *     encoder-norm-0000.params
 *     decoder-0-0000.params, decoder-1-0000.params, ...
 *     decoder-norm-0000.params
 *     out-proj-0000.params
 *     training-state.txt
 */
public class Inference implements AutoCloseable {

    private final Model model;

    /**
     * @param checkpointDir path to a checkpoint directory, e.g. "checkpoints/best"
     */
    public Inference(String checkpointDir)
        throws IOException, MalformedModelException {
        this.model = new Model();

        Path ckpt = Paths.get(checkpointDir);
        model.load(ckpt);

        System.out.printf("Loaded checkpoint: %s%n", ckpt.toAbsolutePath());
    }

    /**
     * Translate a single English sentence to Tagalog/Hiligaynon.
     *
     * @param sourceText the input sentence in English
     * @return the decoded translation
     */
    public String translate(String sourceText) {
        return model.greedyTranslate(sourceText, model.getConfig().maxSeqLen);
    }

    /**
     * Translate with a custom max output length.
     *
     * @param sourceText the input sentence in English
     * @param maxLen     maximum number of target tokens to generate
     * @return the decoded translation
     */
    public String translate(String sourceText, int maxLen) {
        return model.greedyTranslate(sourceText, maxLen);
    }

    @Override
    public void close() throws Exception {
        model.close();
    }

    /**
     * Reload weights from a different checkpoint without restarting.
     * Used by the :ckpt command in the interactive REPL.
     */
    public void loadCheckpointInPlace(String checkpointDir)
        throws IOException, MalformedModelException {
        model.load(Paths.get(checkpointDir));
        System.out.printf("Switched to checkpoint: %s%n", checkpointDir);
    }

    /**
     * Interactive translation loop.
     *
     * Usage:
     *   just infer                          # uses checkpoints/best
     *   just infer checkpoints/epoch-028    # specific checkpoint
     *
     * Commands at the prompt:
     *   <any text>   → translate and print result
     *   :ckpt <path> → switch to a different checkpoint mid-session
     *   :quit / :q   → exit
     */
    public static void main(String[] args) throws Exception {
        String ckptDir = args.length > 0 ? args[0] : "checkpoints/best";

        System.out.println("═══════════════════════════════════════════════");
        System.out.println("  maral-b1.58  —  interactive translation");
        System.out.println("  checkpoint : " + ckptDir);
        System.out.println("  commands   : :ckpt <path>  :quit / :q");
        System.out.println("═══════════════════════════════════════════════");

        try (
            Inference inf = new Inference(ckptDir);
            java.io.BufferedReader console = new java.io.BufferedReader(
                new java.io.InputStreamReader(System.in)
            )
        ) {
            while (true) {
                System.out.print("\nen> ");
                System.out.flush();

                String line = console.readLine();
                if (line == null) break; // EOF / Ctrl-D
                line = line.trim();
                if (line.isEmpty()) continue;

                // ── Commands ─────────────────────────────────────────────
                if (":quit".equals(line) || ":q".equals(line)) {
                    System.out.println("bye.");
                    break;
                }

                if (line.startsWith(":ckpt ")) {
                    String newCkpt = line.substring(6).trim();
                    System.out.println("switching checkpoint → " + newCkpt);
                    inf.loadCheckpointInPlace(newCkpt);
                    System.out.println("loaded.");
                    continue;
                }

                // ── Translate ─────────────────────────────────────────────
                long t0 = System.currentTimeMillis();
                String result = inf.translate(line);
                long ms = System.currentTimeMillis() - t0;

                System.out.println("tl> " + result);
                System.out.printf("    (%.2f s)%n", ms / 1000.0);
            }
        }
    }
}
