package com.rcst;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.rcst.layers.BitDecoder;
import com.rcst.layers.BitEncoder;
import com.rcst.layers.BitLinear;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Greedy inference for maral-b1.58.
 *
 * ── Checkpoint format ─────────────────────────────────────────────────────────
 * The checkpoints are DJL's native binary .params format — NOT PyTorch .pt.
 *
 * Layout on disk (same for both "best" and "epoch-NNN"):
 *
 *   checkpoints/best/
 *     src-embed-0000.params   ← TokenEmbeddingTable (source side)
 *     tgt-embed-0000.params   ← TokenEmbeddingTable (target side)
 *     encoder-0000.params     ← BitEncoder (all N blocks + final RMSNorm)
 *     decoder-0000.params     ← BitDecoder (all N blocks + final RMSNorm)
 *     out-proj-0000.params    ← BitLinear (dModel → vocabSize)
 *     training-state.txt      ← bestEpoch, globalStep, trainLoss, valLoss
 *
 * The .params files contain NDArray parameter values serialized by DJL's
 * NDArraySerializer. They are portable across machines but require the same
 * DJL version and architecture class hierarchy to deserialize.
 *
 * ── Can I load these in Python / PyTorch? ─────────────────────────────────────
 * Not directly. DJL .params != PyTorch .pt/.bin.
 * Options if Python interop is needed:
 *   1. Export to ONNX: DJL can trace the model to ONNX via OnnxTranslator.
 *      Then use onnxruntime-python for inference.
 *   2. Manual weight copy: iterate parameters, write to .npy, reload in Python.
 *   3. Stay in Java: this class is the production inference path.
 *
 * ── Greedy decode algorithm ───────────────────────────────────────────────────
 *
 *   1. Encode source text with the tokenizer.
 *   2. Run encoder → memory  (B=1, S, dModel)
 *   3. Start decoder with [BOS] token.
 *   4. At each step t:
 *        logits = outProj(decoder([generated_so_far], memory))
 *        next   = argmax(logits[0, t, :])
 *        append next to generated
 *   5. Stop when EOS is produced or maxSeqLen is reached.
 *   6. Decode token IDs back to text with the tokenizer.
 *
 * Usage:
 *   Inference inf = new Inference("checkpoints/best");
 *   String translation = inf.translate("Good morning, how are you?");
 *   System.out.println(translation);
 */
public class Inference implements AutoCloseable {

    private static final int BOS_ID = 2;
    private static final int EOS_ID = 3;

    private final ModelConfig cfg;
    private final NDManager manager;
    private final Tokenizer tokenizer;
    private final TokenEmbeddingTable srcEmbed;
    private final TokenEmbeddingTable tgtEmbed;
    private final BitEncoder encoder;
    private final BitDecoder decoder;
    private final BitLinear outProj;

    // Model wrappers kept open so their NDManagers stay alive for the
    // lifetime of this Inference instance. Closed in close().
    private Model mSrcEmbed, mTgtEmbed, mEncoder, mDecoder, mOutProj;

    /**
     * @param checkpointDir path to a checkpoint directory, e.g. "checkpoints/best"
     */
    public Inference(String checkpointDir)
        throws IOException, MalformedModelException {
        this.cfg = ModelConfig.get();
        this.manager = NDManager.newBaseManager();
        this.tokenizer = new Tokenizer();

        // Rebuild the exact same architecture that was trained
        Shape embedInput = new Shape(1, cfg.maxSeqLen);
        Shape seqShape = new Shape(1, cfg.maxSeqLen, cfg.dModel);

        srcEmbed = new TokenEmbeddingTable(cfg.vocabSize, cfg.dModel);
        tgtEmbed = new TokenEmbeddingTable(cfg.vocabSize, cfg.dModel);
        encoder = new BitEncoder(
            cfg.nEncoderLayers,
            cfg.dModel,
            cfg.nHeads,
            cfg.dFfn,
            cfg.ropeBase,
            cfg.maxSeqLen,
            cfg.eps,
            cfg.quantEps
        );
        decoder = new BitDecoder(
            cfg.nDecoderLayers,
            cfg.dModel,
            cfg.nHeads,
            cfg.dFfn,
            cfg.ropeBase,
            cfg.maxSeqLen,
            cfg.eps,
            cfg.quantEps
        );
        outProj = new BitLinear(cfg.dModel, cfg.vocabSize, cfg.quantEps);

        // Initialize shapes BEFORE loading weights — DJL requires this
        srcEmbed.initialize(manager, DataType.FLOAT32, embedInput);
        tgtEmbed.initialize(manager, DataType.FLOAT32, embedInput);
        encoder.initialize(manager, DataType.FLOAT32, seqShape);
        decoder.initialize(manager, DataType.FLOAT32, seqShape, seqShape);
        outProj.initialize(manager, DataType.FLOAT32, seqShape);

        // Load saved weights — Models are kept open so their NDManagers
        // do not free the parameters (closed in close()).
        Path ckpt = Paths.get(checkpointDir);
        mSrcEmbed = openBlock(srcEmbed, ckpt, "src-embed");
        mTgtEmbed = openBlock(tgtEmbed, ckpt, "tgt-embed");
        mEncoder = openBlock(encoder, ckpt, "encoder");
        mDecoder = openBlock(decoder, ckpt, "decoder");
        mOutProj = openBlock(outProj, ckpt, "out-proj");

        System.out.printf("Loaded checkpoint: %s%n", ckpt.toAbsolutePath());
    }

    /**
     * Translate a single English sentence to Tagalog/Hiligaynon.
     *
     * @param sourceText the input sentence in English
     * @return the decoded translation
     */
    public String translate(String sourceText) {
        return translate(sourceText, cfg.maxSeqLen);
    }

    /**
     * Translate with a custom max output length.
     *
     * @param sourceText the input sentence in English
     * @param maxLen     maximum number of target tokens to generate
     * @return the decoded translation
     */
    public String translate(String sourceText, int maxLen) {
        try (NDManager stepMgr = manager.newSubManager()) {
            ParameterStore ps = new ParameterStore(stepMgr, false);

            // ── 1. Encode source ──────────────────────────────────────────
            List<Integer> srcTokens = tokenizer.encode(sourceText.trim());
            long[] srcArr = new long[cfg.maxSeqLen];
            for (
                int i = 0;
                i < Math.min(srcTokens.size(), cfg.maxSeqLen);
                i++
            ) {
                srcArr[i] = srcTokens.get(i);
            }
            NDArray srcIds = stepMgr.create(srcArr).reshape(1, cfg.maxSeqLen);

            // ── 2. Run encoder ────────────────────────────────────────────
            NDArray srcEmb = srcEmbed
                .forward(ps, new NDList(srcIds), false)
                .singletonOrThrow();
            NDArray memory = encoder
                .forward(ps, new NDList(srcEmb), false)
                .singletonOrThrow();

            // ── 3. Greedy decode ──────────────────────────────────────────
            List<Integer> generated = new ArrayList<>();
            generated.add(BOS_ID);

            for (int step = 0; step < maxLen; step++) {
                // Build decoder input: generated tokens padded to maxSeqLen
                long[] tgtArr = new long[cfg.maxSeqLen];
                for (
                    int i = 0;
                    i < Math.min(generated.size(), cfg.maxSeqLen);
                    i++
                ) {
                    tgtArr[i] = generated.get(i);
                }
                NDArray tgtIds = stepMgr
                    .create(tgtArr)
                    .reshape(1, cfg.maxSeqLen);

                NDArray tgtEmb = tgtEmbed
                    .forward(ps, new NDList(tgtIds), false)
                    .singletonOrThrow();
                NDArray decoded = decoder
                    .forward(ps, new NDList(tgtEmb, memory), false)
                    .singletonOrThrow();
                NDArray logits = outProj
                    .forward(ps, new NDList(decoded), false)
                    .singletonOrThrow();

                // Argmax at the current position
                int pos = Math.min(step, cfg.maxSeqLen - 1);
                NDArray logitsAtPos = logits.get("0, " + pos + ", :"); // (vocabSize,)
                long nextToken = logitsAtPos.argMax().getLong();

                if (nextToken == EOS_ID) break;
                generated.add((int) nextToken);
            }

            // Remove the leading BOS token before decoding
            List<Integer> outputIds = generated.subList(1, generated.size());
            return tokenizer.decode(outputIds);
        }
    }

    @Override
    public void close() throws Exception {
        // Close Model wrappers first — this releases the per-model NDManagers
        // that own the loaded parameter NDArrays.
        if (mSrcEmbed != null) mSrcEmbed.close();
        if (mTgtEmbed != null) mTgtEmbed.close();
        if (mEncoder != null) mEncoder.close();
        if (mDecoder != null) mDecoder.close();
        if (mOutProj != null) mOutProj.close();
        tokenizer.close();
        manager.close();
    }

    // ── Hot-swap checkpoint ───────────────────────────────────────────────────

    /**
     * Reload weights from a different checkpoint without restarting.
     * Used by the :ckpt command in the interactive REPL.
     */
    public void loadCheckpointInPlace(String checkpointDir)
        throws IOException, MalformedModelException {
        // Close old Model wrappers before opening new ones to avoid leaking
        // the previous checkpoint's NDManagers.
        if (mSrcEmbed != null) mSrcEmbed.close();
        if (mTgtEmbed != null) mTgtEmbed.close();
        if (mEncoder != null) mEncoder.close();
        if (mDecoder != null) mDecoder.close();
        if (mOutProj != null) mOutProj.close();

        Path ckpt = Paths.get(checkpointDir);
        mSrcEmbed = openBlock(srcEmbed, ckpt, "src-embed");
        mTgtEmbed = openBlock(tgtEmbed, ckpt, "tgt-embed");
        mEncoder = openBlock(encoder, ckpt, "encoder");
        mDecoder = openBlock(decoder, ckpt, "decoder");
        mOutProj = openBlock(outProj, ckpt, "out-proj");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Open a checkpoint block and return the Model WITHOUT closing it.
     * The caller must keep the returned Model alive and close it when done —
     * closing it would free the NDManager that owns the loaded parameters.
     */
    private Model openBlock(
        ai.djl.nn.AbstractBlock block,
        Path dir,
        String name
    ) throws IOException, MalformedModelException {
        Model m = Model.newInstance(name);
        m.setBlock(block);
        m.load(dir, name);
        return m; // intentionally NOT closed here
    }

    // ── Main — interactive REPL ───────────────────────────────────────────────

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
