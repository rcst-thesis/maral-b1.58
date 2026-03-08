package com.rcst;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import org.yaml.snakeyaml.Yaml;

/**
 * Loads model-config.yaml and exposes all hyperparameters as typed, final fields.
 *
 * Default lookup: src/main/resources/model-config.yaml
 * Override via:   ModelConfig.load(Path)  — useful in tests.
 *
 * Singleton — call ModelConfig.get() everywhere; the file is parsed once.
 */
public final class ModelConfig {

    // ── Tokenizer ─────────────────────────────────────────────────────────────
    public final String tokenizerModelPath;
    public final int vocabSize;
    public final int padId;
    public final int unkId;
    public final int bosId;
    public final int eosId;

    // ── Architecture ──────────────────────────────────────────────────────────
    public final int dModel;
    public final int nHeads;
    public final int nEncoderLayers;
    public final int nDecoderLayers;
    public final int dFfn;
    public final int maxSeqLen;
    public final float dropout;
    public final float eps;
    public final int ropeBase;

    // ── Quantization ──────────────────────────────────────────────────────────
    public final float quantEps;

    // ── Training ──────────────────────────────────────────────────────────────
    public final int batchSize;
    public final int blockSize;
    public final double trainRatio;
    public final long seed;
    public final int maxEpochs;
    public final float learningRate;
    public final float weightDecay;
    public final int warmupSteps;
    public final float gradClip;

    // ── Checkpoint ────────────────────────────────────────────────────────────
    /** Root directory where epoch sub-folders are written. */
    public final String checkpointDir;
    /** Write a checkpoint every N epochs (1 = every epoch). */
    public final int saveEveryNEpochs;
    /** Keep only the N most recent checkpoints; older ones are deleted. */
    public final int keepLastN;
    /**
     * If non-empty, resume training from this checkpoint directory
     * (e.g. "checkpoints/epoch-005").  Empty string means start fresh.
     */
    public final String resumeFrom;

    // ── Data ──────────────────────────────────────────────────────────────────
    public final String corpusPath;
    public final String parallelPath;
    public final String sourceLang;
    public final String targetLang;

    // ── Singleton ─────────────────────────────────────────────────────────────

    private static final String DEFAULT_PATH =
        "src/main/resources/model-config.yaml";

    private static volatile ModelConfig instance;

    public static ModelConfig get() {
        if (instance == null) {
            synchronized (ModelConfig.class) {
                if (instance == null) {
                    try {
                        instance = load(Paths.get(DEFAULT_PATH));
                    } catch (IOException e) {
                        throw new RuntimeException(
                            "Cannot load model-config.yaml from: " +
                                DEFAULT_PATH,
                            e
                        );
                    }
                }
            }
        }
        return instance;
    }

    public static ModelConfig load(Path path) throws IOException {
        try (InputStream is = Files.newInputStream(path)) {
            return new ModelConfig(new Yaml().load(is));
        }
    }

    // ── Constructor ───────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private ModelConfig(Map<String, Object> root) {
        Map<String, Object> tok = (Map<String, Object>) root.get("tokenizer");
        Map<String, Object> arch = (Map<String, Object>) root.get(
            "architecture"
        );
        Map<String, Object> quant = (Map<String, Object>) root.get(
            "quantization"
        );
        Map<String, Object> train = (Map<String, Object>) root.get("training");
        Map<String, Object> ckpt = (Map<String, Object>) root.get("checkpoint");
        Map<String, Object> data = (Map<String, Object>) root.get("data");

        tokenizerModelPath = (String) tok.get("model_path");
        vocabSize = (int) tok.get("vocab_size");
        padId = (int) tok.get("pad_id");
        unkId = (int) tok.get("unk_id");
        bosId = (int) tok.get("bos_id");
        eosId = (int) tok.get("eos_id");

        dModel = (int) arch.get("d_model");
        nHeads = (int) arch.get("n_heads");
        nEncoderLayers = (int) arch.get("n_encoder_layers");
        nDecoderLayers = (int) arch.get("n_decoder_layers");
        dFfn = (int) arch.get("d_ffn");
        maxSeqLen = (int) arch.get("max_seq_len");
        dropout = ((Number) arch.get("dropout")).floatValue();
        eps = ((Number) arch.get("eps")).floatValue();
        ropeBase = (int) arch.get("rope_base");

        quantEps = ((Number) quant.get("eps")).floatValue();

        batchSize = (int) train.get("batch_size");
        blockSize = (int) train.get("block_size");
        trainRatio = ((Number) train.get("train_ratio")).doubleValue();
        seed = ((Number) train.get("seed")).longValue();
        maxEpochs = (int) train.get("max_epochs");
        learningRate = ((Number) train.get("learning_rate")).floatValue();
        weightDecay = ((Number) train.get("weight_decay")).floatValue();
        warmupSteps = (int) train.get("warmup_steps");
        gradClip = ((Number) train.get("grad_clip")).floatValue();

        checkpointDir = (String) ckpt.get("dir");
        saveEveryNEpochs = (int) ckpt.get("save_every_n_epochs");
        keepLastN = (int) ckpt.get("keep_last_n");
        String rf = (String) ckpt.get("resume_from");
        resumeFrom = (rf == null) ? "" : rf.trim();

        corpusPath = (String) data.get("corpus");
        parallelPath = (String) data.get("parallel");
        sourceLang = (String) data.get("source_lang");
        targetLang = (String) data.get("target_lang");
    }

    @Override
    public String toString() {
        return String.format(
            "ModelConfig{name=maral-b1.58-125m, vocab=%d, dModel=%d, " +
                "nHeads=%d, enc=%d, dec=%d, maxSeq=%d, ckptDir=%s}",
            vocabSize,
            dModel,
            nHeads,
            nEncoderLayers,
            nDecoderLayers,
            maxSeqLen,
            checkpointDir
        );
    }
}
